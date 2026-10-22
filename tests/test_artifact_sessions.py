# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Artifact downloads that start sooner: the sessions a client keeps between
downloads, its warm-up, both listings at once, the experiment run read once per
job, and logins without the Connector's cloud-metadata probes."""

import gzip
import importlib
import threading
import time

import pytest

from cortex_training import _artifact_download
from tests import test_client as _client_tests

# The client module as test_client loads it: _make_client's clients are of its class.
client_module = _client_tests.nc
_make_client = _client_tests._make_client
# The stage-SQL fake of the execution-log tests (bound to a name pytest won't collect).
_FakeArtifactConnection = _client_tests.TestExecutionLogDownload.FakeArtifactConnection

RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
RUN = {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN_ABC"}
SUB = "job-1:training:0"
KeptSessions = _artifact_download.KeptSessions
SharedConnection = _artifact_download.SharedConnection


class _Session:
    """A connector session that counts its closes."""

    def __init__(self, *, fail_close=False):
        self.closes = 0
        self._fail_close = fail_close

    def close(self):
        self.closes += 1
        if self._fail_close:
            raise RuntimeError("close failed")


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _chunks(count=3):
    """GPU metric chunks a minute apart for SUB: chunk ``i`` holds row ``"<SUB> i"``."""
    return {
        f"{SUB}/gpu.20260904-12{i:02d}00.c{i}.gz": gzip.compress(f"{SUB} {i}\n".encode())
        for i in range(count)
    }


def _staged(monkeypatch, files, root="_metrics"):
    """A client whose job's run stages ``files`` under ``root``; every session it logs in
    is appended to the list returned with it."""
    client = _make_client(get_json=RUN)
    sessions = []

    def log_in():
        sessions.append(_FakeArtifactConnection(
            {RUN_URI + root + "/": [(f"/versions/RUN_ABC/{root}/{path}",) for path in files]},
            {RUN_URI + f"{root}/{path}": content for path, content in files.items()},
        ))
        return sessions[-1]

    monkeypatch.setattr(client, "_open_experiment_artifact_connection", log_in)
    return client, sessions


# ─── KeptSessions ───────────────────────────────────────────────────────────


def test_a_kept_session_goes_to_one_taker():
    kept, session = KeptSessions(), _Session()
    assert kept.take("k") is None and not kept.has("k")
    kept.keep("k", session)
    assert kept.has("k")
    assert kept.take("k") is session
    assert kept.take("k") is None and not kept.has("k")  # the taker owns it now
    assert session.closes == 0


def test_two_sessions_are_kept_per_key_the_last_kept_taken_first_and_a_third_closed():
    kept = KeptSessions()
    first, second, third, elsewhere = _Session(), _Session(), _Session(), _Session()
    for session in (first, second, third):
        kept.keep("k", session)
    kept.keep("other", elsewhere)
    assert (first.closes, second.closes, third.closes, elsewhere.closes) == (0, 0, 1, 0)
    assert kept.take("k") is second and kept.take("k") is first and kept.take("k") is None
    assert kept.take("other") is elsewhere


def test_sessions_idle_too_long_make_room_for_a_fresh_one():
    clock = _Clock()
    kept = KeptSessions(idle_seconds=60, clock=clock)
    old = [_Session(), _Session()]
    for session in old:
        kept.keep("k", session)
    clock.now += 61
    fresh = _Session()
    kept.keep("k", fresh)
    assert [session.closes for session in old] == [1, 1] and fresh.closes == 0
    assert kept.take("k") is fresh and kept.take("k") is None


def test_a_session_idle_too_long_is_closed_instead_of_reused():
    clock = _Clock()
    kept, session = KeptSessions(idle_seconds=60, clock=clock), _Session()
    kept.keep("k", session)
    clock.now += 60
    assert kept.has("k")  # at the limit, still reused
    clock.now += 1
    assert not kept.has("k")
    assert kept.take("k") is None
    assert session.closes == 1


def test_closing_closes_the_kept_sessions_and_any_kept_after():
    kept = KeptSessions()
    failing, late = _Session(fail_close=True), _Session()
    kept.keep("k", failing)
    kept.close()  # a close that fails isn't raised: the session expires by itself
    assert failing.closes == 1 and kept.take("k") is None
    kept.keep("k", late)
    assert late.closes == 1 and kept.take("k") is None


# ─── SharedConnection with kept sessions ────────────────────────────────────


def test_a_download_starts_on_the_kept_session_and_gives_it_back_once_it_succeeds():
    kept, warm, opened = KeptSessions(), _Session(), []
    kept.keep("k", warm)
    shared = SharedConnection(lambda: opened.append(_Session()) or opened[-1], kept=kept, key="k")
    assert shared.current()[0] is warm and opened == []  # no login
    shared.close(keep=True)
    assert warm.closes == 0 and kept.take("k") is warm


def test_a_download_that_failed_closes_its_session():
    kept = KeptSessions()
    shared = SharedConnection(_Session, kept=kept, key="k")
    session = shared.current()[0]
    shared.close()  # what a download that failed, or was cancelled, does
    assert session.closes == 1 and kept.take("k") is None


def test_a_session_with_an_abandoned_attempt_is_closed_not_kept():
    kept = KeptSessions()
    shared = SharedConnection(_Session, kept=kept, key="k")
    session = shared.current()[0]
    shared.note_attempt_failure(_artifact_download.TransferTimeout("stalled"), retried=True)
    shared.close(keep=True)  # that attempt may still be running on it
    assert session.closes == 1 and kept.take("k") is None


def test_a_reopened_download_keeps_its_new_session_and_closes_the_old_one():
    kept, opened = KeptSessions(), []
    shared = SharedConnection(lambda: opened.append(_Session()) or opened[-1], kept=kept, key="k")
    _session, generation = shared.current()
    shared.reopen(generation)
    shared.close(keep=True)
    assert [session.closes for session in opened] == [1, 0]
    assert kept.take("k") is opened[1]


def test_without_kept_sessions_a_download_closes_its_session_as_before():
    shared = SharedConnection(_Session)
    session = shared.current()[0]
    shared.close(keep=True)
    assert session.closes == 1


def test_beside_runs_while_the_caller_works_and_hands_back_its_value_or_error():
    running, finish = threading.Event(), threading.Event()

    def work():
        running.set()
        assert finish.wait(5)
        return 7

    beside = _artifact_download.Beside(work, name="test-beside")
    assert running.wait(5)  # started before result() is asked for
    finish.set()
    assert beside.result() == 7

    def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _artifact_download.Beside(failing, name="test-beside").result()


# ─── The client ─────────────────────────────────────────────────────────────


def test_by_default_each_download_logs_in_and_closes_its_session(tmp_path, monkeypatch):
    client, sessions = _staged(monkeypatch, _chunks())
    assert not client.keep_artifact_sessions
    client.download_metrics("job-1", tmp_path, resume=True)
    client.download_metrics("job-1", tmp_path, resume=True)
    assert len(sessions) == 2 and all(session.closed for session in sessions)


def test_a_kept_session_saves_the_next_download_its_login(tmp_path, monkeypatch):
    client, sessions = _staged(monkeypatch, _chunks())
    client.keep_artifact_sessions = True
    client.download_metrics("job-1", tmp_path, resume=True)
    client.download_metrics("job-1", tmp_path / "again")
    assert len(sessions) == 1 and not sessions[0].closed  # one login for both
    assert (tmp_path / "again" / SUB / "gpu.jsonl").read_text() == "".join(f"{SUB} {i}\n" for i in range(3))
    client.close()
    assert sessions[0].closed


def test_downloads_that_ran_at_once_each_keep_their_session_for_the_next_two(tmp_path, monkeypatch):
    # A saved console still downloading while the metrics are asked for: each logs in, and
    # both sessions are kept, so the next two downloads log in neither.
    client, sessions = _staged(monkeypatch, _chunks())
    client.keep_artifact_sessions = True
    reached, release = threading.Event(), threading.Event()

    def held(_entry):
        reached.set()
        assert release.wait(5)

    first = threading.Thread(target=lambda: client.download_metrics("job-1", tmp_path / "a", on_commit=held))
    first.start()
    assert reached.wait(5)
    client.download_metrics("job-1", tmp_path / "b")  # while the first runs: its own login
    release.set()
    first.join(5)
    assert len(sessions) == 2 and not any(session.closed for session in sessions)
    both = [threading.Thread(target=lambda name=name: client.download_metrics("job-1", tmp_path / name))
            for name in ("c", "d")]
    for thread in both:
        thread.start()
    for thread in both:
        thread.join(5)
    assert len(sessions) == 2  # each started on a kept one
    client.close()
    assert all(session.closed for session in sessions)


def test_a_download_that_fails_leaves_no_session_kept(tmp_path, monkeypatch):
    client, sessions = _staged(monkeypatch, _chunks())
    client.keep_artifact_sessions = True

    def stop(_entry):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        client.download_metrics("job-1", tmp_path, on_commit=stop)
    assert sessions[0].closed
    client.download_metrics("job-1", tmp_path, resume=True)
    assert len(sessions) == 2  # a new login


def test_turning_kept_sessions_off_closes_them(tmp_path, monkeypatch):
    client, sessions = _staged(monkeypatch, _chunks())
    client.keep_artifact_sessions = True
    client.download_metrics("job-1", tmp_path)
    assert not sessions[0].closed
    client.keep_artifact_sessions = False
    assert sessions[0].closed and not client.keep_artifact_sessions


def test_a_for_job_client_shares_the_kept_sessions_under_its_own_schema():
    client = _make_client()
    client.keep_artifact_sessions = True
    child = client.for_job({"database_name": "OTHER_DB", "schema_name": "OTHER_SCH"})
    assert child.keep_artifact_sessions and child._kept_sessions is client._kept_sessions
    assert child._artifact_session_key() == ("OTHER_DB", "OTHER_SCH") != client._artifact_session_key()
    session = _Session()
    client._kept_sessions.keep(child._artifact_session_key(), session)
    child.close()  # it borrows them: closing it, or turning them off on it, closes nothing
    child.keep_artifact_sessions = False
    assert session.closes == 0 and client.keep_artifact_sessions
    client.close()
    assert session.closes == 1


def test_a_warm_up_reads_the_run_and_logs_in_once_for_the_download(tmp_path, monkeypatch):
    client, sessions = _staged(monkeypatch, _chunks())
    client.warm_artifact_session("job-1")  # sessions aren't kept: the run alone
    assert sessions == []
    client.keep_artifact_sessions = True
    client.warm_artifact_session("job-1")
    client.warm_artifact_session("job-1")  # one is kept already
    assert len(sessions) == 1 and not sessions[0].closed
    client.download_metrics("job-1", tmp_path, stale_after_seconds=None)  # as the TUI asks
    assert len(sessions) == 1  # the download started on it
    assert sessions[0].commands[0].startswith("LIST ")
    assert client._session.get.call_count == 1  # the experiment run, read once for all of them


def test_a_warm_up_that_cannot_log_in_raises_and_keeps_nothing(monkeypatch):
    client = _make_client(get_json=RUN)
    client.keep_artifact_sessions = True

    def refused():
        raise RuntimeError("login refused")

    monkeypatch.setattr(client, "_open_experiment_artifact_connection", refused)
    with pytest.raises(RuntimeError, match="login refused"):
        client.warm_artifact_session("job-1")
    assert not client._kept_sessions.has(client._artifact_session_key())


def test_a_job_s_experiment_run_is_read_once_and_a_failed_read_is_tried_again(monkeypatch):
    client = _make_client()
    answers = [RuntimeError("not yet"), {"experiment_name": "DB.SCH.EXP"}, RUN, {**RUN, "experiment_run_name": "R2"}]

    def read(job_id):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(client, "get_experiment_run", read)
    with pytest.raises(RuntimeError, match="not yet"):
        client._experiment_run_uri("job-1")
    with pytest.raises(ValueError, match="experiment_run_name"):
        client._experiment_run_uri("job-1")
    assert client._experiment_run_uri("job-1") == (RUN_URI, "RUN_ABC")
    assert client._experiment_run_uri("job-1") == (RUN_URI, "RUN_ABC")  # not read again
    assert client._experiment_run_uri("job-2") == ("snow://experiment/DB.SCH.EXP/versions/R2/", "R2")
    assert answers == []


def test_both_listings_of_a_download_run_at_once(tmp_path, monkeypatch):
    client, _sessions = _staged(monkeypatch, _chunks())
    started = {"_metrics": threading.Event(), "checkpoints/_metrics": threading.Event()}
    real = client_module.CortexTrainingClient._list_experiment_artifacts

    def listing(connection, run_uri, run_name, artifact_root):
        # Each waits for the other to start: one after the other, neither would.
        started[artifact_root].set()
        other = next(event for root, event in started.items() if root != artifact_root)
        assert other.wait(5), f"LIST {artifact_root} ran alone"
        return real(connection, run_uri, run_name, artifact_root)

    monkeypatch.setattr(client_module.CortexTrainingClient, "_list_experiment_artifacts", staticmethod(listing))
    client.download_metrics("job-1", tmp_path)
    assert all(event.is_set() for event in started.values())
    assert (tmp_path / SUB / "gpu.jsonl").read_text() == "".join(f"{SUB} {i}\n" for i in range(3))


@pytest.mark.parametrize("failing", ["_metrics", "checkpoints/_metrics"])
def test_a_listing_that_fails_fails_the_download(tmp_path, monkeypatch, failing):
    client, sessions = _staged(monkeypatch, _chunks())
    client.keep_artifact_sessions = True
    real = client_module.CortexTrainingClient._list_experiment_artifacts

    def listing(connection, run_uri, run_name, artifact_root):
        if artifact_root == failing:
            raise RuntimeError(f"LIST {artifact_root} denied")
        return real(connection, run_uri, run_name, artifact_root)

    monkeypatch.setattr(client_module.CortexTrainingClient, "_list_experiment_artifacts", staticmethod(listing))
    with pytest.raises(RuntimeError, match=f"LIST {failing} denied"):
        client.download_metrics("job-1", tmp_path)
    assert sessions[0].closed and not client._kept_sessions.has(client._artifact_session_key())
    assert not (tmp_path / SUB / "gpu.jsonl").exists()


def test_artifact_logins_skip_the_connector_s_cloud_metadata_probes(monkeypatch):
    configuration = importlib.import_module("snowflake.connector.connection").DEFAULT_CONFIGURATION
    assert client_module._artifact_login_options() == {"platform_detection_timeout_seconds": 0.0}
    monkeypatch.delitem(configuration, "platform_detection_timeout_seconds")
    assert client_module._artifact_login_options() == {}  # a Connector without the option: as it is


def test_a_kept_session_survives_neither_an_idle_half_hour_nor_a_closed_client(tmp_path, monkeypatch):
    client, sessions = _staged(monkeypatch, _chunks())
    client.keep_artifact_sessions = True
    clock = _Clock()
    client._kept_sessions._clock = clock
    client.download_metrics("job-1", tmp_path)
    clock.now += KeptSessions.IDLE_SECONDS + 1
    client.download_metrics("job-1", tmp_path / "later")
    assert len(sessions) == 2 and sessions[0].closed  # logged in again; the idle one closed
    client.close()
    assert sessions[1].closed
    client.download_metrics("job-1", tmp_path / "closed")
    assert len(sessions) == 3 and sessions[2].closed  # a closed client keeps none


def test_closing_a_client_waits_for_no_download_to_finish(tmp_path, monkeypatch):
    # A download still running when the client closes closes its own session when it ends.
    client, sessions = _staged(monkeypatch, _chunks())
    client.keep_artifact_sessions = True
    reached, release = threading.Event(), threading.Event()

    def commit(_entry):
        reached.set()
        assert release.wait(5)

    downloading = threading.Thread(target=lambda: client.download_metrics("job-1", tmp_path, on_commit=commit))
    downloading.start()
    assert reached.wait(5)
    started = time.monotonic()
    client.close()
    assert time.monotonic() - started < 1
    release.set()
    downloading.join(5)
    assert sessions[0].closed  # given back after close(): closed at once
