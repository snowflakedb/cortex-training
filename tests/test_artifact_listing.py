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

"""Listing one sub-job's chunks alone, and remembering which root holds a sub-job's
files (``CortexTrainingClient._list_gzip_chunks``).

A download that names one sub-job LISTs that sub-job's folder in each root rather than
each root whole, and once a listing of both roots has found the sub-job's files under
one, only that one. None of it may change what the caller sees, so most tests compare a
download with the same download by a client that lists each root whole and remembers
nothing, as every client did before.
"""

from __future__ import annotations

import enum
import gzip
import importlib
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import unquote
from urllib.parse import urlparse

import pytest

from tests import test_client as _client_tests

# The module under test, loaded as test_client.py loads it.
nc = _client_tests.nc
_make_client = _client_tests._make_client
# The stage-SQL fake of the execution-log tests (bound to a name pytest won't collect).
_FakeArtifactConnection = _client_tests.TestExecutionLogDownload.FakeArtifactConnection

RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
# A LIST row's name: the file's path from ``/versions/``.
ROW_PREFIX = "/versions/RUN_ABC/"
JOB = "job-1"
TRAINING = "job-1:training:0"
SAMPLING = "job-1:sampling:0"
SAMPLING_1 = "job-1:sampling:1"
# A sub-job of the job with no files at all.
NO_FILES = "job-1:sampling:2"
OTHER_JOB = "job-2:training:0"

# stream -> (download method, the root its chunks are under, chunk name prefix, file written)
STREAMS = {
    "stdout": ("download_stdout_logs", "_stdout", "console", "stdout.log"),
    "metrics": ("download_metrics", "_metrics", "gpu", "gpu.jsonl"),
    "training-metrics": ("download_training_metrics", "_metrics", "training", "training.jsonl"),
}
_ROOT_OF_KIND = {kind: root for _method, root, kind, _file in STREAMS.values()}

START = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
# The newest chunk of _three_sub_jobs(): the training sub-job's third, from 12:00:02.
NEWEST = (START + timedelta(seconds=2)).timestamp()


# ─── A run's stage ─────────────────────────────────────────────────────────


def _path(sub, kind, index, *, under="", tag="c", step=1):
    """The path of ``sub``'s chunk ``index`` of ``kind``, named for 12:00:00 plus
    ``index * step`` seconds, under ``under`` and the kind's root."""
    stamp = START + timedelta(seconds=index * step)
    return f"{under}{_ROOT_OF_KIND[kind]}/{sub}/{kind}.{stamp:%Y%m%d-%H%M%S}.{tag}{index}.gz"


def _chunks(sub, kinds=("console", "gpu", "training"), *, count=3, first=0, **naming):
    """``count`` chunks of each kind for ``sub``, from ``first`` on. Each holds its own path
    as its one row, so a reconstructed file says which chunks went into it, in what order."""
    paths = [_path(sub, kind, index, **naming) for kind in kinds for index in range(first, first + count)]
    return {path: gzip.compress(f"{path}\n".encode()) for path in paths}


def _three_sub_jobs():
    """A training sub-job with all three streams and two sampling sub-jobs with stdout and
    GPU metrics, beside files every listing passes over: another job's chunks, a file that
    is not a chunk, a chunk a folder too deep, and one directly under a root."""
    return {
        **_chunks(TRAINING),
        **_chunks(SAMPLING, ("console", "gpu"), count=2),
        **_chunks(SAMPLING_1, ("console", "gpu"), count=2),
        **_chunks(OTHER_JOB),
        f"_metrics/{TRAINING}/notes.txt": b"not a chunk\n",
        f"_metrics/{SAMPLING}/nested/gpu.20260904-120000.n0.gz": gzip.compress(b"a folder too deep\n"),
        "_stdout/console.20260904-120000.top.gz": gzip.compress(b"no sub-job\n"),
    }


class _Stage(_FakeArtifactConnection):
    """The execution-log fake holding a whole run: a LIST answers by path prefix, as a
    stage does, and each LIST is kept with the rows it returned."""

    def __init__(self, files=(), *, on_list=None):
        super().__init__({RUN_URI: []})
        # (run-relative location, run-relative paths returned) per LIST, in order.
        self.listings: list[tuple[str, list[str]]] = []
        # Called with each LIST's run-relative location before it answers; may block or raise.
        self.on_list = on_list
        self.add(dict(files))

    def add(self, files):
        """Stage ``files`` (run-relative path: bytes), as chunks uploaded since."""
        for path, body in files.items():
            self.list_rows[RUN_URI].append((ROW_PREFIX + path,))
            self.content_by_uri[RUN_URI + path] = body

    def listed(self, uri):
        location = uri.removeprefix(RUN_URI)
        if self.on_list is not None:
            self.on_list(location)
        rows = super().listed(uri)
        self.listings.append((location, [row[0].removeprefix(ROW_PREFIX) for row in rows]))
        return rows

    class Cursor(_FakeArtifactConnection.Cursor):
        def execute(self, statement):
            if not statement.startswith("GET "):
                return super().execute(statement)
            # As the base fake's GET, which splits the statement at its first space; a
            # source may hold one here (a sub-job id such as "job-1:a b").
            self.connection.commands.append(statement)
            source_literal, target_literal = statement.removeprefix("GET ").rsplit(" ", 1)
            source = self._unquote(source_literal)
            if self.connection.on_get is not None:
                self.connection.on_get(source)
            target = Path(unquote(urlparse(self._unquote(target_literal)).path))
            target.mkdir(parents=True, exist_ok=True)
            assert not list(target.iterdir())
            (target / PurePosixPath(source).name).write_bytes(self.connection.content_by_uri[source])
            self.rows = [(source, "DOWNLOADED")]
            return self


def _fail_at(*locations):
    """An ``on_list`` that fails the LIST of each of ``locations``, as no retry can mend."""

    def on_list(location):
        if location in locations:
            raise RuntimeError(f"LIST {location} failed")

    return on_list


def _gets(stage, start=0):
    """The run-relative paths GET fetched, from command ``start`` on."""
    return [
        command.removeprefix(f"GET '{RUN_URI}").split("' ", 1)[0]
        for command in stage.commands[start:]
        if command.startswith("GET ")
    ]


def _lists(stage, start=0):
    """The run-relative locations LISTed, from command ``start`` on, sorted: a download lists
    both roots at once, so the two can start in either order."""
    return sorted(
        command.removeprefix(f"LIST '{RUN_URI}").removesuffix("'")
        for command in stage.commands[start:]
        if command.startswith("LIST ")
    )


def _iso(seconds_after_newest):
    """An API time ``seconds_after_newest`` after the newest chunk of _three_sub_jobs()."""
    return datetime.fromtimestamp(NEWEST + seconds_after_newest, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Clients ───────────────────────────────────────────────────────────────


def _bind(monkeypatch, client, stage, *, status="DONE", minutes=60, **job_fields):
    """Point ``client`` at ``stage``'s run. For the stall check, the job is ``status``
    (with ``job_fields``; an exception fails the lookup) and the local clock ``minutes``
    past the newest chunk."""

    def job_for_upload_warnings(job_id):
        if isinstance(status, Exception):
            raise status
        return {"job_id": job_id, "status": status, **job_fields}, None

    monkeypatch.setattr(client, "_experiment_run_uri", lambda _job: (RUN_URI, "RUN_ABC"))
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: stage)
    monkeypatch.setattr(client, "_job_for_upload_warnings", job_for_upload_warnings)
    client._wall_clock = lambda: NEWEST + minutes * 60
    client._artifact_retry_policy = nc._artifact_download.RetryPolicy(sleep=lambda _delay: None)
    return client


def _client(monkeypatch, stage, *, scoped=True, **job):
    """A client of ``stage``'s run. ``scoped=False`` is one as every client was before
    scoping: each listing covers both roots whole, and no root is remembered."""
    client = _bind(monkeypatch, _make_client(), stage, **job)
    if not scoped:
        monkeypatch.setattr(client, "_scoped_sub_jobs", lambda *_args: None)
        monkeypatch.setattr(client, "_known_artifact_root", lambda *_args: None)
    return client


# A remembered root is kept per stream, under the stream's chunk pattern.
_STREAM_OF_PATTERN = {
    nc._STDOUT_CHUNK_RE.pattern: "stdout",
    nc._GPU_METRICS_CHUNK_RE.pattern: "metrics",
    nc._TRAINING_METRICS_CHUNK_RE.pattern: "training-metrics",
}


def _remembered(client):
    """The roots ``client`` remembers, as {(stream, sub-job): root}."""
    with client._artifact_roots_lock:
        seen = dict(client._artifact_roots_seen)
    assert all(key[:2] == (RUN_URI, JOB) for key in seen), seen
    return {(_STREAM_OF_PATTERN[key[2]], key[3]): root for key, (root, _when) in seen.items()}


def _both(root, sub):
    """The two locations a sub-job's chunks of a stream may be under."""
    return [f"{root}/{sub}/", f"checkpoints/{root}/{sub}/"]


class _Clock:
    """The module's ``time``, with ``monotonic`` standing at ``now`` until it is moved."""

    def __init__(self, now=1000.0):
        self.now = now

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self):
        return self.now


def _telemetry(monkeypatch, client):
    """``client`` recording its metrics; downloads send their successes by default."""
    monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
    client._metric_emitter = MagicMock()
    return client


def _emitted(client, operation):
    """The value of the one ``operation`` event ``client`` sent since its emitter was reset."""
    [call] = [call for call in client._metric_emitter.emit.call_args_list if call.args[0] == operation]
    return call.args[1]


# ─── Downloads ─────────────────────────────────────────────────────────────


def _relative(entry, out):
    """A result entry with ``saved_path`` relative to ``out``."""
    entry = dict(entry)
    if entry.get("saved_path") is not None:
        entry["saved_path"] = Path(entry["saved_path"]).relative_to(out).as_posix()
    return entry


def _tree(out):
    """Every file under ``out`` and its bytes, hidden manifests included."""
    if not out.exists():
        return {}
    return {path.relative_to(out).as_posix(): path.read_bytes() for path in sorted(out.rglob("*")) if path.is_file()}


# Event values that listing less may change: the call's timings.
_LISTING_VALUES = frozenset({"duration_ms", "list_ms", "fetch_ms", "preview_ms", "first_commit_ms"})


def _observe(client, stage, out, stream, sub_job_ids, *, preview=False, **options):
    """All a caller sees of one download: what it returned or raised, the files it left,
    the chunks it fetched, what its callbacks were given and, for a client from
    _telemetry(), the event it sent, with paths relative to ``out`` so that two output
    directories compare equal. ``preview`` asks the first named sub-job's newest chunks
    first, as the TUI does."""
    emitter = client._metric_emitter if isinstance(client._metric_emitter, MagicMock) else None
    sent = len(emitter.emit.call_args_list) if emitter is not None else 0
    progress, counts, commits, previews = [], [], [], []
    callbacks: dict[str, Any] = {"progress": progress.append, "on_chunk": lambda *args: counts.append(args)}
    if stream != "stdout":
        callbacks["on_commit"] = lambda entry: commits.append(_relative(entry, out))
    if preview:

        def on_preview(entry):
            shown = {key: value for key, value in entry.items() if key != "path"}
            previews.append({**shown, "rows": Path(entry["path"]).read_bytes()})

        first = list(sub_job_ids)[:1]
        if stream == "metrics":
            callbacks.update(priority=first, preview_seconds=600, on_preview=on_preview)
        elif stream == "stdout":
            callbacks.update(preview_seconds=600, on_preview=on_preview)
        else:  # no preview for training metrics: the first sub-job is written first
            callbacks.update(priority=first)
    start = len(stage.commands)
    try:
        result = getattr(client, STREAMS[stream][0])(JOB, out, sub_job_ids=sub_job_ids, **callbacks, **options)
    except Exception as exc:  # noqa: BLE001 - what was raised is compared
        outcome = (
            "raised",
            type(exc),
            str(exc),
            getattr(exc, "sub_job_ids", None),
            getattr(exc, "available", None),
            getattr(exc, "download_progress_saved", None),
        )
    else:
        outcome = ("returned", [_relative(entry, out) for entry in result])
    events = []
    for call in emitter.emit.call_args_list[sent:] if emitter is not None else []:
        value = {key: value for key, value in call.args[1].items() if key not in _LISTING_VALUES}
        events.append((call.args[0], value, call.kwargs["attributes"]))
    return {
        "outcome": outcome,
        "files": _tree(out),
        "gets": sorted(_gets(stage, start)),
        "progress": progress,
        "on_chunk": counts,
        "on_commit": commits,
        "on_preview": previews,
        "events": events,
    }


def _poll(client, stage, stream, sub_job_ids, out, **options):
    """One download as a TUI refresh makes it (``resume``; no stall check unless asked):
    its result entries, and the locations it LISTed."""
    options.setdefault("stale_after_seconds", None)
    start = len(stage.commands)
    result = getattr(client, STREAMS[stream][0])(JOB, out, sub_job_ids=sub_job_ids, resume=True, **options)
    return [_relative(entry, out) for entry in result], _lists(stage, start)


# ─── Which named sub-job is listed alone ──────────────────────────────────


@pytest.mark.parametrize(
    ("sub_job_ids", "scoped"),
    [
        pytest.param(None, None, id="none-named"),
        pytest.param([TRAINING], [TRAINING], id="one"),
        pytest.param([TRAINING, TRAINING], [TRAINING], id="one-named-twice"),
        pytest.param({TRAINING}, [TRAINING], id="a-set-as-downloads-pass-it"),
        pytest.param([JOB], [JOB], id="the-job-s-own-sub-job"),
        pytest.param(["job-1:Train_ing-1.x:0"], ["job-1:Train_ing-1.x:0"], id="letters-digits-and-punctuation"),
        # Several: one LIST of each root whole, not one apiece (each costs a listing of the run).
        pytest.param([TRAINING, SAMPLING], None, id="two"),
        # Another job's sub-job has no chunks in this run, so it is not listed.
        pytest.param([OTHER_JOB, TRAINING], [TRAINING], id="another-job-s-beside-one"),
        pytest.param([OTHER_JOB], [], id="another-job-s"),
        pytest.param(["job-10:training:0"], [], id="a-job-whose-id-starts-with-the-job-s"),
        pytest.param(["", ".", ".."], [], id="empty-and-dots"),
        pytest.param(["job-1:x/y"], [], id="a-folder-below"),
        pytest.param(["job-1:x\\y"], [], id="a-backslash"),
        pytest.param([7], [], id="not-a-string"),
        # A LIST path is taken literally, so these are listed with the rest of their root.
        pytest.param(["job-1:a b"], None, id="a-space"),
        pytest.param(["job-1:a%3A0"], None, id="percent-encoded"),
        pytest.param(["job-1:a'b"], None, id="a-quote"),
        pytest.param(["job-1:" + chr(0xE9)], None, id="not-ascii"),
    ],
)
def test_which_named_sub_job_is_listed_alone(sub_job_ids, scoped):
    assert nc.CortexTrainingClient._scoped_sub_jobs(JOB, sub_job_ids) == scoped


# ─── What a download LISTs ────────────────────────────────────────────────


@pytest.mark.parametrize("stream", list(STREAMS))
def test_one_named_sub_job_is_listed_under_its_own_folder_and_gets_what_a_whole_listing_gives_it(
    tmp_path, monkeypatch, stream
):
    _method, root, _kind, _file = STREAMS[stream]
    files = _three_sub_jobs()
    stage = _Stage(files)
    client = _client(monkeypatch, stage)

    seen = _observe(client, stage, tmp_path / "scoped", stream, [TRAINING])

    # Its folder in each root, ending in "/" so that no sibling whose id starts with
    # its id is listed; the GPU stall check lists the same sub-job's stdout.
    expected = [f"LIST '{RUN_URI}{root}/{TRAINING}/'", f"LIST '{RUN_URI}checkpoints/{root}/{TRAINING}/'"]
    if stream == "metrics":
        expected += [f"LIST '{RUN_URI}_stdout/{TRAINING}/'", f"LIST '{RUN_URI}checkpoints/_stdout/{TRAINING}/'"]
    assert sorted(command for command in stage.commands if command.startswith("LIST ")) == sorted(expected)
    # Every row those LISTs returned is one of its files, and they returned each of them.
    returned = [row for _location, rows in stage.listings for row in rows]
    assert {row.split("/")[-2] for row in returned} == {TRAINING}
    listed_roots = (f"{root}/{TRAINING}/",) + ((f"_stdout/{TRAINING}/",) if stream == "metrics" else ())
    assert sorted(returned) == sorted(path for path in files if path.startswith(listed_roots))

    # What a download of every sub-job by a client listing both roots whole has of it.
    whole_stage = _Stage(files)
    whole = _observe(_client(monkeypatch, whole_stage, scoped=False), whole_stage, tmp_path / "whole", stream, None)
    assert seen["outcome"] == (
        "returned",
        [entry for entry in whole["outcome"][1] if entry["sub_job_id"] == TRAINING],
    )
    assert seen["files"] and seen["files"] == {
        path: data for path, data in whole["files"].items() if path.startswith(f"{TRAINING}/")
    }
    assert seen["gets"] and seen["gets"] == [path for path in whole["gets"] if f"/{TRAINING}/" in path]
    # ...and exactly what the same download by such a client has.
    same_stage = _Stage(files)
    assert seen == _observe(
        _client(monkeypatch, same_stage, scoped=False), same_stage, tmp_path / "same", stream, [TRAINING]
    )

    # Without sub_job_ids, each root is listed whole, as before.
    start = len(stage.commands)
    _observe(client, stage, tmp_path / "every", stream, None)
    expected = [f"{root}/", f"checkpoints/{root}/"]
    if stream == "metrics":
        expected += ["_stdout/", "checkpoints/_stdout/"]
    assert _lists(stage, start) == sorted(expected)


@pytest.mark.parametrize(
    "sub_job_ids",
    [
        pytest.param([TRAINING, SAMPLING], id="two"),
        pytest.param(["job-1:a b"], id="a-space"),
    ],
)
def test_sub_jobs_a_listing_cannot_name_alone_are_listed_with_their_root_whole(tmp_path, monkeypatch, sub_job_ids):
    stage = _Stage({**_three_sub_jobs(), **_chunks("job-1:a b")})
    client = _client(monkeypatch, stage)
    for _poll_number in range(2):  # and nothing a listing of each root whole learns changes that
        start = len(stage.commands)
        result = client.download_metrics(JOB, tmp_path, sub_job_ids=sub_job_ids, resume=True)
        assert [entry["sub_job_id"] for entry in result] == sorted(sub_job_ids)
        assert _lists(stage, start) == sorted(
            ["_metrics/", "checkpoints/_metrics/", "_stdout/", "checkpoints/_stdout/"]
        )


def test_naming_no_sub_job_of_the_job_lists_nothing_but_still_says_which_have_chunks(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)

    # Training metrics leave such a sub-job out, so there is nothing to list at all.
    assert _poll(client, stage, "training-metrics", [OTHER_JOB], tmp_path) == ([], [])
    # GPU metrics raise for it, after one listing of each root whole for the error.
    with pytest.raises(nc.NoChunksError) as raised:
        client.download_metrics(JOB, tmp_path, sub_job_ids=[OTHER_JOB])
    assert str(raised.value) == (
        f"no metrics chunks for sub-job(s) {OTHER_JOB}; "
        f"sub-jobs with metrics: {SAMPLING}, {SAMPLING_1}, {TRAINING}"
    )
    assert _lists(stage) == ["_metrics/", "checkpoints/_metrics/"]
    assert _gets(stage) == []


class _Named(str, enum.Enum):
    """Sub-job ids as a caller's own ``str`` Enum: from Python 3.11 on, a member formats as
    ``_Named.TRAINING``, not as the id."""

    TRAINING = TRAINING


@pytest.mark.parametrize("stream", list(STREAMS))
def test_an_id_of_a_str_subclass_is_listed_under_the_folder_of_its_value(tmp_path, monkeypatch, stream):
    _method, root, _kind, _file = STREAMS[stream]
    stage = _Stage(_three_sub_jobs())
    seen = _observe(_client(monkeypatch, stage), stage, tmp_path / "scoped", stream, [_Named.TRAINING])
    assert f"{root}/{TRAINING}/" in _lists(stage)
    assert not [location for location in _lists(stage) if "_Named" in location]
    whole_stage = _Stage(_three_sub_jobs())
    whole = _client(monkeypatch, whole_stage, scoped=False)
    assert seen == _observe(whole, whole_stage, tmp_path / "whole", stream, [_Named.TRAINING])
    assert seen["outcome"][0] == "returned" and seen["files"]


# ─── Same result as a listing of both roots whole ─────────────────────────


def _job_level_run():
    """The job's own sub-job (whose id is the job id) beside one of its sub-jobs."""
    return {**_chunks(JOB), **_chunks("job-1:a:0")}


def _prefix_siblings():
    """Two sub-jobs whose ids differ only by a final digit."""
    return {**_chunks("job-1:training:1"), **_chunks("job-1:training:10")}


def _odd_ids():
    """Beside the three sub-jobs: one whose id has a space, and chunks a folder below a
    sub-job's, which no listing takes for a sub-job of their own."""
    return {**_three_sub_jobs(), **_chunks("job-1:a b"), **_chunks("job-1:x/y")}


def _both_roots():
    """The training sub-job's chunks under both roots: some under checkpoints/ fall between
    those under the top-level root, and one name is under both (its top-level copy is kept)."""
    return {
        **_three_sub_jobs(),
        **_chunks(TRAINING, count=3, first=1, under="checkpoints/", tag="b"),
        **_chunks(TRAINING, count=1, first=2, under="checkpoints/"),
    }


def _checkpoints_only():
    """A run whose stage is anchored under checkpoints/."""
    return {
        **_chunks(TRAINING, under="checkpoints/"),
        **_chunks(SAMPLING, ("console", "gpu"), count=2, under="checkpoints/"),
        **_chunks(SAMPLING_1, ("console", "gpu"), count=2, under="checkpoints/"),
    }


def _mixed_roots():
    """The training sub-job under checkpoints/, the sampling ones under the top-level roots."""
    return {
        **_chunks(TRAINING, under="checkpoints/"),
        **_chunks(SAMPLING, ("console", "gpu"), count=2),
        **_chunks(SAMPLING_1, ("console", "gpu"), count=2),
    }


def _damaged():
    """The training sub-job's second chunk of each stream will not decompress."""
    files = _three_sub_jobs()
    for kind in ("console", "gpu", "training"):
        files[_path(TRAINING, kind, 1)] = b"not gzip"
    return files


def _stdout_only_sub_job():
    """The second sampling sub-job has saved stdout but no metrics."""
    return {path: body for path, body in _three_sub_jobs().items() if not path.startswith(f"_metrics/{SAMPLING_1}/")}


def _stdout_only_under_checkpoints():
    """A run anchored under checkpoints/ whose second sampling sub-job has saved stdout but
    no metrics."""
    return {
        path: body
        for path, body in _checkpoints_only().items()
        if not path.startswith(f"checkpoints/_metrics/{SAMPLING_1}/")
    }


def _long_run():
    """Forty minutes of the training sub-job's chunks, a minute apart: more than twice a
    ten-minute preview's."""
    return {**_chunks(TRAINING, count=40, step=60), **_chunks(SAMPLING, ("console", "gpu"), count=2)}


def _later_chunks():
    """Chunks uploaded since: two more of the training sub-job's, one more of a sampling one's."""
    return {**_chunks(TRAINING, count=2, first=3), **_chunks(SAMPLING, ("console", "gpu"), count=1, first=2)}


def _an_earlier_chunk():
    """A chunk of the training sub-job whose name sorts before one already saved."""
    return _chunks(TRAINING, count=1, first=1, tag="a")


# A named sub-job without chunks: a GPU or stdout download raises NoChunksError for it,
# while a training-metrics one leaves it out.
_RAISE = ("stdout", "metrics")


@dataclass(frozen=True)
class Scenario:
    """A run, the sub-jobs a download names, and the job's state for the stall check:
    ``status`` (with ``job_fields``) ``minutes`` after the newest chunk. ``arrive`` is
    staged between the first download and the second. ``raises`` names the streams whose
    download raises NoChunksError."""

    sub_job_ids: tuple[str, ...] | None
    files: Callable[[], dict[str, bytes]] = _three_sub_jobs
    status: str | Exception = "DONE"
    minutes: float = 60
    job_fields: dict[str, Any] = field(default_factory=dict)
    arrive: Callable[[], dict[str, bytes]] | None = None
    options: dict[str, Any] = field(default_factory=lambda: {"resume": True})
    raises: tuple[str, ...] = ()


SCENARIOS = [
    pytest.param(Scenario(None), id="every-sub-job"),
    pytest.param(Scenario((TRAINING,)), id="one-of-three"),
    pytest.param(Scenario((SAMPLING,)), id="one-of-three-without-training-metrics"),
    pytest.param(Scenario((TRAINING,), options={}), id="without-resume"),
    pytest.param(Scenario((JOB,), files=_job_level_run), id="the-job-s-own-sub-job"),
    pytest.param(Scenario(("job-1:a:0",), files=_job_level_run), id="beside-the-job-s-own-sub-job"),
    pytest.param(Scenario(("job-1:training:1",), files=_prefix_siblings), id="training-1-beside-training-10"),
    pytest.param(Scenario(("job-1:training:10",), files=_prefix_siblings), id="training-10-beside-training-1"),
    pytest.param(Scenario((NO_FILES,), raises=_RAISE), id="no-chunks"),
    pytest.param(Scenario((OTHER_JOB,), raises=_RAISE), id="another-job-s-sub-job"),
    pytest.param(Scenario((OTHER_JOB, TRAINING), raises=_RAISE), id="another-job-s-beside-one"),
    pytest.param(Scenario(("",), raises=_RAISE), id="an-empty-id"),
    pytest.param(Scenario(("..",), raises=_RAISE), id="dot-dot"),
    pytest.param(Scenario(("job-1:x/y",), files=_odd_ids, raises=_RAISE), id="a-folder-below-a-sub-job"),
    pytest.param(Scenario(("job-1:a b",), files=_odd_ids), id="an-id-with-a-space"),
    pytest.param(Scenario((TRAINING, SAMPLING)), id="two-named"),
    pytest.param(Scenario((TRAINING,), files=_both_roots), id="chunks-under-both-roots"),
    pytest.param(Scenario((SAMPLING,), files=_both_roots), id="beside-chunks-under-both-roots"),
    pytest.param(Scenario((TRAINING,), files=_checkpoints_only), id="checkpoints-only"),
    pytest.param(Scenario((TRAINING,), files=_mixed_roots), id="this-one-under-checkpoints"),
    pytest.param(Scenario((SAMPLING,), files=_mixed_roots), id="another-under-checkpoints"),
    pytest.param(Scenario((TRAINING,), arrive=_later_chunks), id="resume-appends-new-chunks"),
    pytest.param(Scenario((TRAINING,), arrive=_an_earlier_chunk), id="resume-rebuilds-for-an-earlier-chunk"),
    pytest.param(Scenario((TRAINING,), files=_damaged), id="a-damaged-chunk"),
    pytest.param(
        Scenario((SAMPLING_1,), files=_stdout_only_sub_job, status="RUNNING", minutes=3),
        id="running-with-stdout-but-no-metrics",
    ),
    pytest.param(
        Scenario((TRAINING,), files=_stdout_only_sub_job, status="RUNNING", minutes=3),
        id="running-beside-one-with-stdout-but-no-metrics",
    ),
    pytest.param(
        Scenario((SAMPLING_1,), files=_stdout_only_under_checkpoints, status="RUNNING", minutes=3),
        id="running-with-stdout-but-no-metrics-under-checkpoints",
    ),
    pytest.param(Scenario((SAMPLING_1,), files=_stdout_only_sub_job), id="ended-with-stdout-but-no-metrics"),
    # Without the job's status there is no warning to stand for its missing metrics.
    pytest.param(
        Scenario(
            (SAMPLING_1,), files=_stdout_only_sub_job, status=RuntimeError("status unavailable"), raises=("metrics",)
        ),
        id="status-lookup-fails",
    ),
    pytest.param(Scenario((TRAINING,), status="RUNNING", minutes=11), id="uploads-stalled"),
    pytest.param(
        Scenario((TRAINING,), status="CANCELLED", minutes=1, job_fields={"state_changed_at": _iso(30)}),
        id="just-ended",
    ),
    pytest.param(Scenario((TRAINING,), files=_long_run, options={"resume": True, "preview": True}), id="preview"),
]


@pytest.mark.parametrize("stream", list(STREAMS))
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_download_shows_what_a_listing_of_both_roots_whole_shows(tmp_path, monkeypatch, scenario, stream):
    sides = []
    for name, scoped in (("scoped", True), ("whole", False)):
        stage = _Stage(scenario.files())
        client = _client(
            monkeypatch, stage, scoped=scoped, status=scenario.status, minutes=scenario.minutes, **scenario.job_fields
        )
        sides.append((_telemetry(monkeypatch, client), stage, tmp_path / name))
    for poll in range(2):  # the second can go by the roots the first found
        if poll and scenario.arrive is not None:
            for _side_client, stage, _out in sides:
                stage.add(scenario.arrive())
        scoped, whole = [
            _observe(client, stage, out, stream, scenario.sub_job_ids, **scenario.options)
            for client, stage, out in sides
        ]
        assert scoped == whole, f"poll {poll + 1}"
        # Not an equality of two failures the scenario did not mean.
        raised = scoped["outcome"][0] == "raised"
        assert raised == (stream in scenario.raises), scoped["outcome"]
        assert not raised or scoped["outcome"][1] is nc.NoChunksError


def _split_streams():
    """The training sub-job's streams kept apart: GPU chunks under _metrics/ alone, its
    training rows under both metrics roots, its stdout under checkpoints/_stdout/ alone."""
    return {
        **_chunks(TRAINING, ("gpu",)),
        **_chunks(TRAINING, ("training",), count=2),
        **_chunks(TRAINING, ("training",), count=2, first=2, under="checkpoints/"),
        **_chunks(TRAINING, ("console",), under="checkpoints/"),
        **_chunks(SAMPLING, ("console", "gpu"), count=2),
    }


def _split_streams_later():
    """More of each stream, each where that stream's chunks already are."""
    return {
        **_chunks(TRAINING, ("gpu",), count=1, first=3),
        **_chunks(TRAINING, ("training",), count=1, first=4, under="checkpoints/"),
        **_chunks(TRAINING, ("console",), count=1, first=3, under="checkpoints/"),
    }


@pytest.mark.parametrize(
    ("files", "arrive"),
    [
        pytest.param(_three_sub_jobs, _later_chunks, id="one-root"),
        pytest.param(_split_streams, _split_streams_later, id="streams-under-different-roots"),
        pytest.param(_both_roots, _later_chunks, id="both-roots"),
        pytest.param(_mixed_roots, None, id="mixed-roots"),
    ],
)
def test_polls_of_each_stream_in_turn_show_what_a_listing_of_both_roots_whole_shows(
    tmp_path, monkeypatch, files, arrive
):
    # The Resources and Training tabs poll one sub-job's GPU and training streams, which
    # share a folder and so what is remembered of it, while its stdout is downloaded too.
    sides = []
    for name, scoped in (("scoped", True), ("whole", False)):
        stage = _Stage(files())
        sides.append((_telemetry(monkeypatch, _client(monkeypatch, stage, scoped=scoped)), stage, tmp_path / name))
    turns = ["metrics", "training-metrics", "stdout"] * 2 + ["training-metrics", "metrics", "stdout"]
    for turn, stream in enumerate(turns):
        if turn == 3 and arrive is not None:
            for _side_client, stage, _out in sides:
                stage.add(arrive())
        scoped, whole = [
            _observe(client, stage, out / stream, stream, [TRAINING], resume=True) for client, stage, out in sides
        ]
        assert scoped == whole, f"turn {turn + 1} ({stream})"
        assert scoped["outcome"][0] == "returned" and scoped["files"]


def _mixed_roots_later():
    """More chunks for _mixed_roots(), each where that sub-job's are."""
    return {
        **_chunks(TRAINING, ("gpu", "training"), count=2, first=3, under="checkpoints/"),
        **_chunks(SAMPLING, ("gpu",), count=1, first=2),
    }


def test_whole_and_single_sub_job_downloads_into_one_directory_show_what_whole_listings_show(tmp_path, monkeypatch):
    # As the TUI makes them: a first download and "r" fetch every sub-job, a refresh the one
    # shown, and switching to another sub-job refreshes that one, all into one directory.
    sides = []
    for name, scoped in (("scoped", True), ("whole", False)):
        stage = _Stage(_mixed_roots())
        sides.append((_telemetry(monkeypatch, _client(monkeypatch, stage, scoped=scoped)), stage, tmp_path / name))
    steps = [None, (TRAINING,), "arrive", (TRAINING,), (SAMPLING,), None, (SAMPLING,), (TRAINING,)]
    for step, sub_job_ids in enumerate(steps):
        if sub_job_ids == "arrive":
            for _side_client, stage, _out in sides:
                stage.add(_mixed_roots_later())
            continue
        for stream in ("metrics", "training-metrics"):
            scoped, whole = [
                _observe(client, stage, out, stream, sub_job_ids, resume=True) for client, stage, out in sides
            ]
            assert scoped == whole, f"step {step + 1} ({stream})"
    # The training sub-job's file holds each of its GPU chunks, the later ones too, once and in order.
    (client, stage, out), _whole = sides
    expected = sorted(path for path in stage.content_by_uri if f"/{TRAINING}/gpu." in path)
    assert [RUN_URI + row for row in (out / TRAINING / "gpu.jsonl").read_text().splitlines()] == expected


# ─── Remembering a sub-job's root ─────────────────────────────────────────


@pytest.mark.parametrize("stream", list(STREAMS))
def test_a_second_poll_lists_only_the_root_the_first_found_the_sub_job_under(tmp_path, monkeypatch, stream):
    _method, root, _kind, _file = STREAMS[stream]
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)

    first, listed = _poll(client, stage, stream, [TRAINING], tmp_path)
    assert listed == _both(root, TRAINING)
    again, listed = _poll(client, stage, stream, [TRAINING], tmp_path)
    assert listed == [f"{root}/{TRAINING}/"]
    assert again == first and again[0]["chunk_count"] == 3


@pytest.mark.parametrize(("arrive", "mode"), [(_later_chunks, "append"), (_an_earlier_chunk, "rebuild")])
def test_a_poll_by_the_remembered_root_appends_or_rebuilds_as_the_chunks_say(tmp_path, monkeypatch, arrive, mode):
    stage = _Stage(_three_sub_jobs())
    client = _telemetry(monkeypatch, _client(monkeypatch, stage))
    _poll(client, stage, "metrics", [TRAINING], tmp_path)
    stage.add(arrive())
    client._metric_emitter.reset_mock()

    [entry], listed = _poll(client, stage, "metrics", [TRAINING], tmp_path)

    assert listed == [f"_metrics/{TRAINING}/"]
    assert _emitted(client, "download_metrics")["resume_mode"] == mode
    # Each chunk's one row is its path: the file holds every chunk once, in name order.
    chunks = [path for path in {**_three_sub_jobs(), **arrive()} if path.startswith(f"_metrics/{TRAINING}/gpu.")]
    assert (tmp_path / TRAINING / "gpu.jsonl").read_text().splitlines() == sorted(chunks)
    assert entry["chunk_count"] == len(chunks)


@pytest.mark.parametrize("stream", list(STREAMS))
def test_a_run_anchored_under_checkpoints_is_remembered_there(tmp_path, monkeypatch, stream):
    _method, root, _kind, _file = STREAMS[stream]
    stage = _Stage(_checkpoints_only())
    client = _client(monkeypatch, stage)

    assert _poll(client, stage, stream, [TRAINING], tmp_path)[1] == _both(root, TRAINING)
    result, listed = _poll(client, stage, stream, [TRAINING], tmp_path)
    assert listed == [f"checkpoints/{root}/{TRAINING}/"]
    assert result[0]["first_artifact_uri"] == RUN_URI + _path(TRAINING, STREAMS[stream][2], 0, under="checkpoints/")
    assert _remembered(client)[(stream, TRAINING)] == f"checkpoints/{root}"


def test_a_sub_job_with_files_under_both_roots_keeps_both_listed(tmp_path, monkeypatch):
    stage = _Stage(_both_roots())
    client = _client(monkeypatch, stage)
    for _poll_number in range(3):
        result, listed = _poll(client, stage, "metrics", [TRAINING], tmp_path)
        assert listed == _both("_metrics", TRAINING)
        assert result[0]["chunk_count"] == 6  # three under each root, the one under both taken once
        assert "merged in chunk time order" in result[0]["warnings"][0]
    assert ("metrics", TRAINING) not in _remembered(client)


@pytest.mark.parametrize("under", ["", "checkpoints/"])
def test_a_sub_job_without_files_keeps_both_listed_and_its_first_chunks_are_found_under_either(
    tmp_path, monkeypatch, under
):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    new = "job-1:training:1"
    for _poll_number in range(2):
        assert _poll(client, stage, "training-metrics", [new], tmp_path) == ([], _both("_metrics", new))
    assert ("training-metrics", new) not in _remembered(client)

    stage.add(_chunks(new, ("training",), count=2, under=under))
    result, listed = _poll(client, stage, "training-metrics", [new], tmp_path)
    assert listed == _both("_metrics", new) and result[0]["chunk_count"] == 2
    # Now that its files are known to be there, that root alone.
    assert _poll(client, stage, "training-metrics", [new], tmp_path)[1] == [f"{under}_metrics/{new}/"]


def test_roots_an_unscoped_download_found_are_used_by_later_scoped_ones(tmp_path, monkeypatch):
    stage = _Stage(_mixed_roots())
    client = _client(monkeypatch, stage)

    assert _poll(client, stage, "metrics", None, tmp_path / "all")[1] == ["_metrics/", "checkpoints/_metrics/"]
    assert _remembered(client) == {
        ("metrics", TRAINING): "checkpoints/_metrics",
        ("metrics", SAMPLING): "_metrics",
        ("metrics", SAMPLING_1): "_metrics",
    }
    assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == [f"checkpoints/_metrics/{TRAINING}/"]
    assert _poll(client, stage, "metrics", [SAMPLING_1], tmp_path)[1] == [f"_metrics/{SAMPLING_1}/"]
    # An unscoped download never goes by them: each root is listed whole.
    assert _poll(client, stage, "metrics", None, tmp_path / "all")[1] == ["_metrics/", "checkpoints/_metrics/"]
    assert _poll(client, stage, "training-metrics", None, tmp_path / "all")[1] == [
        "_metrics/",
        "checkpoints/_metrics/",
    ]


def test_stdout_roots_are_remembered_apart_from_metrics_roots(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    for _poll_number in range(2):
        _poll(client, stage, "metrics", [TRAINING], tmp_path)
    assert _remembered(client) == {("metrics", TRAINING): "_metrics"}

    assert _poll(client, stage, "stdout", [TRAINING], tmp_path)[1] == _both("_stdout", TRAINING)
    assert _poll(client, stage, "stdout", [TRAINING], tmp_path)[1] == [f"_stdout/{TRAINING}/"]
    assert _remembered(client) == {("metrics", TRAINING): "_metrics", ("stdout", TRAINING): "_stdout"}


@pytest.mark.parametrize(("learns", "uses"), [("metrics", "training-metrics"), ("training-metrics", "metrics")])
def test_gpu_and_training_metrics_each_learn_their_own_root(tmp_path, monkeypatch, learns, uses):
    # Their chunks share a folder, but each stream learns its root from its own chunks: one
    # stream starting under the other root later must not be hidden by the other's root.
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    assert _poll(client, stage, learns, [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)
    result, listed = _poll(client, stage, uses, [TRAINING], tmp_path)
    assert listed == _both("_metrics", TRAINING) and result[0]["chunk_count"] == 3
    result, listed = _poll(client, stage, uses, [TRAINING], tmp_path)
    assert listed == [f"_metrics/{TRAINING}/"] and result[0]["chunk_count"] == 3
    assert _remembered(client) == {(learns, TRAINING): "_metrics", (uses, TRAINING): "_metrics"}


def test_another_stream_under_the_other_root_is_found_while_one_root_is_remembered(tmp_path, monkeypatch):
    # The sampling sub-job's GPU metrics are under _metrics/ alone; its training rows start
    # later, under checkpoints/ only. Remembering the GPU stream's root hides nothing.
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    _poll(client, stage, "metrics", [SAMPLING], tmp_path)
    assert _remembered(client) == {("metrics", SAMPLING): "_metrics"}
    # No training rows yet: both roots are listed for them, and nothing is learned.
    assert _poll(client, stage, "training-metrics", [SAMPLING], tmp_path) == ([], _both("_metrics", SAMPLING))
    assert ("training-metrics", SAMPLING) not in _remembered(client)

    stage.add(_chunks(SAMPLING, ("training",), count=1, under="checkpoints/"))
    result, listed = _poll(client, stage, "training-metrics", [SAMPLING], tmp_path)
    assert listed == _both("_metrics", SAMPLING)
    assert result[0]["first_artifact_uri"] == RUN_URI + _path(SAMPLING, "training", 0, under="checkpoints/")
    # Each stream now goes by its own root.
    assert _remembered(client) == {
        ("metrics", SAMPLING): "_metrics",
        ("training-metrics", SAMPLING): "checkpoints/_metrics",
    }
    assert _poll(client, stage, "metrics", [SAMPLING], tmp_path)[1] == [f"_metrics/{SAMPLING}/"]
    assert _poll(client, stage, "training-metrics", [SAMPLING], tmp_path)[1] == [f"checkpoints/_metrics/{SAMPLING}/"]


def test_a_remembered_root_without_this_stream_s_chunks_is_not_taken_for_the_answer(tmp_path, monkeypatch):
    # The training sub-job's GPU chunks are remembered under _metrics/; then they are gone
    # from there and show up under checkpoints/ instead.
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    _poll(client, stage, "metrics", [TRAINING], tmp_path)
    assert _remembered(client)[("metrics", TRAINING)] == "_metrics"
    gpu = [row for row in stage.list_rows[RUN_URI] if row[0].startswith(f"{ROW_PREFIX}_metrics/{TRAINING}/gpu.")]
    stage.list_rows[RUN_URI] = [row for row in stage.list_rows[RUN_URI] if row not in gpu]
    stage.add({
        path.replace("_metrics/", "checkpoints/_metrics/", 1): stage.content_by_uri[RUN_URI + path]
        for path in (row[0].removeprefix(ROW_PREFIX) for row in gpu)
    })

    # None where they were: the other root is listed before the sub-job counts as having none.
    result, listed = _poll(client, stage, "metrics", [TRAINING], tmp_path / "moved")
    assert listed == _both("_metrics", TRAINING) and result[0]["chunk_count"] == 3
    assert result[0]["first_artifact_uri"] == RUN_URI + _path(TRAINING, "gpu", 0, under="checkpoints/")
    assert _remembered(client)[("metrics", TRAINING)] == "checkpoints/_metrics"


def test_a_remembered_sub_job_found_under_neither_root_is_forgotten(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    _poll(client, stage, "training-metrics", [TRAINING], tmp_path)
    assert _remembered(client)[("training-metrics", TRAINING)] == "_metrics"
    stage.list_rows[RUN_URI] = [row for row in stage.list_rows[RUN_URI] if f"/{TRAINING}/training." not in row[0]]

    # The remembered root, then the other, hold none of its training chunks: it is forgotten ...
    assert _poll(client, stage, "training-metrics", [TRAINING], tmp_path / "gone") == ([], _both("_metrics", TRAINING))
    assert ("training-metrics", TRAINING) not in _remembered(client)
    # ... so its next poll lists both roots' folders at once, rather than one after the other.
    started = {location: threading.Event() for location in _both("_metrics", TRAINING)}

    def at_once(location):
        started[location].set()
        other = next(event for name, event in started.items() if name != location)
        assert other.wait(5), f"LIST {location} ran alone"

    stage.on_list = at_once
    assert _poll(client, stage, "training-metrics", [TRAINING], tmp_path / "gone") == ([], _both("_metrics", TRAINING))


def test_a_remembered_root_is_looked_up_again_after_ten_minutes(tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(nc, "time", clock)
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    one = [f"_metrics/{TRAINING}/"]
    learned = clock.now

    assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)
    # Polls that go by it do not make it younger.
    for seconds in (1.0, 300.0, 599.999):
        clock.now = learned + seconds
        assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == one
    clock.now = learned + 600.0
    assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)
    # Learned again from that listing, for ten minutes more.
    clock.now = learned + 1199.999
    assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == one
    clock.now = learned + 1200.0
    assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)


def test_chunks_saved_under_the_other_root_later_are_found_once_the_root_is_looked_up_again(tmp_path, monkeypatch):
    """The cost of remembering a root, as documented: what lands under the other root
    afterwards is found by the next listing of both, within ten minutes."""
    clock = _Clock()
    monkeypatch.setattr(nc, "time", clock)
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    learned = clock.now
    _poll(client, stage, "metrics", [TRAINING], tmp_path)
    stage.add(_chunks(TRAINING, ("gpu",), count=1, first=3, under="checkpoints/"))

    clock.now = learned + 300.0
    result, listed = _poll(client, stage, "metrics", [TRAINING], tmp_path)
    assert listed == [f"_metrics/{TRAINING}/"] and result[0]["chunk_count"] == 3
    clock.now = learned + 600.0
    result, listed = _poll(client, stage, "metrics", [TRAINING], tmp_path)
    assert listed == _both("_metrics", TRAINING) and result[0]["chunk_count"] == 4
    assert "merged in chunk time order" in result[0]["warnings"][0]
    assert (tmp_path / TRAINING / "gpu.jsonl").read_text().splitlines()[-1] == _path(
        TRAINING, "gpu", 3, under="checkpoints/"
    )


def test_with_no_recheck_interval_every_poll_lists_both_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(nc.CortexTrainingClient, "_ARTIFACT_ROOT_RECHECK_SECONDS", 0.0)
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    for _poll_number in range(3):
        assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)


def test_the_remembered_roots_are_capped_and_the_first_learned_is_forgotten_first(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    monkeypatch.setattr(client, "_ARTIFACT_ROOTS_KEPT", 2)

    def lists(sub):
        return _poll(client, stage, "metrics", [sub], tmp_path)[1]

    for sub in (TRAINING, SAMPLING, SAMPLING_1):
        assert lists(sub) == _both("_metrics", sub)
    assert set(_remembered(client)) == {("metrics", SAMPLING), ("metrics", SAMPLING_1)}
    assert lists(TRAINING) == _both("_metrics", TRAINING)  # forgotten: learned first
    # A poll that goes by a remembered root learns nothing, so its place stays.
    assert lists(SAMPLING_1) == [f"_metrics/{SAMPLING_1}/"]
    assert lists(SAMPLING) == _both("_metrics", SAMPLING)
    assert set(_remembered(client)) == {("metrics", TRAINING), ("metrics", SAMPLING)}
    assert lists(SAMPLING_1) == _both("_metrics", SAMPLING_1)


def test_the_remembered_roots_are_capped_at_their_default_size():
    client = _make_client()
    kept = nc.CortexTrainingClient._ARTIFACT_ROOTS_KEPT
    by_root = {f"job-1:sampling:{index}": {"_metrics": {}} for index in range(kept + 10)}
    client._note_artifact_roots(RUN_URI, JOB, nc._GPU_METRICS_CHUNK_RE.pattern, by_root)
    assert list(_remembered(client)) == [("metrics", f"job-1:sampling:{index}") for index in range(10, kept + 10)]


@pytest.mark.parametrize(
    "failing",
    [pytest.param(f"_metrics/{TRAINING}/", id="first"), pytest.param(f"checkpoints/_metrics/{TRAINING}/", id="second")],
)
def test_a_listing_that_fails_remembers_nothing(tmp_path, monkeypatch, failing):
    stage = _Stage(_three_sub_jobs(), on_list=_fail_at(failing))
    client = _client(monkeypatch, stage)
    with pytest.raises(RuntimeError, match="LIST .* failed"):
        _poll(client, stage, "metrics", [TRAINING], tmp_path)
    assert _remembered(client) == {}

    stage.on_list = _fail_at("checkpoints/_metrics/")
    with pytest.raises(RuntimeError, match="LIST .* failed"):
        _poll(client, stage, "metrics", None, tmp_path)
    assert _remembered(client) == {}

    stage.on_list = None
    assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)


def test_a_failed_listing_of_a_remembered_root_leaves_it_remembered_as_it_was(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    _poll(client, stage, "metrics", [TRAINING], tmp_path)
    before = dict(client._artifact_roots_seen)

    stage.on_list = _fail_at(f"_metrics/{TRAINING}/")
    with pytest.raises(RuntimeError, match="LIST .* failed"):
        _poll(client, stage, "metrics", [TRAINING], tmp_path)
    assert dict(client._artifact_roots_seen) == before
    stage.on_list = None
    assert _poll(client, stage, "metrics", [TRAINING], tmp_path)[1] == [f"_metrics/{TRAINING}/"]


def test_clients_for_jobs_share_the_remembered_roots(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    parent = _client(monkeypatch, stage)
    row = {"job_id": JOB, "database_name": "DB", "schema_name": "SCH"}
    child = _bind(monkeypatch, parent.for_job(row), stage)
    grandchild = _bind(monkeypatch, child.for_job(row), stage)
    assert child._artifact_roots_seen is parent._artifact_roots_seen is grandchild._artifact_roots_seen
    assert child._artifact_roots_lock is parent._artifact_roots_lock is grandchild._artifact_roots_lock

    # What one learns, the others go by.
    assert _poll(child, stage, "metrics", [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)
    assert _poll(parent, stage, "metrics", [TRAINING], tmp_path)[1] == [f"_metrics/{TRAINING}/"]
    assert _poll(parent, stage, "stdout", [SAMPLING], tmp_path)[1] == _both("_stdout", SAMPLING)
    assert _poll(grandchild, stage, "stdout", [SAMPLING], tmp_path)[1] == [f"_stdout/{SAMPLING}/"]
    # A client made apart from them has its own.
    other = _client(monkeypatch, stage)
    assert _poll(other, stage, "metrics", [TRAINING], tmp_path)[1] == _both("_metrics", TRAINING)


# ─── A named sub-job without chunks ───────────────────────────────────────


@pytest.mark.parametrize(
    ("stream", "stall_check"),
    [
        # The GPU stall check lists the sub-job's stdout, finding none, so no warning.
        ("metrics", _both("_stdout", NO_FILES)),
        ("stdout", []),
    ],
)
def test_a_named_sub_job_without_chunks_lists_each_root_whole_once_for_the_error(
    tmp_path, monkeypatch, stream, stall_check
):
    _method, root, _kind, _file = STREAMS[stream]
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)

    with pytest.raises(nc.NoChunksError) as raised:
        getattr(client, STREAMS[stream][0])(JOB, tmp_path, sub_job_ids=[NO_FILES])

    label = root.removeprefix("_")
    assert str(raised.value) == (
        f"no {label} chunks for sub-job(s) {NO_FILES}; sub-jobs with {label}: {SAMPLING}, {SAMPLING_1}, {TRAINING}"
    )
    assert (raised.value.sub_job_ids, raised.value.available) == ([NO_FILES], [SAMPLING, SAMPLING_1, TRAINING])
    # Its own folders, then (after the stall check) each root whole, for the error alone.
    assert _lists(stage) == sorted([*_both(root, NO_FILES), *stall_check, f"{root}/", f"checkpoints/{root}/"])
    assert _gets(stage) == [] and not tmp_path.joinpath(NO_FILES).exists()


def test_no_whole_listing_when_a_warning_stands_for_the_missing_chunks(tmp_path, monkeypatch):
    stage = _Stage(_stdout_only_sub_job())
    client = _client(monkeypatch, stage, status="RUNNING", minutes=3)

    [entry] = client.download_metrics(JOB, tmp_path, sub_job_ids=[SAMPLING_1])

    assert entry["saved_path"] is None and entry["chunk_count"] == 0
    assert entry["warnings"][0].startswith(f"no GPU metrics have been saved for {SAMPLING_1} while the job")
    assert _lists(stage) == sorted([*_both("_metrics", SAMPLING_1), *_both("_stdout", SAMPLING_1)])


def test_no_whole_listing_for_a_stream_only_some_sub_jobs_write(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    assert client.download_training_metrics(JOB, tmp_path, sub_job_ids=[SAMPLING]) == []
    assert _lists(stage) == _both("_metrics", SAMPLING)


def test_no_second_listing_after_a_listing_of_each_root_whole(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    with pytest.raises(nc.NoChunksError) as raised:
        client.download_metrics(JOB, tmp_path, sub_job_ids=[TRAINING, NO_FILES])
    assert raised.value.available == [SAMPLING, SAMPLING_1, TRAINING]
    assert _lists(stage) == sorted(["_metrics/", "checkpoints/_metrics/", "_stdout/", "checkpoints/_stdout/"])


def test_without_a_stall_check_the_error_still_lists_each_root_whole_once(tmp_path, monkeypatch):
    # No warning can stand for the missing chunks, and there is no stdout listing.
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    with pytest.raises(nc.NoChunksError):
        client.download_metrics(JOB, tmp_path, sub_job_ids=[NO_FILES], stale_after_seconds=None)
    assert _lists(stage) == sorted([*_both("_metrics", NO_FILES), "_metrics/", "checkpoints/_metrics/"])


def test_a_whole_listing_for_the_error_that_fails_fails_the_download_as_a_whole_listing_did(tmp_path, monkeypatch):
    # The scoped listing found the named sub-job has none; the whole one that would name the
    # others fails: the download raises that failure, as one whole listing failing did.
    stage = _Stage(_three_sub_jobs(), on_list=_fail_at("checkpoints/_metrics/"))
    client = _client(monkeypatch, stage)
    with pytest.raises(RuntimeError, match="LIST checkpoints/_metrics/ failed"):
        client.download_metrics(JOB, tmp_path, sub_job_ids=[NO_FILES])
    reference = _client(monkeypatch, _Stage(_three_sub_jobs(), on_list=_fail_at("checkpoints/_metrics/")), scoped=False)
    with pytest.raises(RuntimeError, match="LIST checkpoints/_metrics/ failed"):
        reference.download_metrics(JOB, tmp_path / "whole", sub_job_ids=[NO_FILES])
    assert _gets(stage) == []


def test_the_error_never_names_a_missing_sub_job_whose_first_chunk_lands_meanwhile(tmp_path, monkeypatch):
    # The missing sub-job's own listing finds nothing; its first chunk lands before the whole
    # listing for the error: the error does not name it among the sub-jobs with chunks.
    late, landed = NO_FILES, []

    def first_chunk_lands(location):
        if location == "_metrics/" and not landed:  # the whole listing, after its own two
            landed.append(location)
            stage.add(_chunks(late, ("gpu",), count=1))

    stage = _Stage(_three_sub_jobs(), on_list=first_chunk_lands)
    client = _client(monkeypatch, stage)
    with pytest.raises(nc.NoChunksError) as raised:
        client.download_metrics(JOB, tmp_path, sub_job_ids=[late], stale_after_seconds=None)
    assert raised.value.sub_job_ids == [late]
    assert raised.value.available == [SAMPLING, SAMPLING_1, TRAINING]
    assert late not in str(raised.value).split("sub-jobs with metrics: ", 1)[1]


def test_the_whole_listing_for_the_error_is_timed_with_the_listing(tmp_path, monkeypatch):
    delay = 0.1

    def slow_whole_lists(location):
        if location in ("_metrics/", "checkpoints/_metrics/"):
            time.sleep(delay)

    stage = _Stage(_three_sub_jobs(), on_list=slow_whole_lists)
    client = _telemetry(monkeypatch, _client(monkeypatch, stage))
    with pytest.raises(nc.NoChunksError):
        client.download_metrics(JOB, tmp_path, sub_job_ids=[NO_FILES])

    value = _emitted(client, "download_metrics")
    assert value["success"] is False
    # The two roots are listed at once, so the whole listing takes one delay, not two.
    assert value["list_ms"] >= delay * 1000
    assert value["list_ms"] <= value["duration_ms"]


# ─── A LIST that matches more than the folder ─────────────────────────────


class _SlashlessStage(_Stage):
    """A stage whose LIST ignores a location's final "/": ``<root>/<id>/`` matches every
    folder whose name starts with ``<id>``."""

    def listed(self, uri):
        return super().listed(uri.removesuffix("/"))


@pytest.mark.parametrize("stream", list(STREAMS))
def test_a_listing_matching_a_sibling_folder_too_keeps_only_the_named_sub_job_s_chunks(tmp_path, monkeypatch, stream):
    stage = _SlashlessStage(_prefix_siblings())
    client = _client(monkeypatch, stage)
    whole_stage = _Stage(_prefix_siblings())
    whole = _client(monkeypatch, whole_stage, scoped=False)
    named = ["job-1:training:1"]

    for _poll_number in range(2):
        seen = _observe(client, stage, tmp_path / "scoped", stream, named, resume=True)
        assert seen == _observe(whole, whole_stage, tmp_path / "whole", stream, named, resume=True)
    # The LIST did return job-1:training:10's files; they were left out, and not remembered.
    returned = [row for _location, rows in stage.listings for row in rows]
    assert any(row.startswith(f"{STREAMS[stream][1]}/job-1:training:10/") for row in returned)
    assert seen["files"] and all(path.startswith("job-1:training:1/") for path in seen["files"])
    assert {sub for _stream, sub in _remembered(client)} == {"job-1:training:1"}


def test_a_listed_file_outside_the_root_is_still_an_error(tmp_path, monkeypatch):
    class StrayRow(_Stage):
        def listed(self, uri):
            return [*super().listed(uri), (f"{ROW_PREFIX}_logs/{TRAINING}/execution.jsonl",)]

    stage = StrayRow(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    expected = f"outside the requested root '_metrics/': '_logs/{TRAINING}/execution.jsonl'"
    with pytest.raises(ValueError, match=re.escape(expected)):
        client.download_metrics(JOB, tmp_path, sub_job_ids=[TRAINING])
    assert _remembered(client) == {}


def test_within_leaves_out_files_beside_the_root_and_refuses_those_outside_it():
    mine = f"_metrics/{TRAINING}/gpu.20260904-120000.c0.gz"
    sibling = f"_metrics/{TRAINING}0/gpu.20260904-120000.c0.gz"
    connection = _FakeArtifactConnection(
        {RUN_URI + f"_metrics/{TRAINING}/": [(ROW_PREFIX + mine, 7), (ROW_PREFIX + sibling, 9)]}
    )
    list_files = nc.CortexTrainingClient._list_experiment_artifact_files

    # As from a LIST that matched the folder's name without its final "/".
    files = list_files(connection, RUN_URI, "RUN_ABC", f"_metrics/{TRAINING}", within="_metrics")
    assert files == {mine: {"size": 7, "md5": None, "last_modified": None}}
    assert connection.commands == [f"LIST '{RUN_URI}_metrics/{TRAINING}/'"]
    # Without ``within`` that is an error, as it always was.
    with pytest.raises(ValueError, match=re.escape(f"outside the requested root '_metrics/{TRAINING}/'")):
        list_files(connection, RUN_URI, "RUN_ABC", f"_metrics/{TRAINING}")
    # And a file outside ``within`` is one either way.
    connection.list_rows[RUN_URI + f"_metrics/{TRAINING}/"].append((f"{ROW_PREFIX}_stdout/{TRAINING}/x.gz",))
    with pytest.raises(ValueError, match=re.escape("outside the requested root '_metrics/'")):
        list_files(connection, RUN_URI, "RUN_ABC", f"_metrics/{TRAINING}", within="_metrics")


# ─── The stall check's stdout listing ─────────────────────────────────────


def test_the_gpu_stall_check_lists_the_named_sub_job_s_stdout_alone(tmp_path, monkeypatch):
    sides = []
    for name, scoped in (("scoped", True), ("whole", False)):
        stage = _Stage(_stdout_only_sub_job())
        sides.append((_client(monkeypatch, stage, scoped=scoped, status="RUNNING", minutes=3), stage, tmp_path / name))
    (client, stage, out), (whole, whole_stage, whole_out) = sides

    seen = _observe(client, stage, out, "metrics", [SAMPLING_1], resume=True)
    assert seen == _observe(whole, whole_stage, whole_out, "metrics", [SAMPLING_1], resume=True)
    [entry] = seen["outcome"][1]
    assert entry["warnings"] == [
        f"no GPU metrics have been saved for {SAMPLING_1} while the job is still running, although its "
        "stdout output has been; metric uploads may be stalled. Contact support if this persists."
    ]
    assert _lists(stage) == sorted([*_both("_metrics", SAMPLING_1), *_both("_stdout", SAMPLING_1)])
    # Each root of the whole-listing client, as before.
    assert _lists(whole_stage) == sorted(["_metrics/", "checkpoints/_metrics/", "_stdout/", "checkpoints/_stdout/"])

    # The stall check learned the sub-job's stdout root; it has no metrics files to learn from.
    start = len(stage.commands)
    assert _observe(client, stage, out, "metrics", [SAMPLING_1], resume=True) == _observe(
        whole, whole_stage, whole_out, "metrics", [SAMPLING_1], resume=True
    )
    assert _lists(stage, start) == sorted([*_both("_metrics", SAMPLING_1), f"_stdout/{SAMPLING_1}/"])
    # ...which a stdout download goes by.
    assert _poll(client, stage, "stdout", [SAMPLING_1], tmp_path / "stdout")[1] == [f"_stdout/{SAMPLING_1}/"]


def test_the_ended_job_s_missing_metrics_check_lists_the_named_sub_job_s_stdout_alone(tmp_path, monkeypatch):
    stage = _Stage(_stdout_only_sub_job())
    client = _client(monkeypatch, stage, status="FAILED")
    [entry] = client.download_metrics(JOB, tmp_path, sub_job_ids=[SAMPLING_1])
    assert entry["warnings"][0].startswith(f"no GPU metrics were saved for {SAMPLING_1}, although its stdout")
    assert _lists(stage) == sorted([*_both("_metrics", SAMPLING_1), *_both("_stdout", SAMPLING_1)])


# ─── Logs ─────────────────────────────────────────────────────────────────


def test_a_scoped_listing_logs_the_both_roots_warning_only_for_the_named_sub_job(tmp_path, monkeypatch, caplog):
    files = {**_both_roots(), **_chunks(SAMPLING, ("gpu",), count=1, first=5, under="checkpoints/")}
    stage = _Stage(files)
    client = _client(monkeypatch, stage)
    with caplog.at_level(logging.WARNING):
        [entry] = client.download_metrics(JOB, tmp_path, sub_job_ids=[SAMPLING])
    assert entry["warnings"] == [
        f"metrics chunks for {SAMPLING} were saved under both _metrics/ and checkpoints/_metrics/; "
        "they were merged in chunk time order"
    ]
    assert f"for {SAMPLING} were saved under both" in caplog.text
    assert f"for {TRAINING} were saved under both" not in caplog.text  # not listed

    # Nor does the listing of each root whole behind an error's message say it of any.
    caplog.clear()
    start = len(stage.commands)
    with caplog.at_level(logging.WARNING), pytest.raises(nc.NoChunksError):
        client.download_metrics(JOB, tmp_path, sub_job_ids=[NO_FILES])
    assert {"_metrics/", "checkpoints/_metrics/"} <= set(_lists(stage, start))
    assert "were saved under both" not in caplog.text


def test_a_retried_scoped_listing_is_described_by_its_location(tmp_path, monkeypatch, caplog):
    errors = importlib.import_module("snowflake.connector.errors")
    failures = [errors.ServiceUnavailableError(msg="503")]

    def flaky(location):
        if location == f"_metrics/{TRAINING}/" and failures:
            raise failures.pop()

    stage = _Stage(_three_sub_jobs(), on_list=flaky)
    client = _telemetry(monkeypatch, _client(monkeypatch, stage))
    with caplog.at_level(logging.WARNING):
        [entry] = client.download_metrics(JOB, tmp_path, sub_job_ids=[TRAINING], stale_after_seconds=None)
    assert entry["chunk_count"] == 3
    assert f"LIST _metrics/{TRAINING} failed (ServiceUnavailableError" in caplog.text
    assert _emitted(client, "download_metrics")["artifact_retry_count"] == 1


# ─── Concurrent downloads on one client ───────────────────────────────────


def _spread_run(count):
    """``count`` sampling sub-jobs with stdout and GPU metrics: a quarter of them under
    checkpoints/, a quarter under both roots, and the rest under the top-level roots."""
    files = {}
    for index in range(count):
        sub = f"job-1:sampling:{index}"
        if index % 4 == 1:
            files.update(_chunks(sub, ("console", "gpu"), under="checkpoints/"))
        elif index % 4 == 2:
            files.update(_chunks(sub, ("console", "gpu"), count=2))
            files.update(_chunks(sub, ("console", "gpu"), count=2, first=2, under="checkpoints/"))
        else:
            files.update(_chunks(sub, ("console", "gpu")))
    return files


def test_concurrent_polls_of_different_sub_jobs_see_what_polls_one_at_a_time_see(tmp_path, monkeypatch):
    subs = [f"job-1:sampling:{index}" for index in range(8)]
    files = _spread_run(len(subs))

    def polls(client, out, sub):
        """Two polls of ``sub``, as a TUI's refreshes make them, and what each saw."""
        seen = []
        for _poll_number in range(2):
            result = client.download_metrics(JOB, out / sub, sub_job_ids=[sub], resume=True, max_workers=2)
            seen.append(([_relative(entry, out / sub) for entry in result], _tree(out / sub)))
        return seen

    sequential_stage = _Stage(files)
    sequential = _client(monkeypatch, sequential_stage)
    expected = {sub: polls(sequential, tmp_path / "sequential", sub) for sub in subs}

    stage = _Stage(files)
    client = _client(monkeypatch, stage)
    start = threading.Barrier(len(subs))
    seen, errors = {}, []

    def run(sub):
        try:
            start.wait(10)
            seen[sub] = polls(client, tmp_path / "concurrent", sub)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(sub,)) for sub in subs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert seen == expected
    assert sorted(_gets(stage)) == sorted(_gets(sequential_stage))
    assert sorted(_lists(stage)) == sorted(_lists(sequential_stage))
    # One root for each sub-job whose files are under one, for both streams; none for the rest.
    remembered = {}
    for index, sub in enumerate(subs):
        if index % 4 != 2:
            under = "checkpoints/" if index % 4 == 1 else ""
            remembered[("metrics", sub)] = f"{under}_metrics"
            remembered[("stdout", sub)] = f"{under}_stdout"
    assert _remembered(client) == _remembered(sequential) == remembered


def test_the_remembered_roots_are_read_and_changed_only_under_their_lock(tmp_path, monkeypatch):
    stage = _Stage(_three_sub_jobs())
    client = _client(monkeypatch, stage)
    monkeypatch.setattr(client, "_ARTIFACT_ROOTS_KEPT", 2)  # so that entries are forgotten too
    lock = client._artifact_roots_lock
    unlocked = []

    class Table(OrderedDict):
        """The table, noting each use of it made while its lock is free."""

        def _use(self, how):
            if not lock.locked():
                unlocked.append(how)

        def __getitem__(self, key):
            self._use("read")
            return super().__getitem__(key)

        def get(self, key, default=None):
            self._use("read")
            return super().get(key, default)

        def __setitem__(self, key, value):
            self._use("write")
            super().__setitem__(key, value)

        def __delitem__(self, key):
            self._use("delete")
            super().__delitem__(key)

        def pop(self, *args):
            self._use("delete")
            return super().pop(*args)

        def popitem(self, last=True):
            self._use("delete")
            return super().popitem(last=last)

        def __iter__(self):
            self._use("iterate")
            return super().__iter__()

        def __len__(self):
            self._use("count")
            return super().__len__()

    client._artifact_roots_seen = Table()
    for stream in STREAMS:
        for sub in (TRAINING, SAMPLING, SAMPLING_1, TRAINING):
            _poll(client, stage, stream, [sub], tmp_path / stream, stale_after_seconds=600)
    _poll(client, stage, "metrics", None, tmp_path / "every")
    assert unlocked == []
    assert len(_remembered(client)) == 2


def test_the_remembered_roots_stay_consistent_under_concurrent_listings(monkeypatch):
    client = _make_client()
    monkeypatch.setattr(client, "_ARTIFACT_ROOTS_KEPT", 16)
    gpu = nc._GPU_METRICS_CHUNK_RE.pattern
    both = {"_metrics": {}, "checkpoints/_metrics": {}}

    def work(worker):
        for index in range(400):
            sub = f"job-1:sampling:{(worker * 7 + index) % 40}"
            client._note_artifact_roots(RUN_URI, JOB, gpu, {sub: {"_metrics": {}} if index % 3 else both})
            client._known_artifact_root((RUN_URI, JOB, gpu, sub))

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    remembered = _remembered(client)
    assert 0 < len(remembered) <= 16 and set(remembered.values()) == {"_metrics"}
