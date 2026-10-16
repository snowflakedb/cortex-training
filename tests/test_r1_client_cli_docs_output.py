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

"""What the CLI prints on stdout: UTF-8 JSON text, and the order ``list`` prints jobs in."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

import cortex_training._cli as cli

BASE = ["--base-url", "http://test.local", "--database", "DB"]
SAMPLING = {"job_type": "sampling", "model_name": "gpt2", "inference_config": {"max_seq_len": 128, "n_gpus": 1}}
NAME = "na\u00efve \u2192 r\u00e9sum\u00e9 \u00b7 \u6a21\u578b \U0001f600"
# Characters a terminal could act on, or that hide text: they stay escapes.
UNSAFE = "a\x1b]0;title\x07\u202e\x9b\x7f\u2028\u200b\udcff\U000e0001z"


@pytest.fixture(autouse=True)
def _no_login_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)


def _dry_run(tmp_path, comment: str, *global_args: str, stdout=None) -> tuple[int, str]:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"sub_job_configs": [SAMPLING], "comment": comment}), encoding="utf-8")
    stdout = io.StringIO() if stdout is None else stdout
    rc = cli.main([*global_args, *BASE, "submit", str(path), "--dry-run"], stdout=stdout, stderr=io.StringIO())
    return rc, stdout.getvalue() if isinstance(stdout, io.StringIO) else ""


# ─── (a) non-ASCII text prints as UTF-8 ──────────────────────────────────────


@pytest.mark.parametrize("compact", [False, True])
def test_submit_dry_run_prints_the_comment_as_written(tmp_path, compact):
    rc, out = _dry_run(tmp_path, NAME, *(["--compact"] if compact else []))
    assert rc == 0
    assert NAME in out, "the name reads as written, not as \\u escapes"
    assert "\\u" not in out
    assert json.loads(out)["comment"] == NAME


class _Client:
    def __init__(self, comment: str):
        self.comment = comment

    def get_job(self, job_id):
        return {"job_id": job_id, "comment": self.comment}


def _get(comment: str) -> tuple[int, str]:
    stdout = io.StringIO()
    rc = cli.main([*BASE, "get", "JOB_ID"], client_factory=lambda _args: _Client(comment), stdout=stdout,
                  stderr=io.StringIO())
    return rc, stdout.getvalue()


def test_unsafe_characters_stay_escapes_and_decode_to_the_same_text():
    # As the service could answer them: JSON can carry any of these, a lone surrogate included.
    rc, out = _get(f"{NAME} {UNSAFE}")
    assert rc == 0
    assert NAME in out
    unsafe = [char for char in UNSAFE if cli._is_unsafe_char(char)]
    assert len(unsafe) == 9
    for char in unsafe:
        assert char not in out, f"{char!r} printed raw"
    assert "\\u001b]0;title\\u0007\\u202e\\u009b\\u007f\\u2028\\u200b\\udcff\\udb40\\udc01z" in out
    assert json.loads(out)["comment"] == f"{NAME} {UNSAFE}"


@pytest.mark.parametrize("encoding", ["ascii", "latin-1", "cp1252"])
def test_a_stream_that_isnt_utf8_gets_ascii_escapes(tmp_path, encoding):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, newline="\n")
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"sub_job_configs": [SAMPLING], "comment": NAME}), encoding="utf-8")
    rc = cli.main([*BASE, "submit", str(path), "--dry-run"], stdout=stream, stderr=io.StringIO())
    stream.flush()
    assert rc == 0
    printed = raw.getvalue()
    assert printed.isascii(), "valid UTF-8 JSON whatever the stream's own encoding"
    assert json.loads(printed.decode("ascii"))["comment"] == NAME


def test_every_command_prints_utf8():
    rc, out = _get(NAME)
    assert rc == 0 and NAME in out


def _cli_output(tmp_path, env_extra: dict) -> bytes:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"sub_job_configs": [SAMPLING], "comment": NAME}), encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONIOENCODING", "PYTHONUTF8", "LC_ALL", "LC_CTYPE", "LANG", "CORTEX_TRAINING_CONFIG")}
    # The child imports the same cortex_training as this test.
    env.update({"PYTHONPATH": os.path.dirname(os.path.dirname(cli.__file__)),
                "CORTEX_TRAINING_LOGIN_FILE": str(tmp_path / "no-login.json"), **env_extra})
    done = subprocess.run([sys.executable, "-m", "cortex_training", "--compact", "submit", str(path), "--dry-run"],
                          capture_output=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.parametrize("env", [{"LC_ALL": "C.UTF-8"}, {"PYTHONIOENCODING": "utf-8"}, {"LC_ALL": "C"}],
                         ids=["utf8-locale", "utf8-io", "c-locale"])
def test_the_cli_writes_utf8_bytes(tmp_path, env):
    out = _cli_output(tmp_path, env)
    assert NAME.encode("utf-8") in out
    assert json.loads(out.decode("utf-8"))["comment"] == NAME


@pytest.mark.parametrize("encoding", ["ascii", "latin-1"])
def test_the_cli_escapes_for_a_stdout_that_isnt_utf8(tmp_path, encoding):
    out = _cli_output(tmp_path, {"PYTHONIOENCODING": encoding})
    assert out.isascii()
    assert json.loads(out.decode("ascii"))["comment"] == NAME


# ─── (b) list prints oldest first; a job without created_at counts as oldest ──


class _ListClient:
    """The schema's list in the service's order (settled by id, then live ones), and an account walk."""

    def __init__(self, jobs, account_jobs=()):
        self.jobs = jobs
        self.account_jobs = list(account_jobs)

    def list_jobs(self, status=None):
        return list(self.jobs)

    def iter_jobs(self, **_filters):
        return iter(self.account_jobs)


def _job(job_id: str, created_at: str | None, status: str = "CANCELLED") -> dict:
    job = {"job_id": job_id, "status": status}
    if created_at is not None:
        job["created_at"] = created_at
    return job


SCHEMA_JOBS = [
    _job("c-mid", "2026-10-02T00:00:00Z"),
    _job("a-new", "2026-10-03T00:00:00Z", "FAILED"),
    _job("undated", None, "UNKNOWN"),
    _job("b-old", "2026-10-01T00:00:00.5Z", "TERMINATED"),
    _job("d-live", "2026-10-04T08:00:00+00:00", "RUNNING"),
]


def _printed(argv, client) -> list[str]:
    stdout = io.StringIO()
    rc = cli.main([*BASE, "list", *argv], client_factory=lambda _args: client, stdout=stdout,
                  stderr=io.StringIO())
    assert rc == 0
    return [job["job_id"] for job in json.loads(stdout.getvalue())["jobs"]]


def test_list_prints_the_oldest_job_first_and_the_newest_last():
    # --limit counts a job without created_at as the oldest, so it prints first,
    # and the newest dated job stays at the bottom of the terminal.
    assert _printed([], _ListClient(SCHEMA_JOBS)) == ["undated", "b-old", "c-mid", "a-new", "d-live"]


def test_the_last_job_printed_is_the_newest_match():
    # `list --status terminal | jq '.jobs[-1]'` is the newest ended job ...
    assert _printed(["--status", "terminal"], _ListClient(SCHEMA_JOBS)) == ["b-old", "c-mid", "a-new"]
    # ... and a job without created_at never takes its place.
    jobs = SCHEMA_JOBS + [_job("undated-failed", None, "FAILED")]
    assert _printed(["--status", "failed"], _ListClient(jobs)) == ["undated-failed", "a-new"]


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5, 9])
def test_limit_prints_the_end_of_the_full_listing(limit):
    everything = _printed([], _ListClient(SCHEMA_JOBS))
    assert _printed(["--limit", str(limit)], _ListClient(SCHEMA_JOBS)) == everything[-limit:]


def test_list_all_prints_oldest_first_whatever_order_the_walk_yields():
    # After a page-token restart, iter_jobs yields jobs created since the walk began after older ones.
    walk = [_job("old", "2026-10-01T00:00:00Z"), _job("new", "2026-10-05T00:00:00Z"),
            _job("mid", "2026-10-03T00:00:00Z")]
    assert _printed(["--all"], _ListClient([], walk)) == ["old", "mid", "new"]


def test_list_all_prints_the_services_newest_first_answer_reversed():
    newest_first = [_job(f"job-{n}", f"2026-10-0{9 - n}T00:00:00Z") for n in range(5)]
    assert _printed(["--all"], _ListClient([], newest_first)) == [f"job-{n}" for n in reversed(range(5))]


def test_jobs_created_at_the_same_time_print_as_limit_ranks_them():
    # Ties rank in the service's order, the account-level list answering newest
    # first, and print in the reverse of that ranking: --limit N keeps the last N.
    same = "2026-10-01T00:00:00Z"
    jobs = [_job("first", same), _job("second", same), _job("undated-1", None), _job("undated-2", None)]
    assert _printed([], _ListClient(jobs)) == ["undated-2", "undated-1", "second", "first"]
    assert _printed(["--limit", "1"], _ListClient(jobs)) == ["first"]
    assert _printed(["--limit", "3"], _ListClient(jobs)) == ["undated-1", "second", "first"]
