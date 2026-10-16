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

"""The create-job ``comment``: one rule for blank comments on every path, and text that isn't UTF-8."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

import cortex_training._cli as cli
from cortex_training.client import CortexTrainingClient
from cortex_training.client import SubJobConfig
from cortex_training.client import _body_with_checked_comment

BASE = ["--base-url", "http://test.local", "--database", "DB"]
SAMPLING = {"job_type": "sampling", "model_name": "gpt2", "inference_config": {"max_seq_len": 128, "n_gpus": 1}}
DEPLOYMENT = {"job_type": "sampling", "model_name": "gpt2",
              "inference_config": {"max_seq_len": 128, "n_gpus": 1, "deployment_class": "Model"}}
BLANKS = ["", "   ", "\n", " \t\n "]


@pytest.fixture(autouse=True)
def _no_login_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)


def _client() -> CortexTrainingClient:
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.post.return_value.status_code = 200
    client._session.post.return_value.json.return_value = {"job_id": "JOB_ID"}
    return client


def _sent_body(client: CortexTrainingClient) -> dict:
    (call,) = client._session.post.call_args_list
    return call.kwargs["json"]


def _dry_run(tmp_path, body: dict, *extra: str) -> tuple[int, str, str]:
    path = tmp_path / "job.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = cli.main(BASE + ["submit", str(path), "--dry-run", *extra], stdout=stdout, stderr=stderr)
    return rc, stdout.getvalue(), stderr.getvalue()


# ─── R1-102: a blank comment is never sent, whichever path creates the job ───


@pytest.mark.parametrize("blank", BLANKS)
def test_create_job_from_body_sends_no_blank_comment(blank):
    client = _client()
    body = {"sub_job_configs": [SAMPLING], "comment": blank}
    assert client.create_job_from_body(body) == {"job_id": "JOB_ID"}
    assert "comment" not in _sent_body(client)
    assert body["comment"] == blank, "the caller's dict is unchanged"


@pytest.mark.parametrize("blank", BLANKS)
def test_create_job_sends_no_blank_comment(blank):
    client = _client()
    client.create_job(sub_jobs=[SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)],
                      comment=blank)
    assert "comment" not in _sent_body(client)


@pytest.mark.parametrize("blank", BLANKS)
def test_submit_dry_run_prints_the_body_without_a_blank_comment(tmp_path, blank):
    rc, out, err = _dry_run(tmp_path, {"sub_job_configs": [SAMPLING], "comment": blank})
    assert rc == 0, err
    assert "comment" not in json.loads(out)


@pytest.mark.parametrize("blank", BLANKS)
def test_submit_sends_no_blank_comment_from_the_json(tmp_path, blank):
    client = _client()
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"sub_job_configs": [SAMPLING], "comment": blank}), encoding="utf-8")
    stdout = io.StringIO()
    rc = cli.main(BASE + ["submit", str(path)], client_factory=lambda _args: client, stdout=stdout,
                  stderr=io.StringIO())
    assert rc == 0
    assert "comment" not in _sent_body(client)
    assert json.loads(stdout.getvalue()) == {"job_id": "JOB_ID"}


def test_an_empty_comment_option_still_removes_the_jsons(tmp_path):
    rc, out, err = _dry_run(tmp_path, {"sub_job_configs": [SAMPLING], "comment": "from the json"}, "--comment", "")
    assert rc == 0, err
    assert "comment" not in json.loads(out)


def test_a_comment_with_text_is_kept_as_written():
    body = {"sub_job_configs": [SAMPLING], "comment": "  sweep #3  \n second line "}
    assert _body_with_checked_comment(body) is body


def test_a_blank_comment_on_a_deployment_is_dropped_without_a_warning(caplog):
    body = {"sub_job_configs": [DEPLOYMENT], "comment": "  "}
    assert "comment" not in _body_with_checked_comment(body)
    assert not caplog.records


# ─── R1-103: a comment that is not UTF-8 text ────────────────────────────────


def test_a_comment_with_a_byte_that_isnt_utf8_names_the_byte():
    # A command-line argument holding the byte 0xFF arrives as U+DCFF.
    body = {"sub_job_configs": [SAMPLING], "comment": "name \udcff"}
    with pytest.raises(ValueError) as caught:
        _body_with_checked_comment(body)
    assert str(caught.value) == ("comment must be valid UTF-8 text; character 6 stands for the byte 0xFF, "
                                 "which is not UTF-8")


def test_a_comment_with_an_unpaired_surrogate_says_so(tmp_path):
    # JSON can spell one: "\ud800" with no low surrogate after it.
    path = tmp_path / "job.json"
    path.write_text('{"sub_job_configs": [%s], "comment": "ab\\ud800"}' % json.dumps(SAMPLING), encoding="utf-8")
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = cli.main(BASE + ["submit", str(path), "--dry-run"], stdout=stdout, stderr=stderr)
    assert rc == 1 and stdout.getvalue() == ""
    assert stderr.getvalue() == ("error: comment must be valid UTF-8 text; character 3 is U+D800, an unpaired "
                                 "surrogate\n")


@pytest.mark.skipif(os.name == "nt", reason="POSIX argv bytes")
def test_submit_with_a_comment_that_isnt_utf8_says_what_is_wrong(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"sub_job_configs": [SAMPLING]}), encoding="utf-8")
    # The child imports the same cortex_training as this test.
    env = {**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
           "PYTHONPATH": os.path.dirname(os.path.dirname(cli.__file__)),
           "CORTEX_TRAINING_LOGIN_FILE": str(tmp_path / "no-login.json")}
    env.pop("CORTEX_TRAINING_CONFIG", None)
    done = subprocess.run(
        [os.fsencode(sys.executable), b"-m", b"cortex_training", b"submit", os.fsencode(path), b"--dry-run",
         b"--comment", b"name \xff"],
        capture_output=True, env=env, timeout=120,
    )
    assert done.returncode == 1 and done.stdout == b""
    assert done.stderr == (b"error: comment must be valid UTF-8 text; character 6 stands for the byte 0xFF, "
                           b"which is not UTF-8\n")
