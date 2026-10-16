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

"""``cortex-training checkpoints``: the Experiment hint on its errors."""

from __future__ import annotations

import io
import json

import pytest
import requests

import cortex_training._cli as cli

MARKER_MESSAGE = "Linked experiment is not found or not authorized for the training job"
BASE = ["--base-url", "http://test.local", "--database", "CORTEX_TRAINING_DB"]


@pytest.fixture(autouse=True)
def _no_login_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)


class FakeClient:
    def __init__(self, error=None):
        self.error = error

    def list_checkpoints(self, job_id):
        if self.error is not None:
            raise self.error
        return []


def _run(argv, client):
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = cli.main(BASE + argv, client_factory=lambda _args: client, stdout=stdout, stderr=stderr)
    return rc, stdout.getvalue(), stderr.getvalue()


def _http_error(status: int, body: dict) -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://test.local/api"
    resp._content = json.dumps(body).encode("utf-8")
    return requests.exceptions.HTTPError(f"{status} Server Error: for url: {resp.url}", response=resp)


def test_the_experiment_access_error_gets_a_hint():
    error = _http_error(500, {"message": MARKER_MESSAGE})
    rc, out, err = _run(["checkpoints", "job-1"], FakeClient(error=error))
    assert rc == 1 and out == ""
    assert MARKER_MESSAGE in err  # the service's body is still shown
    assert "hint: this role can't read the job's Experiment" in err
    assert "use the role that created the job" in err


def test_other_errors_get_no_experiment_hint():
    rc, _, err = _run(["checkpoints", "job-404"], FakeClient(error=_http_error(404, {"message": "no such job"})))
    assert rc == 1 and "no such job" in err and "hint:" not in err
