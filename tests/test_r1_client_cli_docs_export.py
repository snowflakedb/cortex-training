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

"""What reaches the terminal: service error bodies and the client's own warnings, as escapes."""

from __future__ import annotations

import json
import logging

import pytest
import requests

import cortex_training._cli as cli
from tests.test_checkpoints_cli import FakeClient
from tests.test_checkpoints_cli import _http_error
from tests.test_checkpoints_cli import _run

ESC = "\x1b"


@pytest.fixture(autouse=True)
def _no_login_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)


def _plain_cli_logging(monkeypatch):
    """No logging configured, as in a plain CLI process: warnings get the stderr handler.

    Called in the test body: pytest adds its own root handler after fixtures run.
    """
    monkeypatch.setattr(logging.getLogger(), "handlers", [])


# ─── R1-22: control characters in an error body ──────────────────────────────


def _raw_http_error(status: int, content: bytes) -> requests.exceptions.HTTPError:
    """An HTTP error whose body is these bytes as sent: a proxy's page, or a malformed answer."""
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://test.local/api"
    resp.encoding = "utf-8"
    resp._content = content
    return requests.exceptions.HTTPError(f"{status} Client Error: for url: {resp.url}", response=resp)


def test_an_error_body_with_raw_escape_sequences_is_shown_as_escapes():
    # OSC 52 writes the clipboard; the bidi override reorders what follows.
    content = f"<html>bad{ESC}]52;c;cHduZWQ=\x07{ESC}[7m \u202egnirts</html>".encode()
    rc, out, err = _run(["checkpoints", "job-1"], FakeClient(error=_raw_http_error(400, content)))
    assert rc == 1 and out == ""
    assert err.startswith("error: 400 Client Error")
    assert ESC not in err and "\x07" not in err and "\u202e" not in err
    assert r"response body: <html>bad\x1b]52;c;cHduZWQ=\x07\x1b[7m \u202egnirts</html>" in err
    assert err.count("\n") == 2  # the message's own lines are kept


def test_an_unchanged_body_still_has_the_service_message_as_sent():
    body = {"code": "390000", "message": "caf\u00e9 \u2192 failed"}
    rc, out, err = _run(["checkpoints", "job-1"], FakeClient(error=_http_error(400, body)))
    assert rc == 1 and out == ""
    assert f"response body: {json.dumps(body)}" in err


def test_printable_escapes_only_unsafe_characters():
    text = "caf\u00e9 \u2192 \u6a21\u578b \U0001f600\nline two\ttab\rCR\x1b[7m\u202e\x9b\udcff\U000e0001"
    shown = cli._printable(text)
    assert shown == ("caf\u00e9 \u2192 \u6a21\u578b \U0001f600\nline two\\ttab\\rCR\\x1b[7m\\u202e\\x9b"
                     "\\udcff\\U000e0001")
    assert cli._printable("plain ascii, nothing to do") == "plain ascii, nothing to do"


# ─── R1-22: the client's own warnings reach stderr through the same pass ────


def test_a_client_warning_quoting_the_service_reaches_stderr_as_escapes(monkeypatch):
    from unittest.mock import MagicMock

    import cortex_training.client as client_module

    _plain_cli_logging(monkeypatch)
    reason = f"oom{ESC}]0;LEAK\x07{ESC}[7mINV{ESC}[27m"
    resp = requests.Response()
    resp.status_code = 200
    resp.headers["x-snowflake-request-id"] = "REQUEST_ID"
    resp._content = json.dumps({"job_id": "JOB_ID", "status": "FAILED", "reason": reason}).encode()
    client = client_module.CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.get.return_value = resp
    rc, out, err = _run(["wait", "JOB_ID"], client)
    assert rc == 1 and out == ""
    # The client's "[last snowflake request id: ...] wait_for_job failed: ..." line, then the error.
    assert "[last snowflake request id: REQUEST_ID] wait_for_job failed: " in err
    assert err.count(r"oom\x1b]0;LEAK\x07\x1b[7mINV\x1b[27m") == 2
    assert ESC not in err and "\x07" not in err
    assert not logging.getLogger("cortex_training").handlers, "the handler is removed after the command"
