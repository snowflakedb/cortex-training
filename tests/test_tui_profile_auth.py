"""The TUI signs in the way a customer does, end to end except for Snowflake itself.

``cortex-training tui`` runs as a user starts it: the profile comes from
``connections.toml`` and the choice ``cortex-training login`` saves, both in the
Snowflake folder (``SNOWFLAKE_HOME``, see conftest). Only the Snowflake Connector
is faked, and the Cortex Training API is a local HTTP server, so these tests
check what the TUI sends: the profile's session token, and a new one after the
session expires. Skipped unless the optional ``textual`` extra is installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

import snowflake.connector  # noqa: E402

from cortex_training.tui import __main__ as tui_main  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.job_list import JobListScreen  # noqa: E402

JOBS = [
    {"job_id": "job-a", "status": "RUNNING", "sub_jobs": [{"job_type": "training"}]},
    {"job_id": "job-b", "status": "FAILED"},
]
# The configured schema's job list, which the TUI shows when the account-wide one isn't deployed.
JOB_LIST = "/api/v2/databases/CT_DB/schemas/PUBLIC/cortex-training"
# Snowflake's "session expired" error, as the REST API returns it.
EXPIRED = {"code": "390112", "message": "Your session has expired. Please login again."}


class FakeAPI:
    """A local Cortex Training API: records each request's token, and answers the job list."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.expired_tokens: set[str] = set()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's name
                token = self.headers.get("Authorization", "")
                api.requests.append((self.path, token))
                if any(token == f'Snowflake Token="{t}"' for t in api.expired_tokens):
                    self._send(401, EXPIRED)
                elif self.path.split("?")[0] == JOB_LIST:
                    self._send(200, {"jobs": JOBS})
                else:
                    # The account-wide job list isn't deployed (as for accounts today), and
                    # telemetry has nothing to say.
                    self._send(404, {"code": "404", "message": "not here"})

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tokens(self) -> list[str]:
        return [token for _, token in self.requests]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class FakeConnector:
    """``snowflake.connector.connect``: each sign-in gets the next session token."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        token = f"session-{len(self.calls)}"
        return SimpleNamespace(
            rest=SimpleNamespace(token=token, server_url=self.url),
            database="CT_DB", schema="PUBLIC", role="TRAINER", close=lambda: None,
        )


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()


@pytest.fixture
def connector(api, monkeypatch):
    fake = FakeConnector(api.url)
    monkeypatch.setattr(snowflake.connector, "connect", fake)
    return fake


@pytest.fixture
def customer(monkeypatch):
    """A customer's machine: one profile, saved by ``cortex-training login``, and no old settings."""
    for name in ("CORTEX_TRAINING_CONFIG", "CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_HOST",
                 "CORTEX_TRAINING_PAT", "CORTEX_TRAINING_DATABASE", "CORTEX_TRAINING_SCHEMA",
                 "CORTEX_TRAINING_BASE_URL", "SNOWFLAKE_HOST", "SNOWFLAKE_PAT", "SNOWFLAKE_DATABASE",
                 "SNOWFLAKE_SCHEMA", "SNOWFLAKE_DEFAULT_CONNECTION_NAME"):
        monkeypatch.delenv(name, raising=False)
    home = Path(os.environ["SNOWFLAKE_HOME"])
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(home / "no-login.json"))
    (home / "connections.toml").write_text(
        '[cortex-training]\naccount = "ACME"\nuser = "ME"\n'
        'authenticator = "PROGRAMMATIC_ACCESS_TOKEN"\ntoken = "not-sent-by-the-tui"\n'
        'database = "CT_DB"\nschema = "PUBLIC"\n'
    )
    (home / "connections.toml").chmod(0o600)
    (home / "cortex-training").mkdir(exist_ok=True)
    (home / "cortex-training" / "config.json").write_text(json.dumps({"profile": "cortex-training"}))


def _run_tui(monkeypatch, argv=(), before=None) -> dict:
    """Run ``cortex-training tui`` headless until its job list shows both jobs."""
    seen = {}

    def run_headless(app: CortexTrainingLogTUI) -> None:
        async def drive() -> None:
            async with app.run_test() as pilot:
                if before is not None:
                    before(app)
                for _ in range(200):
                    await pilot.pause()
                    screen = app.screen
                    if isinstance(screen, JobListScreen) and len(screen._job_by_item) >= len(JOBS):
                        seen["jobs"] = sorted(screen.whole_id(item) for item in screen._job_by_item)
                        break
                seen["sign_in"] = app.sign_in
                app.workers.cancel_all()
                for _ in range(5):
                    await pilot.pause()

        asyncio.run(drive())

    monkeypatch.setattr(CortexTrainingLogTUI, "run", run_headless)
    seen["rc"] = tui_main.run(list(argv))
    return seen


def test_the_tui_signs_in_with_the_profile_login_saved(customer, api, connector, monkeypatch, capsys):
    seen = _run_tui(monkeypatch)

    assert seen["rc"] == 0
    assert seen.get("jobs") == ["job-a", "job-b"], "the job list did not load"
    assert seen["sign_in"] == {"kind": "profile"}
    # The saved profile, through the Connector: the TUI never reads the PAT itself.
    assert connector.calls[0] == {"connection_name": "cortex-training"}
    assert api.tokens() and set(api.tokens()) == {'Snowflake Token="session-1"'}
    assert any(path == JOB_LIST for path, _ in api.requests)
    err = capsys.readouterr().err
    assert "deprecated" not in err
    assert "not-sent-by-the-tui" not in err


def test_the_tui_takes_a_named_profile(customer, api, connector, monkeypatch):
    home = Path(os.environ["SNOWFLAKE_HOME"])
    with (home / "connections.toml").open("a") as f:
        f.write('\n[other]\naccount = "ACME"\nuser = "ME"\ndatabase = "CT_DB"\nschema = "PUBLIC"\n')

    seen = _run_tui(monkeypatch, ["--connection", "other"])

    assert seen.get("jobs") == ["job-a", "job-b"]
    assert connector.calls[0] == {"connection_name": "other"}


def test_the_tui_signs_in_again_when_the_session_expires(customer, api, connector, monkeypatch):
    # The first session has expired by the time the TUI lists jobs.
    api.expired_tokens.add("session-1")

    seen = _run_tui(monkeypatch)

    assert seen.get("jobs") == ["job-a", "job-b"], "the job list did not load after the session expired"
    # One new sign-in, with the same profile, and the new token from then on.
    assert connector.calls == [{"connection_name": "cortex-training"}] * 2
    tokens = api.tokens()
    assert tokens[0] == 'Snowflake Token="session-1"'
    assert tokens[-1] == 'Snowflake Token="session-2"'


def test_without_a_profile_the_tui_says_how_to_connect(customer, api, connector, monkeypatch, capsys):
    home = Path(os.environ["SNOWFLAKE_HOME"])
    (home / "connections.toml").unlink()
    (home / "cortex-training" / "config.json").unlink()

    with pytest.raises(SystemExit) as exc:
        _run_tui(monkeypatch)

    assert exc.value.code == 2
    assert tui_main._CONNECT_HINT in capsys.readouterr().err
    assert connector.calls == [] and api.requests == []
