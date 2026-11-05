# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the TUI reports about itself: how long it took to start, and the errors it logs."""

import asyncio
import os
import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

pytest.importorskip("textual")

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training import CortexTrainingClient  # noqa: E402
from cortex_training.client.telemetry import _telemetry_base_url  # noqa: E402
from cortex_training.tui import __main__ as tui_main  # noqa: E402
from cortex_training.tui import telemetry  # noqa: E402
from cortex_training.tui import widgets  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.job_list import JobListScreen  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402


def _client():
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [
        {"job_id": "7", "status": "RUNNING", "sub_jobs": [{"job_type": "training"}]},
        {"job_id": "old", "status": "FAILED"},
    ]
    c.get_job.return_value = {
        "status": "RUNNING",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    return c


async def _wait(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause(0.01)
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def _startups(client):
    return [call for call in client._emit_operation_outcome.call_args_list if call.args[0] == "tui_startup"]


# ─── tui_startup ────────────────────────────────────────────────────────


async def _run_first_list_is_the_startup():
    client = _client()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    now = time.monotonic()
    app.launched_at, app.client_ready_at = now - 2.0, now - 0.5
    async with app.run_test() as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        assert await _wait(pilot, lambda: _startups(client))
        app.first_view_drawn("job_list")  # the list drawn again (a refresh): still one report
        await _settle(app, pilot)
    (call,) = _startups(client)
    assert call.args == ("tui_startup", app.launched_at, {})
    assert call.kwargs["values"] == {"view": "job_list", "client_build_ms": 1500.0}


def test_the_first_job_list_reports_how_long_the_tui_took_to_start():
    asyncio.run(_run_first_list_is_the_startup())


async def _run_a_job_opened_by_id():
    client = _client()
    app = CortexTrainingLogTUI(client, "7", poll_interval=0.01)
    app.launched_at = time.monotonic()
    async with app.run_test() as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and _startups(client))
        await _settle(app, pilot)
    (call,) = _startups(client)
    assert call.kwargs["values"] == {"view": "job"}  # no client_ready_at: no client_build_ms


def test_a_tui_opened_on_a_job_reports_its_start_when_the_job_is_read():
    asyncio.run(_run_a_job_opened_by_id())


def test_only_the_screen_the_tui_opened_with_is_its_start():
    client = _client()
    app = CortexTrainingLogTUI(client)
    app.launched_at = time.monotonic()

    app.first_view_drawn("job")  # a job opened from the list: its wait began with a key press
    assert not _startups(client)
    app.first_view_drawn("job_list")
    assert len(_startups(client)) == 1


def test_an_app_nobody_launched_reports_no_start():
    client = _client()
    app = CortexTrainingLogTUI(client)  # a test's, or the screenshot tool's: no launched_at
    app.first_view_drawn("job_list")
    assert not _startups(client)


def test_a_startup_report_that_raises_is_ignored():
    client = _client()
    client._emit_operation_outcome.side_effect = RuntimeError("telemetry is broken")
    telemetry.report_startup(client, "job_list", time.monotonic(), None)  # does not raise


# ─── tui_error ──────────────────────────────────────────────────────────


def _raised_in_the_package():
    try:
        _telemetry_base_url("")  # raises inside cortex_training/client/telemetry.py
    except ValueError as exc:
        return exc
    raise AssertionError("expected a ValueError")


def test_an_error_is_reported_with_where_it_was_logged_and_raised_but_not_its_text():
    client = MagicMock()
    reporter = telemetry.ErrorReporter(client, clock=lambda: 100.0)
    error = _raised_in_the_package()
    error._telemetry_operation = "get_job"  # what the client marks an error it reported with

    reporter(error, "tui/logs.py:1147 _read_sources")

    operation, value = client.emit_metric.call_args.args
    attributes = client.emit_metric.call_args.kwargs["attributes"]
    assert operation == "tui_error"
    assert value["where"] == "tui/logs.py:1147 _read_sources"
    assert value["raised_at"].startswith("client/telemetry.py:") and value["raised_at"].endswith(" _telemetry_base_url")
    assert (value["operation"], value["occurrences"]) == ("get_job", 1)
    assert attributes == {"error.type": "ValueError"}
    assert "hostname" not in repr(client.emit_metric.call_args)  # the error's text stays local


def test_an_http_error_says_its_status():
    client = MagicMock()
    response = MagicMock(status_code=503)
    response.json.return_value = {"code": "SERVICE_UNAVAILABLE"}
    error = requests.exceptions.HTTPError("503 Server Error", response=response)

    telemetry.ErrorReporter(client)(error, "tui/resource_view.py:408 _refresh")

    attributes = client.emit_metric.call_args.kwargs["attributes"]
    assert (attributes["error.type"], attributes["error.http_status"]) == ("HTTPError", 503)


def test_each_kind_of_error_is_reported_at_most_once_a_minute():
    client = MagicMock()
    now = [100.0]
    reporter = telemetry.ErrorReporter(client, clock=lambda: now[0])
    tail, refresh = "tui/logs.py:2607 _tail", "tui/resource_view.py:408 _refresh"

    for at, where in ((100, tail), (110, tail), (120, tail), (130, refresh), (170, tail)):
        now[0] = float(at)
        reporter(KeyError("status"), where)

    sent = [(call.args[1]["where"], call.args[1]["occurrences"]) for call in client.emit_metric.call_args_list]
    # The tail's three in a minute are one report; the next says how many came since.
    assert sent == [(tail, 1), (refresh, 1), (tail, 3)]


def test_a_reporter_whose_client_raises_does_not_raise():
    client = MagicMock()
    client.emit_metric.side_effect = RuntimeError("telemetry is broken")
    telemetry.ErrorReporter(client)(KeyError("status"), "tui/logs.py:1 f")  # does not raise


def test_code_locations_are_relative_to_the_package_and_nothing_outside_it():
    inside = os.path.join(telemetry._PACKAGE, "tui", "logs.py")
    assert telemetry.code_location(inside, 12, "_on_page") == "tui/logs.py:12 _on_page"
    assert telemetry.code_location("/usr/lib/python3.12/json/decoder.py", 3, "decode") is None


def test_log_traceback_keeps_the_label_local_and_tells_the_reporter_where(tmp_path, monkeypatch):
    log = tmp_path / "errors.log"
    monkeypatch.setattr(widgets, "_ERROR_LOG", str(log))
    seen = []
    widgets.set_error_reporter(lambda error, where: seen.append((type(error).__name__, where)))
    try:
        try:
            raise KeyError("status")
        except KeyError:
            widgets.log_traceback("reading job j-1 of DB.SCHEMA failed")
        widgets.log_traceback("outside an except")  # nothing to report
    finally:
        widgets.set_error_reporter(None)
    assert "reading job j-1 of DB.SCHEMA failed" in log.read_text()
    # This test logged it, from outside the package: no path of the test's is sent.
    assert seen == [("KeyError", "unknown")]


def test_log_traceback_survives_a_reporter_that_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(widgets, "_ERROR_LOG", str(tmp_path / "errors.log"))

    def reporter(_error, _where):
        raise RuntimeError("telemetry is broken")

    widgets.set_error_reporter(reporter)
    try:
        try:
            raise KeyError("status")
        except KeyError:
            widgets.log_traceback("a failure")  # does not raise
    finally:
        widgets.set_error_reporter(None)


# ─── cortex-training tui: wiring and quitting ───────────────────────────


def _run_tui_main(monkeypatch, client, during_run):
    class FakeApp:
        return_code = 0
        sign_in = None

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            self.client = client

        def run(self):
            during_run(self)

    monkeypatch.delenv("CORTEX_TRAINING_CONNECTION", raising=False)
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training.cli.commands.build_client", lambda args, cls: client)
    return tui_main.run(["--base-url", "http://localhost:1", "--database", "DB"])


def test_the_tui_times_its_start_and_reports_its_errors_only_while_it_runs(monkeypatch):
    client = CortexTrainingClient(base_url="http://localhost:1", database="DB", schema="PUBLIC")
    seen = {}

    def during_run(app):
        seen["reporter"] = widgets._error_reporter
        seen["times"] = (app.launched_at, app.client_ready_at)

    assert _run_tui_main(monkeypatch, client, during_run) == 0
    assert isinstance(seen["reporter"], telemetry.ErrorReporter)
    launched, ready = seen["times"]
    assert launched <= ready <= time.monotonic()
    assert widgets._error_reporter is None


def test_quitting_sends_the_telemetry_left_beside_closing_the_sessions(monkeypatch):
    monkeypatch.setattr(tui_main, "SESSION_CLOSE_SECONDS", 0.3)
    monkeypatch.setattr(tui_main, "TELEMETRY_CLOSE_SECONDS", 0.3)
    release = threading.Event()

    class Client:
        def __init__(self):
            self._keep, self.timeouts = True, []

        @property
        def keep_artifact_sessions(self):
            return self._keep

        @keep_artifact_sessions.setter
        def keep_artifact_sessions(self, value):
            release.wait(5)  # an unreachable service: the sessions' close hangs
            self._keep = value

        def close_telemetry(self, timeout):
            self.timeouts.append(timeout)
            release.wait(5)  # and so does the telemetry's

    client = Client()
    started = time.monotonic()
    try:
        tui_main._close_sessions(client)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert client.timeouts == [0.3]
    assert elapsed < 0.55  # side by side: about the longer of the two waits, not their sum


def test_quitting_with_telemetry_off_or_nothing_kept_still_works():
    tui_main._close_sessions(object())  # neither sessions nor telemetry: nothing to wait for
    client = CortexTrainingClient(base_url="http://localhost:1", database="DB", schema="PUBLIC")
    tui_main._close_sessions(client)  # no emitter: close_telemetry is a no-op


def test_quitting_while_the_collector_hangs_waits_only_the_telemetry_budget(monkeypatch):
    monkeypatch.setattr(tui_main, "TELEMETRY_CLOSE_SECONDS", 0.2)
    from cortex_training.client import telemetry as client_telemetry

    client = CortexTrainingClient(base_url="http://localhost:1", database="DB", schema="PUBLIC")
    provider = MagicMock()
    provider.get_token.return_value = "t"
    emitter = client_telemetry.OtlpMetricEmitter("https://account.test", provider)
    emitter._session = MagicMock()
    hostname = MagicMock(status_code=200)
    hostname.json.return_value = {"hostname": "c.snowflakecomputing.com"}
    emitter._session.get.return_value = hostname
    release = threading.Event()
    emitter._session.post.side_effect = lambda *_args, **_kwargs: release.wait(5) and hostname
    client._metric_emitter = emitter
    try:
        client.emit_metric("tail_logs", {"success": False})  # still being posted at quit
        time.sleep(0.1)
        started = time.monotonic()
        tui_main._close_sessions(client)
        client_telemetry._flush_emitters_at_exit()  # what exit runs next
        assert time.monotonic() - started < 0.5  # the budget, not the budget and exit's second
    finally:
        release.set()


def test_the_tui_waits_no_longer_on_telemetry_than_on_its_sessions():
    assert tui_main.TELEMETRY_CLOSE_SECONDS <= tui_main.SESSION_CLOSE_SECONDS
