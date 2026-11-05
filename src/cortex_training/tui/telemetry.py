# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the TUI reports about itself, beside its client's calls: how long it took to
show its first screen (``tui_startup``), and the errors it logs (``tui_error``).

Both go through the TUI's client, so they carry its account, ``client.surface``
(``tui``) and ``service.instance.id`` (one per TUI session), and send nothing when
telemetry is off. Neither sends what an error's text or a log label can quote: job
ids, database and schema names, paths.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Any

# One report per kind of error (where it was logged, and its type) in this long; the
# next says how many happened meanwhile. A failing refresh logs every few seconds.
ERROR_REPORT_SECONDS = 60.0

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_location(filename: str, lineno: int | None, function: str) -> str | None:
    """``path:line function``, with ``path`` relative to this package (``tui/logs.py``);
    None for code outside it, whose path would say where it was installed."""
    path = os.path.abspath(filename)
    if not path.startswith(_PACKAGE + os.sep):
        return None
    return f"{os.path.relpath(path, _PACKAGE).replace(os.sep, '/')}:{lineno} {function}"


def _raised_at(error: BaseException) -> str | None:
    """Where in this package ``error`` was raised: its traceback's innermost frame here."""
    found = None
    for frame in traceback.extract_tb(error.__traceback__):
        found = code_location(frame.filename, frame.lineno, frame.name) or found
    return found


def report_startup(client: Any, view: str, launched: float, client_ready: float | None) -> None:
    """Report ``tui_startup``: from ``launched`` (a ``time.monotonic()``) to now, when
    the screen the TUI opened with (``view``: ``job_list``, or ``job`` for one opened by
    id) drew its first data. ``client_build_ms`` is the part before the client was ready:
    reading the config, importing the Connector, and signing in."""
    emit = getattr(client, "_emit_operation_outcome", None)
    if emit is None:
        return
    values: dict[str, Any] = {"view": view}
    if client_ready is not None:
        values["client_build_ms"] = round((client_ready - launched) * 1000, 3)
    try:
        emit("tui_startup", launched, {}, values=values)
    except Exception:  # noqa: BLE001 - telemetry must never stop the TUI
        pass


class ErrorReporter:
    """Reports each error :func:`~cortex_training.tui.widgets.log_traceback` logs as a
    ``tui_error`` event, each kind at most once in :data:`ERROR_REPORT_SECONDS`.

    The event has the error's type (``error.type``, with ``error.http_status`` and
    ``error.code`` for an HTTP error), ``where`` it was logged and ``raised_at`` (both
    ``path:line function`` in this package), the client ``operation`` that failed with
    it when one did, and the ``occurrences`` since its kind was last reported. Not its
    text, nor the log's label.
    """

    def __init__(self, client: Any, *, clock=time.monotonic) -> None:
        self._client = client
        self._clock = clock
        self._lock = threading.Lock()
        # (where, error type) -> [when last reported, occurrences not reported since]
        self._kinds: dict[tuple[str, str], list] = {}

    def __call__(self, error: BaseException, where: str) -> None:
        kind = (where, type(error).__name__)
        now = self._clock()
        with self._lock:
            seen = self._kinds.get(kind)
            if seen is not None and now - seen[0] < ERROR_REPORT_SECONDS:
                seen[1] += 1
                return
            occurrences = 1 + (seen[1] if seen is not None else 0)
            self._kinds[kind] = [now, 0]
        try:
            from cortex_training.client.core import _metric_error_attributes

            value: dict[str, Any] = {"where": where, "occurrences": occurrences}
            raised_at = _raised_at(error)
            if raised_at:
                value["raised_at"] = raised_at
            operation = getattr(error, "_telemetry_operation", None)
            if isinstance(operation, str):
                value["operation"] = operation
            self._client.emit_metric("tui_error", value, attributes=_metric_error_attributes(error))
        except Exception:  # noqa: BLE001 - reporting an error must not raise one
            pass
