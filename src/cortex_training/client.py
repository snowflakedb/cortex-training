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

"""HTTP client for the Cortex Training REST API.

Single-purpose, production-shaped client:

- Snowflake connection profiles — see
  ``CortexTrainingClient.from_connection_name``.
- Direct Programmatic Access Token (PAT) compatibility — see
  ``CortexTrainingClient.from_pat``.
- One CreateJob shape: a list of typed :class:`SubJobConfig` (each carries either
  a :class:`TrainingConfig` or an :class:`InferenceConfig`). The client-side
  validators mirror the server's own required-field checks so an invalid job
  fails locally instead of after a round trip.
- One wire format: ``{job_id?, sub_job_configs:[SubJobConfig]}`` with per-sub-job
  ``training_config`` / ``inference_config`` blocks. See
  ``docs/reference/rest-api.md`` for the full schema.

The server keeps the config blocks open (``additionalProperties``), so unknown
keys pass through unchanged. The strict, typed shape lives here in the client;
use the ``extra`` fields for anything the typed layer does not model yet.
"""

from __future__ import annotations

import base64
import codecs
import contextlib
import functools
import gzip
import hashlib
import inspect
import json
import logging
import math
import numbers
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import TypeVar
from urllib.parse import quote

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field as PydanticField
from pydantic import model_validator

import requests
from requests.adapters import HTTPAdapter
from tenacity import Retrying
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential_jitter
from urllib3.exceptions import NewConnectionError

from cortex_training.optimizer import normalize_optimizer_config
from cortex_training.peft import normalize_lora_peft_config
from cortex_training.snowflake_auth import SnowflakeProfileAuth
from cortex_training.snowflake_auth import SnowflakeTelemetryTokenProvider
from cortex_training.telemetry import CachedSessionTokenProvider
from cortex_training.telemetry import OtlpMetricEmitter

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    # cortex_training.client.wire, as before: the frame codec imports PyTorch, so
    # this module imports it where a frame is encoded or decoded, on first use.
    if name == "wire":
        import importlib

        return importlib.import_module("cortex_training.wire")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_STDOUT_CHUNK_RE = re.compile(
    r"^console\.(?P<timestamp>\d{8}-\d{6})\.(?P<suffix>[^./\\]+)\.gz$"
)
_GPU_METRICS_CHUNK_RE = re.compile(
    r"^gpu\.(?P<timestamp>\d{8}-\d{6})\.(?P<suffix>[^./\\]+)\.gz$"
)
_STREAM_COPY_BUFFER_BYTES = 1024 * 1024
_ARTIFACT_GET_WORKERS = 8
_ARTIFACT_GET_MIN_CHUNKS = 128

# Env var that unlocks create-job debug options. These are an internal-only
# capability and are deliberately not documented for external use: the client
# refuses to forward a create-job request carrying a `debug` block unless this
# is set to a truthy value. The server gates them independently.
DEBUG_OPTIONS_ENV = "CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS"
DISABLE_TELEMETRY_ENV = "CORTEX_TRAINING_DISABLE_TELEMETRY"
ENABLE_SUCCESS_TELEMETRY_ENV = "CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY"

# Escape hatch for the tensor encoding of pre-tokenized prompts. Prompts are
# packed as tensors by default because every server accepts both encodings.
# Setting this puts the token ids back in the frame's JSON header; it does not
# change the request body, which is a DSSST1 frame either way.
DISABLE_TENSOR_PROMPTS_ENV = "CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_options_enabled() -> bool:
    """True when create-job debug options are explicitly enabled via env var."""
    return _env_flag_enabled(DEBUG_OPTIONS_ENV)


def _telemetry_disabled() -> bool:
    """True when OTLP client metrics are explicitly disabled via env var."""
    return _env_flag_enabled(DISABLE_TELEMETRY_ENV)


def _success_telemetry_enabled() -> bool:
    """True when successful automatic operation outcomes should be emitted."""
    return _env_flag_enabled(ENABLE_SUCCESS_TELEMETRY_ENV)


# HTTP statuses worth retrying: the request was well-formed, so the same call
# may succeed once transient load/infra conditions clear.
#   429 Too Many Requests   - rate limited; back off and retry
#   500 Internal Server Err - transient server-side failure
#   502 Bad Gateway         - upstream/proxy hiccup in front of the backend
#   503 Service Unavailable - server temporarily overloaded or restarting
#   504 Gateway Timeout     - upstream didn't respond in time
#   409 Conflict            - the target zone is restarting
#   307/308 Redirect        - a leaked internal redirect; retried on the original
#                             URL, and `Location` is never followed.
#   404 Not Found           - a just-created job may not be visible yet, so a
#                             valid id can 404 briefly and succeed on retry.
# Note the trade-off on 404: a *mistyped* job id is also retried (up to
# `max_retries`, with backoff) before the error surfaces. Every other 4xx is
# excluded because it signals a client/config error that won't fix itself.
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504, 404, 409, 307, 308}
# The service answers 500 with this message when the caller's role can't
# resolve the job's linked Experiment. That is a permission answer, not a
# transient failure, so retrying it only delays the error by a minute.
_EXPERIMENT_ACCESS_DENIED = "Linked experiment is not found or not authorized"
_SESSION_EXPIRED_CODES = frozenset({"390111", "390112", "390114"})
_CHUNK_GROUP_RESTART_REQUIRED = "chunk_group_restart_required"
_CHUNK_GROUP_ERROR_CODES = {
    _CHUNK_GROUP_RESTART_REQUIRED,
    "chunk_group_conflict",
    "chunk_group_missing_chunks",
}

# Ceiling on one ``/operation`` JSON request body. A binary payload is
# base64-encoded and inflated 4/3, so this default is conservative relative
# to the 60 MiB octet-stream budget. Raise it via the env var once the real
# ceiling is measured.
_OPERATION_MAX_JSON_BYTES_ENV = "CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES"
_DEFAULT_OPERATION_MAX_JSON_BYTES = 16 * 1024 * 1024
_MIN_OPERATION_MAX_JSON_BYTES = 16 * 1024

# Headroom for everything in the envelope that is not the base64 payload:
# ``operation_type``, the routing hints, the JSON punctuation and key names.
_OPERATION_ENVELOPE_OVERHEAD_BYTES = 4096


def _operation_max_json_bytes() -> int:
    """Read the per-request ``/operation`` JSON body ceiling from the env."""
    raw = os.environ.get(_OPERATION_MAX_JSON_BYTES_ENV, "").strip()
    if not raw:
        return _DEFAULT_OPERATION_MAX_JSON_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_OPERATION_MAX_JSON_BYTES_ENV} must be an integer, got {raw!r}"
        ) from exc
    if value < _MIN_OPERATION_MAX_JSON_BYTES:
        raise ValueError(
            f"{_OPERATION_MAX_JSON_BYTES_ENV} must be at least "
            f"{_MIN_OPERATION_MAX_JSON_BYTES} bytes"
        )
    return value


def _operation_chunk_max_bytes(max_json_bytes: int) -> int:
    """Raw frame bytes that fit one ``/operation`` body once base64-inflated."""
    return max(1, (max_json_bytes - _OPERATION_ENVELOPE_OVERHEAD_BYTES) * 3 // 4)


def _operation_body_json_bytes(body: dict) -> int:
    """Encoded size of an operation body without re-serializing its payload.

    ``json.dumps`` on a body holding a multi-megabyte base64 string copies that
    string. The base64 alphabet needs no JSON escaping and is single-byte in
    UTF-8, so the payload's encoded length is its character count: measure the
    envelope around an empty placeholder and add that count back.
    """
    payload = body.get("payload")
    raw = payload.get("payload_b64") if isinstance(payload, dict) else None
    if not isinstance(raw, str):
        return len(json.dumps(body).encode("utf-8"))
    skeleton = {**body, "payload": {**payload, "payload_b64": ""}}
    return len(json.dumps(skeleton).encode("utf-8")) + len(raw)


def _log_safe_operation_body(body: dict) -> dict:
    """Replace an operation body's base64 payload with its length for logging."""
    payload = body.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("payload_b64"), str):
        return body
    return {
        **body,
        "payload": {
            **payload,
            "payload_b64": f"<{len(payload['payload_b64'])} base64 chars>",
        },
    }


class ChunkGroupError(requests.exceptions.HTTPError):
    """Base class for structured chunk-group failures."""

    def __init__(self, message: str, *, response: requests.Response, detail: dict):
        super().__init__(
            message,
            response=response,
            request=getattr(response, "request", None),
        )
        self.detail = detail


class ChunkGroupRestartError(ChunkGroupError):
    """The server lost an incomplete group and the bounded replay also failed."""


class ChunkGroupConflictError(ChunkGroupError):
    """The same chunk-group identity was reused inconsistently."""


# Default per-request HTTP timeout, as a (connect, read) pair in seconds. The
# read leg accommodates long-running data-plane calls; fast control-plane calls
# such as GetCapacity apply a smaller bound explicitly.
_DEFAULT_REQUEST_TIMEOUT = (30.0, 600.0)
_CAPACITY_REQUEST_TIMEOUT = (10.0, 30.0)


def _validated_request_timeout(
    request_timeout: float | tuple[float, float],
) -> tuple[float, float]:
    values = (
        request_timeout
        if isinstance(request_timeout, tuple)
        else (request_timeout, request_timeout)
    )
    if len(values) != 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError(
            "request_timeout must be a positive finite number of seconds, or a "
            f"(connect, read) pair of them; got {request_timeout!r}"
        )
    return float(values[0]), float(values[1])


class _DefaultTimeoutSession(requests.Session):
    """Apply a default timeout to every request that does not provide one.

    Redirects are never followed: a leaked internal redirect points at an
    address the client cannot reach, so the original URL is retried instead.
    """

    def __init__(self, default_timeout: tuple[float, float]):
        super().__init__()
        self.default_timeout = default_timeout

    def request(
        self,
        *args: Any,
        timeout: Any = None,
        **kwargs: Any,
    ) -> requests.Response:  # type: ignore[override]
        # Session.get() injects allow_redirects=True before it calls request().
        kwargs["allow_redirects"] = False
        return super().request(
            *args,
            timeout=self.default_timeout if timeout is None else timeout,
            **kwargs,
        )


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = exc.response
        if resp is None or resp.status_code not in _TRANSIENT_STATUSES:
            return False
        return not _is_experiment_access_denied(resp)
    return False


def _is_experiment_access_denied(response: requests.Response) -> bool:
    """Whether an error body says the caller can't resolve the job's Experiment."""
    try:
        text = response.text
    except Exception:
        return False
    return isinstance(text, str) and _EXPERIMENT_ACCESS_DENIED in text


def _iter_error_dicts(value: Any, *, depth: int = 0) -> Iterator[dict]:
    """Yield dictionaries from direct or service-wrapped JSON error bodies."""
    if depth > 8:
        return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_error_dicts(child, depth=depth + 1)
        return
    if isinstance(value, list):
        for child in value:
            yield from _iter_error_dicts(child, depth=depth + 1)
        return
    if not isinstance(value, str):
        return

    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char not in "[{":
            continue
        try:
            decoded, _ = decoder.raw_decode(value[index:])
        except (TypeError, ValueError):
            continue
        yield from _iter_error_dicts(decoded, depth=depth + 1)


def _is_session_auth_expired_response(response: requests.Response) -> bool:
    """Return whether the API explicitly rejected an expired session token."""
    structured_code_seen = False
    try:
        body = response.json()
    except Exception:
        body = None
    for candidate in _iter_error_dicts(body):
        code = candidate.get("code") or candidate.get("error_code")
        if code is None:
            continue
        structured_code_seen = True
        if str(code) in _SESSION_EXPIRED_CODES:
            return True
    # A bare 401 is an authentication rejection, but a structured non-expiry
    # code must not be turned into a reconnect loop.
    return response.status_code == 401 and not structured_code_seen


def _chunk_group_error_detail(response: requests.Response | None) -> dict | None:
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    for candidate in _iter_error_dicts(body):
        if candidate.get("code") in _CHUNK_GROUP_ERROR_CODES:
            return candidate
        message = str(candidate.get("message") or "")
        if candidate.get("chunk_group_id") and message == "request chunk group is missing chunks":
            return {
                **candidate,
                "code": "chunk_group_missing_chunks",
            }
    return None


_METRIC_IDENTITY_FIELDS = (
    "job_id",
    "sub_job_id",
    "sub_job_type",
    "source_job_id",
    "source_sub_job_id",
    "target_sub_job_id",
    "target_sub_job_ids",
    "request_id",
    "checkpoint_id",
)
_REQUEST_ID_RESULT_OPERATIONS = {
    "forward_backward",
    "forward",
    "generate",
    "generate_stream",
    "step",
    "save",
    "load",
    "weight_sync",
    "bootstrap_router_replay",
    "router_replay_discard",
    "reset_prefix_cache",
    "cancel_request",
}
_METRIC_ERROR_MESSAGE_LIMIT = 512
_SECRET_IN_ERROR_RE = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*"
    r"(?:bearer\s+|snowflake\s+token\s*=\s*)?|"
    r"bearer\s+|snowflake\s+token\s*=\s*|"
    r"(?:password|secret|credential|pat|api[_-]?key|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|token)"
    r"[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|\S+)"
)
# A URL's query string carries what the caller sent, such as the account-level
# list's filters and page token, so it is cut from every telemetry message.
_URL_QUERY_IN_ERROR_RE = re.compile(r"((?:https?://|/)[^\s?#]*)\?[^\s#]*", re.IGNORECASE)


def _safe_metric_error_message(exc: BaseException) -> str:
    """Return a bounded, single-line error message without URL queries or obvious secrets."""
    # Cut before redacting so the patterns never scan an unbounded message;
    # cutting cannot expose more of a URL query or a secret than redaction would.
    message = " ".join(str(exc).split())[: 4 * _METRIC_ERROR_MESSAGE_LIMIT]
    message = _URL_QUERY_IN_ERROR_RE.sub(r"\1?<redacted>", message)
    message = _SECRET_IN_ERROR_RE.sub(r"\1<redacted>", message)
    return message[:_METRIC_ERROR_MESSAGE_LIMIT]


_SNOWFLAKE_REQUEST_ID_HEADER = "x-snowflake-request-id"
# Bound per-operation history so long polls (wait_for_job) stay cheap.
_MAX_TRACKED_SNOWFLAKE_REQUEST_IDS = 10


def _response_snowflake_request_id(response: Any) -> str | None:
    headers = getattr(response, "headers", None) or {}
    request_id = headers.get(_SNOWFLAKE_REQUEST_ID_HEADER)
    return request_id if isinstance(request_id, str) and request_id else None


def _add_snowflake_request_id_to_error(
    exc: BaseException, request_id: str | None
) -> None:
    """Prefix the Snowflake request id to an error message so tracebacks carry it.

    The id goes first so it survives truncation of long messages, e.g. the
    512-character telemetry ``error_message``.
    """
    if not request_id or not exc.args or not isinstance(exc.args[0], str):
        return
    if request_id in exc.args[0]:
        return
    exc.args = (
        f"[snowflake request id: {request_id}] {exc.args[0]}",
        *exc.args[1:],
    )


def _snowflake_request_attributes(
    error: BaseException | None,
    request_ids: list[str],
) -> dict[str, Any]:
    """Snowflake request id attributes for an operation outcome.

    ``error.snowflake_request_id`` is the failing response's id, set only when
    that response carried the header. ``snowflake.request_ids`` is the
    operation's recent Snowflake request ids, newest last; no entry in it is
    implied to have failed.
    """
    attributes: dict[str, Any] = {}
    if error is not None:
        request_id = _response_snowflake_request_id(getattr(error, "response", None))
        if request_id is not None:
            attributes["error.snowflake_request_id"] = request_id
            # Deprecated name for the same value; remove after one release.
            attributes["snowflake.request_id"] = request_id
    if request_ids:
        attributes["snowflake.request_ids"] = list(request_ids)
    return attributes


_E = TypeVar("_E", bound=BaseException)
_SERVER_FAILURE_ATTR = "_snowflake_server_failure"


def _server_failure(exc: _E, failed: bool = True) -> _E:
    """Mark ``exc`` as caused by a server outcome when ``failed`` is true.

    Marked errors (a request or job that ended ``failed``, a poll that timed
    out, a response payload that could not be decoded) carry no response of
    their own, so :func:`_log_untagged_request_id` names the operation's last
    Snowflake request id for them.
    """
    if failed:
        try:
            setattr(exc, _SERVER_FAILURE_ATTR, True)
        except Exception:
            pass
    return exc


@contextlib.contextmanager
def _server_payload() -> Iterator[None]:
    """Treat decode errors in the enclosed code as server failures."""
    try:
        yield
    except ValueError as exc:
        raise _server_failure(exc)


def _log_untagged_request_id(
    operation: str,
    exc: BaseException,
    request_ids: list[str],
) -> None:
    """Log the last Snowflake request id for a server-side failure with no response.

    This is the fallback for :func:`_add_snowflake_request_id_to_error`, which
    tags any error that has a response. It covers:

    - transport errors with no response, such as a connection error or
      timeout after retries;
    - errors marked by :func:`_server_failure`: a request or job that ended
      ``failed``, a poll that timed out, or an undecodable response payload.

    It logs ``request_ids[-1]`` labeled "last" because that request did not
    necessarily fail. It logs nothing when the operation made no Snowflake
    requests, when the error has a response (already tagged, or the response
    lacked the header), or for any other error, such as client-side
    validation or a job that reached ``done`` or ``cancelled``. Telemetry
    still records ``snowflake.request_ids`` for those.
    """
    try:
        if getattr(exc, "response", None) is not None or not request_ids:
            return
        if not (
            isinstance(exc, requests.exceptions.RequestException)
            or getattr(exc, _SERVER_FAILURE_ATTR, False)
        ):
            return
        request_id = request_ids[-1]
        if request_id in str(exc):
            return
        logger.warning(
            "[last snowflake request id: %s] %s failed: %s",
            request_id,
            operation,
            _safe_metric_error_message(exc),
        )
    except Exception:
        logger.debug("client operation request id logging failed", exc_info=True)


def _metric_error_attributes(exc: BaseException) -> dict[str, Any]:
    attributes: dict[str, Any] = {"error.type": type(exc).__name__}
    response = getattr(exc, "response", None)
    if response is None:
        return attributes
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        attributes["error.http_status"] = status_code
    try:
        body = response.json()
    except Exception:
        return attributes
    for candidate in _iter_error_dicts(body):
        code = candidate.get("code") or candidate.get("error_code")
        if code is not None:
            attributes["error.code"] = str(code)
            break
    return attributes


def _metric_identity(
    operation: str,
    arguments: Mapping[str, Any],
    result: Any = None,
) -> dict[str, Any]:
    attributes = {}
    for field in _METRIC_IDENTITY_FIELDS:
        value = arguments.get(field)
        if value is not None:
            attributes[field] = (
                list(value) if isinstance(value, (list, tuple)) else value
            )
    if operation == "create_job" and "job_id" not in attributes:
        if isinstance(result, str):
            attributes["job_id"] = result
        elif isinstance(result, dict) and result.get("job_id") is not None:
            attributes["job_id"] = result["job_id"]
    if operation in _REQUEST_ID_RESULT_OPERATIONS and "request_id" not in attributes:
        if isinstance(result, str):
            attributes["request_id"] = result
        elif isinstance(result, dict) and result.get("request_id") is not None:
            attributes["request_id"] = result["request_id"]
    return attributes


def _track_operation(operation: str):
    """Record one outcome metric around a public client operation."""

    def decorate(method):
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            state = getattr(self, "_operation_metric_state", None)
            if state is None:
                return method(self, *args, **kwargs)
            active = getattr(state, "active", set())
            if active:
                return method(self, *args, **kwargs)
            # Snowflake request ids are tracked even with telemetry off so a
            # failure can still be logged with the id; only metrics need the
            # arguments.
            arguments = None
            if getattr(self, "_metric_emitter", None) is not None:
                try:
                    bound = signature.bind(self, *args, **kwargs)
                    bound.apply_defaults()
                    arguments = bound.arguments
                except TypeError:
                    return method(self, *args, **kwargs)
                except Exception:
                    logger.debug("client operation metric setup failed", exc_info=True)
                    return method(self, *args, **kwargs)
            active = set()
            active.add(operation)
            state.active = active
            state.attempt_count = 0
            state.request_count = 0
            state.snowflake_request_ids = []
            started = time.monotonic()
            try:
                result = method(self, *args, **kwargs)
            except Exception as exc:
                # Errors wrapped after _send (e.g. chunk-group errors) still
                # carry the failing response.
                _add_snowflake_request_id_to_error(
                    exc, _response_snowflake_request_id(getattr(exc, "response", None))
                )
                _log_untagged_request_id(operation, exc, state.snowflake_request_ids)
                if arguments is not None:
                    self._emit_operation_outcome(
                        operation,
                        started,
                        arguments,
                        error=exc,
                    )
                raise
            else:
                if arguments is not None:
                    self._emit_operation_outcome(
                        operation,
                        started,
                        arguments,
                        result=result,
                    )
                return result
            finally:
                state.active = set()
                state.attempt_count = 0
                state.request_count = 0
                state.snowflake_request_ids = []

        return wrapped

    return decorate


def _is_chunk_post_transient(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        if _chunk_group_error_detail(exc.response) is not None:
            return False
    return _is_transient(exc)


# ─── Account-level job list ──────────────────────────────────────────────

# One fixed path, whatever ``endpoint`` says; a listed job is reached through
# its own schema with for_job().
_ACCOUNT_JOBS_PATH = "/api/v2/cortex-training"
_ACCOUNT_JOBS_MAX_PAGE_SIZE = 1000
# `done` used to be a group, and the service now answers 400 for it; `terminal`
# replaced it and also takes `failed`. The client keeps `done`'s old meaning by
# sending what it stood for. Legacy DONE rows match no filter on this route.
_ACCOUNT_STATUS_ALIASES = {"done": ("cancelled", "terminated")}
# The service's message when the route is deployed but not enabled for the
# account. Any other 404 means the route is missing; the service sends those
# with an empty body.
_ACCOUNT_LIST_NOT_ENABLED_MESSAGE = "account-level job listing is not enabled for this account"
_INVALID_PAGE_TOKEN_MESSAGE = "invalid page_token"
# How much of the service's error message an exception's own message quotes.
_SERVICE_MESSAGE_LIMIT = 300


class AccountJobListUnavailable(requests.exceptions.HTTPError):
    """The account-level job list is not available to this caller.

    Raised by :meth:`CortexTrainingClient.list_jobs_page` and
    :meth:`CortexTrainingClient.iter_jobs` at once, without retries. ``reason``
    is one of:

    - ``"not_deployed"``: 404 without the service's not-enabled error, normally
      with an empty body: the service does not have the route yet. With some
      other body, a proxy in front of the service may have answered instead.
    - ``"disabled"``: 404 whose JSON body says account-level job listing is not
      enabled for this account.
    - ``"forbidden"``: 403. When the answer carries the service's reason, the
      message quotes it: Cortex Training is not enabled for the account, its
      trial has ended, or, under a ``status`` filter, the service was refused
      a status lookup. A 403 without one can also come from a network policy
      or a proxy in front of the service.
    - ``"rest_disabled"``: 405. The Cortex Training REST API is turned off for
      the account.

    ``not_deployed`` and ``disabled`` concern this route only, so
    :meth:`CortexTrainingClient.list_jobs` still lists the configured schema.
    ``forbidden`` and ``rest_disabled`` mostly come from gates the per-schema
    routes share, so expect those to fail the same way. A subclass of
    ``requests.HTTPError``, so code that catches that keeps working;
    ``response`` is the service's answer, and ``str()`` is one line, which can
    quote the service's own message.
    """

    NOT_DEPLOYED = "not_deployed"
    DISABLED = "disabled"
    FORBIDDEN = "forbidden"
    REST_DISABLED = "rest_disabled"
    REASONS = frozenset({NOT_DEPLOYED, DISABLED, FORBIDDEN, REST_DISABLED})

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        response: requests.Response | None = None,
        request: requests.Request | requests.PreparedRequest | None = None,
    ):
        super().__init__(message, response=response, request=request)
        self.reason = reason

    def __reduce__(self):
        # copy and pickle rebuild an exception from its args, which leave out
        # the keyword-only reason.
        return (functools.partial(type(self), reason=self.reason), self.args, self.__dict__)


class InvalidPageToken(requests.exceptions.HTTPError):
    """The service could not open a ``page_token``: 400 ``Invalid page_token``.

    It happens, for example, after the service rotates the key that seals its
    tokens. Start the walk again from the first page.
    :meth:`CortexTrainingClient.iter_jobs` does that once by itself.
    """


def _error_body_candidates(response: requests.Response) -> Iterator[dict]:
    try:
        body = response.json()
    except Exception:
        return iter(())
    return _iter_error_dicts(body)


def _service_error_message(response: requests.Response) -> str | None:
    """The message in the service's JSON error body, on one bounded line."""
    for candidate in _error_body_candidates(response):
        message = candidate.get("message")
        if isinstance(message, str) and message.strip():
            message = " ".join(message.split())
            if len(message) > _SERVICE_MESSAGE_LIMIT:
                message = message[: _SERVICE_MESSAGE_LIMIT - 3] + "..."
            return message
    return None


def _has_body(response: requests.Response) -> bool:
    try:
        return bool((response.content or b"").strip())
    except Exception:
        return True


def _account_list_unavailable_message(reason: str, response: requests.Response) -> str:
    """Why the account-level list is unavailable, claiming no more than the answer shows."""
    if reason == AccountJobListUnavailable.NOT_DEPLOYED:
        if not _has_body(response):
            return (
                "account-level job listing is not available on this service yet: "
                "GET /api/v2/cortex-training answered 404 with no error body, so "
                "the route is not deployed"
            )
        # A body without the service's error: often a proxy's or gateway's page,
        # which says nothing about whether the route is deployed.
        return (
            "account-level job listing is not available: GET /api/v2/cortex-training "
            "answered 404 without the service's not-enabled error; the route may not "
            "be deployed yet, or a proxy in front of the service answered"
        )
    if reason == AccountJobListUnavailable.DISABLED:
        return (
            "account-level job listing is not enabled for this account (HTTP 404); "
            "Snowflake enables it per account"
        )
    if reason == AccountJobListUnavailable.FORBIDDEN:
        message = _service_error_message(response)
        if message is not None:
            return f"account-level job listing was refused (HTTP 403): {message}"
        return (
            "account-level job listing was refused (HTTP 403) with no error "
            "message: a network policy or a proxy may have blocked the request, "
            "or Cortex Training is not enabled for this account or its trial has "
            "ended"
        )
    return (
        "account-level job listing is unavailable: the Cortex Training REST API "
        "is turned off for this account (HTTP 405)"
    )


def _says_account_list_not_enabled(response: requests.Response) -> bool:
    return any(
        isinstance(candidate.get("message"), str)
        and _ACCOUNT_LIST_NOT_ENABLED_MESSAGE in candidate["message"].lower()
        for candidate in _error_body_candidates(response)
    )


def _says_invalid_page_token(response: requests.Response) -> bool:
    # A prefix, should the service add detail. Not "page_token does not match
    # the request filters": that one is a caller sending different filters,
    # which a restart would only hide.
    return any(
        isinstance(candidate.get("message"), str)
        and candidate["message"].strip().lower().startswith(_INVALID_PAGE_TOKEN_MESSAGE)
        for candidate in _error_body_candidates(response)
    )


def _account_list_unavailable_reason(response: requests.Response) -> str | None:
    status_code = getattr(response, "status_code", None)
    if status_code == 405:
        return AccountJobListUnavailable.REST_DISABLED
    if status_code == 403:
        return AccountJobListUnavailable.FORBIDDEN
    if status_code != 404:
        return None
    if _says_account_list_not_enabled(response):
        return AccountJobListUnavailable.DISABLED
    return AccountJobListUnavailable.NOT_DEPLOYED


def _account_list_error(
    exc: requests.exceptions.HTTPError, *, page_token_sent: bool
) -> requests.exceptions.HTTPError | None:
    """The typed error for an account-list failure, or None to raise ``exc`` as is."""
    response = exc.response
    if response is None:
        return None
    reason = _account_list_unavailable_reason(response)
    if reason is not None:
        error: requests.exceptions.HTTPError = AccountJobListUnavailable(
            _account_list_unavailable_message(reason, response),
            reason=reason,
            response=response,
            request=exc.request,
        )
    elif (
        page_token_sent
        and response.status_code == 400
        and _says_invalid_page_token(response)
    ):
        error = InvalidPageToken(
            "the service could not open page_token (HTTP 400 Invalid page_token); "
            "start again from the first page",
            response=response,
            request=exc.request,
        )
    else:
        return None
    _add_snowflake_request_id_to_error(error, _response_snowflake_request_id(response))
    return error


def _page_token_rejected_again(exc: InvalidPageToken) -> InvalidPageToken:
    """``iter_jobs``' error when a token fails again after the walk started over.

    Starting again from the first page, the advice of the first such error,
    has already been tried.
    """
    error = InvalidPageToken(
        "the service could not open page_token (HTTP 400 Invalid page_token) a second "
        "time, after the walk had started over from the first page; list the jobs again",
        response=exc.response,
        request=exc.request,
    )
    _add_snowflake_request_id_to_error(error, _response_snowflake_request_id(exc.response))
    return error


def _optional_filter(value: Any, name: str) -> str | None:
    """A string filter, or None when it is None or empty, as the service reads it."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _account_status_param(status: str | Iterable[str] | None) -> str | None:
    """The ``status`` query value: a comma list with ``done`` spelled out.

    Words keep their case (the service ignores it); empty words and repeats are
    dropped, and an empty list means no filter, as it does on the service.
    """
    if status is None:
        return None
    if isinstance(status, str):
        words: Iterable[Any] = status.split(",")
    elif isinstance(status, Iterable):
        words = status
    else:
        raise ValueError(
            f"status must be a string or a list of strings, got {type(status).__name__}"
        )
    sent: list[str] = []
    seen: set[str] = set()
    for word in words:
        if not isinstance(word, str):
            raise ValueError(
                f"status must be a string or a list of strings, got a {type(word).__name__} in it"
            )
        word = word.strip()
        if not word:
            continue
        for value in _ACCOUNT_STATUS_ALIASES.get(word.lower(), (word,)):
            if value.lower() not in seen:
                seen.add(value.lower())
                sent.append(value)
    return ",".join(sent) or None


def _is_whole_number(value: Any) -> bool:
    """An int, or another integral type such as numpy's, but not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _account_page_size(page_size: Any) -> int:
    if not _is_whole_number(page_size):
        raise ValueError(
            f"page_size must be an int, got {type(page_size).__name__} {page_size!r}"
        )
    if not 1 <= page_size <= _ACCOUNT_JOBS_MAX_PAGE_SIZE:
        raise ValueError(
            f"page_size must be between 1 and {_ACCOUNT_JOBS_MAX_PAGE_SIZE}, got {page_size}"
        )
    return int(page_size)


def _account_page_bounds(
    max_retries: Any, request_timeout: Any
) -> tuple[int | None, tuple[float, float] | None]:
    """Validated per-call ``max_retries`` and ``request_timeout`` (one page of
    the account-level list, or one ``list_jobs``); None keeps the client's."""
    if max_retries is not None:
        if not _is_whole_number(max_retries) or max_retries < 0:
            raise ValueError(f"max_retries must be an int >= 0, got {max_retries!r}")
        max_retries = int(max_retries)
    timeout = None if request_timeout is None else _validated_request_timeout(request_timeout)
    return max_retries, timeout


# A response body at least this large is decoded in pieces; see _json_body. (The
# growth it avoids began between bodies of 1.0 and 1.4 MB, with a 4-byte character.)
_PIECEWISE_DECODE_BYTES = 1 << 18
_DECODE_PIECE_BYTES = 1 << 16


def _json_body(resp: requests.Response) -> Any:
    """``resp.json()``, decoded so that reading a large job list again and again
    doesn't keep growing the process.

    Decoding a large body that holds non-ASCII text (a ``·`` in a comment, say)
    in one call makes a string that CPython first sizes for the byte count and
    then shrinks in place. Some allocators (the macOS one among them) never
    reuse that memory, so the TUI, which lists the jobs again every minute, grew
    by the size of the decoded list each time: hundreds of MB within the hour
    for a few thousand jobs. Decoding in 64 KiB pieces and joining them sizes
    the string once. A smaller body, and a body that isn't JSON, go through
    ``resp.json()``, which raises as before.
    """
    content = resp.content
    if not content or len(content) < _PIECEWISE_DECODE_BYTES:
        return resp.json()
    try:
        decoder = codecs.getincrementaldecoder(json.detect_encoding(content))(errors="surrogatepass")
        pieces = [decoder.decode(content[at:at + _DECODE_PIECE_BYTES])
                  for at in range(0, len(content), _DECODE_PIECE_BYTES)]
        pieces.append(decoder.decode(b"", final=True))
        return json.loads("".join(pieces))
    except ValueError:  # not text, or not JSON: raise what resp.json() raises
        return resp.json()


def _account_jobs_page(body: Any) -> dict:
    """``{"jobs": [...], "next_page_token": str | None}`` from a page's JSON body."""
    if not isinstance(body, dict):
        raise _server_failure(
            ValueError("the account-level job list answered with a non-object JSON body")
        )
    jobs = body.get("jobs")
    if jobs is None:
        jobs = []
    if not isinstance(jobs, list):
        raise _server_failure(
            ValueError("the account-level job list answered with a non-list 'jobs'")
        )
    token = body.get("next_page_token")
    return {"jobs": jobs, "next_page_token": token if isinstance(token, str) and token else None}


def _is_account_list_transient(exc: BaseException) -> bool:
    # 403, 404 and 405 are the gates in front of the account-level list, which
    # no retry opens.
    if isinstance(exc, requests.exceptions.HTTPError):
        if exc.response is not None and exc.response.status_code in (403, 404, 405):
            return False
    return _is_transient(exc)


def _is_tail_logs_transient(exc: BaseException) -> bool:
    # tail-logs answers 409 while the job is not running (still pending
    # placement, or ended) and 404 once its session is gone. Neither clears on
    # a retry; get_job says which it is.
    if isinstance(exc, requests.exceptions.HTTPError):
        if exc.response is not None and exc.response.status_code in (404, 409):
            return False
    return _is_transient(exc)


# The service answers 500 "Linked experiment is not found for job ..." from the
# experiment-run route (and "... or not authorized for the training job" from
# others) when it can't resolve the job's Experiment for the caller's role.
_LINKED_EXPERIMENT_NOT_FOUND = "Linked experiment is not found"


def _is_experiment_run_transient(exc: BaseException) -> bool:
    # A retry can't make the job's Experiment readable; it only delays the
    # error by a minute.
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        try:
            text = exc.response.text
        except Exception:  # noqa: BLE001 - an unreadable body falls back to the general rules
            text = ""
        if isinstance(text, str) and _LINKED_EXPERIMENT_NOT_FOUND in text:
            return False
    return _is_transient(exc)


def _is_connect_error(exc: BaseException) -> bool:
    """Only failures proving the request never reached the server (safe for create_job)."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        cause = exc.args[0] if exc.args else None
        reason = getattr(cause, "reason", None)
        return isinstance(cause, NewConnectionError) or isinstance(reason, NewConnectionError)
    return False


# ─── Canonical types ─────────────────────────────────────────────────────


class JobType(str, Enum):
    """Sub-job types defined by the Cortex Training schema.

    Matches the ``job_type`` enum in the REST schema; see
    ``docs/reference/rest-api.md`` section 8.1. Not every schema value can
    currently be submitted: ``LOG_PROBABILITY`` configs still validate and
    serialize, but the client and CLI reject them at submission.
    """

    TRAINING = "training"
    SAMPLING = "sampling"
    LOG_PROBABILITY = "log_probability"  # Schema value; submission is unsupported.


_LOG_PROBABILITY_JOB_TYPE_ALIASES = frozenset({"log_probability", "log_prob"})
_UNSUPPORTED_LOG_PROBABILITY_MESSAGE = (
    "log_probability sub-jobs are not currently supported"
)


def _normalized_job_type(job_type: Any) -> str:
    """Short-form ``job_type`` for raw bodies, which may use the enum spelling."""
    if not isinstance(job_type, str):
        return ""
    return job_type.strip().lower().removeprefix("job_type_")


def _is_log_probability_job_type(job_type: Any) -> bool:
    return _normalized_job_type(job_type) in _LOG_PROBABILITY_JOB_TYPE_ALIASES


def _reject_log_probability_sub_job(job_type: Any, *, location: str) -> None:
    if _is_log_probability_job_type(job_type):
        raise ValueError(f"{location}: {_UNSUPPORTED_LOG_PROBABILITY_MESSAGE}")


# The create-job ``comment``: free text the service stores with the job and
# returns unchanged from get and list. It is set at creation only (there is no
# route to change it) and holds at most this many bytes of UTF-8.
MAX_COMMENT_BYTES = 1024


def _is_model_deployment(sub_job_configs: Any) -> bool:
    """Whether a create body is a model deployment, the way the server decides:
    some sub-job's ``inference_config`` sets ``deployment_class``."""
    if not isinstance(sub_job_configs, list):
        return False
    for cfg in sub_job_configs:
        inference = cfg.get("inference_config") if isinstance(cfg, dict) else None
        if isinstance(inference, dict) and inference.get("deployment_class") is not None:
            return True
    return False


def _body_with_checked_comment(body: dict) -> dict:
    """Return the create body to send, with its ``comment`` checked.

    Every create path sends its body through this: :meth:`create_job`,
    :meth:`create_job_from_body` and ``cortex-training submit`` (also with
    ``--dry-run``). A model deployment never carries a comment, because the
    server rejects one there: it is left out of the returned body, unchecked,
    with a warning when it had text. A blank comment (empty, or only
    whitespace) names nothing and is left out as well. For any other job a
    comment that is not a string, is not valid UTF-8 text, or is longer than
    :data:`MAX_COMMENT_BYTES` bytes of UTF-8 raises :class:`ValueError` before
    anything is sent. The caller's dict is never changed.
    """
    comment = body.get("comment")
    if comment is None:
        return body
    without_comment = {key: value for key, value in body.items() if key != "comment"}
    if _is_model_deployment(body.get("sub_job_configs")):
        if str(comment).strip():
            logger.warning(
                "comment not sent: this create request is a model deployment, "
                "which does not accept one"
            )
        return without_comment
    if not isinstance(comment, str):
        raise ValueError(f"comment must be a string, got {type(comment).__name__}")
    if not comment.strip():
        return without_comment
    try:
        size = len(comment.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"comment must be valid UTF-8 text; {_not_utf8(comment, exc.start)}") from None
    if size > MAX_COMMENT_BYTES:
        raise ValueError(
            f"comment must be at most {MAX_COMMENT_BYTES} bytes of UTF-8, got {size}"
        )
    return body


def _not_utf8(text: str, index: int) -> str:
    """Why ``text[index]``, an unpaired surrogate, has no UTF-8 form."""
    code = ord(text[index])
    if 0xDC80 <= code <= 0xDCFF:
        # How Python decodes a byte that isn't UTF-8 in a command-line argument.
        return f"character {index + 1} stands for the byte 0x{code - 0xDC00:02X}, which is not UTF-8"
    return f"character {index + 1} is U+{code:04X}, an unpaired surrogate"


class Hardware(str, Enum):
    """GPU hardware for create job and capacity.

    Wire values match the REST ``hardware`` field; see
    ``docs/reference/rest-api.md`` sections 5.1 and 5.4. Omitting it defaults to
    H200 on the server. Unknown values are rejected client-side.
    """

    H200 = "H200"
    B200 = "B200"
    B300 = "B300"


def _hardware_wire(hardware: Hardware | str) -> str:
    """Return the wire spelling for ``hardware``."""
    if isinstance(hardware, Hardware):
        return hardware.value
    try:
        return Hardware(hardware).value
    except ValueError as exc:
        allowed = ", ".join(member.value for member in Hardware)
        raise ValueError(f"hardware must be one of {allowed}, got {hardware!r}") from exc


# ``get_capacity_breakdown`` dimensions, statuses, and the job types always
# reported under ``job_type``, in order.
_CAPACITY_BREAKDOWN_DIMENSIONS = ("job_type", "submitted_by")
_CAPACITY_BREAKDOWN_JOB_TYPES = ("TRAINING", "SAMPLING", "LOG_PROBABILITY")


def _capacity_totals(body: Mapping[str, Any]) -> dict:
    """Return the six ``/capacity`` totals, filling proto3-omitted defaults."""
    return {
        "has_reservation": bool(body.get("has_reservation", False)),
        "max_total_gpus": int(body.get("max_total_gpus", 0)),
        "reserved_gpus": int(body.get("reserved_gpus", 0)),
        "in_use_gpus": int(body.get("in_use_gpus", 0)),
        "pending_gpus": int(body.get("pending_gpus", 0)),
        "available_gpus": int(body.get("available_gpus", 0)),
    }


def _capacity_breakdown_dimensions(dimensions: Any) -> tuple[str, ...]:
    """Validate ``get_capacity_breakdown(dimensions=...)`` before any request."""
    try:
        names = tuple(dimensions)
    except TypeError:
        names = ()
    if not names or not all(
        isinstance(name, str) and name in _CAPACITY_BREAKDOWN_DIMENSIONS
        for name in names
    ):
        allowed = ", ".join(_CAPACITY_BREAKDOWN_DIMENSIONS)
        raise ValueError(
            f"dimensions must be a non-empty iterable of {allowed} "
            f"(not a string), got {dimensions!r}"
        )
    # Use our own strings: a str Enum member equal to a name formats differently.
    return tuple(
        _CAPACITY_BREAKDOWN_DIMENSIONS[_CAPACITY_BREAKDOWN_DIMENSIONS.index(name)]
        for name in names
    )


def _capacity_breakdown(
    body: Mapping[str, Any], dimensions: tuple[str, ...]
) -> dict[str, Any]:
    """Normalize ``breakdown``, trusting the data over the status label:
    groups are kept whenever the server sends them, and ``status`` is passed
    through as sent. Only a missing status is derived: ``OK`` with groups,
    ``UNAVAILABLE`` without (also for a dimension the server left out)."""
    raw = body.get("breakdown")
    if not isinstance(raw, dict):
        raw = {}
    breakdown: dict[str, Any] = {
        "job_count": int(raw.get("job_count", 0)),
        "truncated": bool(raw.get("truncated", False)),
        "unsupported_dimensions": list(raw.get("unsupported_dimensions") or []),
    }
    for dimension in dict.fromkeys(dimensions):
        section = raw.get(f"by_{dimension}")
        if not isinstance(section, dict):
            section = {}
        groups = section.get("groups")
        has_groups = isinstance(groups, list) and bool(groups)
        status = section.get("status") or ("OK" if has_groups else "UNAVAILABLE")
        if not has_groups:
            rendered_groups = []
        elif dimension == "job_type":
            rendered_groups = _capacity_job_type_groups(groups)
        else:
            rendered_groups = [_capacity_submitted_by_group(group) for group in groups]
        breakdown[f"by_{dimension}"] = {"status": status, "groups": rendered_groups}
    return breakdown


def _capacity_job_type_groups(groups: list[Mapping[str, Any]]) -> list[dict]:
    """Known job types first, zero-filled and in order, then any others as sent."""
    counts = ("in_use_gpus", "pending_gpus", "sub_job_count")
    known: dict[str, dict[str, Any]] = {
        job_type: {"job_type": job_type, **dict.fromkeys(counts, 0)}
        for job_type in _CAPACITY_BREAKDOWN_JOB_TYPES
    }
    others = []
    for group in groups:
        job_type = group.get("job_type")
        row: dict[str, Any] | None = None
        if isinstance(job_type, str):
            job_type = job_type.removeprefix("JOB_TYPE_")
            row = known.get(job_type)
        if row is None:
            row = {"job_type": job_type, **dict.fromkeys(counts, 0)}
            others.append(row)
        for key in counts:
            row[key] += int(group.get(key, 0))
    return [*known.values(), *others]


def _capacity_submitted_by_group(group: Mapping[str, Any]) -> dict:
    """Normalize one ``by_submitted_by`` group; only a ``USER`` row names anyone."""
    user = group.get("kind") == "USER"
    return {
        "kind": group.get("kind"),
        "submitted_by": group.get("submitted_by") if user else None,
        "is_caller": bool(group.get("is_caller", False)) if user else False,
        "in_use_gpus": int(group.get("in_use_gpus", 0)),
        "pending_gpus": int(group.get("pending_gpus", 0)),
        "job_count": int(group.get("job_count", 0)),
    }


def _effective_primerl_config(extra: dict) -> dict:
    prime_rl = extra.get("prime_rl")
    if isinstance(prime_rl, dict):
        # The training runtime expands the nested prime_rl block over the training
        # config, so validate the same effective values here.
        return {**extra, **prime_rl}
    return extra


def _validate_primerl_lm_head_config(extra: dict, *, location: str) -> None:
    cfg = _effective_primerl_config(extra)
    fused_cross_entropy = cfg.get("fused_cross_entropy", "liger")
    if fused_cross_entropy is False:
        return

    token_chunk_size = cfg.get("fused_lm_head_token_chunk_size")
    token_chunked_lm_head = isinstance(token_chunk_size, int) and not isinstance(token_chunk_size, bool)
    if cfg.get("fp32_lm_head", False) or token_chunked_lm_head:
        raise ValueError(
            f"{location} cannot combine fused_cross_entropy with "
            "fp32_lm_head or fused_lm_head_token_chunk_size. "
            "Set fused_cross_entropy=false to use fp32/chunked LM head, "
            "or remove fp32/chunked LM-head knobs to use fused CE."
        )


class TrainingConfig(BaseModel):
    """Training hyperparameters for a training sub-job.

    Required fields mirror the server-side validator: ``max_seq_len > 0``,
    ``train_batch_size > 0``, ``n_gpus > 0``, and an ``optimizer`` that carries
    at least one setting. ``optimizer`` is canonicalized on construction (see
    :func:`cortex_training.optimizer.normalize_optimizer_config`), so a block
    holding only ``gradient_clipping`` is hoisted onto the typed field and
    leaves ``optimizer`` empty.

    Unknown fields are passed through to the server unchanged
    (accessible via ``model_extra``). See ``docs/reference/rest-api.md`` section 8.2.
    """

    optimizer: dict = PydanticField(description="Optimizer config dict (e.g. {name: AdamW, lr: 1e-4}).")
    max_seq_len: int = PydanticField(gt=0, description="Maximum sequence length for training.")
    train_batch_size: int = PydanticField(gt=0, description="Global batch size across all GPUs.")
    n_gpus: int = PydanticField(gt=0, description="Number of GPUs (data-parallel size).")
    gradient_clipping: float | None = PydanticField(default=None, description="Optional gradient clipping threshold.")
    multiplex_job_id: str | None = PydanticField(default=None, description="Optional multiplex job identifier.")
    load_optimizer_states: bool | None = PydanticField(default=None, description="Whether to load optimizer states from checkpoints. Set to False when changing DP size.")

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _flatten_legacy_extra(cls, data: Any) -> Any:
        if isinstance(data, dict) and "extra" in data:
            extra = data.pop("extra")
            if isinstance(extra, dict):
                for k, v in extra.items():
                    data.setdefault(k, v)
        return data

    @model_validator(mode="after")
    def _check_optimizer(self) -> "TrainingConfig":
        # object.__setattr__ writes the already-normalized values without
        # re-entering assignment validation. create_job posts to_wire() of this
        # instance, so the rewrite has to land here.
        optimizer, hoisted_gradient_clipping = normalize_optimizer_config(
            self.optimizer, location="training.optimizer"
        )
        object.__setattr__(self, "optimizer", optimizer)
        if self.gradient_clipping is None and hoisted_gradient_clipping is not None:
            object.__setattr__(self, "gradient_clipping", hoisted_gradient_clipping)
        # Checked after the hoist: a clip-only block canonicalizes to ``{}`` and
        # is still a complete config, so only a block that says nothing at all
        # is an error.
        if not self.optimizer and self.gradient_clipping is None:
            raise ValueError("training.optimizer is required and must be a non-empty dict")
        extra = self.model_extra or {}
        _validate_primerl_lm_head_config(extra, location="training.extra")
        prime_rl = extra.get("prime_rl")
        if isinstance(prime_rl, dict):
            _validate_primerl_lm_head_config(prime_rl, location="training.extra.prime_rl")
        if "peft_config" in extra:
            extra["peft_config"] = normalize_lora_peft_config(
                extra["peft_config"], location="training.extra.peft_config"
            )
        return self

    def validate(self) -> None:
        """Re-validate the current state. Backward-compatible with the old dataclass API."""
        self.__class__.model_validate(self.model_dump(by_alias=True))

    def to_wire(self) -> dict:
        out: dict = {
            "optimizer": self.optimizer,
            "max_seq_len": self.max_seq_len,
            "train_batch_size": self.train_batch_size,
            "n_gpus": self.n_gpus,
        }
        if self.gradient_clipping is not None:
            out["gradient_clipping"] = self.gradient_clipping
        if self.multiplex_job_id is not None:
            out["multiplex_job_id"] = self.multiplex_job_id
        if self.load_optimizer_states is not None:
            out["load_optimizer_states"] = self.load_optimizer_states
        for k, v in (self.model_extra or {}).items():
            out.setdefault(k, v)
        return out


class InferenceConfig(BaseModel):
    """Sampling/log-probability config for an inference sub-job.

    Required fields mirror the server-side validator: ``max_seq_len > 0`` and
    ``n_gpus > 0``. Unknown fields are passed through to the server unchanged
    (accessible via ``model_extra``). See ``docs/reference/rest-api.md`` section 8.3.
    """

    max_seq_len: int = PydanticField(gt=0, description="Maximum sequence length for inference.")
    n_gpus: int = PydanticField(gt=0, description="Number of GPUs for inference.")
    multiplex_job_id: str | None = PydanticField(default=None, description="Optional multiplex job identifier.")

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _flatten_legacy_extra(cls, data: Any) -> Any:
        if isinstance(data, dict) and "extra" in data:
            extra = data.pop("extra")
            if isinstance(extra, dict):
                for k, v in extra.items():
                    data.setdefault(k, v)
        return data

    @model_validator(mode="after")
    def _check_peft_config(self) -> "InferenceConfig":
        extra = self.model_extra or {}
        if "peft_config" in extra:
            extra["peft_config"] = normalize_lora_peft_config(
                extra["peft_config"], location="sampling.extra.peft_config"
            )
        return self

    def validate(self) -> None:
        """Re-validate the current state. Backward-compatible with the old dataclass API."""
        self.__class__.model_validate(self.model_dump(by_alias=True))

    def to_wire(self) -> dict:
        out: dict = {"max_seq_len": self.max_seq_len, "n_gpus": self.n_gpus}
        if self.multiplex_job_id is not None:
            out["multiplex_job_id"] = self.multiplex_job_id
        for k, v in (self.model_extra or {}).items():
            out.setdefault(k, v)
        return out


class SubJobConfig(BaseModel):
    """One sub-job within a CreateJob request.

    Matches the ``SubJobConfig`` schema in ``docs/reference/rest-api.md``
    section 8.1.

    Exactly one of ``training`` or ``sampling`` must be set, matching the
    sub-job's ``job_type``.
    """

    job_type: JobType = PydanticField(description="Sub-job type (training, sampling, or log_probability).")
    model_name: str = PydanticField(min_length=1, description="Model identifier (e.g. Qwen/Qwen3-8B).")
    training: TrainingConfig | None = PydanticField(default=None, description="Training config; required for training sub-jobs.")
    sampling: InferenceConfig | None = PydanticField(default=None, description="Inference config; required for sampling/log-prob sub-jobs.")
    global_batch_size: int | None = PydanticField(default=None, description="Global batch size across all GPUs.")
    dtype: str | None = PydanticField(default=None, description="Model dtype (e.g. bfloat16).")
    seed: int | None = PydanticField(default=None, description="Random seed.")
    model_post_init_ops: list[str] | None = PydanticField(default=None, alias="model_post_init", description="List of post-initialization operations.")
    source_checkpoint_info: dict | None = PydanticField(default=None, description="Checkpoint to resume from at creation time.")

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    @model_validator(mode="after")
    def _check_job_type(self) -> "SubJobConfig":
        if self.training is not None and self.sampling is not None:
            raise ValueError("sub_job.training and sub_job.sampling are mutually exclusive")
        if self.job_type == JobType.TRAINING:
            if self.training is None:
                raise ValueError("training sub-job requires a `training` block")
        elif self.job_type in (JobType.SAMPLING, JobType.LOG_PROBABILITY):
            if self.sampling is None:
                raise ValueError(f"{self.job_type.value} sub-job requires a `sampling` block")
        return self

    @classmethod
    def training_job(
        cls,
        model_name: str,
        *,
        optimizer: dict,
        max_seq_len: int,
        train_batch_size: int,
        n_gpus: int,
        gradient_clipping: float | None = None,
        multiplex_job_id: str | None = None,
        load_optimizer_states: bool | None = None,
        extra_training: dict | None = None,
        global_batch_size: int | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        model_post_init: list[str] | None = None,
        source_checkpoint_info: dict | None = None,
    ) -> "SubJobConfig":
        """Build a training :class:`SubJobConfig`. All training fields are explicit.

        Args:
            model_name: Model identifier (e.g., "Qwen/Qwen3-1.7B").
            optimizer: Optimizer config dict (e.g., {"name": "AdamW", "lr": 0.0001}).
            max_seq_len: Maximum sequence length for training.
            train_batch_size: Per-device batch size.
            n_gpus: Number of GPUs (DP size).
            gradient_clipping: Optional gradient clipping threshold.
            multiplex_job_id: Optional multiplex job identifier.
            load_optimizer_states: Whether to load optimizer states from
                checkpoints. Set to ``False`` when loading across different DP
                sizes (``n_gpus``) since optimizer states are DP-sharded and
                cannot be resized. When ``None``, uses server default (True).
                **This is a creation-time setting and cannot be changed later.**
            extra_training: Additional training config keys (passthrough).
            global_batch_size: Global batch size across all GPUs.
            dtype: Model dtype (e.g., "bfloat16").
            seed: Random seed.
            model_post_init: List of post-initialization operations.
            source_checkpoint_info: Optional checkpoint to resume from at
                creation time (use ``{"checkpoint_id": "...", "source_job_id": "..."}`).
        """
        _typed_training = set(TrainingConfig.model_fields.keys())
        passthrough_training = {k: v for k, v in (extra_training or {}).items() if k not in _typed_training}
        return cls(
            job_type=JobType.TRAINING,
            model_name=model_name,
            training=TrainingConfig(
                optimizer=optimizer,
                max_seq_len=max_seq_len,
                train_batch_size=train_batch_size,
                n_gpus=n_gpus,
                gradient_clipping=gradient_clipping,
                multiplex_job_id=multiplex_job_id,
                load_optimizer_states=load_optimizer_states,
                **passthrough_training,
            ),
            global_batch_size=global_batch_size,
            dtype=dtype,
            seed=seed,
            model_post_init=model_post_init,
            source_checkpoint_info=source_checkpoint_info,
        )

    @classmethod
    def sampling_job(
        cls,
        model_name: str,
        *,
        max_seq_len: int,
        n_gpus: int,
        multiplex_job_id: str | None = None,
        extra_sampling: dict | None = None,
        job_type: JobType = JobType.SAMPLING,
        global_batch_size: int | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        model_post_init: list[str] | None = None,
        source_checkpoint_info: dict | None = None,
    ) -> "SubJobConfig":
        """Build a sampling :class:`SubJobConfig`.

        ``job_type`` remains for source compatibility, but only ``SAMPLING`` is
        currently accepted.
        """
        _reject_log_probability_sub_job(job_type, location="sampling_job().job_type")
        if job_type != JobType.SAMPLING:
            raise ValueError(f"sampling_job() only accepts SAMPLING, got {job_type!r}")
        _typed_sampling = set(InferenceConfig.model_fields.keys())
        passthrough_sampling = {k: v for k, v in (extra_sampling or {}).items() if k not in _typed_sampling}
        return cls(
            job_type=job_type,
            model_name=model_name,
            sampling=InferenceConfig(
                max_seq_len=max_seq_len,
                n_gpus=n_gpus,
                multiplex_job_id=multiplex_job_id,
                **passthrough_sampling,
            ),
            global_batch_size=global_batch_size,
            dtype=dtype,
            seed=seed,
            model_post_init=model_post_init,
            source_checkpoint_info=source_checkpoint_info,
        )

    def validate(self) -> None:
        """Re-validate the current state. Backward-compatible with the old dataclass API."""
        self.__class__.model_validate(self.model_dump(by_alias=True))

    def to_wire(self) -> dict:
        wire: dict = {
            "job_type": self.job_type.value,
            "model_name": self.model_name,
        }
        if self.global_batch_size is not None:
            wire["global_batch_size"] = self.global_batch_size
        if self.dtype is not None:
            wire["dtype"] = self.dtype
        if self.seed is not None:
            wire["seed"] = self.seed
        if self.model_post_init_ops is not None:
            wire["model_post_init"] = list(self.model_post_init_ops)
        if self.training is not None:
            wire["training_config"] = self.training.to_wire()
        if self.sampling is not None:
            wire["inference_config"] = self.sampling.to_wire()
        if self.source_checkpoint_info is not None:
            wire["source_checkpoint_info"] = self.source_checkpoint_info
        return wire


def _validate_raw_sub_job_configs(sub_job_configs: list) -> None:
    """Canonicalize every ``peft_config`` in a raw create-job body, in place.

    The typed path validates LoRA through :class:`TrainingConfig` /
    :class:`InferenceConfig`; this gives raw bodies the same guarantee so an
    adapter that weight sync cannot carry is rejected before the request is
    sent. Everything else in the body, including ``optimizer``, is forwarded as
    given.
    """
    for index, sub_job in enumerate(sub_job_configs):
        if not isinstance(sub_job, dict):
            continue
        job_type = _normalized_job_type(sub_job.get("job_type"))
        config_key = "training_config" if job_type == JobType.TRAINING.value else "inference_config"
        job_config = sub_job.get(config_key)
        peft_config = job_config.get("peft_config") if isinstance(job_config, dict) else None
        if peft_config is None:
            continue
        job_config["peft_config"] = normalize_lora_peft_config(
            peft_config,
            location=f"sub_job_configs[{index}].{config_key}.peft_config",
        )


# ─── Forward-backward payload helpers ────────────────────────────────────


def _load_torch():
    import torch

    return torch


def _tensor_dtype(torch, dtype_name: str | None):
    if dtype_name is None:
        return torch.long
    aliases = {
        "bool": torch.bool,
        "boolean": torch.bool,
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "int": torch.int32,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.long,
    }
    try:
        return aliases[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype: {dtype_name}") from exc


def _json_tensor(value: Any, *, default_dtype: str | None = "long"):
    torch = _load_torch()
    if isinstance(value, dict):
        if "data" not in value:
            raise ValueError("tensor object must contain a data field")
        dtype_name = value.get("dtype", default_dtype)
        value = value["data"]
    else:
        dtype_name = default_dtype
    return torch.tensor(value, dtype=_tensor_dtype(torch, dtype_name))


def _read_fwd_bwd_args(payload: dict[str, Any]) -> tuple:
    args = payload.get("args", [])
    if args is None:
        return ()
    if not isinstance(args, list):
        raise ValueError("fwd-bwd payload args must be a JSON list")
    return tuple(args)


def _tensorize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and "data" in value:
            out[key] = _json_tensor(value)
        elif isinstance(value, list):
            out[key] = _json_tensor(value)
        else:
            out[key] = value
    return out


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _repeat_texts(texts: Any, batch_size: Any) -> list[str]:
    if isinstance(texts, str):
        text_list = [texts]
    elif isinstance(texts, list) and texts and all(isinstance(x, str) for x in texts):
        text_list = texts
    else:
        raise ValueError("fwd-bwd payload texts must be a non-empty string list")

    if batch_size is None:
        return text_list
    batch = _positive_int(batch_size, "fwd-bwd payload batch_size")
    return [text_list[i % len(text_list)] for i in range(batch)]


def _load_tokenizer(tokenizer_spec: Any):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No module named 'transformers'",
            name="transformers",
        ) from exc

    if isinstance(tokenizer_spec, str):
        model_name = tokenizer_spec
        kwargs: dict[str, Any] = {}
    elif isinstance(tokenizer_spec, dict):
        model_name = tokenizer_spec.get("model_name") or tokenizer_spec.get("name") or tokenizer_spec.get("path")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("tokenizer.model_name is required")
        kwargs = {}
        for key in ("trust_remote_code", "use_fast", "revision"):
            if key in tokenizer_spec:
                kwargs[key] = tokenizer_spec[key]
    else:
        raise ValueError("fwd-bwd payload tokenizer must be a string or object")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
    return tokenizer


def _tokenize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tokenizer_spec = payload.get("tokenizer", payload.get("model_name"))
    if tokenizer_spec is None:
        raise ValueError("fwd-bwd payload requires tokenizer or input_ids")

    tokenizer = _load_tokenizer(tokenizer_spec)
    texts = _repeat_texts(payload.get("texts"), payload.get("batch_size"))
    padding = payload.get("padding", "max_length")
    truncation = payload.get("truncation", True)
    add_special_tokens = payload.get("add_special_tokens", True)
    max_length = payload.get("max_length")
    if padding == "max_length":
        max_length = _positive_int(max_length, "fwd-bwd payload max_length")
    elif max_length is not None:
        max_length = _positive_int(max_length, "fwd-bwd payload max_length")
    _require_bool(truncation, "fwd-bwd payload truncation")
    _require_bool(add_special_tokens, "fwd-bwd payload add_special_tokens")

    encode_kwargs = {
        "return_tensors": "pt",
        "padding": padding,
        "truncation": truncation,
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        encode_kwargs["max_length"] = max_length
    encoded = tokenizer(texts, **encode_kwargs)
    if "input_ids" not in encoded:
        raise ValueError("tokenizer output did not include input_ids")
    return encoded


def _ensure_rank2(tensor, name: str) -> None:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} must have shape [batch, seq_len]")


def _build_position_ids(input_ids, position_spec: Any):
    torch = _load_torch()
    if position_spec is None or position_spec is False:
        return None
    if position_spec is True or position_spec == "arange":
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
    if isinstance(position_spec, (dict, list)):
        return _json_tensor(position_spec)
    raise ValueError("fwd-bwd payload position_ids must be true, false, arange, or tensor data")


def _label_config(payload: dict[str, Any]) -> tuple[Any, int, bool]:
    labels = payload.get("labels", payload.get("label_strategy", "next_token"))
    if isinstance(labels, dict) and "data" in labels:
        return labels, int(payload.get("ignore_index", -100)), True
    if isinstance(labels, dict):
        strategy = labels.get("strategy", "next_token")
        ignore_index = int(labels.get("ignore_index", payload.get("ignore_index", -100)))
        mask_padding = labels.get("mask_padding", payload.get("mask_padding", True))
        return strategy, ignore_index, _require_bool(mask_padding, "labels.mask_padding")
    return (
        labels,
        int(payload.get("ignore_index", -100)),
        _require_bool(
            payload.get("mask_padding", True),
            "fwd-bwd payload mask_padding",
        ),
    )


def _build_labels(input_ids, attention_mask, payload: dict[str, Any]):
    torch = _load_torch()
    labels, ignore_index, mask_padding = _label_config(payload)
    if labels is None or labels == "none":
        return None
    if isinstance(labels, (dict, list)):
        return _json_tensor(labels)
    if labels in ("next_token", "shifted_input_ids"):
        out = torch.roll(input_ids, shifts=-1, dims=1)
        out[:, -1] = ignore_index
        if mask_padding and attention_mask is not None:
            target_mask = torch.roll(attention_mask, shifts=-1, dims=1)
            target_mask[:, -1] = 0
            out = out.masked_fill(target_mask == 0, ignore_index)
        return out
    if labels in ("input_ids", "self"):
        out = input_ids.clone()
        if mask_padding and attention_mask is not None:
            out = out.masked_fill(attention_mask == 0, ignore_index)
        return out
    raise ValueError("fwd-bwd labels strategy must be next_token, input_ids, none, or tensor data")


def build_forward_backward_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Build tensor kwargs for a forward-backward training batch.

    ``payload`` is intentionally JSON-shaped so CLI examples can stay readable:
    callers may provide either tokenizable ``texts`` plus a ``tokenizer`` or
    direct tensor data under ``input_ids`` / ``labels`` / ``position_ids``.
    """
    if not isinstance(payload, dict):
        raise ValueError("fwd-bwd payload must be a JSON object")

    if "kwargs" in payload:
        kwargs = payload["kwargs"]
        if not isinstance(kwargs, dict):
            raise ValueError("fwd-bwd payload kwargs must be an object")
        return _tensorize_kwargs(kwargs)

    torch = _load_torch()
    encoded = None
    if "input_ids" in payload:
        input_ids = _json_tensor(payload["input_ids"])
        attention_mask = (
            _json_tensor(payload["attention_mask"])
            if isinstance(payload.get("attention_mask"), (dict, list))
            else None
        )
    else:
        encoded = _tokenize_payload(payload)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    _ensure_rank2(input_ids, "input_ids")
    if attention_mask is not None:
        _ensure_rank2(attention_mask, "attention_mask")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

    kwargs: dict[str, Any] = {"input_ids": input_ids.contiguous()}

    include_attention = payload.get("include_attention_mask", False)
    if "attention_mask" in payload and isinstance(payload["attention_mask"], bool):
        include_attention = payload["attention_mask"]
    include_attention = _require_bool(
        include_attention,
        "fwd-bwd payload include_attention_mask",
    )
    if include_attention:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        kwargs["attention_mask"] = attention_mask.contiguous()

    position_spec = payload.get("position_ids")
    if position_spec is None and payload.get("include_position_ids", False):
        position_spec = "arange"
    position_ids = _build_position_ids(input_ids, position_spec)
    if position_ids is not None:
        _ensure_rank2(position_ids, "position_ids")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")
        kwargs["position_ids"] = position_ids.contiguous()

    labels = _build_labels(input_ids, attention_mask, payload)
    if labels is not None:
        _ensure_rank2(labels, "labels")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        kwargs["labels"] = labels.contiguous()

    return kwargs


def serialize_forward_backward_args(args: tuple = (), kwargs: dict[str, Any] | None = None) -> bytes:
    """Serialize ``{"args": args, "kwargs": kwargs}`` for Cortex Training fwd-bwd."""
    from cortex_training import wire  # torch loads on first use, not at import
    if not isinstance(args, tuple):
        raise ValueError("forward-backward args must be a tuple")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("forward-backward kwargs must be a dict")

    return wire.dumps(
        {"args": args, "kwargs": kwargs},
        metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
    )


def build_forward_backward_payload(spec: dict[str, Any]) -> bytes:
    """Build a serialized fwd-bwd byte payload from readable JSON."""
    if not isinstance(spec, dict):
        raise ValueError("fwd-bwd JSON must be an object")
    payload = spec.get("payload", spec)
    if not isinstance(payload, dict):
        raise ValueError("fwd-bwd payload must be an object")
    args = _read_fwd_bwd_args(payload)
    kwargs = build_forward_backward_kwargs(payload)
    return serialize_forward_backward_args(args, kwargs)


# ─── Generate prompt helpers ─────────────────────────────────────────────

# Wire dtype for pre-tokenized prompts. int32, not int64: a token id needs 18
# bits for today's vocabularies, and the frame size is the thing being cut.
# Measured on 8 prompts x 32,768 tokens, int32 gives 4.00 B/token and a 1.05 MB
# frame; int64 gives 8.00 B/token and 2.10 MB, which is larger than the
# 7.00 B/token of the JSON encoding it replaces.
_PROMPT_TOKEN_DTYPE_NAME = "int32"


def _tensor_prompts_disabled() -> bool:
    """True when tensor packing of pre-tokenized prompts is disabled via env var."""
    return _env_flag_enabled(DISABLE_TENSOR_PROMPTS_ENV)


def _as_token_tensor(prompt: list[int]):
    """One pre-tokenized prompt as a wire-dtype tensor.

    Falls back to int64 rather than failing. A token id outside int32 means a
    vocabulary of over two billion, so this should never fire -- but torch
    raises on overflow instead of wrapping, and turning a working request into
    an exception is a worse trade than sending a larger frame.
    """
    torch = _load_torch()
    try:
        return torch.tensor(prompt, dtype=_tensor_dtype(torch, _PROMPT_TOKEN_DTYPE_NAME))
    except (RuntimeError, OverflowError):
        logger.warning(
            "token id outside %s; sending this prompt as int64, which doubles its wire size",
            _PROMPT_TOKEN_DTYPE_NAME,
        )
        return torch.tensor(prompt, dtype=torch.int64)


def _is_token_id(value: object) -> bool:
    """True for an integer token id. ``bool`` is an ``int`` subclass and is not one."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require_token_ids(prompt: list, *, location: str) -> None:
    for index, value in enumerate(prompt):
        if not _is_token_id(value):
            raise ValueError(f"{location}[{index}] must be an integer token id, got {value!r}")


def _pack_token_prompts(prompts):
    """Move pre-tokenized prompts into the frame's tensor section.

    ``wire`` routes a tensor to the safetensors tensor section and everything
    else into the header as JSON, so a ``list[int]`` prompt travels as JSON
    text inside what is nominally a binary frame: roughly 7 bytes per token,
    plus a per-token encode cost paid inside the caller's submit latency.
    Handing ``wire`` tensors instead puts the token ids in the section built
    for them.

    Strings are left alone -- they are not token ids and the server tokenizes
    them. Empty lists are left alone so the server's own non-empty check is
    the one that rejects them, rather than this becoming a second, differently
    worded validation.
    """
    if _tensor_prompts_disabled():
        return prompts
    torch = _load_torch()
    if torch.is_tensor(prompts) or not isinstance(prompts, list) or not prompts:
        return prompts
    # A flat list of token ids is one prompt. Every element has to be an int;
    # a later float or bool would otherwise be truncated into the tensor.
    if _is_token_id(prompts[0]):
        _require_token_ids(prompts, location="prompts")
        return _as_token_tensor(prompts)
    packed = []
    for index, prompt in enumerate(prompts):
        if isinstance(prompt, list) and prompt and _is_token_id(prompt[0]):
            _require_token_ids(prompt, location=f"prompts[{index}]")
            packed.append(_as_token_tensor(prompt))
        else:
            packed.append(prompt)
    return packed


def _sql_string_literal(value: str) -> str:
    """Return ``value`` as a single-quoted Snowflake SQL string."""
    return "'" + value.replace("'", "''") + "'"


_PLAIN_IDENTIFIER_RE = re.compile(r"[A-Z_][A-Z0-9_$]*")


def _sql_identifier(name: str) -> str:
    """Return ``name`` as a Snowflake identifier that resolves to exactly ``name``.

    A plain uppercase name stays bare; any other is double-quoted, with embedded
    quotes doubled, so its case and characters survive identifier resolution.
    """
    if _PLAIN_IDENTIFIER_RE.fullmatch(name):
        return name
    return '"' + name.replace('"', '""') + '"'


# The strings created_after / created_before take: YYYY-MM-DD, or
# YYYY-MM-DDTHH:MM[:SS[.fraction]] with "T" or a space between date and time,
# and an optional "Z" or +HH[:MM] / -HH[:MM] offset.
_CREATED_BOUND_RE = re.compile(
    r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})"
    r"(?:[Tt ](?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})(?:[.,](?P<fraction>[0-9]+))?)?"
    r"(?P<offset>[Zz]|[+-][0-9]{2}(?::?[0-9]{2})?)?)?"
)


def _parse_created_bound(text: str, name: str) -> datetime:
    """``text`` as a datetime; naive when it names no offset."""
    problem = (
        f"{name} must be YYYY-MM-DD, or YYYY-MM-DDTHH:MM[:SS[.fraction]] (a space "
        f"may replace the T) with an optional Z or +HH[:MM] / -HH[:MM] offset; "
        f"got {text!r}"
    )
    match = _CREATED_BOUND_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(problem)
    try:
        day = date.fromisoformat(match["date"])
        if match["hour"] is None:
            return datetime(day.year, day.month, day.day)
        zone = None
        offset = match["offset"]
        if offset in ("Z", "z"):
            zone = timezone.utc
        elif offset:
            digits = offset[1:].replace(":", "")
            hours, minutes = int(digits[:2]), int(digits[2:] or 0)
            if minutes >= 60:
                raise ValueError(f"offset minutes out of range: {offset}")
            shift = timedelta(hours=hours, minutes=minutes)
            zone = timezone(-shift if offset[0] == "-" else shift)
        return datetime(
            day.year,
            day.month,
            day.day,
            int(match["hour"]),
            int(match["minute"]),
            int(match["second"] or 0),
            # Microseconds are the finest a datetime holds; later digits go.
            int((match["fraction"] or "")[:6].ljust(6, "0")),
            tzinfo=zone,
        )
    except ValueError as exc:
        raise ValueError(problem) from exc


def _created_param(value: date | datetime | str | None, name: str) -> str | None:
    """Render a ``created_after`` / ``created_before`` value as RFC 3339 in UTC with ``Z``.

    A ``date``, or a ``YYYY-MM-DD`` string, is midnight UTC. A naive
    ``datetime``, or a string without an offset, is taken as UTC. ``None`` and
    ``""`` mean no bound.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        moment = _parse_created_bound(value, name)
    else:
        raise ValueError(
            f"{name} must be a date, a datetime or a string, got {type(value).__name__}"
        )
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        moment = moment.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is out of range once converted to UTC: {value!r}") from exc
    # A plain datetime, so a subclass's own isoformat() can't change the form.
    plain = datetime(*moment.timetuple()[:6], moment.microsecond)
    return plain.isoformat() + "Z"


def _validate_artifact_relative_path(path: str) -> str:
    """Reject paths that could escape or ambiguously address the run stage."""
    candidate = PurePosixPath(path)
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or candidate.name in {"", ".", ".."}
    ):
        raise ValueError(f"unsafe experiment artifact path: {path!r}")
    return path


# ─── HTTP client ─────────────────────────────────────────────────────────


class CortexTrainingClient:
    """HTTP client for the Cortex Training REST API (``cortex-training``).

    Construct with a named Snowflake connection profile::

        client = CortexTrainingClient.from_connection_name("training")

    Direct PAT authentication remains available through :meth:`from_pat`::

        client = CortexTrainingClient.from_pat(
            host="ACCOUNT.snowflakecomputing.com",
            pat="...",
            database="CORTEX_TRAINING_DB",
            schema="PUBLIC",
        )

    Against a local or otherwise compatible server, construct the client
    directly with an explicit ``base_url``; this skips PAT auth::

        client = CortexTrainingClient(
            base_url="http://localhost:8084",
            database="MY_DB",
            schema="PUBLIC",
        )

    Then build one or more sub-jobs and submit::

        sub = SubJobConfig.training_job(
            model_name="Qwen/Qwen3-1.7B",
            optimizer={"name": "AdamW", "lr": 1e-5},
            max_seq_len=2048,
            train_batch_size=1,
            n_gpus=1,
        )
        job_id = client.create_job(sub_jobs=[sub])
    """

    def __init__(
        self,
        base_url: str,
        database: str,
        schema: str,
        endpoint: str = "cortex-training",
        poll_interval: float = 0.5,
        poll_timeout: float = 1800.0,
        poll_backoff_multiplier: float = 1.25,
        poll_max_interval: float = 6.0,
        pool_maxsize: int = 1024,
        max_retries: int = 10,
        request_timeout: float | tuple[float, float] = _DEFAULT_REQUEST_TIMEOUT,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if poll_timeout <= 0:
            raise ValueError("poll_timeout must be > 0")
        if poll_backoff_multiplier < 1.0:
            raise ValueError("poll_backoff_multiplier must be >= 1.0")
        if poll_max_interval < poll_interval:
            raise ValueError("poll_max_interval must be >= poll_interval")
        if pool_maxsize <= 0:
            raise ValueError("pool_maxsize must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.request_timeout = _validated_request_timeout(request_timeout)
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.schema = schema
        self.endpoint = endpoint
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.poll_backoff_multiplier = poll_backoff_multiplier
        self.poll_max_interval = poll_max_interval
        self.max_retries = max_retries
        self._fwd_bwd_send_count = 0
        self._fwd_bwd_request_debug: dict[str, dict[str, Any]] = {}
        self._generate_request_ids: set[str] = set()
        # Forward request ids awaiting poll, so a result that is only
        # ``{"job_id", "payload_b64"}`` (no ``wire_format``) can still be decoded.
        self._forward_request_ids: set[str] = set()
        # Per-job cache of the sampling sub-job's ``inference_config.max_seq_len``
        # (the value the backend launches vLLM with as ``max_model_len``). Used
        # to validate generate prompt lengths client-side. A cached ``None``
        # means the job was fetched but exposes no sampling max_seq_len (skip
        # validation); a missing key means it has not been resolved yet (e.g. a
        # transient get_job failure), so a later call will retry.
        self._sampling_max_seq_len: dict[str, int | None] = {}
        self._artifact_connection_config: dict[str, str] | None = None
        self._artifact_connection_factory: Callable[[], Any] | None = None
        self._auth_provider: SnowflakeProfileAuth | None = None
        self._db_ensured = False
        self._db_creating = False
        # Both are cleared on a for_job() child; see for_job().
        self._owns_session = True
        self._auto_create_database = True
        self._metric_emitter: OtlpMetricEmitter | None = None
        self._operation_metric_state = threading.local()
        self._session = _DefaultTimeoutSession(self.request_timeout)
        adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @property
    def auto_create_database(self) -> bool:
        """Whether a request answered "database or schema not found" creates them and retries.

        True by default. Set it to False for a client that only reads, as the TUI
        does: the error is then raised as it is. A :meth:`for_job` client never
        creates one.
        """
        return self._auto_create_database

    @auto_create_database.setter
    def auto_create_database(self, value: bool) -> None:
        self._auto_create_database = bool(value)

    @classmethod
    def from_pat(
        cls,
        host: str,
        pat: str,
        database: str,
        schema: str,
        endpoint: str = "cortex-training",
        verify_ssl: bool = True,
        *,
        telemetry_timeout: float = 3.0,
        **kwargs: Any,
    ) -> "CortexTrainingClient":
        """Authenticate with a Programmatic Access Token (PAT).

        Sends ``Authorization: Bearer <pat>`` plus the
        ``X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN`` header
        on every request. No login round-trip.
        """
        client = cls(
            base_url=f"https://{host}",
            database=database,
            schema=schema,
            endpoint=endpoint,
            **kwargs,
        )
        client._session.headers["Authorization"] = f"Bearer {pat}"
        client._session.headers["X-Snowflake-Authorization-Token-Type"] = "PROGRAMMATIC_ACCESS_TOKEN"
        client._session.verify = verify_ssl
        client._artifact_connection_config = {"host": host, "pat": pat}
        if not _telemetry_disabled():
            token_provider = CachedSessionTokenProvider(
                client.base_url,
                pat,
                verify_ssl=verify_ssl,
                timeout=telemetry_timeout,
            )
            client._metric_emitter = OtlpMetricEmitter(
                client.base_url,
                token_provider,
                verify_ssl=verify_ssl,
                timeout=telemetry_timeout,
            )
        return client

    @classmethod
    def from_connection_name(
        cls,
        connection_name: str | None = None,
        *,
        database: str | None = None,
        schema: str | None = None,
        endpoint: str = "cortex-training",
        verify_ssl: bool = True,
        telemetry_timeout: float = 3.0,
        **kwargs: Any,
    ) -> "CortexTrainingClient":
        """Authenticate through a named or configured-default Snowflake profile.

        Profile discovery and authentication are delegated to
        ``snowflake-connector-python``. The live Connector session token is sent
        to the Cortex Training API, and the Connector session is recreated once
        if the API reports token expiry.
        """
        auth = SnowflakeProfileAuth(connection_name)
        resolved_database = database or auth.database
        if not resolved_database:
            auth.close()
            raise ValueError(
                "Snowflake connection profile must set database, or pass --database"
            )
        resolved_schema = schema or auth.schema or "PUBLIC"
        try:
            client = cls(
                base_url=auth.base_url,
                database=resolved_database,
                schema=resolved_schema,
                endpoint=endpoint,
                **kwargs,
            )
        except Exception:
            auth.close()
            raise
        client._auth_provider = auth
        client._artifact_connection_factory = auth.open_artifact_connection
        client._session.verify = verify_ssl
        if not _telemetry_disabled():
            client._metric_emitter = OtlpMetricEmitter(
                client.base_url,
                SnowflakeTelemetryTokenProvider(auth),
                verify_ssl=verify_ssl,
                timeout=telemetry_timeout,
            )
        return client

    @property
    def _prefix(self) -> str:
        # "$" is legal in a path segment and common in plain identifiers, so it
        # stays bare; anything else that could end or split the segment is encoded.
        return (
            f"{self.base_url}/api/v2/databases/{quote(self.database, safe='$')}"
            f"/schemas/{quote(self.schema, safe='$')}/{self.endpoint}"
        )

    def emit_metric(
        self,
        operation: str,
        value: Any = 1,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Emit a best-effort OTLP log record used as a client metric.

        Profile clients reuse their live Snowflake session token. PAT clients
        lazily exchange the PAT for a cached session token on the first call.
        Local/mock clients, or clients constructed with
        ``CORTEX_TRAINING_DISABLE_TELEMETRY`` set, treat this method as a no-op.
        All authentication, discovery, and export errors are ignored.
        """
        try:
            if self._metric_emitter is not None:
                self._metric_emitter.emit(operation, value, attributes=attributes)
        except Exception:
            return

    def close(self) -> None:
        """Flush telemetry briefly and close this client's HTTP sessions.

        A client from :meth:`for_job` borrows these, so closing it is a no-op.
        """
        if not self._owns_session:
            return
        try:
            if self._metric_emitter is not None:
                self._metric_emitter.close()
        except Exception:
            logger.debug("client telemetry close failed", exc_info=True)
        finally:
            try:
                self._session.close()
            finally:
                if self._auth_provider is not None:
                    self._auth_provider.close()

    def __enter__(self) -> "CortexTrainingClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _emit_operation_outcome(
        self,
        operation: str,
        started: float,
        arguments: Mapping[str, Any],
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """Emit one structured outcome without changing the operation result."""
        if self._metric_emitter is None:
            return
        if error is None and not _success_telemetry_enabled():
            return
        try:
            attempt_count = int(
                getattr(self._operation_metric_state, "attempt_count", 0)
            )
            request_count = int(
                getattr(self._operation_metric_state, "request_count", 0)
            )
            value: dict[str, Any] = {
                "attempt_count": attempt_count,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "request_count": request_count,
                "retry_count": max(attempt_count - request_count, 0),
                "success": error is None,
            }
            attributes = _metric_identity(operation, arguments, result)
            if error is not None:
                value["error_message"] = _safe_metric_error_message(error)
                attributes.update(_metric_error_attributes(error))
            attributes.update(
                _snowflake_request_attributes(
                    error,
                    list(
                        getattr(self._operation_metric_state, "snowflake_request_ids", [])
                    ),
                )
            )
            self.emit_metric(operation, value, attributes=attributes)
        except Exception:
            logger.debug("client operation metric construction failed", exc_info=True)

    @staticmethod
    def _debug_json_summary(value: Any, *, limit: int = 2000) -> str:
        try:
            text = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            text = repr(value)
        if len(text) > limit:
            return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
        return text

    @classmethod
    def _debug_response_summary(cls, resp: requests.Response, *, limit: int = 2000) -> str:
        try:
            return cls._debug_json_summary(resp.json(), limit=limit)
        except ValueError:
            text = getattr(resp, "text", "")
        except Exception as exc:
            text = f"<failed to read response body: {type(exc).__name__}: {exc}>"
        text = str(text).replace("\n", "\\n")
        if len(text) > limit:
            return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
        return text

    @staticmethod
    def _debug_context_label(debug_context: dict[str, Any]) -> str:
        parts = ["[cortex-training fwd-bwd]"]
        for key in ("phase", "call", "job_id", "request_id", "payload_bytes"):
            value = debug_context.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def _create_database(self) -> None:
        """Attempt to create the configured database (and schema if non-PUBLIC).

        Called only when an API request fails with a database-not-found error.
        Routes through ``_send`` so auth works for both PAT and connection
        profile users. Raises on failure with an actionable error message.
        """
        db_stmt = f"CREATE DATABASE IF NOT EXISTS {self.database}"
        manual_hint = (
            f"Create it manually in Snowsight: CREATE DATABASE IF NOT EXISTS {self.database}; "
            "— or if the database already exists, check that your role has USAGE on it."
        )
        self._db_creating = True  # prevent recursion from inner _send call
        try:
            logger.info("Database '%s' not found. Attempting to create it...", self.database)
            resp = self._send(
                "POST",
                f"{self.base_url}/api/v2/statements",
                json={"statement": db_stmt, "timeout": 60},
                max_retries=1,
            )
            if resp.status_code == 202:
                raise RuntimeError(
                    f"Database creation for '{self.database}' was accepted but has not "
                    f"completed yet. {manual_hint}"
                )
            logger.info("Database '%s' created successfully.", self.database)

            if self.schema.upper() != "PUBLIC":
                schema_stmt = f"CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema}"
                logger.info("Schema '%s' does not exist. Attempting to create it...", self.schema)
                schema_resp = self._send(
                    "POST",
                    f"{self.base_url}/api/v2/statements",
                    json={"statement": schema_stmt, "timeout": 60},
                    max_retries=1,
                )
                if schema_resp.status_code == 202:
                    raise RuntimeError(
                        f"Schema creation for '{self.database}.{self.schema}' was accepted but has not "
                        f"completed yet. Create it manually: CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema};"
                    )
                logger.info("Schema '%s' created successfully.", self.schema)

            self._db_ensured = True
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Could not create database '{self.database}': {exc}. {manual_hint}"
            ) from exc
        finally:
            self._db_creating = False

    @staticmethod
    def _is_database_not_found(resp: requests.Response) -> bool:
        """Return True if the response indicates the configured database does not exist.

        The Cortex Training API returns 400 with error code 517602 and a message
        like ``Schema MY_DB.PUBLIC is not found or not authorized`` when the
        database in the URL path does not exist.
        """
        if resp.status_code not in (400, 404, 422):
            return False
        try:
            text = resp.text.lower()
        except Exception:
            return False
        return "not found or not authorized" in text or "does not exist or not authorized" in text

    def _send(
        self,
        method: str,
        url: str,
        *,
        retry_on=_is_transient,
        max_retries: int | None = None,
        ensure_database: bool = True,
        **kwargs,
    ) -> requests.Response:
        """Send one request with retries.

        ``ensure_database=False`` skips creating the configured database on a
        not-found error, for routes whose path names no database, and for reads
        that must never create it (a database that doesn't exist can't hold the
        job they read). A client from :meth:`for_job` never creates one,
        whatever the caller passes.
        """
        auto_create = ensure_database and self._auto_create_database
        fn = getattr(self._session, method.lower())
        debug_context = kwargs.pop("debug_context", None)
        debug_label = self._debug_context_label(debug_context) if debug_context is not None else None
        attempt_no = 0
        auth_retry_used = False
        retry_limit = self.max_retries if max_retries is None else max_retries
        max_attempts = 1 + retry_limit
        displayed_max_attempts = max_attempts + (
            1 if self._auth_provider is not None else 0
        )
        try:
            state = self._operation_metric_state
            if getattr(state, "active", set()):
                state.request_count = getattr(state, "request_count", 0) + 1
        except Exception:
            logger.debug("client operation request counting failed", exc_info=True)

        def attempt() -> requests.Response:
            nonlocal attempt_no, auth_retry_used
            while True:
                attempt_no += 1
                try:
                    state = self._operation_metric_state
                    if getattr(state, "active", set()):
                        state.attempt_count = getattr(state, "attempt_count", 0) + 1
                except Exception:
                    logger.debug(
                        "client operation attempt counting failed", exc_info=True
                    )
                if debug_label is not None:
                    logger.debug(
                        "%s sending %s %s attempt=%d/%d",
                        debug_label,
                        method.upper(),
                        url,
                        attempt_no,
                        displayed_max_attempts,
                    )

                request_kwargs = kwargs
                observed_token: str | None = None
                if self._auth_provider is not None:
                    observed_token = self._auth_provider.get_token()
                    request_headers = dict(kwargs.get("headers") or {})
                    request_headers["Authorization"] = (
                        f'Snowflake Token="{observed_token}"'
                    )
                    request_kwargs = {**kwargs, "headers": request_headers}
                try:
                    resp = fn(url, **request_kwargs)
                except Exception as exc:
                    if debug_label is not None:
                        logger.debug(
                            "%s request exception %s %s attempt=%d/%d: %s: %s",
                            debug_label,
                            method.upper(),
                            url,
                            attempt_no,
                            displayed_max_attempts,
                            type(exc).__name__,
                            exc,
                        )
                    raise

                status_code = getattr(resp, "status_code", None)
                sf_request_id = _response_snowflake_request_id(resp)
                if sf_request_id:
                    req = getattr(resp, "request", None)
                    logger.debug(
                        "snowflake request_id=%s  %s %s  status=%d",
                        sf_request_id,
                        getattr(req, "method", method.upper()),
                        getattr(req, "path_url", url),
                        status_code,
                    )
                    try:
                        state = self._operation_metric_state
                        if getattr(state, "active", set()):
                            seen = state.snowflake_request_ids
                            seen.append(sf_request_id)
                            del seen[:-_MAX_TRACKED_SNOWFLAKE_REQUEST_IDS]
                    except Exception:
                        logger.debug(
                            "client operation request id tracking failed",
                            exc_info=True,
                        )

                if (
                    self._auth_provider is not None
                    and not auth_retry_used
                    and _is_session_auth_expired_response(resp)
                ):
                    auth_retry_used = True
                    resp.close()
                    self._auth_provider.refresh(observed_token)
                    continue

                if debug_label is not None:
                    try:
                        status_int = int(status_code)
                    except (TypeError, ValueError):
                        status_int = None
                    outcome = (
                        "successful response"
                        if status_int is not None and 200 <= status_int < 300
                        else "failed response"
                    )
                    snowflake = (
                        f" snowflake_request_id={sf_request_id}"
                        if sf_request_id
                        else ""
                    )
                    logger.debug(
                        "%s %s %s %s attempt=%d/%d status=%s%s body=%s",
                        debug_label,
                        outcome,
                        method.upper(),
                        url,
                        attempt_no,
                        displayed_max_attempts,
                        status_code,
                        snowflake,
                        self._debug_response_summary(resp),
                    )
                if (
                    auto_create
                    and not self._db_ensured
                    and not self._db_creating
                    and self._is_database_not_found(resp)
                ):
                    resp.close()
                    self._create_database()
                    # Retry the original request once after creating the DB.
                    continue
                if (
                    auto_create
                    and self._db_ensured
                    and self._is_database_not_found(resp)
                ):
                    raise RuntimeError(
                        f"Database '{self.database}' still not found after creation attempt. "
                        f"The database may already exist but your role lacks USAGE on it. "
                        f"Create it manually in Snowsight: CREATE DATABASE IF NOT EXISTS {self.database};"
                    )
                try:
                    # requests.Response.raise_for_status() intentionally ignores 3xx.
                    if 300 <= resp.status_code < 400:
                        raise requests.exceptions.HTTPError(
                            f"Unexpected redirect {resp.status_code} for url: {url}",
                            response=resp,
                            request=getattr(resp, "request", None),
                        )
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as exc:
                    _add_snowflake_request_id_to_error(exc, sf_request_id)
                    raise
                return resp

        retryer = Retrying(
            retry=retry_if_exception(retry_on),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return retryer(attempt)

    # ─── Job management ──────────────────────────────────────────────────

    @_track_operation("create_job")
    def create_job(
        self,
        sub_jobs: list[SubJobConfig],
        job_id: str | None = None,
        experiment_name: str | None = None,
        hardware: Hardware | str | None = None,
        idle_timeout_seconds: int | None = None,
        pending_timeout_seconds: int | None = None,
        comment: str | None = None,
    ) -> str:
        """Create a job from a list of sub-jobs and return its server job_id.

        Each :class:`SubJobConfig` is validated client-side before the request
        is sent (see :meth:`SubJobConfig.validate`). A job supports zero or one
        ``training`` sub-job and any number of ``sampling`` sub-jobs;
        ``log_probability`` sub-jobs are rejected here. ``job_id`` is optional;
        when omitted the server generates one. ``experiment_name`` is optional;
        when omitted the server auto-creates an experiment for the job.
        ``hardware`` is optional (:class:`Hardware`.H200 / .B200 / .B300); when
        set, all sub-jobs in the job run on that GPU type. Omitted defaults to
        H200.
        ``idle_timeout_seconds`` bounds how long the job may sit idle before the
        server reclaims it: ``0`` disables reclamation, any other value must be
        between 300 and 604,800. ``pending_timeout_seconds`` bounds how long the
        job may wait for capacity before failing with ``pending_timeout``: between
        300 and 604,800, with no disable value. Omit either to use the server
        default, currently 30 minutes for idle and 24 hours for pending.
        ``comment`` names the job: free text stored with it and returned by
        :meth:`get_job` and :meth:`list_jobs` (the TUI shows its first line as
        the job's name). It can't be changed after creation. It is never sent
        for a model deployment, which rejects one; for any other job a comment
        over 1,024 bytes of UTF-8, or not valid UTF-8 text, raises
        :class:`ValueError` before sending. A blank comment is not sent.
        """
        if not sub_jobs:
            raise ValueError("create_job requires a non-empty sub_jobs list")
        if idle_timeout_seconds is not None:
            if isinstance(idle_timeout_seconds, bool) or not isinstance(
                idle_timeout_seconds, int
            ):
                raise ValueError("idle_timeout_seconds must be an integer")
            if idle_timeout_seconds != 0 and not (
                300 <= idle_timeout_seconds <= 604_800
            ):
                raise ValueError(
                    "idle_timeout_seconds must be 0 or between 300 and 604800"
                )
        if pending_timeout_seconds is not None:
            if isinstance(pending_timeout_seconds, bool) or not isinstance(
                pending_timeout_seconds, int
            ):
                raise ValueError("pending_timeout_seconds must be an integer")
            if not (300 <= pending_timeout_seconds <= 604_800):
                raise ValueError(
                    "pending_timeout_seconds must be between 300 and 604800"
                )
        for index, sj in enumerate(sub_jobs):
            _reject_log_probability_sub_job(
                sj.job_type, location=f"sub_jobs[{index}].job_type"
            )
            SubJobConfig.model_validate(sj.model_dump(by_alias=True))
        body: dict = {"sub_job_configs": [sj.to_wire() for sj in sub_jobs]}
        if job_id is not None:
            body["job_id"] = job_id
        if experiment_name is not None:
            body["experiment_name"] = experiment_name
        if hardware is not None:
            body["hardware"] = _hardware_wire(hardware)
        if idle_timeout_seconds is not None:
            body["idle_timeout_seconds"] = idle_timeout_seconds
        if pending_timeout_seconds is not None:
            body["pending_timeout_seconds"] = pending_timeout_seconds
        if comment is not None:
            body["comment"] = comment  # checked, and left out when blank, on the way out
        return self.create_job_from_body(body)["job_id"]

    @_track_operation("create_job")
    def create_job_from_body(self, body: dict) -> dict:
        """Create a job from a raw REST CreateJob request body.

        This is useful for tooling that already has the REST JSON payload,
        while :meth:`create_job` remains the typed path for Python callers.

        A job supports zero or one ``training`` sub-job and any number of
        ``sampling`` sub-jobs; a second training sub-job or any
        ``log_probability`` sub-job raises :class:`ValueError` before the
        request is sent.

        Every ``peft_config`` present in the body is canonicalized in place and
        an invalid one raises :class:`ValueError` before the request is sent.
        A ``comment`` is checked as in :meth:`create_job`: a model deployment is
        sent without it, a blank one is not sent, and for any other job one
        over 1,024 bytes of UTF-8, or not valid UTF-8 text, raises. The rest of
        the body, including ``optimizer``, is forwarded as given.
        """
        if not isinstance(body, dict):
            raise ValueError("create_job_from_body requires a JSON object")
        sub_job_configs = body.get("sub_job_configs")
        if not isinstance(sub_job_configs, list) or not sub_job_configs:
            raise ValueError("create_job_from_body requires a non-empty sub_job_configs list")
        training_sub_jobs = 0
        for index, cfg in enumerate(sub_job_configs):
            if not isinstance(cfg, dict):
                continue
            job_type = cfg.get("job_type")
            _reject_log_probability_sub_job(
                job_type, location=f"sub_job_configs[{index}].job_type"
            )
            if str(job_type or "").strip().lower() == JobType.TRAINING.value:
                training_sub_jobs += 1
                if training_sub_jobs > 1:
                    raise ValueError("at most one training sub-job is supported per job")
        _validate_raw_sub_job_configs(sub_job_configs)
        if body.get("debug") and not _debug_options_enabled():
            raise ValueError(
                "create-job debug options are an internal-only capability; set "
                f"{DEBUG_OPTIONS_ENV}=1 to send a request with a `debug` block"
            )
        body = _body_with_checked_comment(body)
        resp = self._send("POST", self._prefix, json=body, retry_on=_is_connect_error)
        return resp.json()

    @staticmethod
    def _normalize_job_status(status) -> str:
        """Normalize full enum names (``JOB_STATE_RUNNING``) to short form (``running``).

        Coerces to ``str`` first: a backend running ahead of the API layer can
        return a JobState enum value the API layer doesn't yet know (e.g. a
        newly added state), which is then rendered as a raw integer rather than
        a name. Tolerating it keeps :meth:`wait_for_job` polling — an unknown
        status is neither ``running`` nor terminal, so it stays in-progress —
        instead of crashing on ``int.lower()``.
        """
        return str(status or "").lower().removeprefix("job_state_")

    def _sleep_with_backoff(self, delay: float, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return delay
        time.sleep(min(delay, remaining))
        return min(delay * self.poll_backoff_multiplier, self.poll_max_interval)

    @_track_operation("wait_for_job")
    def wait_for_job(self, job_id: str) -> dict:
        """Poll until the job reaches ``running``. Raises on terminal states or timeout."""
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            status = self._normalize_job_status(job.get("status", ""))
            if status == "running":
                return job
            if status in ("failed", "done", "cancelled", "canceled"):
                reason = job.get("reason", "")
                raise _server_failure(
                    RuntimeError(f"Job {job_id} reached terminal state '{status}': {reason}"),
                    failed=status == "failed",
                )
            delay = self._sleep_with_backoff(delay, deadline)
        raise _server_failure(
            TimeoutError(f"Job {job_id} did not become running within {self.poll_timeout}s")
        )

    @_track_operation("get_job")
    def get_job(self, job_id: str) -> dict:
        resp = self._send("GET", f"{self._prefix}/{job_id}")
        return resp.json()

    @_track_operation("list_jobs")
    def list_jobs(
        self,
        status: str | None = None,
        *,
        max_retries: int | None = None,
        request_timeout: float | tuple[float, float] | None = None,
    ) -> list:
        """List the jobs of this client's database and schema.

        ``max_retries`` (retries after the first attempt; ``0`` means one
        attempt) and ``request_timeout`` (seconds, or a ``(connect, read)``
        pair) bound this one call, as on :meth:`list_jobs_page`; None keeps
        the client's own (10 retries, 30 s to connect and 600 s to read, by
        default).
        """
        max_retries, timeout = _account_page_bounds(max_retries, request_timeout)
        params = {}
        if status is not None:
            params["status"] = status
        resp = self._send(
            "GET",
            self._prefix,
            params=params,
            max_retries=max_retries,
            **({} if timeout is None else {"timeout": timeout}),
        )
        return _json_body(resp).get("jobs", [])

    def iter_jobs(
        self,
        *,
        database: str | None = None,
        schema: str | None = None,
        status: str | Iterable[str] | None = None,
        submitted_by: str | None = None,
        created_after: date | datetime | str | None = None,
        created_before: date | datetime | str | None = None,
        page_size: int = 100,
        max_retries: int | None = None,
        request_timeout: float | tuple[float, float] | None = None,
    ) -> Iterator[dict]:
        """Yield the jobs the caller can see across the account, newest first.

        Reads :meth:`list_jobs_page` page after page as you iterate, with the
        same filters each time, until no ``next_page_token`` is left; a page can
        hold fewer than ``page_size`` jobs, or none, before the end. The
        arguments are those of :meth:`list_jobs_page` without ``page_token``,
        and are checked on the call. ``max_retries`` and ``request_timeout``
        bound each page, not the whole walk.

        If the service stops accepting the page token partway (400 ``Invalid
        page_token``, for example after it rotated the key that seals tokens),
        the walk starts again from the first page, once, and skips the jobs it
        already yielded, by ``job_id``. Jobs created since the walk began then
        come after older jobs already yielded, so sort by ``created_at`` when
        order matters. A second such answer raises :class:`InvalidPageToken`.
        Other failures raise as :meth:`list_jobs_page` describes, on the page
        that fails.
        """
        params = self._account_jobs_params(
            database=database,
            schema=schema,
            status=status,
            submitted_by=submitted_by,
            created_after=created_after,
            created_before=created_before,
            page_size=page_size,
        )
        max_retries, timeout = _account_page_bounds(max_retries, request_timeout)
        return self._iter_account_jobs(params, max_retries, timeout)

    def list_jobs_page(
        self,
        *,
        database: str | None = None,
        schema: str | None = None,
        status: str | Iterable[str] | None = None,
        submitted_by: str | None = None,
        created_after: date | datetime | str | None = None,
        created_before: date | datetime | str | None = None,
        page_size: int = 100,
        page_token: str | None = None,
        max_retries: int | None = None,
        request_timeout: float | tuple[float, float] | None = None,
    ) -> dict:
        """Read one page of the jobs the caller can see across the account.

        Returns ``{"jobs": [...], "next_page_token": str | None}``. Jobs come
        newest first and also carry ``database_name`` and ``schema_name``; use
        :meth:`for_job` for per-job calls on one. For the next page, pass
        ``next_page_token`` back as ``page_token`` with the same filters
        (``page_size`` may change), so compute a moving bound such as
        ``created_after`` once per walk; ``None`` means there is no next page.
        The service examines at most 5,000 jobs per call, so a page can hold
        fewer than ``page_size`` jobs, or none, and still carry a token.

        This reads the account-level ``GET /api/v2/cortex-training``, a fixed
        path whatever this client's ``endpoint`` is, and never creates a
        database. The service applies every filter, and ``None`` or ``""``
        means none:

        - ``database``, ``schema``: SQL identifier rules apply (unquoted is
          case-insensitive, ``"quoted"`` is exact); ``schema`` needs
          ``database``.
        - ``status``: a comma-separated string or a list of ``pending``,
          ``placing``, ``initializing``, ``running``, ``failed``,
          ``cancelled``, ``terminated`` and the groups ``active`` (``pending``
          through ``running``) and ``terminal`` (``cancelled``, ``terminated``,
          ``failed``). ``done``, which the service no longer takes, is sent as
          ``cancelled,terminated``, its older meaning. Under any status filter
          the service leaves out jobs whose live status it can't read; without
          one they come back as ``UNKNOWN`` with a ``reason``. The per-schema
          :meth:`list_jobs` reads ``status`` its own way.
        - ``submitted_by``: a user name, by SQL identifier rules. A name that
          matches no active user in the account is a 400.
        - ``created_after`` (inclusive), ``created_before`` (exclusive): a
          ``date`` (midnight UTC), a ``datetime`` (naive means UTC), or a
          string: ``YYYY-MM-DD``, or ``YYYY-MM-DDTHH:MM[:SS[.fraction]]`` (a
          space may replace the ``T``) with an optional ``Z`` or
          ``+HH[:MM]`` / ``-HH[:MM]`` offset (none means UTC). Sent as RFC
          3339 in UTC with ``Z``.
        - ``page_size``: an int from 1 to 1000.

        The page uses this client's ``max_retries`` (10 by default, so up to
        11 attempts) and ``request_timeout`` (30 s to connect and 600 s to
        read, by default) unless bounded here: ``max_retries`` counts retries
        after the first attempt, and ``request_timeout`` is seconds or a
        ``(connect, read)`` pair. Connection errors, timeouts, and 307, 308,
        409, 429, 500, 502, 503 and 504 answers are retried. Bound
        interactive calls: under a status filter the service can wait up to
        300 s on a failing status lookup before it fails the page.

        Raises, with the arguments checked before any request:

        - :class:`AccountJobListUnavailable` for 403, 404 and 405, without a
          retry;
        - :class:`InvalidPageToken` for 400 ``Invalid page_token``;
        - ``requests.HTTPError`` for any other failed answer, including other
          400s;
        - ``requests.ConnectionError`` or ``requests.Timeout`` once retries
          are spent;
        - ``ValueError`` for bad arguments, or a page body that is not a job
          list.

        All but the last are ``requests.RequestException``, an ``OSError``.
        """
        params = self._account_jobs_params(
            database=database,
            schema=schema,
            status=status,
            submitted_by=submitted_by,
            created_after=created_after,
            created_before=created_before,
            page_size=page_size,
        )
        page_token = _optional_filter(page_token, "page_token")
        if page_token is not None:
            params["page_token"] = page_token
        max_retries, timeout = _account_page_bounds(max_retries, request_timeout)
        return self._list_jobs_page(params, max_retries, timeout)

    @staticmethod
    def _account_jobs_params(
        *,
        database: Any,
        schema: Any,
        status: Any,
        submitted_by: Any,
        created_after: Any,
        created_before: Any,
        page_size: Any,
    ) -> dict:
        """The account-level list's query parameters, checked before any request."""
        database = _optional_filter(database, "database")
        schema = _optional_filter(schema, "schema")
        if schema is not None and database is None:
            raise ValueError(
                "schema=... needs database=... with it: a schema name alone does "
                "not name one schema."
            )
        query = {
            "database": database,
            "schema": schema,
            "status": _account_status_param(status),
            "submitted_by": _optional_filter(submitted_by, "submitted_by"),
            "created_after": _created_param(created_after, "created_after"),
            "created_before": _created_param(created_before, "created_before"),
            "limit": _account_page_size(page_size),
        }
        return {key: value for key, value in query.items() if value is not None}

    def _iter_account_jobs(
        self,
        params: dict,
        max_retries: int | None,
        timeout: tuple[float, float] | None,
    ) -> Iterator[dict]:
        yielded: set[str] = set()
        restarted = False
        token: str | None = None
        while True:
            page_params = params if token is None else {**params, "page_token": token}
            try:
                page = self._iter_jobs_page(page_params, max_retries, timeout)
            except InvalidPageToken as exc:
                if restarted:
                    raise _page_token_rejected_again(exc) from exc
                restarted = True
                token = None
                logger.info(
                    "account-level job list: the service could not open the page "
                    "token, so the walk starts again from the first page"
                )
                continue
            for job in page["jobs"]:
                job_id = job.get("job_id") if isinstance(job, dict) else None
                if isinstance(job_id, str):
                    if job_id in yielded:
                        continue
                    yielded.add(job_id)
                yield job
            token = page["next_page_token"]
            if token is None:
                return

    # One event per page, named for the public call that read it. Tracked per
    # page: around the generator, _track_operation would record iter_jobs as
    # finished before its first request.
    @_track_operation("iter_jobs")
    def _iter_jobs_page(
        self,
        params: dict,
        max_retries: int | None,
        timeout: tuple[float, float] | None,
    ) -> dict:
        return self._get_account_jobs_page(params, max_retries, timeout)

    @_track_operation("list_jobs_page")
    def _list_jobs_page(
        self,
        params: dict,
        max_retries: int | None,
        timeout: tuple[float, float] | None,
    ) -> dict:
        return self._get_account_jobs_page(params, max_retries, timeout)

    def _get_account_jobs_page(
        self,
        params: dict,
        max_retries: int | None,
        timeout: tuple[float, float] | None,
    ) -> dict:
        try:
            resp = self._send(
                "GET",
                f"{self.base_url}{_ACCOUNT_JOBS_PATH}",
                params=params,
                retry_on=_is_account_list_transient,
                max_retries=max_retries,
                # The path names no database, so a not-found error here is not
                # about the configured one.
                ensure_database=False,
                **({} if timeout is None else {"timeout": timeout}),
            )
        except requests.exceptions.HTTPError as exc:
            error = _account_list_error(exc, page_token_sent="page_token" in params)
            if error is None:
                raise
            raise error from exc
        with _server_payload():
            return _account_jobs_page(_json_body(resp))

    def for_job(self, job: dict) -> "CortexTrainingClient":
        """Return a client bound to ``job``'s schema, for per-job calls on a
        row from :meth:`iter_jobs` or :meth:`list_jobs_page`.

        The new client borrows this client's session, auth and telemetry, so
        keep this one open and close it rather than the new one. It never
        creates the job's database, which exists and may be another role's: a
        not-found answer surfaces as the HTTP error it is.
        """
        client = type(self)(
            base_url=self.base_url,
            database=_sql_identifier(job["database_name"]),
            schema=_sql_identifier(job["schema_name"]),
            endpoint=self.endpoint,
            poll_interval=self.poll_interval,
            poll_timeout=self.poll_timeout,
            poll_backoff_multiplier=self.poll_backoff_multiplier,
            poll_max_interval=self.poll_max_interval,
            max_retries=self.max_retries,
            request_timeout=self.request_timeout,
        )
        # Release the session and pool the constructor just opened.
        client._session.close()
        client._session = self._session
        client._owns_session = False
        client._auto_create_database = False
        client._auth_provider = self._auth_provider
        client._metric_emitter = self._metric_emitter
        client._artifact_connection_config = self._artifact_connection_config
        client._artifact_connection_factory = self._artifact_connection_factory
        return client

    @_track_operation("cancel_job")
    def cancel_job(self, job_id: str) -> None:
        # The REST API uses colon-action syntax: /{jobId}:cancel
        self._send("POST", f"{self._prefix}/{job_id}:cancel")

    @_track_operation("get_capacity")
    def get_capacity(self, hardware: Hardware | str | None = None) -> dict:
        """Return the calling account's GPU ceiling and current usage.

        Backed by the account-scoped endpoint ``/cortex-training/capacity``
        (not under ``/{job_id}``). The account is resolved server-side from the
        caller's session — never from a request field — so a caller can only
        ever read its own account's capacity. Optional ``hardware``
        (:class:`Hardware`.H200 / .B200 / .B300) is sent as a query parameter and
        scopes the numbers to that GPU type; omitted defaults to H200.

        The returned dict always carries all six fields. The server emits
        proto3 JSON, which omits zero/false fields (an account holding nothing
        under a zero ceiling is literally ``{}``), so we fill in the documented
        defaults:

        - ``has_reservation`` (bool): whether ``max_total_gpus`` is a
          *guaranteed* commitment rather than a best-effort per-account cap.
          When ``False`` the account draws on the shared pool, so
          ``available_gpus`` is an upper bound.
        - ``max_total_gpus`` (int): the account's ceiling — its commitment when
          ``has_reservation``, else its per-account cap. ``-1`` means no
          ceiling, ``0`` a real quota of zero, ``> 0`` the limit. This is the
          canonical ceiling; prefer it over ``reserved_gpus``.
        - ``reserved_gpus`` (int, **deprecated**): mirrors ``max_total_gpus``
          when ``has_reservation`` is ``True``, else 0.
        - ``in_use_gpus`` (int): GPUs the account holds — jobs in ``placing``,
          ``initializing``, or ``running``. Queued work is *not* counted here;
          it is in ``pending_gpus``.
        - ``pending_gpus`` (int): GPUs requested by jobs still waiting for
          capacity (``pending``). It claims quota, so it reduces
          ``available_gpus``.
        - ``available_gpus`` (int): what a new job could start with right now —
          ceiling headroom capped by what is schedulable. ``0`` does not mean
          blocked: a submit within the ceiling is still accepted and queued.

        See ``docs/reference/rest-api.md`` section 5.4 for the authoritative
        field list, and :meth:`get_capacity_breakdown` to split ``in_use_gpus``
        and ``pending_gpus`` by sub-job type or by submitting user.
        """
        params = {}
        if hardware is not None:
            params["hardware"] = _hardware_wire(hardware)
        return _capacity_totals(self._fetch_capacity(params))

    @_track_operation("get_capacity_breakdown")
    def get_capacity_breakdown(
        self,
        hardware: Hardware | str | None = None,
        *,
        dimensions: Iterable[str] = _CAPACITY_BREAKDOWN_DIMENSIONS,
    ) -> dict:
        """Return :meth:`get_capacity` plus a ``"breakdown"`` of its GPU usage.

        Sends the same single request as :meth:`get_capacity` with
        ``?breakdown=`` naming ``dimensions``: ``"job_type"``,
        ``"submitted_by"``, or both (the default). Any other value raises
        ``ValueError`` before sending.

        ``breakdown`` holds ``job_count``, ``truncated``,
        ``unsupported_dimensions``, and ``by_job_type`` / ``by_submitted_by``
        for the requested dimensions. Each has a ``status`` (``OK``,
        ``DISABLED``, ``UNAVAILABLE``, or any newer value the server sends)
        and ``groups``; under ``OK`` the groups add up to ``in_use_gpus`` and
        ``pending_gpus``. Missing counts are 0; on non-``USER`` groups
        ``submitted_by`` is ``None`` and ``is_caller`` is ``False``.

        See ``docs/reference/rest-api.md`` section 5.4 for every field.
        """
        dimensions = _capacity_breakdown_dimensions(dimensions)
        params = {}
        if hardware is not None:
            params["hardware"] = _hardware_wire(hardware)
        params["breakdown"] = ",".join(dimensions)
        body = self._fetch_capacity(params)
        return {
            **_capacity_totals(body),
            "breakdown": _capacity_breakdown(body, dimensions),
        }

    def _fetch_capacity(self, params: Mapping[str, str]) -> Any:
        """``GET /capacity`` once, under the capacity timeout bound; return its JSON."""
        resp = self._send(
            "GET",
            f"{self._prefix}/capacity",
            timeout=(
                min(self.request_timeout[0], _CAPACITY_REQUEST_TIMEOUT[0]),
                min(self.request_timeout[1], _CAPACITY_REQUEST_TIMEOUT[1]),
            ),
            max_retries=0,
            **({"params": params} if params else {}),
        )
        return resp.json()

    @_track_operation("get_experiment_run")
    def get_experiment_run(self, job_id: str) -> dict[str, str]:
        """Return ``{experiment_name, experiment_run_name}`` for ``job_id``.

        Backed by the endpoint ``/cortex-training/{job_id}/experiment-run``.
        A 500 saying ``Linked experiment is not found`` (the caller's role
        can't resolve the job's Experiment) is raised at once, not retried.
        """
        resp = self._send(
            "GET", f"{self._prefix}/{job_id}/experiment-run", retry_on=_is_experiment_run_transient
        )
        return resp.json()

    def _query_sql_row(self, statement: str) -> list[Any]:
        """Execute a synchronous SQL statement and return its first row."""
        resp = self._send(
            "POST",
            f"{self.base_url}/api/v2/statements",
            json={"statement": statement, "database": self.database, "schema": self.schema},
        )
        rows = resp.json().get("data")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], list) or not rows[0]:
            raise ValueError(f"SQL query returned no rows: {statement}")
        return rows[0]

    def _experiment_run_uri(self, job_id: str) -> tuple[str, str]:
        """Return the experiment run's ``snow://`` URI and run name."""
        run = self.get_experiment_run(job_id)
        try:
            experiment_name = run["experiment_name"]
            run_name = run["experiment_run_name"]
        except KeyError as exc:
            raise ValueError(f"experiment-run response missing field: {exc.args[0]}") from exc
        if not isinstance(experiment_name, str) or not experiment_name:
            raise ValueError("experiment-run response has invalid experiment_name")
        if not isinstance(run_name, str) or not run_name:
            raise ValueError("experiment-run response has invalid experiment_run_name")
        return (
            f"snow://experiment/{experiment_name}/versions/{run_name}/",
            run_name,
        )

    def _snowflake_connection_kwargs(self) -> dict[str, Any]:
        """Return connection kwargs derived from this client's PAT credentials."""
        config = self._artifact_connection_config
        if config is None:
            raise RuntimeError("requires a PAT-authenticated client")
        user, role = self._query_sql_row("SELECT CURRENT_USER(), CURRENT_ROLE()")
        if not isinstance(user, str) or not user:
            raise ValueError("SQL identity response missing current user")
        kwargs: dict[str, Any] = {
            "host": config["host"],
            # The login must name the account this host addresses; given the host,
            # the connector's parse_account derives it (first label, or a .global
            # host without its external id). CURRENT_ACCOUNT_NAME() is not that
            # on a locator host and that pairing returns 404.
            "account": config["host"],
            "user": user,
            "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
            "token": config["pat"],
            "database": self.database,
            "schema": self.schema,
        }
        if isinstance(role, str) and role:
            kwargs["role"] = role
        return kwargs

    def _open_experiment_artifact_connection(self) -> Any:
        """Open a connector session for Snowflake experiment artifact LIST/GET."""
        if self._artifact_connection_factory is not None:
            return self._artifact_connection_factory()
        if self._artifact_connection_config is None:
            raise RuntimeError(
                "experiment artifact download requires a Snowflake-authenticated client"
            )
        import snowflake.connector

        return snowflake.connector.connect(**self._snowflake_connection_kwargs())

    def create_snowpark_session(self) -> Any:
        """Create a Snowpark ``Session`` using this client's PAT credentials."""
        from snowflake.snowpark import Session

        return Session.builder.configs(self._snowflake_connection_kwargs()).create()

    @staticmethod
    def _list_experiment_artifacts(
        connection: Any,
        run_uri: str,
        run_name: str,
        artifact_root: str,
    ) -> list[str]:
        """List safe run-relative file paths below ``artifact_root``."""
        root = artifact_root.strip("/") + "/"
        _validate_artifact_relative_path(root.removesuffix("/"))
        cursor = connection.cursor()
        try:
            rows = cursor.execute(
                f"LIST {_sql_string_literal(run_uri + root)}"
            ).fetchall()
        finally:
            cursor.close()

        run_prefix = f"/versions/{run_name}/"
        paths: list[str] = []
        for row in rows:
            name = row[0] if row else None
            if not isinstance(name, str) or not name.startswith(run_prefix):
                raise ValueError(
                    f"experiment artifact LIST returned an unexpected path: {name!r}"
                )
            relative_path = _validate_artifact_relative_path(
                name.removeprefix(run_prefix)
            )
            if not relative_path.startswith(root):
                raise ValueError(
                    "experiment artifact LIST returned a file outside the "
                    f"requested root {root!r}: {relative_path!r}"
                )
            paths.append(relative_path)
        return sorted(set(paths))

    @staticmethod
    def _download_experiment_artifact(
        connection: Any,
        run_uri: str,
        relative_path: str,
        target_dir: Path,
    ) -> Path:
        """Download one exact experiment artifact and return its local path."""
        relative_path = _validate_artifact_relative_path(relative_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_uri = target_dir.resolve().as_uri() + "/"
        cursor = connection.cursor()
        try:
            cursor.execute(
                "GET "
                f"{_sql_string_literal(run_uri + relative_path)} "
                f"{_sql_string_literal(target_uri)}"
            ).fetchall()
        finally:
            cursor.close()
        downloaded = target_dir / PurePosixPath(relative_path).name
        if not downloaded.is_file():
            raise RuntimeError(
                f"Snowflake GET did not create expected artifact {downloaded}"
            )
        return downloaded

    @_track_operation("fetch_execution_logs")
    def fetch_execution_logs(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        filename: str | None = None,
    ) -> list[dict[str, str]]:
        """Download every log file under the job's experiment run stage.

        Returns a list of
        ``{"sub_job_id", "filename", "artifact_uri", "content"}``
        entries, one per object found under a ``_logs/<sub_job_id>/`` subtree
        in the run's stage. Filenames such as ``execution.jsonl`` and
        ``server.log`` are both included; bodies are decoded as UTF-8.

        ``sub_job_id`` and ``filename``, when given, download only the files
        of that sub-job and with that name (for example one sub-job's
        ``execution.jsonl``); the rest are listed but not fetched.

        ``artifact_uri`` is the logical ``snow://`` URI. Files are retrieved
        through Snowflake's supported experiment artifact ``LIST``/``GET``
        interface; no object-store credentials are exposed to the caller.
        """
        run_uri, run_name = self._experiment_run_uri(job_id)
        connection = self._open_experiment_artifact_connection()
        try:
            paths: list[str] = []
            for artifact_root in ("_logs", "checkpoints/_logs"):
                paths.extend(
                    self._list_experiment_artifacts(
                        connection, run_uri, run_name, artifact_root
                    )
                )

            results: list[dict[str, str]] = []
            with tempfile.TemporaryDirectory(prefix="cortex-training-execution-logs-") as temp:
                temp_root = Path(temp)
                for relative_path in sorted(set(paths)):
                    if relative_path.startswith("_logs/"):
                        after_logs = relative_path.removeprefix("_logs/")
                    elif relative_path.startswith("checkpoints/_logs/"):
                        after_logs = relative_path.removeprefix("checkpoints/_logs/")
                    else:
                        continue
                    file_sub_job, separator, file_name = after_logs.partition("/")
                    if (
                        not separator
                        or not file_name
                        or not (
                            file_sub_job == job_id
                            or file_sub_job.startswith(f"{job_id}:")
                        )
                        or file_sub_job in {".", ".."}
                        or "\\" in file_sub_job
                        or (sub_job_id is not None and file_sub_job != sub_job_id)
                        or (filename is not None and file_name != filename)
                    ):
                        continue
                    downloaded = self._download_experiment_artifact(
                        connection,
                        run_uri,
                        relative_path,
                        temp_root,
                    )
                    artifact_uri = run_uri + relative_path
                    try:
                        content = downloaded.read_text(encoding="utf-8")
                    finally:
                        downloaded.unlink(missing_ok=True)
                    results.append(
                        {
                            "sub_job_id": file_sub_job,
                            "filename": file_name,
                            "artifact_uri": artifact_uri,
                            "content": content,
                        }
                    )
            return results
        finally:
            connection.close()

    @staticmethod
    def _resume_manifest_path(destination: Path) -> Path:
        return destination.with_name("." + destination.name + ".manifest.json")

    @classmethod
    def _read_resume_manifest(cls, destination: Path) -> tuple[list[str], int, bool] | None:
        """(chunk paths, committed bytes, partial) from the manifest beside ``destination``.

        ``partial``: the file holds only the newest chunks; the older ones are
        still to be downloaded (see ``newest_first_seconds``).
        """
        path = cls._resume_manifest_path(destination)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict) or doc.get("version") != 1:
            return None
        names = doc.get("names")
        nbytes = doc.get("bytes")
        partial = doc.get("partial", False)
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
            or isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes < 0
            or not isinstance(partial, bool)
        ):
            return None
        return list(names), nbytes, partial

    @classmethod
    def _write_resume_manifest(
        cls, destination: Path, names: list[str], nbytes: int, *, partial: bool = False
    ) -> None:
        path = cls._resume_manifest_path(destination)
        doc: dict[str, Any] = {"bytes": nbytes, "names": names, "version": 1}
        if partial:
            doc["partial"] = True
        payload = json.dumps(doc, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _copy_gzip_member(downloaded: Path, output) -> None:
        with downloaded.open("rb") as compressed:
            with gzip.GzipFile(fileobj=compressed, mode="rb") as source:
                shutil.copyfileobj(source, output, length=_STREAM_COPY_BUFFER_BYTES)

    @staticmethod
    def _artifact_result(
        destination: Path,
        run_uri: str,
        names: list[str],
    ) -> dict[str, Any]:
        return {
            "sub_job_id": destination.parent.name,
            "filename": destination.name,
            "saved_path": str(destination),
            "chunk_count": len(names),
            "first_artifact_uri": run_uri + names[0],
            "last_artifact_uri": run_uri + names[-1],
        }

    def _download_gzip_member(
        self,
        connection: Any,
        run_uri: str,
        relative_path: str,
        artifact_temp: Path,
        output,
    ) -> None:
        downloaded = self._download_experiment_artifact(
            connection,
            run_uri,
            relative_path,
            artifact_temp,
        )
        try:
            self._copy_gzip_member(downloaded, output)
        finally:
            downloaded.unlink(missing_ok=True)

    @staticmethod
    def _artifact_progress_status(
        artifact_name: str, done: int, total: int, sub_job_id: str
    ) -> str:
        width = 20
        filled = (width * done) // total
        bar = "#" * filled + "-" * (width - filled)
        return f"{artifact_name}: [{bar}] {done}/{total} {sub_job_id}"

    @staticmethod
    def _report_artifact_progress(progress: Callable[[str], None] | None, message: str) -> None:
        if progress is not None:
            progress(message)

    @contextlib.contextmanager
    def _artifact_progress_line(
        self, progress: Callable[[str], None] | None
    ) -> Iterator[None]:
        """Close the status line with ``"\\n"`` whether the body succeeds or fails.

        On failure the download error wins over any error from the newline.
        """
        try:
            yield
        except BaseException:
            try:
                self._report_artifact_progress(progress, "\n")
            except BaseException:
                pass
            raise
        self._report_artifact_progress(progress, "\n")

    def _write_ordered_gzip_chunks(
        self,
        connection: Any,
        run_uri: str,
        artifact_temp: Path,
        paths: list[str],
        output,
        on_chunk: Callable[[int], None] | None = None,
    ) -> None:
        """GET sessions are not shared across workers; bytes append in path order."""
        if len(paths) < _ARTIFACT_GET_MIN_CHUNKS:
            for index, relative_path in enumerate(paths, start=1):
                self._download_gzip_member(
                    connection,
                    run_uri,
                    relative_path,
                    artifact_temp,
                    output,
                )
                if on_chunk is not None:
                    on_chunk(index)
            return
        pool = [connection]
        try:
            limit = min(_ARTIFACT_GET_WORKERS, len(paths))
            warned = False
            while len(pool) < limit:
                try:
                    extra = self._open_experiment_artifact_connection()
                except Exception:
                    if not warned:
                        logger.warning(
                            "could not open an extra session for artifact download; "
                            "continuing with %d",
                            len(pool),
                        )
                        warned = True
                    break
                if any(extra is existing for existing in pool):
                    break
                pool.append(extra)
            self._write_gzip_chunk_batches(
                pool, run_uri, artifact_temp, paths, output, on_chunk
            )
        finally:
            for extra in pool[1:]:
                try:
                    extra.close()
                except Exception:
                    logger.warning("failed to close an extra artifact-download session")

    def _write_gzip_chunk_batches(
        self,
        pool: list[Any],
        run_uri: str,
        artifact_temp: Path,
        paths: list[str],
        output,
        on_chunk: Callable[[int], None] | None,
    ) -> None:
        batch_size = len(pool)
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            chunk_dirs = []
            for offset, _relative_path in enumerate(batch):
                chunk_dir = artifact_temp / f"chunk-{start + offset}"
                chunk_dir.mkdir()
                chunk_dirs.append(chunk_dir)
            try:
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    futures = [
                        executor.submit(
                            self._download_experiment_artifact,
                            pool[index],
                            run_uri,
                            relative_path,
                            chunk_dir,
                        )
                        for index, (relative_path, chunk_dir) in enumerate(
                            zip(batch, chunk_dirs)
                        )
                    ]
                    for offset, (future, chunk_dir) in enumerate(
                        zip(futures, chunk_dirs), start=1
                    ):
                        downloaded = future.result()
                        self._copy_gzip_member(downloaded, output)
                        shutil.rmtree(chunk_dir, ignore_errors=True)
                        if on_chunk is not None:
                            on_chunk(start + offset)
            finally:
                for chunk_dir in chunk_dirs:
                    shutil.rmtree(chunk_dir, ignore_errors=True)

    @staticmethod
    def _run_index(chunk_paths: list[str], known: list[str]) -> int | None:
        """Where ``known`` appears in ``chunk_paths`` as one contiguous run, or None."""
        if not known or len(known) > len(chunk_paths):
            return None
        try:
            start = chunk_paths.index(known[0])
        except ValueError:
            return None
        return start if chunk_paths[start : start + len(known)] == known else None

    @staticmethod
    def _copy_committed_bytes(destination: Path, output, nbytes: int) -> None:
        with destination.open("rb") as source:
            remaining = nbytes
            while remaining > 0:
                block = source.read(min(remaining, _STREAM_COPY_BUFFER_BYTES))
                if not block:
                    raise OSError(f"{destination} is shorter than its manifest records")
                output.write(block)
                remaining -= len(block)

    @staticmethod
    def _newest_chunk_paths(
        chunk_paths: list[str], chunk_pattern: re.Pattern[str], seconds: float
    ) -> list[str] | None:
        """The chunks named within ``seconds`` of the newest one, and the one before them.

        Chunk names start with a UTC time (``gpu.YYYYMMDD-HHMMSS.<suffix>.gz``), so
        ``chunk_paths`` (sorted) is in time order. The chunk just before the cut
        is included because a name may give the time its rows start rather than
        end. None when that is every chunk, or a name carries no time.
        """
        times: list[datetime] = []
        for path in chunk_paths:
            match = chunk_pattern.fullmatch(PurePosixPath(path).name)
            if match is None or "timestamp" not in match.groupdict():
                return None
            try:
                times.append(datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S"))
            except ValueError:
                return None
        if not times:
            return None
        cut = times[-1] - timedelta(seconds=max(0.0, seconds))
        start = next(index for index, moment in enumerate(times) if moment >= cut) - 1
        return chunk_paths[start:] if start > 0 else None

    def _chunk_reporter(
        self,
        artifact_name: str,
        sub_job_id: str,
        total: int,
        progress: Callable[[str], None] | None,
        chunk_progress: Callable[[str, int, int], None] | None,
    ) -> Callable[[int], None]:
        """Report ``done`` of ``total`` chunks in place: to ``progress`` as the status
        line, and to ``chunk_progress`` as ``(sub_job_id, done, total)``."""

        def report(done: int) -> None:
            self._report_artifact_progress(
                progress,
                self._artifact_progress_status(artifact_name, done, total, sub_job_id),
            )
            if chunk_progress is not None:
                chunk_progress(sub_job_id, done, total)

        return report

    def _rebuild_gzip_file(
        self,
        connection: Any,
        run_uri: str,
        artifact_temp: Path,
        destination: Path,
        chunk_paths: list[str],
        *,
        temporary_prefix: str,
        artifact_name: str,
        progress: Callable[[str], None] | None,
        reuse: tuple[list[str], int] | None = None,
        partial: bool = False,
        chunk_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Write every chunk to a temp file, then replace the destination.

        ``reuse``: the destination's current (chunk paths, committed bytes). When
        those paths are one contiguous run of ``chunk_paths``, their bytes are
        copied from the destination instead of downloaded again. ``partial``
        records in the manifest that ``chunk_paths`` are only the newest chunks.
        ``chunk_progress(sub_job_id, done, total)`` is called as chunks are in
        place (a reused run counts at once).

        The manifest is removed only after the temp file is fsynced, so a failed
        chunk download leaves the previous manifest and destination in place.
        """
        sub_job_id = destination.parent.name
        chunk_count = len(chunk_paths)
        keep: tuple[int, int, int] | None = None  # (first index, end index, bytes) copied
        if reuse is not None:
            known, committed = reuse
            start = self._run_index(chunk_paths, known)
            try:
                size = destination.stat().st_size
            except OSError:
                size = -1
            if start is not None and size >= committed:
                keep = (start, start + len(known), committed)
        report_chunk = self._chunk_reporter(
            artifact_name, sub_job_id, chunk_count, progress, chunk_progress
        )

        with self._artifact_progress_line(progress):
            self._report_artifact_progress(
                progress,
                self._artifact_progress_status(artifact_name, 0, chunk_count, sub_job_id),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(destination.parent),
                prefix=temporary_prefix,
                suffix=".tmp",
                delete=False,
            )
            temporary = Path(output.name)
            try:
                if keep is None:
                    self._write_ordered_gzip_chunks(
                        connection,
                        run_uri,
                        artifact_temp,
                        chunk_paths,
                        output,
                        report_chunk,
                    )
                else:
                    # The chunks before the reused run, its bytes, then the chunks after it.
                    first, end, nbytes = keep
                    self._write_ordered_gzip_chunks(
                        connection,
                        run_uri,
                        artifact_temp,
                        chunk_paths[:first],
                        output,
                        report_chunk,
                    )
                    self._copy_committed_bytes(destination, output, nbytes)
                    report_chunk(end)
                    self._write_ordered_gzip_chunks(
                        connection,
                        run_uri,
                        artifact_temp,
                        chunk_paths[end:],
                        output,
                        lambda done: report_chunk(end + done),
                    )
                output.flush()
                os.fsync(output.fileno())
                output.close()
                self._resume_manifest_path(destination).unlink(missing_ok=True)
                os.replace(temporary, destination)
                nbytes = destination.stat().st_size
                self._write_resume_manifest(destination, chunk_paths, nbytes, partial=partial)
            except BaseException:
                if not output.closed:
                    output.close()
                temporary.unlink(missing_ok=True)
                raise
        return self._artifact_result(destination, run_uri, chunk_paths)

    def _append_gzip_chunks(
        self,
        connection: Any,
        run_uri: str,
        artifact_temp: Path,
        destination: Path,
        chunk_paths: list[str],
        known: list[str],
        committed: int,
        *,
        temporary_prefix: str,
        artifact_name: str,
        progress: Callable[[str], None] | None,
        partial: bool = False,
        chunk_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Append chunk paths that sort after ``known``, or rebuild.

        A rebuild copies the bytes of ``known`` instead of downloading them again
        when they are one contiguous run of ``chunk_paths``: a file that holds
        only the newest chunks (``partial``), or one whose chunk list grew an
        older chunk.
        """
        sub_job_id = destination.parent.name
        if partial or known != chunk_paths[: len(known)]:
            return self._rebuild_gzip_file(
                connection,
                run_uri,
                artifact_temp,
                destination,
                chunk_paths,
                temporary_prefix=temporary_prefix,
                artifact_name=artifact_name,
                progress=progress,
                reuse=(known, committed),
                chunk_progress=chunk_progress,
            )
        new_paths = chunk_paths[len(known) :]
        try:
            size = destination.stat().st_size
        except OSError:
            size = -1
        if not destination.is_file() or size < committed:
            return self._rebuild_gzip_file(
                connection,
                run_uri,
                artifact_temp,
                destination,
                chunk_paths,
                temporary_prefix=temporary_prefix,
                artifact_name=artifact_name,
                progress=progress,
                chunk_progress=chunk_progress,
            )
        if not new_paths:
            self._report_artifact_progress(
                progress,
                f"{artifact_name}: no new chunks for {sub_job_id}\n",
            )
            if size > committed:
                with destination.open("r+b") as output:
                    output.truncate(committed)
                    output.flush()
                    os.fsync(output.fileno())
            return self._artifact_result(destination, run_uri, known)
        names = list(known) + new_paths
        report_chunk = self._chunk_reporter(
            artifact_name, sub_job_id, len(new_paths), progress, chunk_progress
        )

        with self._artifact_progress_line(progress):
            self._report_artifact_progress(
                progress,
                self._artifact_progress_status(
                    artifact_name, 0, len(new_paths), sub_job_id
                ),
            )
            with destination.open("r+b") as output:
                output.seek(committed)
                output.truncate(committed)
                try:
                    self._write_ordered_gzip_chunks(
                        connection,
                        run_uri,
                        artifact_temp,
                        new_paths,
                        output,
                        report_chunk,
                    )
                    output.flush()
                    os.fsync(output.fileno())
                    end = output.tell()
                except BaseException:
                    output.truncate(committed)
                    output.flush()
                    os.fsync(output.fileno())
                    raise
            self._write_resume_manifest(destination, names, end)
        return self._artifact_result(destination, run_uri, names)

    def _download_gzip_artifacts(
        self,
        job_id: str,
        output_dir: str | os.PathLike[str],
        *,
        artifact_name: str,
        chunk_pattern: re.Pattern[str],
        destination_name: str,
        temporary_prefix: str,
        resume: bool = False,
        sub_job_ids: Iterable[str] | None = None,
        progress: Callable[[str], None] | None = None,
        chunk_progress: Callable[[str, int, int], None] | None = None,
        priority: Iterable[str] = (),
        newest_first_seconds: float | None = None,
        on_commit: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Reconstruct staged gzip chunks into one file per sub-job.

        Each sub-job is committed before the next one starts: those named in
        ``priority`` first, in that order, then the rest in id order.
        ``sub_job_ids`` limits the download to those sub-jobs, in that order
        (``priority`` then has nothing to add). ``resume=True`` appends when the
        manifest names are a prefix of the sorted chunk list. ``progress``
        receives the status line (see :meth:`download_stdout_logs`);
        ``chunk_progress(sub_job_id, done, total)`` is called after each chunk,
        from the calling thread.

        ``newest_first_seconds``: a sub-job with no usable manifest first commits
        only its chunks named within that many seconds of its newest one (the
        manifest records ``"partial": true``), then the whole file, reusing those
        bytes. ``on_commit`` is called after every commit with that sub-job's
        result entry plus ``"complete"`` (False after such a partial commit). An
        exception it raises stops the download; committed files stay.
        """
        run_uri, run_name = self._experiment_run_uri(job_id)
        connection = self._open_experiment_artifact_connection()
        root = Path(output_dir).expanduser()
        artifact_roots = (f"_{artifact_name}", f"checkpoints/_{artifact_name}")
        try:
            paths_by_root = {
                artifact_root: self._list_experiment_artifacts(
                    connection, run_uri, run_name, artifact_root
                )
                for artifact_root in artifact_roots
            }
            chunks_by_sub_job: dict[str, dict[str, Any]] = {}
            for artifact_root, paths in paths_by_root.items():
                prefix = artifact_root + "/"
                for relative_path in paths:
                    after_root = relative_path.removeprefix(prefix)
                    sub_job_id, separator, filename = after_root.partition("/")
                    if (
                        not separator
                        or sub_job_id in {"", ".", ".."}
                        or not (
                            sub_job_id == job_id
                            or sub_job_id.startswith(f"{job_id}:")
                        )
                        or "/" in filename
                        or "\\" in sub_job_id
                        or chunk_pattern.fullmatch(filename) is None
                    ):
                        continue
                    chunks = chunks_by_sub_job.get(sub_job_id)
                    if chunks is None:
                        chunks = {"artifact_root": artifact_root, "paths": []}
                        chunks_by_sub_job[sub_job_id] = chunks
                    elif chunks["artifact_root"] != artifact_root:
                        raise ValueError(
                            f"{artifact_name} chunks for {sub_job_id!r} appear "
                            "under multiple artifact roots"
                        )
                    chunks["paths"].append(relative_path)

            if not chunks_by_sub_job:
                self._report_artifact_progress(
                    progress, f"{artifact_name}: found 0 chunks\n"
                )
                return []

            results: list[dict[str, Any]] = []
            with tempfile.TemporaryDirectory(
                prefix=f"cortex-training-{artifact_name}-artifacts-"
            ) as temp:
                artifact_temp = Path(temp)
                wanted = [priority] if isinstance(priority, str) else priority  # one id, not its characters
                first = [sub for sub in dict.fromkeys(wanted) if sub in chunks_by_sub_job]
                order = first + [sub for sub in sorted(chunks_by_sub_job) if sub not in first]
                if sub_job_ids is not None:
                    only = [sub_job_ids] if isinstance(sub_job_ids, str) else sub_job_ids
                    order = [sub for sub in dict.fromkeys(only) if sub in chunks_by_sub_job]
                for sub_job_id in order:
                    chunk_paths = sorted(chunks_by_sub_job[sub_job_id]["paths"])
                    destination = root / sub_job_id / destination_name
                    prior = self._read_resume_manifest(destination) if resume else None
                    newest = (
                        self._newest_chunk_paths(chunk_paths, chunk_pattern, newest_first_seconds)
                        if prior is None and newest_first_seconds is not None
                        else None
                    )
                    if newest is not None:
                        early = self._rebuild_gzip_file(
                            connection,
                            run_uri,
                            artifact_temp,
                            destination,
                            newest,
                            temporary_prefix=temporary_prefix,
                            artifact_name=artifact_name,
                            progress=progress,
                            partial=True,
                            chunk_progress=chunk_progress,
                        )
                        if on_commit is not None:
                            on_commit({**early, "complete": False})
                        prior = self._read_resume_manifest(destination)
                    if prior is None:
                        result = self._rebuild_gzip_file(
                            connection,
                            run_uri,
                            artifact_temp,
                            destination,
                            chunk_paths,
                            temporary_prefix=temporary_prefix,
                            artifact_name=artifact_name,
                            progress=progress,
                            chunk_progress=chunk_progress,
                        )
                    else:
                        known, committed, partial = prior
                        result = self._append_gzip_chunks(
                            connection,
                            run_uri,
                            artifact_temp,
                            destination,
                            chunk_paths,
                            known,
                            committed,
                            temporary_prefix=temporary_prefix,
                            artifact_name=artifact_name,
                            progress=progress,
                            partial=partial,
                            chunk_progress=chunk_progress,
                        )
                    results.append(result)
                    if on_commit is not None:
                        on_commit({**result, "complete": True})
            return results
        finally:
            connection.close()

    def download_stdout_logs(
        self,
        job_id: str,
        output_dir: str | os.PathLike[str],
        *,
        resume: bool = False,
        sub_job_ids: list[str] | tuple[str, ...] | None = None,
        progress: Callable[[str], None] | None = None,
        chunk_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Reconstruct sealed console artifacts into one file per sub-job.

        ``resume=True`` appends chunks that are not already recorded beside the
        file. ``sub_job_ids`` downloads only those sub-jobs, in that order. Two
        downloads of the same output directory at once are unsupported.
        ``progress`` receives status updates without a trailing newline, then
        ``"\\n"`` when that sub-job finishes or fails. A status that needs no
        updating (nothing new, no chunks) arrives as one message ending in a
        newline. An error raised by ``progress`` stops the download.
        ``chunk_progress(sub_job_id, done, total)`` is called after each chunk,
        from the calling thread. The return value is unchanged.
        """
        return self._download_gzip_artifacts(
            job_id,
            output_dir,
            artifact_name="stdout",
            chunk_pattern=_STDOUT_CHUNK_RE,
            destination_name="stdout.log",
            temporary_prefix=".stdout-",
            resume=resume,
            sub_job_ids=sub_job_ids,
            progress=progress,
            chunk_progress=chunk_progress,
        )

    def download_metrics(
        self,
        job_id: str,
        output_dir: str | os.PathLike[str],
        *,
        resume: bool = False,
        sub_job_ids: list[str] | tuple[str, ...] | None = None,
        progress: Callable[[str], None] | None = None,
        chunk_progress: Callable[[str, int, int], None] | None = None,
        priority: Iterable[str] = (),
        newest_first_seconds: float | None = None,
        on_commit: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Reconstruct staged GPU metric chunks into one JSONL file per sub-job.

        ``resume=True`` appends chunks that are not already recorded beside the
        file. ``sub_job_ids`` downloads only those sub-jobs, in that order. Two
        downloads of the same output directory at once are unsupported.
        ``progress`` receives status updates without a trailing newline, then
        ``"\\n"`` when that sub-job finishes or fails. A status that needs no
        updating (nothing new, no chunks) arrives as one message ending in a
        newline. An error raised by ``progress`` stops the download.
        ``chunk_progress(sub_job_id, done, total)`` is called after each chunk,
        from the calling thread. The return value is unchanged.

        Sub-jobs named in ``priority`` are downloaded first, in that order.
        With ``newest_first_seconds``, a sub-job that has to be downloaded from
        the start first commits only the chunks named within that many seconds
        of its newest chunk, so a reader can show the recent samples before the
        rest of the history arrives. ``on_commit`` is called after each commit
        with the sub-job's result entry plus ``"complete"``: False after that
        first, partial commit, True once the file holds every chunk.
        """
        return self._download_gzip_artifacts(
            job_id,
            output_dir,
            artifact_name="metrics",
            chunk_pattern=_GPU_METRICS_CHUNK_RE,
            destination_name="gpu.jsonl",
            temporary_prefix=".gpu-metrics-",
            resume=resume,
            sub_job_ids=sub_job_ids,
            progress=progress,
            chunk_progress=chunk_progress,
            priority=priority,
            newest_first_seconds=newest_first_seconds,
            on_commit=on_commit,
        )

    # ─── Data-plane async operations ─────────────────────────────────────

    # Maximum payload size for forward-backward requests.
    # Payloads beyond this limit will fail fast with a message.
    _MAX_FWD_BWD_BYTES = 60 * 1024 * 1024  # 60 MB

    # Maximum payload size for generate / generate_stream request bodies.
    # The encoded frame must stay under this cap; generate splits an oversized
    # frame into request chunks, while generate_stream fails fast client-side
    # with a clear message.
    _MAX_GENERATE_BYTES = 60 * 1024 * 1024  # 60 MB

    def _build_generate_frame(
        self,
        op: str,
        job_id: str,
        prompts,
        sampling_params,
        routing_key,
        strict,
    ) -> bytes:
        """Encode a generate request body, for both the unary and stream calls.

        The two calls take the same logical fields and the server reads them
        from the same request model, so any difference in how they encode is a
        bug waiting to happen -- and was one: ``generate_stream`` built its
        body with ``json.dumps`` long after ``generate`` had moved token ids
        into the frame's tensor section, so streaming callers silently kept
        paying the JSON encoding for every token.

        ``op`` only names the caller in the length-check error message.
        """
        from cortex_training import wire  # torch loads on first use, not at import
        self._check_prompt_lengths(op, job_id, prompts)
        # Packed after validation so the check above still reads plain lists,
        # and so a rejected prompt is never converted for nothing.
        payload: dict = {"prompts": _pack_token_prompts(prompts)}
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params
        if routing_key is not None:
            payload["routing_key"] = routing_key
        if strict is not None:
            payload["strict"] = strict
        return wire.dumps(
            payload,
            metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
        )

    def _check_generate_payload(self, op: str, body: bytes) -> None:
        if len(body) > self._MAX_GENERATE_BYTES:
            size_mb = len(body) / (1024 * 1024)
            raise ValueError(
                f"{op} payload is {size_mb:.1f} MB, which exceeds the "
                "maximum allowed size of "
                f"{self._MAX_GENERATE_BYTES / (1024 * 1024):.0f} MB. "
                "Reduce the number of prompts or shorten each prompt to "
                "bring the serialized request under the limit."
            )

    def _resolve_sampling_max_seq_len(self, job_id: str) -> int | None:
        """Return the sampling sub-job's ``max_seq_len``, or ``None`` if unknown.

        The value is read from the job's sampling ``inference_config`` (the same
        number the backend launches vLLM with as ``max_model_len``) and cached
        per ``job_id``. A successful lookup — including a definitive "no sampling
        sub-job" miss — is cached; a transient ``get_job`` failure is not, so a
        later call retries rather than disabling validation permanently.
        """
        if job_id in self._sampling_max_seq_len:
            return self._sampling_max_seq_len[job_id]
        try:
            job = self.get_job(job_id)
        except Exception:
            # Don't let a transient lookup failure break generate, and don't
            # cache the miss — the real request will surface any hard error.
            return None
        max_seq_len: int | None = None
        for sub in job.get("sub_jobs", []) or []:
            if not isinstance(sub, dict):
                continue
            cfg = sub.get("inference_config")
            if not isinstance(cfg, dict):
                continue
            raw = cfg.get("max_seq_len")
            # Proto numbers can arrive as floats (e.g. 2048.0); coerce to int.
            if isinstance(raw, (int, float)) and raw > 0:
                max_seq_len = int(raw)
                break
        self._sampling_max_seq_len[job_id] = max_seq_len
        return max_seq_len

    def _check_prompt_lengths(self, op: str, job_id: str, prompts: list[str | list[int]]) -> None:
        """Fail fast when a pre-tokenized prompt cannot fit the sampling window.

        Mirrors vLLM's decoder-prompt check (``vllm/v1/engine/input_processor``):
        a generation prompt must be strictly shorter than ``max_model_len`` so
        at least one output token fits, i.e. it is rejected when
        ``len(prompt_token_ids) >= max_seq_len``. Only pre-tokenized prompts
        are validated here, since their token count is exact; string prompts
        would require replicating the server tokenizer and are left to fail
        server-side.

        Tensors count as pre-tokenized. They are a supported wire shape for
        token ids, and matching on ``list`` alone let one through unchecked --
        silently trading a clear client-side error for vLLM's much later
        ``maximum model length`` failure.
        """
        torch = _load_torch()

        def _pretokenized(p) -> bool:
            return isinstance(p, list) or torch.is_tensor(p)

        # ``prompts`` is either one prompt or a batch of them. Iterating one
        # prompt yields its tokens, which the loop below would then read as
        # prompts: a flat ``list[int]`` lands on ``int``, which is not
        # pre-tokenized, so the whole check is skipped; a bare tensor lands on
        # 0-d scalars, which are tensors, so the check proceeds and ``len()``
        # raises on them. Wrapping first means one prompt is checked as one
        # prompt in either shape, and reads the argument the same way
        # ``_pack_token_prompts`` does.
        if torch.is_tensor(prompts) or (
            isinstance(prompts, list) and prompts and _is_token_id(prompts[0])
        ):
            prompts = [prompts]

        if not any(_pretokenized(p) for p in prompts):
            return
        max_seq_len = self._resolve_sampling_max_seq_len(job_id)
        if max_seq_len is None:
            return
        for i, prompt in enumerate(prompts):
            if not _pretokenized(prompt):
                continue
            n = len(prompt)
            if n >= max_seq_len:
                raise ValueError(
                    f"{op} prompt at index {i} has {n} tokens, which does not "
                    f"fit the sampling job's max_seq_len of {max_seq_len}: a "
                    "prompt must be shorter than max_seq_len to leave room for "
                    "at least one generated token (vLLM rejects "
                    "len(prompt) >= max_model_len). Shorten the prompt to at "
                    f"most {max_seq_len - 1} tokens, or recreate the job with a "
                    "larger max_seq_len."
                )

    def _post_octet_request_chunks(
        self,
        *,
        job_id: str,
        path_suffix: str,
        operation: str,
        frame: bytes,
        max_bytes: int,
        force_chunk: bool = False,
        allow_group_restart: bool = False,
        debug_context: dict[str, Any] | None = None,
    ) -> dict:
        from cortex_training import wire  # torch loads on first use, not at import
        chunks = wire.encode_byte_chunks(
            frame,
            kind="request",
            operation=operation,
            max_bytes=max_bytes,
            force_chunk=force_chunk,
        )
        group_metadata = wire.read_byte_chunk_metadata(chunks[0])
        chunk_group_id = (
            str(group_metadata["chunk_group_id"])
            if group_metadata is not None and group_metadata.get("chunk_group_id")
            else None
        )
        group_restarts = 0
        idx = 0
        final_body: dict | None = None
        while idx < len(chunks):
            chunk = chunks[idx]
            try:
                resp = self._send(
                    "POST",
                    f"{self._prefix}/{job_id}/{path_suffix}",
                    data=chunk,
                    headers={"Content-Type": "application/octet-stream"},
                    retry_on=(_is_chunk_post_transient if allow_group_restart else _is_transient),
                    debug_context=debug_context,
                )
            except requests.exceptions.HTTPError as exc:
                detail = _chunk_group_error_detail(exc.response)
                if detail is None:
                    raise
                code = detail.get("code")
                if allow_group_restart and code == _CHUNK_GROUP_RESTART_REQUIRED and group_restarts < 1:
                    group_restarts += 1
                    idx = 0
                    final_body = None
                    logger.warning(
                        "%s chunk group %s lost server state; replaying once from chunk 0",
                        operation,
                        chunk_group_id,
                    )
                    continue
                error_type = (
                    ChunkGroupRestartError if code == _CHUNK_GROUP_RESTART_REQUIRED else ChunkGroupConflictError
                )
                raise error_type(
                    str(detail.get("message") or code),
                    response=exc.response,
                    detail=detail,
                ) from exc

            body = resp.json()
            if idx < len(chunks) - 1:
                if isinstance(body, dict) and body.get("request_id"):
                    raise RuntimeError(f"{operation} chunk {idx} unexpectedly returned request_id")
            else:
                final_body = body
            idx += 1
        if final_body is None:
            raise RuntimeError(f"{operation} produced no request body")
        return final_body

    @_track_operation("forward_backward")
    def forward_backward(self, job_id: str, data: bytes) -> str:
        """Submit a forward+backward pass. Returns request_id."""
        self._fwd_bwd_send_count += 1
        call = self._fwd_bwd_send_count
        debug_context = {
            "phase": "submit",
            "call": call,
            "job_id": job_id,
            "payload_bytes": len(data),
        }
        debug_label = self._debug_context_label(debug_context)
        logger.debug("%s payload total bytes: %d", debug_label, len(data))
        body = self._post_octet_request_chunks(
            job_id=job_id,
            path_suffix="forward-backward",
            operation="fwd-bwd",
            frame=data,
            max_bytes=self._MAX_FWD_BWD_BYTES,
            force_chunk=True,
            allow_group_restart=True,
            debug_context=debug_context,
        )
        request_id = body["request_id"]
        request_debug_context = {
            "phase": "poll",
            "call": call,
            "job_id": job_id,
            "request_id": request_id,
            "payload_bytes": len(data),
        }
        self._fwd_bwd_request_debug[request_id] = request_debug_context
        logger.debug(
            "%s submitted request_id=%s body=%s",
            debug_label,
            request_id,
            self._debug_json_summary(body),
        )
        return request_id

    @_track_operation("generate")
    def generate(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> str:
        """Submit a generate request to a sampling sub-job. Returns request_id.

        The typed fields are serialized as a DSSST1 safetensors frame and sent
        as ``application/octet-stream``. Oversized frames are byte-chunked
        across multiple POSTs; the final POST returns the pollable request id.
        Poll with :meth:`poll_request`.

        Prompt-length validation is asymmetric by input type:

        * **Pre-tokenized prompts** are checked client-side against the
          sampling sub-job's ``max_seq_len`` and fail fast with a
          ``ValueError`` when ``len(prompt) >= max_seq_len`` — the token count
          is exact, so this never rejects a prompt vLLM would accept. A flat
          ``list[int]``, and a bare 1-D tensor, each count as one prompt.
        * **String prompts** are *not* validated here. Counting their tokens
          would require replicating the server tokenizer (including special
          tokens), so an over-long string instead fails server-side in vLLM
          with a ``maximum model length`` error rather than client-side.
        """
        body = self._build_generate_frame("generate", job_id, prompts, sampling_params, routing_key, strict)
        response = self._post_octet_request_chunks(
            job_id=job_id,
            path_suffix="generate",
            operation="generate",
            frame=body,
            max_bytes=self._MAX_GENERATE_BYTES,
        )
        request_id = response["request_id"]
        self._generate_request_ids.add(request_id)
        return request_id

    @_track_operation("generate_stream")
    def generate_stream(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> dict:
        """Start a streaming generate request. Returns the response body.

        The body is the same DSSST1 safetensors frame :meth:`generate` sends,
        under ``application/octet-stream``. Unlike ``generate`` it is never
        split into request chunks: the frame goes out in one POST and must fit
        under 60 MiB.

        Progress is read with :meth:`get_request_status` using the returned
        ``request_id``. Cancellation uses :meth:`cancel_request`.

        Prompt-length validation matches :meth:`generate`: pre-tokenized
        prompts are checked client-side against the sampling sub-job's
        ``max_seq_len`` and fail fast, while over-long string prompts are left
        to fail server-side in vLLM (see :meth:`generate`).
        """
        body = self._build_generate_frame("generate_stream", job_id, prompts, sampling_params, routing_key, strict)
        self._check_generate_payload("generate_stream", body)
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/generate-stream",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        return resp.json()

    @_track_operation("step")
    def step(self, job_id: str, learning_rate: float | None = None) -> str:
        """Submit an optimizer step. Returns request_id."""
        body: dict = {}
        if learning_rate is not None:
            body["learning_rate"] = learning_rate
        resp = self._send("POST", f"{self._prefix}/{job_id}/step", json=body)
        return resp.json()["request_id"]

    @_track_operation("save")
    def save(
        self,
        job_id: str,
        checkpoint_id: str | None = None,
        checkpoint_type: str | None = None,
    ) -> str:
        """Submit a checkpoint save. Returns request_id."""
        body: dict = {}
        if checkpoint_id is not None:
            body["checkpoint_id"] = checkpoint_id
        if checkpoint_type is not None:
            normalized_type = checkpoint_type.lower()
            if normalized_type not in ("resumable", "weights-only"):
                raise ValueError("checkpoint_type must be 'resumable' or 'weights-only'")
            body["checkpoint_type"] = normalized_type
        resp = self._send("POST", f"{self._prefix}/{job_id}/save", json=body)
        return resp.json()["request_id"]

    @_track_operation("load")
    def load(
        self,
        job_id: str,
        checkpoint_id: str,
        source_job_id: str | None = None,
        *,
        target_sub_job_id: str | None = None,
    ) -> str:
        """Submit a checkpoint load into an existing job. Returns request_id.

        Args:
            job_id: The job to load the checkpoint into.
            checkpoint_id: Checkpoint identifier (from a prior save).
            source_job_id: Optional. Load from another job's checkpoint store.
            target_sub_job_id: Optional. Route the load to a specific training
                sub-job. Format: ``"{job_id}:training:{index}"``. Omit to use
                the session's default training sub-job.

        Returns:
            request_id for polling via ``poll_request()``.

        **When to use target_sub_job_id:**

        A job has at most one training sub-job, so omitting this routes to that
        sub-job. Pass it when you want explicit routing rather than relying on
        the server's default resolution.

        **Discovering sub-job IDs:**

        .. code-block:: python

            job = client.get_job(job_id)
            for sj in job["sub_jobs"]:
                if sj["job_type"] == "training":
                    sub_job_id = sj["sub_job_id"]
                    n_gpus = sj["training_config"]["n_gpus"]
                    print(f"Training sub-job {sub_job_id}: {n_gpus} GPUs")

        **DP size compatibility:**

        If the target sub-job has a different DP size (``n_gpus``) than the
        checkpoint was saved from, you **must** have created the job with
        ``load_optimizer_states=False`` in the ``SubJobConfig``. The optimizer
        states are DP-sharded and cannot be resized. This setting is configured
        at job creation time and cannot be changed at runtime.

        **Server validation:**

        The server validates that ``target_sub_job_id`` belongs to this session
        (returns 400 if not) and is a training sub-job (returns 501 for
        non-training sub-jobs like sampling).
        """
        body: dict = {"checkpoint_id": checkpoint_id}
        if source_job_id is not None:
            body["source_job_id"] = source_job_id
        if target_sub_job_id is not None:
            body["target_sub_job_id"] = target_sub_job_id
        resp = self._send("POST", f"{self._prefix}/{job_id}/load", json=body)
        return resp.json()["request_id"]

    def _operation(
        self,
        job_id: str,
        operation_type: str,
        *,
        payload: dict | bytes | bytearray | memoryview | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        max_json_bytes: int | None = None,
        retry_on: Callable[[BaseException], bool] | None = None,
    ) -> dict:
        """Submit a generic data-plane operation to the routed sub-job.

        ``max_json_bytes``, when set, fails fast if the encoded body exceeds it.
        ``retry_on`` overrides the retry predicate when passed; chunk posts use
        one that excludes structured chunk-group conflicts.
        """
        body: dict = {"operation_type": operation_type}
        if sub_job_id is not None:
            body["sub_job_id"] = sub_job_id
        if sub_job_type is not None:
            body["sub_job_type"] = sub_job_type
        if isinstance(payload, (bytes, bytearray, memoryview)):
            payload = {
                "payload_b64": base64.b64encode(bytes(payload)).decode("ascii"),
                "content_type": "application/octet-stream",
            }
        if payload is not None:
            body["payload"] = payload
        if max_json_bytes is not None:
            encoded_bytes = _operation_body_json_bytes(body)
            if encoded_bytes > max_json_bytes:
                raise ValueError(
                    f"{operation_type} operation body is {encoded_bytes} bytes, which "
                    f"exceeds the {max_json_bytes}-byte limit for one /operation "
                    f"request. Lower {_OPERATION_MAX_JSON_BYTES_ENV} so the payload is "
                    f"split into more chunks, or raise it if the deployment allows "
                    f"larger JSON bodies."
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "POST operation %s/%s/operation body=%s",
                self._prefix,
                job_id,
                self._debug_json_summary(_log_safe_operation_body(body)),
            )
        kwargs: dict[str, Any] = {}
        if retry_on is not None:
            kwargs["retry_on"] = retry_on
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/operation",
            json=body,
            **kwargs,
        )
        return resp.json()

    def _post_operation_request_chunks(
        self,
        job_id: str,
        *,
        operation_type: str,
        frame: bytes,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Send a DSSST1 frame through ``/operation``, byte-chunked if oversized."""
        from cortex_training import wire  # torch loads on first use, not at import
        max_json_bytes = _operation_max_json_bytes()
        # The server assembler rejects total_chunks <= 1, so a fitting frame
        # is posted raw.
        chunks = wire.encode_byte_chunks(
            frame,
            kind="request",
            operation=operation_type,
            max_bytes=_operation_chunk_max_bytes(max_json_bytes),
        )
        final_body: dict | None = None
        for idx, chunk in enumerate(chunks):
            try:
                body = self._operation(
                    job_id,
                    operation_type,
                    payload=chunk,
                    sub_job_id=sub_job_id,
                    sub_job_type=sub_job_type,
                    max_json_bytes=max_json_bytes,
                    retry_on=_is_chunk_post_transient,
                )
            except requests.exceptions.HTTPError as exc:
                detail = _chunk_group_error_detail(exc.response)
                if detail is None:
                    raise
                raise ChunkGroupConflictError(
                    str(detail.get("message") or detail.get("code")),
                    response=exc.response,
                    detail=detail,
                ) from exc
            if idx < len(chunks) - 1:
                if isinstance(body, dict) and body.get("request_id"):
                    raise RuntimeError(
                        f"{operation_type} chunk {idx} unexpectedly returned request_id"
                    )
            else:
                final_body = body
        if final_body is None:
            raise RuntimeError(f"{operation_type} produced no request body")
        return final_body

    @_track_operation("forward")
    def forward(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Submit a generic forward operation. Returns the response body.

        A ``bytes`` payload is a DSSST1 frame; it is byte-chunked across as many
        ``/operation`` POSTs as the JSON body ceiling requires and reassembled by
        the backend. Only the final POST returns the pollable ``request_id``;
        poll it with :meth:`poll_request`. Dict payloads stay a single plain
        ``/operation`` POST.
        """
        if isinstance(payload, (bytes, bytearray, memoryview)):
            body = self._post_operation_request_chunks(
                job_id,
                operation_type="forward",
                frame=bytes(payload),
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            )
        else:
            body = self._operation(
                job_id,
                "forward",
                payload=payload,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            )
        request_id = body.get("request_id") if isinstance(body, dict) else None
        if isinstance(request_id, str) and request_id:
            self._forward_request_ids.add(request_id)
        return body

    def fwd(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(
            job_id,
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def fwd_no_grad(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(
            job_id,
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    @_track_operation("weight_sync")
    def weight_sync(
        self,
        job_id: str,
        source_sub_job_id: str,
        target_sub_job_ids: list[str],
        *,
        weight_format: str | None = None,
        bucket_size: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> str:
        """Sync model weights from a training sub-job into one or more sampling
        sub-jobs (used in RL training loops). Returns request_id.

        Multi-sub-job sessions require the operation envelope to include a
        routing hint; by default the operation is routed through the source
        training sub-job.

        Args:
            job_id: The job owning both sub-jobs.
            source_sub_job_id: Training sub-job to read weights from, e.g.
                ``f"{job_id}:training:0"``.
            target_sub_job_ids: Sampling sub-jobs to update, e.g.
                ``[f"{job_id}:sampling:0"]``.
            weight_format: ``"vllm"`` (server default) or ``"hf"`` for full
                weights, or ``"lora"`` to broadcast only the adapter tensors.
            bucket_size: Optional transfer-bucket size in bytes. Increase this
                when one model tensor exceeds the server default bucket.

        Example::

            request_id = client.weight_sync(
                job_id,
                source_sub_job_id=f"{job_id}:training:0",
                target_sub_job_ids=[f"{job_id}:sampling:0"],
                weight_format="lora",
            )
            client.poll_request(job_id, request_id)
        """
        body = {
            "source_sub_job_id": source_sub_job_id,  # training
            "target_sub_job_ids": list(target_sub_job_ids),  # sampling
        }
        if weight_format is not None:
            # "lora" broadcasts only the trained adapter tensors.
            body["weight_format"] = weight_format
        if bucket_size is not None:
            if bucket_size <= 0:
                raise ValueError("bucket_size must be positive")
            body["bucket_size"] = int(bucket_size)
        return self._operation(
            job_id,
            "weight-sync",
            payload=body,
            sub_job_id=sub_job_id or source_sub_job_id,  # training
            sub_job_type=(sub_job_type if sub_job_type is not None else "training"),
        )["request_id"]

    @_track_operation("bootstrap_router_replay")
    def bootstrap_router_replay(
        self,
        job_id: str,
        source_sub_job_id: str,
        target_sub_job_id: str,
        *,
        max_cache_bytes: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Bootstrap router replay from one sub-job to another."""
        payload: dict = {
            "source_sub_job_id": source_sub_job_id,
            "target_sub_job_id": target_sub_job_id,
        }
        if max_cache_bytes is not None:
            payload["max_cache_bytes"] = max_cache_bytes
        return self._operation(
            job_id,
            "bootstrap-router-replay",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    @_track_operation("router_replay_discard")
    def router_replay_discard(
        self,
        job_id: str,
        sample_ids: list[str] | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Discard router replay samples on a routed sub-job."""
        payload = dict(extra or {})
        payload.setdefault("sample_ids", list(sample_ids or []))
        return self._operation(
            job_id,
            "router-replay-discard",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    @_track_operation("reset_prefix_cache")
    def reset_prefix_cache(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = "sampling",
        drain: bool = True,
        timeout_s: float = 60.0,
        retry_interval_s: float = 0.1,
        extra: dict | None = None,
    ) -> dict:
        """Reset the sampling prefix cache for a Cortex Training job.

        Operation-specific fields are carried inside the opaque operation
        payload. ``extra`` allows new reset options to flow through without a
        cortex-training release.
        """
        payload = dict(extra or {})
        payload.setdefault("drain", drain)
        payload.setdefault("timeout_s", timeout_s)
        payload.setdefault("retry_interval_s", retry_interval_s)
        return self._operation(
            job_id,
            "reset-prefix-cache",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    # ─── Async request polling ────────────────────────────────────────────

    @staticmethod
    def _normalize_request_status(status: str) -> str:
        """Normalize full enum names (``REQUEST_STATE_DONE``) to short form (``done``)."""
        return status.lower().removeprefix("request_state_")

    @staticmethod
    def _decode_result_payload(result: dict) -> dict | None:
        from cortex_training import wire  # torch loads on first use, not at import
        if not isinstance(result, dict):
            return None
        if result.get("wire_format") != wire.WIRE_FORMAT_VERSION:
            return None
        if result.get("encoding") != "base64":
            raise RuntimeError("DSSST1 result payload must use base64 encoding")
        raw = result.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            raise RuntimeError("DSSST1 result payload missing payload_b64")
        payload = base64.b64decode(raw)
        decoded = wire.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError("DSSST1 result payload did not decode to a dict")
        return CortexTrainingClient._merge_transport_metrics(decoded, result)

    @staticmethod
    def _merge_transport_metrics(decoded: dict, envelope: dict) -> dict:
        transport_metrics = envelope.get("metrics")
        if not isinstance(transport_metrics, dict):
            return decoded
        decoded = dict(decoded)
        payload_metrics = decoded.get("metrics")
        if isinstance(payload_metrics, dict):
            decoded["metrics"] = {**payload_metrics, **transport_metrics}
        else:
            decoded["metrics"] = dict(transport_metrics)
        return decoded

    @staticmethod
    def _decode_result_chunk_event(event: dict) -> bytes | None:
        if not isinstance(event, dict) or event.get("type") != "result_chunk":
            return None
        raw = event.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            raise RuntimeError("result_chunk event missing payload_b64")
        payload = base64.b64decode(raw)
        expected = event.get("payload_sha256")
        if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError("result_chunk payload_sha256 mismatch")
        return payload

    @staticmethod
    def _restore_generate_result_lists(value: Any, torch_module: Any | None = None) -> Any:
        torch_module = _load_torch() if torch_module is None else torch_module
        if torch_module.is_tensor(value):
            return value.cpu().tolist()
        if isinstance(value, dict):
            return {
                key: CortexTrainingClient._restore_generate_result_lists(item, torch_module)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [CortexTrainingClient._restore_generate_result_lists(item, torch_module) for item in value]
        if isinstance(value, tuple):
            return tuple(CortexTrainingClient._restore_generate_result_lists(item, torch_module) for item in value)
        return value

    def _normalize_generate_result_if_needed(self, request_id: str, result: dict) -> dict:
        if request_id not in self._generate_request_ids:
            return result
        self._generate_request_ids.discard(request_id)
        if isinstance(result, dict) and "results" in result:
            result = dict(result)
            result["results"] = self._restore_generate_result_lists(result["results"])
        return result

    def _normalize_forward_result_if_needed(self, request_id: str, result: dict) -> dict:
        """Decode a forward result that is only a base64 DSSST1 blob.

        A current backend returns a self-describing envelope that
        :meth:`_decode_result_payload` already handles. An older backend
        returns ``{"job_id", "payload_b64"}`` with no ``wire_format``.
        """
        from cortex_training import wire  # torch loads on first use, not at import
        if request_id not in self._forward_request_ids:
            return result
        self._forward_request_ids.discard(request_id)
        if not isinstance(result, dict) or result.get("wire_format"):
            return result
        raw = result.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            return result
        decoded = wire.loads(base64.b64decode(raw))
        if not isinstance(decoded, dict):
            raise RuntimeError("legacy forward payload_b64 did not decode to a dict")
        return self._merge_transport_metrics(decoded, result)

    @staticmethod
    def _decode_stream_result_event(event: Any) -> Any:
        if not isinstance(event, dict) or event.get("type") != "result":
            return event
        result = event.get("result")
        if not isinstance(result, dict):
            return event
        decoded = CortexTrainingClient._decode_result_payload(result)
        if decoded is None:
            return event
        event = dict(event)
        event["result"] = CortexTrainingClient._restore_generate_result_lists(decoded)
        return event

    @staticmethod
    def _decode_stream_result_events(status: dict) -> dict:
        events = status.get("events")
        if not isinstance(events, list):
            return status
        decoded = [CortexTrainingClient._decode_stream_result_event(event) for event in events]
        if decoded == events:
            return status
        status = dict(status)
        status["events"] = decoded
        return status

    @_track_operation("poll_request")
    def poll_request(self, job_id: str, request_id: str) -> dict:
        """Poll until the async request completes. Returns the result dict."""
        from cortex_training import wire  # torch loads on first use, not at import
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        debug_context = self._fwd_bwd_request_debug.get(request_id)
        debug_label = self._debug_context_label(debug_context) if debug_context is not None else None
        result_chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            status = self.get_request_status(job_id, request_id, cursor=cursor)
            state = self._normalize_request_status(status.get("status", ""))
            events = status.get("events") or []
            received_chunk = False
            for event in events:
                with _server_payload():
                    chunk = self._decode_result_chunk_event(event)
                if chunk is not None:
                    result_chunks.append(chunk)
                    received_chunk = True
            next_cursor = status.get("next_cursor")
            if isinstance(next_cursor, str) and next_cursor:
                cursor = next_cursor
                # Drain available result chunks without backing off.
                continue
            if state in ("completed", "done", "succeeded"):
                if result_chunks:
                    with _server_payload():
                        result = wire.decode_result_chunks(result_chunks)
                    result = self._merge_transport_metrics(
                        result,
                        status.get("result") or {},
                    )
                else:
                    result = status.get("result") or {}
                    with _server_payload():
                        decoded = self._decode_result_payload(result)
                    if decoded is not None:
                        result = decoded
                result = self._normalize_generate_result_if_needed(request_id, result)
                result = self._normalize_forward_result_if_needed(request_id, result)
                if debug_label is not None:
                    logger.debug(
                        "%s completed state=%s result=%s",
                        debug_label,
                        state,
                        self._debug_json_summary(result),
                    )
                    self._fwd_bwd_request_debug.pop(request_id, None)
                return result
            if state in ("failed", "cancelled", "canceled"):
                error = status.get("error", "")
                if debug_label is not None:
                    logger.debug(
                        "%s failed state=%s error=%s body=%s",
                        debug_label,
                        state,
                        error,
                        self._debug_json_summary(status),
                    )
                    self._fwd_bwd_request_debug.pop(request_id, None)
                self._generate_request_ids.discard(request_id)
                self._forward_request_ids.discard(request_id)
                raise _server_failure(
                    RuntimeError(f"Request {request_id} ended with state '{state}': {error}"),
                    failed=state == "failed",
                )
            if received_chunk:
                # Defensive: if the server returns a chunk without next_cursor
                # before terminal status, poll again immediately rather than
                # sleeping.
                continue
            delay = self._sleep_with_backoff(delay, deadline)
        if debug_label is not None:
            logger.debug("%s timed out after %ss", debug_label, self.poll_timeout)
            self._fwd_bwd_request_debug.pop(request_id, None)
        self._generate_request_ids.discard(request_id)
        self._forward_request_ids.discard(request_id)
        raise _server_failure(
            TimeoutError(f"Request {request_id} did not complete within {self.poll_timeout}s")
        )

    @_track_operation("get_request_status")
    def get_request_status(
        self,
        job_id: str,
        request_id: str,
        *,
        max_events: int | None = None,
        cursor: str | None = None,
    ) -> dict:
        params = {}
        if max_events is not None:
            params["max_events"] = max_events
        if cursor:
            params["cursor"] = cursor
        debug_context = self._fwd_bwd_request_debug.get(request_id)
        resp = self._send(
            "GET",
            f"{self._prefix}/{job_id}/requests/{request_id}",
            params=params or None,
            debug_context=debug_context,
        )
        with _server_payload():
            return self._decode_stream_result_events(resp.json())

    @_track_operation("cancel_request")
    def cancel_request(
        self,
        job_id: str,
        request_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        return self._operation(
            job_id,
            "cancel-request",
            payload={"request_id": request_id},
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    # ─── Logs (read-only) ────────────────────────────────────────────────

    def tail_logs(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_lines: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Fetch one page of a sub-job's logs from ``cursor`` (read-only).

        The target is the sub-job's zone-manager (Ray head) pod, whose stdout
        already includes worker output (Ray ``log_to_driver``); identify it via
        ``sub_job_id`` (or ``sub_job_type`` when unambiguous). Returns
        ``{"entries": [...], "next_cursor": str, "eof": bool}``. Re-call with
        ``next_cursor`` to continue. Without a cursor (``None`` or empty),
        ``max_lines`` tails the log's last lines; with neither, the read starts
        at the beginning of the log.

        A 409 (the job is pending placement, or no longer running) or a 404
        (its session is gone) is raised at once instead of being retried:
        neither clears on a retry, and :meth:`get_job` tells them apart. See
        ``docs/reference/rest-api.md`` section 7.8.
        """
        payload: dict = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_lines is not None:
            payload["max_lines"] = max_lines
        return self._operation(
            job_id,
            "tail-logs",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
            retry_on=_is_tail_logs_transient,
        )

    def _stream_pages(
        self,
        fetch,
        entries_key: str,
        *,
        follow: bool,
        cursor: str | None = None,
        poll_interval: float | None = None,
    ) -> Iterator[dict]:
        """Drive a cursor-paged tail: call ``fetch(cursor)`` for a page, yield
        its entries (under ``entries_key``), advance the cursor, and poll with
        backoff. With ``follow`` False, stop once a page reports ``eof``.
        """
        base_delay = poll_interval or self.poll_interval
        delay = base_delay
        while True:
            page = fetch(cursor)
            cursor = page.get("next_cursor", cursor)
            entries = page.get(entries_key) or []
            for entry in entries:
                yield entry
            if entries:
                delay = base_delay  # made progress; poll again promptly
                continue
            if page.get("eof", False) and not follow:
                return
            time.sleep(delay)
            delay = min(delay * self.poll_backoff_multiplier, self.poll_max_interval)

    def stream_logs(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        cursor: str | None = None,
        max_lines: int | None = None,
        poll_interval: float | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        """Yield a sub-job's log entries, polling with backoff.

        Drains the log then, with ``follow=True`` (the default), keeps polling
        for new entries until the caller stops iterating — suitable for a live
        tail. With ``follow=False`` it stops once the log reports ``eof``.
        """
        return self._stream_pages(
            lambda cur: self.tail_logs(
                job_id,
                cursor=cur,
                max_lines=max_lines,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            ),
            "entries",
            follow=follow,
            cursor=cursor,
            poll_interval=poll_interval,
        )

    # ─── Zone scheduling events (read-only) ───────────────────────────────

    def tail_events(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_events: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Fetch one page of zone scheduling events for the session.

        Returns ``{"events": [...], "next_cursor": str, "eof": bool}``. Served
        by the zone dispatcher itself, not routed to a Zone Manager.

        Note: the server does not currently accept this operation; see
        ``docs/reference/rest-api.md`` section 7.9.
        """
        payload: dict = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_events is not None:
            payload["max_events"] = max_events
        return self._operation(
            job_id,
            "zmd-events",
            payload=payload or None,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def stream_events(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_events: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        poll_interval: float | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        """Yield zone scheduling events for the session, polling with backoff."""
        return self._stream_pages(
            lambda cur: self.tail_events(
                job_id,
                cursor=cur,
                max_events=max_events,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            ),
            "events",
            follow=follow,
            cursor=cursor,
            poll_interval=poll_interval,
        )

    # ─── Checkpoints ─────────────────────────────────────────────────────

    @_track_operation("list_checkpoints")
    def list_checkpoints(self, job_id: str) -> list:
        """Return the job's checkpoint records (``checkpoint_id``, ``created_at``, ``checkpoint_type``).

        A read never creates the configured database: when it doesn't exist,
        the job can't be in it, so the service's error is raised as it is.
        """
        resp = self._send("GET", f"{self._prefix}/{job_id}/checkpoints", ensure_database=False)
        return resp.json().get("checkpoints", [])

    @_track_operation("export_checkpoint")
    def export_checkpoint(self, job_id: str, checkpoint_id: str) -> dict:
        """Return download links for every file of one checkpoint.

        The service answers ``{checkpoint_id, files: [{filename, url,
        size_bytes}], expires_at}``. Each ``url`` is presigned and lasts 5 hours
        (until ``expires_at``); fetch it without this client's credentials, and
        export again for fresh links. ``filename`` is relative and may include a
        folder. Like :meth:`list_checkpoints`, this
        never creates the configured database. See
        ``docs/reference/rest-api.md`` section 5.8.
        """
        # The REST API uses colon-action syntax: /{jobId}/checkpoints/{cpId}:export
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/checkpoints/{checkpoint_id}:export",
            ensure_database=False,
        )
        return resp.json()

    @_track_operation("delete_checkpoint")
    def delete_checkpoint(self, job_id: str, checkpoint_id: str) -> None:
        """Delete a checkpoint. Returns None on success."""
        self._send("DELETE", f"{self._prefix}/{job_id}/checkpoints/{checkpoint_id}")
