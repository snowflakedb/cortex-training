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

"""Cortex Training client v2, built on the Arctic Platform client.

Same public surface as :class:`cortex_training.client.CortexTrainingClient`.
Request bodies come from ``arctic_platform.client.requests`` and are delivered
by ``arctic_platform``'s ``CortexTransport``. A feature they cannot express
raises ``NotImplementedError``; the comment beside it says what is missing.

Not carried over from v1, because Arctic Platform has no equivalent:

- telemetry around each call, and the local training-metrics log;
- v1's exception types: HTTP errors are ``aiohttp.ClientResponseError``, and a
  missing database or schema is not explained (``DatabaseAccessError``);
- client-side checks made before sending: prompt length against the sampling
  ``max_seq_len``, ``create_job_from_body``'s sub-job checks, log-probability
  sub-job rejection, ``save``'s checkpoint type and ``weight_sync``'s bucket size;
- decoding the legacy ``{"job_id", "payload_b64"}`` forward result.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import aiohttp
from arctic_platform import wire
from arctic_platform.client.base import _next_log_wait
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.requests import bootstrap_router_replay_request
from arctic_platform.client.requests import cancel_job_request
from arctic_platform.client.requests import cancel_request_request
from arctic_platform.client.requests import capacity_request
from arctic_platform.client.requests import delete_checkpoint_request
from arctic_platform.client.requests import export_checkpoint_request
from arctic_platform.client.requests import fwd_bwd_request
from arctic_platform.client.requests import fwd_no_grad_request
from arctic_platform.client.requests import generate_request
from arctic_platform.client.requests import generate_stream_request
from arctic_platform.client.requests import get_experiment_run_request
from arctic_platform.client.requests import get_job_request
from arctic_platform.client.requests import list_checkpoints_request
from arctic_platform.client.requests import list_jobs_request
from arctic_platform.client.requests import load_checkpoint_request
from arctic_platform.client.requests import reset_prefix_cache_request
from arctic_platform.client.requests import router_replay_discard_request
from arctic_platform.client.requests import save_checkpoint_request
from arctic_platform.client.requests import step_request
from arctic_platform.client.requests import sync_weights_request
from arctic_platform.client.requests import tail_logs_request
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _SendPolicy
from arctic_platform.client.transports.cortex import _decode_result
from arctic_platform.client.transports.cortex import _identity
from arctic_platform.client.transports.cortex import _is_job_not_found
from arctic_platform.client.transports.cortex import _parent_id
from arctic_platform.client.transports.cortex import _route
from arctic_platform.client.transports.cortex import _to_python
from arctic_platform.client.transports.cortex import job_create_payload

from cortex_training.client.auth import rest_host
from cortex_training.client.core import _SNOWFLAKE_REQUEST_ID_HEADER
from cortex_training.client.core import Hardware
from cortex_training.client.core import SubJobConfig
from cortex_training.model_deployment import DEPLOYMENT_STATUS_DROPPED
from cortex_training.model_deployment import DEPLOYMENT_STATUS_DROPPING
from cortex_training.model_deployment import DEPLOYMENT_STATUS_FAILED
from cortex_training.model_deployment import DEPLOYMENT_STATUS_TERMINATED
from cortex_training.model_deployment import JOB_FIELD_JOB_ID
from cortex_training.model_deployment import OUT_DEPLOYMENT
from cortex_training.model_deployment import OUT_DROPPED
from cortex_training.model_deployment import OUT_STATUS
from cortex_training.model_deployment import SOURCE_TYPE_CHECKPOINT
from cortex_training.model_deployment import ModelDeploymentSpec
from cortex_training.model_deployment import _deployment_id_from_job_id
from cortex_training.model_deployment import _job_id_from_deployment_id
from cortex_training.model_deployment import _model_deployment_description
from cortex_training.model_deployment import _model_deployment_metadata
from cortex_training.model_deployment import _model_deployment_ready
from cortex_training.model_deployment import _model_deployment_request_error
from cortex_training.model_deployment import _model_deployment_status
from cortex_training.model_deployment import _model_deployment_status_filter
from cortex_training.model_deployment import _model_deployment_sub_job
from cortex_training.model_deployment import _model_deployment_wait_timeout
from cortex_training.model_deployment import _pending_model_deployment_metadata
from cortex_training.model_deployment import _trained_model_name

logger = logging.getLogger(__name__)

# CortexConfig takes one total timeout per request, so v1's (connect, read)
# default becomes its read bound.
_DEFAULT_REQUEST_TIMEOUT = 600.0
# CortexTransport's fixed job-wait backoff; deployment waits use it too.
_POLL_BACKOFF_MULTIPLIER = 1.25
_POLL_MAX_INTERVAL = 6.0
_REQUEST_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError)


def _unsupported(feature: str) -> Callable[..., Any]:
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{feature}: not supported by the Arctic Platform client")

    return method


def _unsupported_unless(condition: bool, feature: str) -> None:
    if not condition:
        raise NotImplementedError(f"{feature}: not supported by the Arctic Platform client")


def _deployment_request_error(exc: Exception, deployment_id: str | None, **kwargs: Any) -> RuntimeError:
    """:func:`_model_deployment_request_error` for a failed ``CortexTransport`` call."""
    # Only HTTP errors carry a response; error_code and error_message come from the
    # body, when CortexTransport could read one.
    headers = getattr(exc, "headers", None) or {}
    return _model_deployment_request_error(
        deployment_id,
        status_code=getattr(exc, "status", None),
        code=getattr(exc, "error_code", None),
        detail=getattr(exc, "error_message", None),
        request_id=headers.get(_SNOWFLAKE_REQUEST_ID_HEADER),
        **kwargs,
    )


def _deployment_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a method whose first argument is a public ``sf-ct-`` deployment ID.

    The ID is validated before any request, and any HTTP error is turned into
    :func:`_model_deployment_request_error`.
    """

    @functools.wraps(method)
    def wrapper(self: Any, deployment_id: str, *args: Any, **kwargs: Any) -> Any:
        job_id = _job_id_from_deployment_id(deployment_id)
        try:
            return method(self, deployment_id, *args, **kwargs)
        except _REQUEST_ERRORS as exc:
            raise _deployment_request_error(
                exc, deployment_id, job_id=job_id, scope=f"{self.database}.{self.schema}"
            ) from exc

    return wrapper


class CortexTrainingClient:
    """HTTP client for the Cortex Training REST API, on the Arctic Platform client.

    Construct it as v1's :class:`cortex_training.client.CortexTrainingClient`:
    directly with a ``base_url`` (no auth), or with :meth:`from_configs`.
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
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        # Needs: CortexConfig fields for the poll backoff and its cap (CortexTransport
        # hard-codes 1.25x up to 6s) and for the connection pool size.
        _unsupported_unless(
            (poll_backoff_multiplier, poll_max_interval, pool_maxsize)
            == (_POLL_BACKOFF_MULTIPLIER, _POLL_MAX_INTERVAL, 1024),
            "poll_backoff_multiplier, poll_max_interval and pool_maxsize",
        )
        # Needs: a (connect, read) timeout pair on CortexConfig.request_timeout.
        _unsupported_unless(not isinstance(request_timeout, tuple), "A (connect, read) request_timeout")
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.schema = schema
        self.endpoint = endpoint
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._generate_request_ids: set[str] = set()
        # CortexTransport polls only the job it holds, so each job gets its own.
        self._transports: dict[str | None, CortexTransport] = {}
        self._lock = threading.Lock()
        self._settings: dict[str, Any] = {}
        # Explicit Nones keep ARCTIC_CORTEX_* environment variables out of the job body.
        self._configure(
            base_url=self.base_url,
            database=database,
            schema=schema,
            endpoint=endpoint,
            max_retries=max_retries,
            request_timeout=request_timeout,
            poll_timeout=poll_timeout,
            job_ready_timeout=poll_timeout,
            job_comment=None,
            idle_timeout_seconds=None,
            pending_timeout_seconds=None,
        )

    def _configure(self, **settings: Any) -> None:
        self._settings.update(settings)
        self._config = CortexConfig(**self._settings)

    @classmethod
    def from_configs(
        cls,
        connection: str | None = None,
        *,
        database: str | None = None,
        schema: str | None = None,
        **kwargs: Any,
    ) -> "CortexTrainingClient":
        """Create a client from ``CORTEX_TRAINING_HOST`` / ``CORTEX_TRAINING_PAT``, as v1 does without a profile."""
        from cortex_training import _picker

        resolution = _picker.resolve(connection=connection, database=database, schema=schema)
        logger.info("Cortex Training target: %s", resolution.describe())
        # Needs: Snowflake connection-profile auth on CortexConfig (a Connector session
        # token, recreated on expiry); CortexTransport only sends a PAT.
        _unsupported_unless(resolution.profile is None, "Snowflake connection-profile auth")
        return cls._from_pat(
            host=resolution.host,
            pat=resolution.pat,
            database=resolution.database,
            schema=resolution.schema,
            **kwargs,
        )

    @classmethod
    def from_pat(cls, *args: Any, **kwargs: Any) -> "CortexTrainingClient":
        """Deprecated: use :meth:`from_configs`. Same arguments as before (``host``, ``pat``, ...)."""
        warnings.warn(
            "CortexTrainingClient.from_pat() is deprecated and will be removed in a future "
            "release. Use CortexTrainingClient.from_configs().",
            FutureWarning,
            stacklevel=2,
        )
        return cls._from_pat(*args, **kwargs)

    # Needs: Snowflake connection-profile auth on CortexConfig.
    from_connection_name = classmethod(_unsupported("Snowflake connection-profile auth"))

    @property
    def auto_create_database(self) -> bool:
        """Deprecated, and always False: the client never creates a database."""
        return False

    @auto_create_database.setter
    def auto_create_database(self, value: bool) -> None:
        if value:
            warnings.warn(
                "CortexTrainingClient.auto_create_database is deprecated and does nothing.",
                FutureWarning,
                stacklevel=2,
            )

    @classmethod
    def _from_pat(
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
        host = rest_host(host)
        client = cls(
            base_url=f"https://{host}", database=database, schema=schema, endpoint=endpoint, **kwargs
        )
        client._configure(base_url=None, host=host, pat=pat, verify_ssl=verify_ssl)
        return client

    def close(self) -> None:
        with self._lock:
            transports, self._transports = list(self._transports.values()), {}
        for transport in transports:
            transport.shutdown()

    def __enter__(self) -> "CortexTrainingClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ─── Delivery ────────────────────────────────────────────────────────

    def _transport(self, job_id: str | None = None) -> CortexTransport:
        key = None if job_id is None else _parent_id(job_id)
        with self._lock:
            if key not in self._transports:
                transport = CortexTransport(self._config)
                transport.job_id = key
                transport.poll_interval = self.poll_interval
                self._transports[key] = transport
            return self._transports[key]

    def _call(self, request: Request, **route_changes: Any) -> Any:
        """Deliver ``request`` on its Cortex route, with ``route_changes`` applied to the route."""
        route = replace(_route(request.op), **route_changes)
        return self._transport(request.job_id)._call(request, route)

    def _submit(self, request: Request) -> dict:
        """Post ``request`` and return the server's reply as-is; :meth:`poll_request` waits for it."""
        return self._call(request, poll=False, normalize=_identity)

    # ─── Job management ──────────────────────────────────────────────────

    def create_job(
        self,
        sub_jobs: list[SubJobConfig],
        job_id: str | None = None,
        experiment_name: str | None = None,
        hardware: Hardware | str | None = None,
        idle_timeout_seconds: int | None = None,
        pending_timeout_seconds: int | None = None,
    ) -> str:
        """Create a job from a list of sub-jobs and return its server job_id."""
        if not sub_jobs:
            raise ValueError("create_job requires a non-empty sub_jobs list")
        # Needs: job_id, experiment_name and hardware fields that job_create_payload sends.
        _unsupported_unless(
            (job_id, experiment_name, hardware) == (None, None, None),
            "create_job job_id, experiment_name and hardware",
        )
        # CortexConfig validates both timeouts the way v1 does.
        config = CortexConfig(
            **{
                **self._settings,
                "idle_timeout_seconds": idle_timeout_seconds,
                "pending_timeout_seconds": pending_timeout_seconds,
            }
        )
        body = job_create_payload([sj.to_wire() for sj in sub_jobs], config)
        return self.create_job_from_body(body)["job_id"]

    def create_job_from_body(self, body: dict) -> dict:
        """Create a job from a raw REST CreateJob request body."""
        # Needs: the internal-only `debug` block gate (CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS).
        _unsupported_unless(not body.get("debug"), "A create-job `debug` block")
        job = self._call(Request("create-job", body=body))
        # A new job can briefly be missing from reads; CortexTransport retries its first one.
        self._transport(job["job_id"])._awaiting_job_visibility = True
        return job

    def wait_for_job(self, job_id: str) -> dict:
        """Poll until the job reaches ``running``. Raises on terminal states or timeout."""
        self._transport(job_id)._wait_running()
        return self.get_job(job_id)

    def get_job(self, job_id: str, *, retry_on: Callable[[BaseException], bool] | None = None) -> dict:
        if retry_on is not None:
            return self._call(get_job_request(job_id), policy=_SendPolicy(retry_on=retry_on))
        return self._transport(job_id)._job()

    def list_jobs(self, status: str | None = None) -> list:
        return self._call(list_jobs_request(status))

    def cancel_job(self, job_id: str) -> None:
        self._call(cancel_job_request(job_id))

    def get_capacity(self, hardware: Hardware | str | None = None) -> dict:
        """Return the calling account's GPU ceiling and current usage."""
        return self._call(capacity_request(None if hardware is None else Hardware(hardware).value))

    def get_experiment_run(self, job_id: str) -> dict[str, str]:
        return self._call(get_experiment_run_request(job_id))

    # ─── Training and sampling ───────────────────────────────────────────

    def forward_backward(self, job_id: str, data: bytes) -> str:
        """Submit a forward+backward pass. Returns request_id."""
        # fwd_bwd_request encodes the body itself, so the caller's frame is decoded first.
        return self._submit(fwd_bwd_request(JobHandles(training=job_id), wire.loads(data)))["request_id"]

    def generate(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> str:
        """Submit a generate request to a sampling sub-job. Returns request_id."""
        request = generate_request(JobHandles(sampling=job_id), prompts, sampling_params, routing_key, strict)
        request_id = self._submit(request)["request_id"]
        self._generate_request_ids.add(request_id)
        return request_id

    def generate_stream(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> dict:
        """Start a streaming generate request. Returns the response body."""
        return self._submit(
            generate_stream_request(
                JobHandles(sampling=job_id), prompts, sampling_params, routing_key, strict
            )
        )

    def step(self, job_id: str, learning_rate: float | None = None) -> str:
        """Submit an optimizer step. Returns request_id."""
        return self._submit(step_request(JobHandles(training=job_id), learning_rate))["request_id"]

    def save(self, job_id: str, checkpoint_id: str | None = None, checkpoint_type: str | None = None) -> str:
        """Submit a checkpoint save. Returns request_id."""
        return self._submit(
            save_checkpoint_request(JobHandles(training=job_id), checkpoint_id, checkpoint_type)
        )["request_id"]

    def load(
        self,
        job_id: str,
        checkpoint_id: str,
        source_job_id: str | None = None,
        *,
        target_sub_job_id: str | None = None,
    ) -> str:
        """Submit a checkpoint load into an existing job. Returns request_id."""
        # Needs: a target_sub_job_id argument on load_checkpoint_request.
        _unsupported_unless(target_sub_job_id is None, "load target_sub_job_id")
        request = load_checkpoint_request(
            JobHandles(training=job_id), checkpoint_id=checkpoint_id, source_job_id=source_job_id
        )
        return self._submit(request)["request_id"]

    def forward(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Submit a forward operation carrying a DSSST1 frame. Returns the response body."""
        # Needs: forward over JSON payloads, and an envelope that leaves sub_job_id to
        # the server or sets sub_job_type; Arctic Platform's forward always sends a
        # DSSST1 frame addressed to the sub-job it is given.
        _unsupported_unless(
            isinstance(payload, (bytes, bytearray, memoryview))
            and sub_job_id is not None
            and sub_job_type is None,
            "forward without a bytes payload and sub_job_id, or with sub_job_type,",
        )
        return self._submit(fwd_no_grad_request(JobHandles(training=sub_job_id), wire.loads(payload)))

    def fwd(
        self, job_id: str, payload: dict | bytes | bytearray | memoryview | None = None, **kwargs: Any
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(job_id, payload, **kwargs)

    def fwd_no_grad(
        self, job_id: str, payload: dict | bytes | bytearray | memoryview | None = None, **kwargs: Any
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(job_id, payload, **kwargs)

    # ─── Operations ──────────────────────────────────────────────────────

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
        """Sync weights from a training sub-job into a sampling sub-job. Returns request_id."""
        # Needs: several targets on sync_weights_request, and routing hints other than
        # the source training sub-job.
        _unsupported_unless(
            len(target_sub_job_ids) == 1
            and sub_job_id in (None, source_sub_job_id)
            and sub_job_type in (None, "training"),
            "weight_sync to several targets, or routed other than through the source,",
        )
        jobs = JobHandles(training=source_sub_job_id, sampling=target_sub_job_ids[0])
        return self._submit(sync_weights_request(jobs, weight_format=weight_format, bucket_size=bucket_size))[
            "request_id"
        ]

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
        # Needs: routing hints on bootstrap_router_replay_request; it always routes to the target.
        _unsupported_unless(
            sub_job_id in (None, target_sub_job_id) and sub_job_type is None,
            "bootstrap_router_replay routed other than through the target",
        )
        jobs = JobHandles(training=target_sub_job_id, sampling=source_sub_job_id)
        return self._submit(bootstrap_router_replay_request(jobs, max_cache_bytes))

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
        # Needs: router_replay_discard_request without a sub_job_id, with a
        # sub_job_type, or with extra payload fields.
        _unsupported_unless(
            sub_job_id is not None and sub_job_type is None and not extra,
            "router_replay_discard without sub_job_id, or with sub_job_type or extra,",
        )
        return self._submit(
            router_replay_discard_request(JobHandles(sampling=sub_job_id), list(sample_ids or []))
        )

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
        """Reset the sampling prefix cache for a Cortex Training job."""
        # Needs: sub_job_id, other sub_job_types and extra payload fields on
        # reset_prefix_cache_request.
        _unsupported_unless(
            sub_job_id is None and sub_job_type == "sampling" and not extra,
            "reset_prefix_cache with sub_job_id, extra, or a sub_job_type other than sampling,",
        )
        return self._submit(
            reset_prefix_cache_request(JobHandles(sampling=job_id), drain, timeout_s, retry_interval_s)
        )

    # ─── Async request polling ───────────────────────────────────────────

    def poll_request(self, job_id: str, request_id: str) -> dict:
        """Poll until the async request completes. Returns the result dict."""
        is_generate = request_id in self._generate_request_ids
        self._generate_request_ids.discard(request_id)
        transport = self._transport(job_id)
        chunks, envelope = transport._on_loop(transport._poll(request_id)).result()
        result = _decode_result(envelope, chunks)
        if is_generate and "results" in result:
            result = {**result, "results": _to_python(result["results"])}
        return result

    def cancel_request(
        self,
        job_id: str,
        request_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        # Needs: cancel_request_request without a sub_job_id, or with a sub_job_type.
        _unsupported_unless(
            sub_job_id is not None and sub_job_type is None,
            "cancel_request without sub_job_id, or with sub_job_type,",
        )
        return self._submit(cancel_request_request(sub_job_id, request_id))

    # ─── Logs ────────────────────────────────────────────────────────────

    def tail_logs(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_lines: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Fetch one page of a sub-job's logs from ``cursor``.

        A job that is gone answers an empty ``eof`` page, where v1 raises with a
        pointer to :meth:`download_stdout_logs`.
        """
        # Needs: tail_logs_request without a sub_job_id (v1 lets sub_job_type pick one).
        _unsupported_unless(
            sub_job_id is not None and sub_job_type is None,
            "tail_logs without sub_job_id, or with sub_job_type,",
        )
        return self._submit(tail_logs_request(sub_job_id, cursor, max_lines))

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
        """Yield a sub-job's log entries, polling with backoff."""
        poll_interval = poll_interval or self.poll_interval
        wait = 0.0
        while True:
            page = self.tail_logs(
                job_id, cursor=cursor, max_lines=max_lines, sub_job_id=sub_job_id, sub_job_type=sub_job_type
            )
            cursor = page["next_cursor"] or cursor
            yield from page["entries"] or []
            wait = _next_log_wait(page, wait, poll_interval, follow)
            if wait is None:
                return
            time.sleep(wait)

    # ─── Checkpoints ─────────────────────────────────────────────────────

    def list_checkpoints(self, job_id: str) -> list:
        return self._call(list_checkpoints_request(job_id))

    def export_checkpoint(self, job_id: str, checkpoint_id: str) -> dict:
        return self._call(export_checkpoint_request(job_id, checkpoint_id))

    def delete_checkpoint(self, job_id: str, checkpoint_id: str) -> None:
        self._call(delete_checkpoint_request(job_id, checkpoint_id))

    # ─── Model deployments ───────────────────────────────────────────────

    @property
    def model_deployment_base_url(self) -> str:
        """OpenAI-compatible base URL for deployments in this database and schema."""
        return f"{self._transport()._prefix}/inference/v1"

    def create_model_deployment(self, spec: ModelDeploymentSpec | Mapping[str, Any]) -> dict[str, Any]:
        """Create an evaluation deployment and return only its public metadata.

        See ``docs/reference/rest-api.md`` section 14.
        """
        raw_spec = spec.to_public_spec() if isinstance(spec, ModelDeploymentSpec) else spec
        spec = ModelDeploymentSpec.from_dict(dict(raw_spec) if isinstance(raw_spec, Mapping) else raw_spec)
        try:
            if spec.source_type == SOURCE_TYPE_CHECKPOINT:
                source_job = self.get_job(spec.source_job_id)
                spec = replace(spec, model_name=_trained_model_name(source_job, spec.source_job_id))
            response = self.create_job_from_body(spec._to_create_job_body())
        except _REQUEST_ERRORS as exc:
            raise _deployment_request_error(exc, None) from exc
        deployment_id = _deployment_id_from_job_id(response.get(JOB_FIELD_JOB_ID))
        return _pending_model_deployment_metadata(spec, deployment_id, self.model_deployment_base_url)

    def _model_deployment_job(self, job_id: str) -> dict[str, Any]:
        """Fetch one job and raise ``ValueError`` if it is not a model deployment."""
        job = self.get_job(job_id)
        if _model_deployment_sub_job(job) is None:
            raise ValueError("the requested resource is not a model deployment")
        return job

    def list_model_deployments(self, status: str | Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Return sanitized metadata for every marked deployment in scope.

        ``status`` keeps only deployments in the given deployment status(es),
        e.g. ``"READY"`` or ``["PENDING", "INITIALIZING"]`` (case-insensitive).
        """
        wanted = _model_deployment_status_filter(status)
        deployments = [
            _model_deployment_metadata(job, _deployment_id_from_job_id(job.get(JOB_FIELD_JOB_ID)))
            for job in self.list_jobs()
            if _model_deployment_sub_job(job) is not None
        ]
        return [d for d in deployments if wanted is None or d[OUT_STATUS] in wanted]

    @_deployment_operation
    def get_model_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Return sanitized metadata for one opaque deployment ID."""
        job = self._model_deployment_job(_job_id_from_deployment_id(deployment_id))
        return _model_deployment_metadata(job, deployment_id, base_url=self.model_deployment_base_url)

    def describe_model_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Return one deployment's status, endpoint, scope, and configuration."""
        return _model_deployment_description(
            self.get_model_deployment(deployment_id), database=self.database, schema=self.schema
        )

    def wait_for_model_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Poll until an opaque deployment ID is READY or terminal.

        Raises :class:`ModelDeploymentWaitError` for a terminal status other
        than READY and :class:`ModelDeploymentWaitTimeout` when the poll timeout
        passes; both carry the last observed metadata as ``deployment``.
        """
        # Not a _deployment_operation: ModelDeploymentWaitTimeout is a TimeoutError,
        # which that would report as a failed request.
        _job_id_from_deployment_id(deployment_id)
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        metadata: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            metadata = self.get_model_deployment(deployment_id)
            if _model_deployment_ready(metadata):
                return metadata
            time.sleep(max(0.0, min(delay, deadline - time.monotonic())))
            delay = min(delay * _POLL_BACKOFF_MULTIPLIER, _POLL_MAX_INTERVAL)
        raise _model_deployment_wait_timeout(deployment_id, self.poll_timeout, metadata)

    @_deployment_operation
    def drop_model_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Idempotently drop one opaque deployment ID.

        A ``DROPPING`` or ``DROPPED`` deployment is reported as dropped without
        a second cancel. A ``FAILED`` or ``TERMINATED`` one can never be dropped,
        so it is returned as ``dropped: False`` with its ``status``. A missing
        deployment is reported as dropped only when the lookup fails with the
        job-not-found code; any other error propagates.
        """
        job_id = _job_id_from_deployment_id(deployment_id)
        try:
            job = self._model_deployment_job(job_id)
        except aiohttp.ClientResponseError as exc:
            if _is_job_not_found(exc):
                return {OUT_DEPLOYMENT: deployment_id, OUT_DROPPED: True}
            raise
        status = _model_deployment_status(job)
        if status in {DEPLOYMENT_STATUS_FAILED, DEPLOYMENT_STATUS_TERMINATED}:
            return {OUT_DEPLOYMENT: deployment_id, OUT_DROPPED: False, OUT_STATUS: status}
        if status not in {DEPLOYMENT_STATUS_DROPPING, DEPLOYMENT_STATUS_DROPPED}:
            self.cancel_job(job_id)
        return {OUT_DEPLOYMENT: deployment_id, OUT_DROPPED: True}

    # ─── Not supported by the Arctic Platform client ─────────────────────

    # Needs: a client-side breakdown normalizer and dimension check; the capacity
    # route returns the server's breakdown unparsed.
    get_capacity_breakdown = _unsupported("get_capacity_breakdown")
    # Needs: a request-status route that sends `cursor` and returns the raw status
    # (status, next_cursor, result); Arctic Platform's reshapes it to {state, error, events}.
    get_request_status = _unsupported("get_request_status")
    # Needs: a `zmd-events` operation.
    tail_events = _unsupported("tail_events")
    stream_events = _unsupported("stream_events")
    # Needs: the account-level GET /api/v2/{endpoint} route with paging, and a client
    # bound to another database and schema that shares this one's session.
    iter_jobs = _unsupported("Account-level job listing")
    list_jobs_page = _unsupported("Account-level job listing")
    for_job = _unsupported("for_job")
    # Needs: Snowflake connector sessions for experiment artifacts (LIST/GET of logs
    # and metrics), Snowpark sessions, and keeping sessions between downloads.
    download_model_deployment_logs = _unsupported("Model deployment log downloads")
    download_model_deployment_metrics = _unsupported("Model deployment metric downloads")
    keep_artifact_sessions = property(_unsupported("Experiment artifact downloads"))
    warm_artifact_session = _unsupported("Experiment artifact downloads")
    create_snowpark_session = _unsupported("Snowpark sessions")
    fetch_execution_logs = _unsupported("Experiment artifact downloads")
    download_execution_logs = _unsupported("Experiment artifact downloads")
    download_stdout_logs = _unsupported("Experiment artifact downloads")
    download_metrics = _unsupported("Experiment artifact downloads")
    download_training_metrics = _unsupported("Experiment artifact downloads")
    # Needs: OTLP client telemetry.
    emit_metric = _unsupported("Client telemetry")
