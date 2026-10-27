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

"""Model deployment spec, constants and metadata for ``cortex-training model-deployment``.

Spec keys, job fields and statuses, deployment statuses, the keys of the
metadata the client returns, and the helpers that read a job into that
metadata, in one place. The client imports this module, so it must not import
the client at module level. See ``docs/reference/rest-api.md`` section 14.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Only EVALUATION deployments are supported today.
MODEL_DEPLOYMENT_CLASS = "EVALUATION"
# Public deployment IDs are sf-ct-<backing job UUID>.
MODEL_DEPLOYMENT_ID_PREFIX = "sf-ct-"
# Allowed GPUs per replica (the replica's tensor-parallel size). This copies a
# server hardware-placement rule so a bad GPU shape fails in the CLI; update it
# together with the server. Which models are supported, and the GPUs each
# needs, is left to the server.
MODEL_DEPLOYMENT_GPU_WIDTHS = frozenset({1, 2, 4, 8})
# Duplicated from ``client.Hardware``, which this module cannot import (the
# client imports it); a test keeps the two copies from drifting apart.
MODEL_DEPLOYMENT_GPU_TYPES = ("H200", "B200", "B300")

# Keys of the customer deployment spec (the JSON passed to ``create``).
SPEC_DEPLOYMENT = "deployment"
SPEC_SOURCE = "source"
SPEC_INFERENCE = "inference"
# Keys inside spec.deployment: the deployment's class and capacity.
DEPLOYMENT_CLASS = "class"
DEPLOYMENT_MIN_REPLICAS = "min_replicas"
DEPLOYMENT_GPUS_PER_REPLICA = "gpus_per_replica"
DEPLOYMENT_GPU_TYPE = "gpu_type"
# Keys inside spec.source, and its two source types. A checkpoint source names
# no model: the server already knows the source job's model, so the client
# reads it.
SOURCE_TYPE = "type"
SOURCE_MODEL = "model"
SOURCE_JOB_ID = "job_id"
SOURCE_CHECKPOINT_ID = "checkpoint_id"
SOURCE_TYPE_BASE_MODEL = "base_model"
SOURCE_TYPE_CHECKPOINT = "checkpoint"
# Keys inside spec.inference.
INFERENCE_MAX_SEQ_LEN = "max_seq_len"
# Keys of the sampling sub-job the server stores for a deployment.
WIRE_INFERENCE_CONFIG = "inference_config"
WIRE_NUM_REPLICAS = "num_replicas"
WIRE_N_GPUS = "n_gpus"
WIRE_VLLM_CONFIG = "vllm_config"
WIRE_TENSOR_PARALLEL_SIZE = "tensor_parallel_size"
WIRE_MAX_MODEL_LEN = "max_model_len"
WIRE_SOURCE_CHECKPOINT_INFO = "source_checkpoint_info"
WIRE_SOURCE_JOB_ID = "source_job_id"
WIRE_CHECKPOINT_ID = "checkpoint_id"
WIRE_DEPLOYMENT_CLASS = "deployment_class"
WIRE_MAX_SEQ_LEN = "max_seq_len"
# Customer-facing deployment statuses.
DEPLOYMENT_STATUS_PENDING = "PENDING"
DEPLOYMENT_STATUS_INITIALIZING = "INITIALIZING"
DEPLOYMENT_STATUS_READY = "READY"
DEPLOYMENT_STATUS_FAILED = "FAILED"
DEPLOYMENT_STATUS_DROPPING = "DROPPING"
DEPLOYMENT_STATUS_DROPPED = "DROPPED"
DEPLOYMENT_STATUS_TERMINATED = "TERMINATED"
DEPLOYMENT_STATUS_UNKNOWN = "UNKNOWN"
DEPLOYMENT_STATUSES = (
    DEPLOYMENT_STATUS_PENDING,
    DEPLOYMENT_STATUS_INITIALIZING,
    DEPLOYMENT_STATUS_READY,
    DEPLOYMENT_STATUS_FAILED,
    DEPLOYMENT_STATUS_DROPPING,
    DEPLOYMENT_STATUS_DROPPED,
    DEPLOYMENT_STATUS_TERMINATED,
    DEPLOYMENT_STATUS_UNKNOWN,
)
# Statuses in which the deployment no longer serves, so no base_url is shown.
DEPLOYMENT_STATUSES_WITHOUT_ENDPOINT = frozenset(
    {
        DEPLOYMENT_STATUS_FAILED,
        DEPLOYMENT_STATUS_DROPPING,
        DEPLOYMENT_STATUS_DROPPED,
        DEPLOYMENT_STATUS_TERMINATED,
    }
)
# Fields read from a job response.
JOB_FIELD_JOB_ID = "job_id"
JOB_FIELD_HARDWARE = "hardware"
JOB_FIELD_STATUS = "status"
# ``reason`` is a stable token (e.g. ``sub_job_failed``) and
# ``reason_details`` an optional display sentence about it; only the token is
# meant to be matched on.
JOB_FIELD_REASON = "reason"
JOB_FIELD_REASON_DETAILS = "reason_details"
JOB_FIELD_CREATED_AT = "created_at"
JOB_FIELD_OWNER_ROLE = "owner_role"
JOB_FIELD_SUBMITTED_BY = "submitted_by"
# The server's reason for a job cancelled through the API, i.e. a drop. The
# DROPPING / DROPPED status already says this, so it is not repeated as
# status_reason.
JOB_REASON_CANCELLED_BY_CLIENT = "cancelled_by_client"
# Keys of the deployment metadata this client returns. ``deployment`` holds
# whatever identifies the deployment; today that is its sf-ct-<UUID> ID.
OUT_DEPLOYMENT = "deployment"
OUT_DEPLOYMENT_CLASS = "deployment_class"
OUT_MAX_SEQ_LEN = "max_seq_len"
OUT_GPUS_PER_REPLICA = "gpus_per_replica"
OUT_GPU_TYPE = "gpu_type"
OUT_STATUS = "status"
OUT_STATUS_REASON = "status_reason"
OUT_STATUS_DETAILS = "status_details"
OUT_MODEL_SOURCE = "model_source"
OUT_REPLICAS = "replicas"
OUT_CREATED_AT = "created_at"
OUT_OWNER_ROLE = "owner_role"
OUT_CREATOR = "creator"
OUT_FAILURE = "failure"
OUT_FAILURE_CODE = "code"
OUT_FAILURE_MESSAGE = "message"
OUT_BASE_URL = "base_url"
OUT_DROPPED = "dropped"
OUT_LOGS = "logs"
OUT_METRICS = "metrics"
# Extra keys ``describe`` adds: the scope it was read from, and the deployment's
# configuration grouped under ``spec`` (a read view, not a create input).
OUT_DATABASE = "database"
OUT_SCHEMA = "schema"
OUT_SPEC = "spec"
# Job statuses, in the short form ``CortexTrainingClient._normalize_job_status``
# returns (``JOB_STATE_RUNNING`` -> ``running``).
JOB_STATUS_PENDING = "pending"
JOB_STATUS_PLACING = "placing"
JOB_STATUS_INITIALIZING = "initializing"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLING = "cancelling"
JOB_STATUS_CANCELLED = "cancelled"
JOB_STATUS_DONE = "done"
JOB_STATUS_TERMINATED = "terminated"
# How each job status is shown for a deployment. A job moves to running only
# once all its sub-jobs are ready, so running is READY.
JOB_STATUS_TO_DEPLOYMENT_STATUS = {
    JOB_STATUS_PENDING: DEPLOYMENT_STATUS_PENDING,
    JOB_STATUS_PLACING: DEPLOYMENT_STATUS_INITIALIZING,
    JOB_STATUS_INITIALIZING: DEPLOYMENT_STATUS_INITIALIZING,
    JOB_STATUS_RUNNING: DEPLOYMENT_STATUS_READY,
    JOB_STATUS_FAILED: DEPLOYMENT_STATUS_FAILED,
    JOB_STATUS_CANCELLING: DEPLOYMENT_STATUS_DROPPING,
    JOB_STATUS_CANCELLED: DEPLOYMENT_STATUS_DROPPED,
    JOB_STATUS_DONE: DEPLOYMENT_STATUS_DROPPED,
    JOB_STATUS_TERMINATED: DEPLOYMENT_STATUS_TERMINATED,
}


def _strict_object(
    value: Any,
    *,
    location: str,
    required: set[str],
    allowed: set[str] | None = None,
) -> dict[str, Any]:
    """Require ``value`` to be an object with every ``required`` key.

    Keys outside ``allowed`` (default: ``required``) are rejected.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    allowed = allowed or required
    # Report unknown keys alongside missing ones, so a typo such as
    # "min_replica" names the mistyped key, not only the one it replaced.
    problems = []
    unknown = sorted(value.keys() - allowed)
    if unknown:
        problems.append(f"contains unsupported field(s): {', '.join(unknown)}")
    missing = sorted(required - value.keys())
    if missing:
        problems.append(f"is missing required field(s): {', '.join(missing)}")
    if problems:
        raise ValueError(f"{location} {' and '.join(problems)}")
    return value


@dataclass(frozen=True)
class ModelDeploymentSpec:
    """Validated evaluation-deployment specification.

    The public JSON shape is intentionally narrower than CreateJob. It cannot
    carry stage credentials, admission settings, arbitrary vLLM options, or a
    customer-selected job identifier.
    """

    # None for a checkpoint source until create reads it from the source job.
    model_name: str | None
    min_replicas: int
    gpus_per_replica: int
    max_seq_len: int
    source_type: str
    gpu_type: str = "H200"
    source_job_id: str | None = None
    checkpoint_id: str | None = None
    deployment_class: str = MODEL_DEPLOYMENT_CLASS

    @classmethod
    def from_dict(cls, value: Any) -> "ModelDeploymentSpec":
        """Validate a deployment JSON spec and return the parsed ``ModelDeploymentSpec``."""
        if not isinstance(value, dict):
            raise ValueError("deployment specification must be an object")
        spec = _strict_object(
            value,
            location="deployment specification",
            required={SPEC_DEPLOYMENT, SPEC_SOURCE, SPEC_INFERENCE},
        )
        deployment = _strict_object(
            spec[SPEC_DEPLOYMENT],
            location=SPEC_DEPLOYMENT,
            required={
                DEPLOYMENT_CLASS,
                DEPLOYMENT_MIN_REPLICAS,
                DEPLOYMENT_GPUS_PER_REPLICA,
                DEPLOYMENT_GPU_TYPE,
            },
        )
        deployment_class = deployment[DEPLOYMENT_CLASS]
        if deployment_class != MODEL_DEPLOYMENT_CLASS:
            raise ValueError(
                f'{SPEC_DEPLOYMENT}.{DEPLOYMENT_CLASS} must be "{MODEL_DEPLOYMENT_CLASS}"; '
                f"only {MODEL_DEPLOYMENT_CLASS} is supported today"
            )

        source = _strict_object(
            spec[SPEC_SOURCE],
            location=SPEC_SOURCE,
            required={SOURCE_TYPE},
            allowed={
                SOURCE_TYPE,
                SOURCE_MODEL,
                SOURCE_JOB_ID,
                SOURCE_CHECKPOINT_ID,
            },
        )
        source_type = source[SOURCE_TYPE]
        if source_type == SOURCE_TYPE_BASE_MODEL:
            _strict_object(
                source,
                location=SPEC_SOURCE,
                required={SOURCE_TYPE, SOURCE_MODEL},
            )
            string_fields = {SOURCE_MODEL: source[SOURCE_MODEL]}
        elif source_type == SOURCE_TYPE_CHECKPOINT:
            _strict_object(
                source,
                location=SPEC_SOURCE,
                required={SOURCE_TYPE, SOURCE_JOB_ID, SOURCE_CHECKPOINT_ID},
            )
            string_fields = {
                key: source[key] for key in (SOURCE_JOB_ID, SOURCE_CHECKPOINT_ID)
            }
        else:
            raise ValueError(
                f"{SPEC_SOURCE}.{SOURCE_TYPE} must be either "
                f"'{SOURCE_TYPE_BASE_MODEL}' or '{SOURCE_TYPE_CHECKPOINT}'"
            )
        # Every source field must be a non-empty string; the error names the field.
        for key, field_value in string_fields.items():
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{SPEC_SOURCE}.{key} must be a non-empty string")
        # A checkpoint's model is read from its source job at create time.
        model_name = string_fields.get(SOURCE_MODEL)
        source_job_id = string_fields.get(SOURCE_JOB_ID)
        checkpoint_id = string_fields.get(SOURCE_CHECKPOINT_ID)
        # Which models can be deployed (and how many GPUs each needs) is left to
        # the server, as for ordinary sampling jobs, so new models need no
        # client release.

        raw_min_replicas = deployment[DEPLOYMENT_MIN_REPLICAS]
        raw_gpus_per_replica = deployment[DEPLOYMENT_GPUS_PER_REPLICA]
        raw_gpu_type = deployment[DEPLOYMENT_GPU_TYPE]
        min_replicas_location = f"{SPEC_DEPLOYMENT}.{DEPLOYMENT_MIN_REPLICAS}"
        gpus_per_replica_location = f"{SPEC_DEPLOYMENT}.{DEPLOYMENT_GPUS_PER_REPLICA}"
        gpu_type_location = f"{SPEC_DEPLOYMENT}.{DEPLOYMENT_GPU_TYPE}"
        inference = _strict_object(
            spec[SPEC_INFERENCE],
            location=SPEC_INFERENCE,
            required={INFERENCE_MAX_SEQ_LEN},
        )
        max_seq_len_location = f"{SPEC_INFERENCE}.{INFERENCE_MAX_SEQ_LEN}"
        # Counts must be positive integers. bool is excluded because Python treats
        # True/False as 1/0.
        for location, field_value in (
            (min_replicas_location, raw_min_replicas),
            (gpus_per_replica_location, raw_gpus_per_replica),
            (max_seq_len_location, inference[INFERENCE_MAX_SEQ_LEN]),
        ):
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value <= 0
            ):
                raise ValueError(f"{location} must be a positive integer")
        min_replicas = raw_min_replicas
        gpus_per_replica = raw_gpus_per_replica
        max_seq_len = inference[INFERENCE_MAX_SEQ_LEN]
        if not isinstance(raw_gpu_type, str) or not raw_gpu_type.strip():
            raise ValueError(f"{gpu_type_location} must be a non-empty string")
        # Case-insensitive ("h200" works); the error names the spec field.
        gpu_type = raw_gpu_type.strip().upper()
        if gpu_type not in MODEL_DEPLOYMENT_GPU_TYPES:
            raise ValueError(
                f"{gpu_type_location} must be one of "
                f"{', '.join(MODEL_DEPLOYMENT_GPU_TYPES)}, got {raw_gpu_type!r}"
            )
        if max_seq_len > 2_147_483_647:
            raise ValueError(f"{max_seq_len_location} must be at most 2147483647")
        # Same hardware-placement rules as the server, checked here so a bad GPU
        # shape fails before anything is sent.
        if gpus_per_replica not in MODEL_DEPLOYMENT_GPU_WIDTHS:
            raise ValueError(f"{gpus_per_replica_location} must be one of 1, 2, 4, or 8")
        total_gpus = min_replicas * gpus_per_replica
        if total_gpus > 8 and total_gpus % 8 != 0:
            raise ValueError(
                f"{min_replicas_location} * {gpus_per_replica_location} must be at most 8 or "
                "a positive multiple of 8"
            )

        return cls(
            model_name=model_name,
            min_replicas=min_replicas,
            gpus_per_replica=gpus_per_replica,
            max_seq_len=max_seq_len,
            source_type=source_type,
            gpu_type=gpu_type,
            source_job_id=source_job_id,
            checkpoint_id=checkpoint_id,
            deployment_class=deployment_class,
        )

    @property
    def total_gpus(self) -> int:
        """Total GPUs across all replicas (``min_replicas * gpus_per_replica``)."""
        return self.min_replicas * self.gpus_per_replica

    def source_to_wire(self) -> dict[str, Any]:
        """Return the spec's model source in its public JSON form.

        Values pass through unchanged, so a directly constructed spec with an
        unknown ``source_type`` or a missing checkpoint ID still fails
        ``from_dict`` instead of being coerced into a valid-looking source.
        """
        if self.source_type == SOURCE_TYPE_CHECKPOINT:
            return {
                SOURCE_TYPE: SOURCE_TYPE_CHECKPOINT,
                SOURCE_JOB_ID: self.source_job_id,
                SOURCE_CHECKPOINT_ID: self.checkpoint_id,
            }
        return {SOURCE_TYPE: self.source_type, SOURCE_MODEL: self.model_name}

    def to_public_spec(self) -> dict[str, Any]:
        """Return the spec as deployment JSON, in the same shape ``from_dict`` accepts."""
        return {
            SPEC_DEPLOYMENT: {
                DEPLOYMENT_CLASS: self.deployment_class,
                DEPLOYMENT_MIN_REPLICAS: self.min_replicas,
                DEPLOYMENT_GPUS_PER_REPLICA: self.gpus_per_replica,
                DEPLOYMENT_GPU_TYPE: self.gpu_type,
            },
            SPEC_SOURCE: self.source_to_wire(),
            SPEC_INFERENCE: {INFERENCE_MAX_SEQ_LEN: self.max_seq_len},
        }

    def _to_create_job_body(self) -> dict[str, Any]:
        """Build the single-sampling-job CreateJob request body for a deployment."""
        # Imported here: the client imports this module.
        from cortex_training.client import SubJobConfig

        if not self.model_name:
            raise ValueError(
                "model_name is required; for a checkpoint source, create_model_deployment "
                "reads it from the source job"
            )
        sub_job = SubJobConfig.sampling_job(
            model_name=self.model_name,
            max_seq_len=self.max_seq_len,
            n_gpus=self.total_gpus,
            extra_sampling={
                WIRE_DEPLOYMENT_CLASS: self.deployment_class,
                WIRE_NUM_REPLICAS: self.min_replicas,
                WIRE_VLLM_CONFIG: {
                    WIRE_TENSOR_PARALLEL_SIZE: self.gpus_per_replica,
                    WIRE_MAX_MODEL_LEN: self.max_seq_len,
                },
            },
            source_checkpoint_info=(
                {
                    WIRE_SOURCE_JOB_ID: self.source_job_id,
                    WIRE_CHECKPOINT_ID: self.checkpoint_id,
                }
                if self.source_type == SOURCE_TYPE_CHECKPOINT
                else None
            ),
        )
        sub_job.validate()
        return {
            "idle_timeout_seconds": 0,
            JOB_FIELD_HARDWARE: self.gpu_type,
            "sub_job_configs": [sub_job.to_wire()],
        }


def _model_deployment_sub_job(job: Any) -> dict[str, Any] | None:
    """Return a job's sub-job if the job is an ``EVALUATION`` deployment, else ``None``."""
    from cortex_training.client.core import JobType
    from cortex_training.client.core import _normalized_job_type

    if not isinstance(job, dict):
        return None
    sub_jobs = job.get("sub_jobs")
    if not isinstance(sub_jobs, list) or len(sub_jobs) != 1:
        return None
    sub_job = sub_jobs[0]
    if not isinstance(sub_job, dict):
        return None
    if _normalized_job_type(sub_job.get("job_type")) != JobType.SAMPLING.value:
        return None
    inference_config = sub_job.get(WIRE_INFERENCE_CONFIG)
    if not isinstance(inference_config, dict):
        return None
    if inference_config.get(WIRE_DEPLOYMENT_CLASS) != MODEL_DEPLOYMENT_CLASS:
        return None
    return sub_job


def _model_deployment_status(job: Mapping[str, Any]) -> str:
    """Derive the customer-facing deployment status from one job response."""
    from cortex_training.client.core import CortexTrainingClient

    status = CortexTrainingClient._normalize_job_status(job.get(JOB_FIELD_STATUS))
    if status in JOB_STATUS_TO_DEPLOYMENT_STATUS:
        return JOB_STATUS_TO_DEPLOYMENT_STATUS[status]
    # A state this client doesn't know yet is shown as-is rather than guessed.
    return status.upper() if status else DEPLOYMENT_STATUS_UNKNOWN


def _deployment_int(value: Any, *, location: str) -> int:
    """Read a positive whole number from a job response, which may arrive as a float."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or int(value) != value
        or value <= 0
    ):
        raise ValueError(f"invalid marked deployment: {location} must be a positive integer")
    return int(value)


def _model_deployment_metadata(
    job: Mapping[str, Any],
    deployment_id: str,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Render one marked job without exposing its backing job or sub-job IDs."""
    from cortex_training.client.core import _hardware_wire

    sub_job = _model_deployment_sub_job(job)
    if sub_job is None:
        raise ValueError("the requested resource is not a model deployment")
    inference_config = sub_job[WIRE_INFERENCE_CONFIG]
    vllm_config = inference_config.get(WIRE_VLLM_CONFIG)
    if not isinstance(vllm_config, dict):
        raise ValueError(
            f"invalid marked deployment: {WIRE_INFERENCE_CONFIG}.{WIRE_VLLM_CONFIG} is required"
        )

    # Every count the server stored for the deployment must be a positive whole
    # number. The error location is built from the key so the two cannot drift
    # apart.
    vllm_location = f"{WIRE_INFERENCE_CONFIG}.{WIRE_VLLM_CONFIG}"
    replicas, total_gpus, max_seq_len, gpus_per_replica, max_model_len = (
        _deployment_int(container.get(key), location=f"{location}.{key}")
        for container, location, key in (
            (inference_config, WIRE_INFERENCE_CONFIG, WIRE_NUM_REPLICAS),
            (inference_config, WIRE_INFERENCE_CONFIG, WIRE_N_GPUS),
            (inference_config, WIRE_INFERENCE_CONFIG, WIRE_MAX_SEQ_LEN),
            (vllm_config, vllm_location, WIRE_TENSOR_PARALLEL_SIZE),
            (vllm_config, vllm_location, WIRE_MAX_MODEL_LEN),
        )
    )
    if total_gpus != replicas * gpus_per_replica:
        raise ValueError("invalid marked deployment: inconsistent replica GPU totals")
    if max_model_len != max_seq_len:
        raise ValueError("invalid marked deployment: inconsistent sequence lengths")

    # The server already admitted this deployment, so admission rules (supported
    # models, GPU minimums and shapes) are not re-checked here: a model or shape
    # this client does not know about must still be listed, not fail the read.
    model_name = sub_job.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("marked deployment model_name must be a non-empty string")
    checkpoint = sub_job.get(WIRE_SOURCE_CHECKPOINT_INFO)
    if checkpoint is None:
        model_source = {SOURCE_TYPE: SOURCE_TYPE_BASE_MODEL, SOURCE_MODEL: model_name}
    elif isinstance(checkpoint, dict):
        # A checkpoint source needs both IDs as non-empty strings.
        for key in (WIRE_SOURCE_JOB_ID, WIRE_CHECKPOINT_ID):
            value = checkpoint.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{WIRE_SOURCE_CHECKPOINT_INFO}.{key} must be a non-empty string")
        model_source = {
            SOURCE_TYPE: SOURCE_TYPE_CHECKPOINT,
            SOURCE_JOB_ID: checkpoint[WIRE_SOURCE_JOB_ID],
            SOURCE_CHECKPOINT_ID: checkpoint[WIRE_CHECKPOINT_ID],
        }
    else:
        raise ValueError("invalid marked deployment: source_checkpoint_info must be an object")

    # None means unknown. A job read while the server cannot reach its
    # scheduler has no hardware, and defaulting to H200 would misreport a B300
    # deployment.
    hardware = job.get(JOB_FIELD_HARDWARE)
    result: dict[str, Any] = {
        OUT_DEPLOYMENT: deployment_id,
        OUT_DEPLOYMENT_CLASS: inference_config[WIRE_DEPLOYMENT_CLASS],
        OUT_STATUS: _model_deployment_status(job),
        OUT_MODEL_SOURCE: model_source,
        OUT_MAX_SEQ_LEN: max_seq_len,
        OUT_GPUS_PER_REPLICA: gpus_per_replica,
        OUT_REPLICAS: replicas,
        OUT_GPU_TYPE: _hardware_wire(hardware) if hardware else None,
    }
    # Copy optional job fields to the output.
    for source_key, output_key in (
        (JOB_FIELD_CREATED_AT, OUT_CREATED_AT),
        (JOB_FIELD_OWNER_ROLE, OUT_OWNER_ROLE),
        (JOB_FIELD_SUBMITTED_BY, OUT_CREATOR),
    ):
        value = job.get(source_key)
        if isinstance(value, str) and value:
            result[output_key] = value
    deployment_status = result[OUT_STATUS]
    reason = job.get(JOB_FIELD_REASON)
    reason = reason if isinstance(reason, str) and reason else None
    reason_details = job.get(JOB_FIELD_REASON_DETAILS)
    if isinstance(reason_details, str) and reason_details:
        # The sentence may quote the backing job or sub-job ID. The sub-job ID
        # and the deployment ID both contain the job ID, so match the longest
        # ID first and rewrite each match to the deployment ID in a single
        # pass; an existing deployment ID is left as it is.
        ids = sorted(
            {
                value
                for value in (
                    deployment_id,
                    sub_job.get("sub_job_id"),
                    job.get(JOB_FIELD_JOB_ID),
                )
                if isinstance(value, str) and value
            },
            key=len,
            reverse=True,
        )
        reason_details = re.sub(
            "|".join(map(re.escape, ids)),
            deployment_id,
            reason_details,
            flags=re.IGNORECASE,
        )
    else:
        reason_details = None
    if deployment_status in {DEPLOYMENT_STATUS_FAILED, DEPLOYMENT_STATUS_TERMINATED}:
        failure: dict[str, str] = {}
        if reason:
            failure[OUT_FAILURE_CODE] = reason
        if reason_details:
            failure[OUT_FAILURE_MESSAGE] = reason_details
        if failure:
            result[OUT_FAILURE] = failure
    elif not (
        deployment_status in {DEPLOYMENT_STATUS_DROPPING, DEPLOYMENT_STATUS_DROPPED}
        and reason == JOB_REASON_CANCELLED_BY_CLIENT
    ):
        if reason:
            result[OUT_STATUS_REASON] = reason
        if reason_details:
            result[OUT_STATUS_DETAILS] = reason_details
    # A deployment that has failed or is being dropped no longer serves.
    if base_url is not None and deployment_status not in DEPLOYMENT_STATUSES_WITHOUT_ENDPOINT:
        result[OUT_BASE_URL] = base_url
    return result


def _model_deployment_status_filter(
    status: str | Iterable[str] | None,
) -> frozenset[str] | None:
    """Normalize a ``list`` status filter, rejecting unknown statuses."""
    if status is None:
        return None
    values = [status] if isinstance(status, str) else list(status)
    wanted = frozenset(str(value).strip().upper() for value in values)
    unknown = sorted(wanted - set(DEPLOYMENT_STATUSES))
    if unknown:
        raise ValueError(
            f"unknown deployment status {', '.join(unknown)}; choose from "
            f"{', '.join(DEPLOYMENT_STATUSES)}"
        )
    return wanted


def _deployment_id_from_job_id(job_id: Any) -> str:
    """Turn a canonical job UUID into the public ``sf-ct-<UUID>`` deployment ID."""
    if not isinstance(job_id, str):
        raise ValueError("create response did not include a valid deployment identifier")
    try:
        parsed = uuid.UUID(job_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "create response did not include a valid deployment identifier"
        ) from exc
    canonical = str(parsed)
    if job_id != canonical:
        raise ValueError("create response deployment identifier was not canonical")
    return MODEL_DEPLOYMENT_ID_PREFIX + canonical


# Deployments are addressed by ID today; names may be accepted later.
_DEPLOYMENT_FORMAT_ERROR = "deployment must currently be an sf-ct-<UUID> ID"


def _job_id_from_deployment_id(deployment_id: Any) -> str:
    """Turn a public ``sf-ct-<UUID>`` deployment ID back into its backing job UUID."""
    if not isinstance(deployment_id, str) or not deployment_id.startswith(
        MODEL_DEPLOYMENT_ID_PREFIX
    ):
        raise ValueError(_DEPLOYMENT_FORMAT_ERROR)
    suffix = deployment_id[len(MODEL_DEPLOYMENT_ID_PREFIX) :]
    try:
        parsed = uuid.UUID(suffix)
    except (ValueError, AttributeError) as exc:
        raise ValueError(_DEPLOYMENT_FORMAT_ERROR) from exc
    canonical = str(parsed)
    if suffix != canonical:
        raise ValueError(f"{_DEPLOYMENT_FORMAT_ERROR} with a canonical lowercase UUID")
    return canonical


def _pending_model_deployment_metadata(
    spec: ModelDeploymentSpec,
    deployment_id: str,
    base_url: str,
) -> dict[str, Any]:
    """Build the metadata ``create`` returns before the deployment is placed."""
    return {
        OUT_DEPLOYMENT: deployment_id,
        OUT_DEPLOYMENT_CLASS: spec.deployment_class,
        OUT_STATUS: DEPLOYMENT_STATUS_PENDING,
        OUT_MODEL_SOURCE: spec.source_to_wire(),
        OUT_MAX_SEQ_LEN: spec.max_seq_len,
        OUT_GPUS_PER_REPLICA: spec.gpus_per_replica,
        OUT_REPLICAS: spec.min_replicas,
        OUT_GPU_TYPE: spec.gpu_type,
        OUT_BASE_URL: base_url,
    }


def _model_deployment_description(
    metadata: Mapping[str, Any],
    *,
    database: str | None,
    schema: str | None,
) -> dict[str, Any]:
    """Reshape deployment metadata into what ``describe`` prints.

    The deployment's configuration is grouped under ``spec`` for readability.
    It is a read view: it may gain read-only fields, and it is not guaranteed
    to be accepted by ``create``. Status, endpoint, scope, and ownership stay
    at the top level.
    """
    result: dict[str, Any] = {
        key: metadata[key]
        for key in (
            OUT_DEPLOYMENT,
            OUT_STATUS,
            OUT_STATUS_REASON,
            OUT_STATUS_DETAILS,
            OUT_FAILURE,
            OUT_BASE_URL,
        )
        if key in metadata
    }
    for key, value in ((OUT_DATABASE, database), (OUT_SCHEMA, schema)):
        if value:
            result[key] = value
    for key in (OUT_CREATED_AT, OUT_CREATOR, OUT_OWNER_ROLE):
        if key in metadata:
            result[key] = metadata[key]
    result[OUT_SPEC] = {
        SPEC_DEPLOYMENT: {
            DEPLOYMENT_CLASS: metadata[OUT_DEPLOYMENT_CLASS],
            DEPLOYMENT_MIN_REPLICAS: metadata[OUT_REPLICAS],
            DEPLOYMENT_GPUS_PER_REPLICA: metadata[OUT_GPUS_PER_REPLICA],
            # None means the server did not report the hardware.
            DEPLOYMENT_GPU_TYPE: metadata[OUT_GPU_TYPE],
        },
        SPEC_SOURCE: dict(metadata[OUT_MODEL_SOURCE]),
        SPEC_INFERENCE: {INFERENCE_MAX_SEQ_LEN: metadata[OUT_MAX_SEQ_LEN]},
    }
    return result


# Longest server error message kept in a model deployment error.
_MAX_SERVER_ERROR_MESSAGE_CHARS = 1000


def _model_deployment_request_error(
    deployment_id: str | None,
    *,
    status_code: int | None,
    code: str | None,
    detail: str | None,
    request_id: str | None,
    job_id: str | None = None,
    scope: str | None = None,
) -> RuntimeError:
    """Return a public error that cannot expose backing job identifiers.

    Built from parts of a failed request, so no request URL or raw body can
    reach it, but the server's error ``code`` and ``message`` (``detail``) are
    kept so "feature disabled", "not authorized" and "wrong ID" stay
    distinguishable. ``job_id`` is replaced by ``deployment_id`` in the
    message, since the server's own messages can name the backing job.
    ``deployment_id`` is ``None`` for ``create``, which has no deployment yet.
    """
    from cortex_training.client.core import _JOB_NOT_FOUND_CODE

    subject = f"model deployment {deployment_id}" if deployment_id else "model deployment"
    if code == _JOB_NOT_FOUND_CODE and deployment_id:
        message = f"{subject} not found" + (f" in {scope}" if scope else "")
    else:
        message = f"{subject} request failed"
        if status_code is not None:
            message += f" with HTTP {status_code}"
        if code:
            message += f" ({code})"
        if detail:
            if job_id and deployment_id:
                detail = detail.replace(job_id, deployment_id)
            if len(detail) > _MAX_SERVER_ERROR_MESSAGE_CHARS:
                detail = detail[:_MAX_SERVER_ERROR_MESSAGE_CHARS] + "..."
            message += f": {detail}"
    if request_id:
        message += f" (snowflake request id: {request_id})"
    return RuntimeError(message)


class ModelDeploymentWaitError(RuntimeError):
    """A deployment reached a terminal status other than ``READY``.

    ``deployment`` is the metadata from the last poll, or ``None``.
    """

    def __init__(self, message: str, deployment: dict[str, Any] | None):
        super().__init__(message)
        self.deployment = deployment


class ModelDeploymentWaitTimeout(TimeoutError):
    """A deployment did not become ``READY`` before the poll timeout.

    The deployment is still running. ``deployment`` is the metadata from the
    last poll, or ``None``.
    """

    def __init__(self, message: str, deployment: dict[str, Any] | None):
        super().__init__(message)
        self.deployment = deployment


def _model_deployment_ready(metadata: dict[str, Any]) -> bool:
    """Return whether a polled deployment is READY.

    Raises :class:`ModelDeploymentWaitError` for a terminal status other than READY.
    """
    deployment_id = metadata[OUT_DEPLOYMENT]
    deployment_status = metadata[OUT_STATUS]
    if deployment_status == DEPLOYMENT_STATUS_READY:
        return True
    if deployment_status == DEPLOYMENT_STATUS_DROPPED:
        raise ModelDeploymentWaitError(
            f"model deployment {deployment_id} was dropped; create a new "
            "deployment to serve this model again",
            metadata,
        )
    if deployment_status in {DEPLOYMENT_STATUS_FAILED, DEPLOYMENT_STATUS_TERMINATED}:
        failure = metadata.get(OUT_FAILURE, {})
        message = f"model deployment {deployment_id} is {deployment_status}"
        if failure.get(OUT_FAILURE_CODE):
            message += f" ({failure[OUT_FAILURE_CODE]})"
        if failure.get(OUT_FAILURE_MESSAGE):
            message += f": {failure[OUT_FAILURE_MESSAGE]}"
        message += (
            "; for details run `cortex-training model-deployment download-log "
            f"{deployment_id}`"
        )
        raise ModelDeploymentWaitError(message, metadata)
    return False


def _model_deployment_wait_timeout(
    deployment_id: str, poll_timeout: float, metadata: dict[str, Any] | None
) -> ModelDeploymentWaitTimeout:
    last_status = metadata[OUT_STATUS] if metadata else DEPLOYMENT_STATUS_UNKNOWN
    return ModelDeploymentWaitTimeout(
        f"model deployment {deployment_id} did not become READY within "
        f"{poll_timeout:g}s (last status: {last_status}). It is still "
        "running and holding GPUs: run `cortex-training model-deployment wait "
        f"{deployment_id}` again, or `cortex-training model-deployment drop "
        f"{deployment_id}` to release them",
        metadata,
    )


def _trained_model_name(job: Mapping[str, Any], job_id: str) -> str:
    """Return the model a checkpoint's source job trained.

    The server requires ``model_name`` on the deployment's sampling sub-job
    and rejects one that differs from the source job's, so it is read from
    the source job rather than asked of the user.
    """
    from cortex_training.client.core import JobType
    from cortex_training.client.core import _normalized_job_type

    for sub_job in job.get("sub_jobs") or ():
        if _normalized_job_type(sub_job.get("job_type")) == JobType.TRAINING.value:
            model_name = sub_job.get("model_name")
            if isinstance(model_name, str) and model_name.strip():
                return model_name
    raise ValueError(
        f"{SPEC_SOURCE}.{SOURCE_JOB_ID} {job_id} has no training sub-job model_name"
    )
