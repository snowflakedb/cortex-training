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

"""Model deployment spec and constants for ``cortex-training model-deployment``.

Spec keys, job fields and statuses, deployment statuses, and the keys of the
metadata the client returns, in one place. The client imports this module, so
it must not import the client at module level. See
``docs/reference/rest-api.md`` section 14.
"""

from __future__ import annotations

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
