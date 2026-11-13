# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fine-tune of ``model`` a draft is trained for, when the target is not ``model`` itself.

A fine-tune is named by the weights-only checkpoint a Cortex Training job saved (``target_job_id`` and
``target_checkpoint_id``), or by a Hugging Face directory uploaded to a stage (``target_stage_path``). ``model``
stays the base it was fine-tuned from: it picks the catalog job files, the draft architecture and the tokenizer.

The three steps use it differently:

- the data step serves the fine-tune from its checkpoint, so the training text is the fine-tune's own;
- the training job loads it as the frozen target (a LoRA fine-tune is merged into its base first), and every
  weights-only save of the draft carries it under ``target/``;
- an endpoint serving that draft serves the target it carries, so ``evaluate.py`` needs only the draft's checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from recipes.utils import source_checkpoint_info

# The folder of a weights-only checkpoint that holds a LoRA job's adapter, and the one of a draft checkpoint that
# holds the fine-tune the draft was trained against.
ADAPTER_DIR = "default"
TARGET_DIR = "target"


@dataclass(frozen=True)
class FineTunedTarget:
    source_checkpoint_info: dict[str, str]
    # Served as its base plus the adapter, so the endpoint needs the catalog's LoRA inference file.
    lora: bool = False

    @property
    def from_checkpoint(self) -> bool:
        """Saved by a Cortex Training job, rather than uploaded to a stage."""
        return "checkpoint_id" in self.source_checkpoint_info

    def training_override(self) -> dict[str, Any]:
        """The layer that trains a draft against this fine-tune instead of ``model``."""
        training: dict[str, Any] = {"job_type": "training", "source_checkpoint_info": self.source_checkpoint_info}
        if self.from_checkpoint:
            # An imported directory is the job's model already; a checkpoint is a resume unless the job says so.
            training["training_config"] = {"dflash2": {"target_from_source_checkpoint": True}}
        return {"sub_job_configs": [training]}


def checkpoint_files(client: Any, job_id: str, checkpoint_id: str) -> list[dict[str, Any]]:
    """The files of a saved checkpoint, each with its ``filename`` relative to the checkpoint root and a ``url``."""
    return list(client.export_checkpoint(job_id, checkpoint_id).get("files") or ())


def find_files(files: Sequence[Mapping[str, Any]], filename: str) -> list[Mapping[str, Any]]:
    """The entries for ``filename`` under the checkpoint root.

    The export lists a checkpoint's files under its tag folder (``global_step5/config.json``). A file listed as
    ``filename`` itself is at the root; only when there is none does one folder down count.
    """
    exact = [f for f in files if f.get("filename") == filename]
    if exact:
        return exact

    def one_folder_down(name: str) -> bool:
        folder, _, rest = name.partition("/")
        return bool(folder) and rest == filename

    return [f for f in files if one_folder_down(str(f.get("filename", "")))]


def has_file(files: Sequence[Mapping[str, Any]], filename: str) -> bool:
    return bool(find_files(files, filename))


def trained_model(client: Any, job_id: str) -> str | None:
    """The model the training sub-job of ``job_id`` trained, if the job has one."""
    for sub_job in client.get_job(job_id).get("sub_jobs") or ():
        if str(sub_job.get("job_type") or "").lower() == "training":
            return sub_job.get("model_name")
    return None


def fine_tuned_target(
    client: Any,
    model: str,
    *,
    job_id: str | None = None,
    checkpoint_id: str | None = None,
    stage_path: str | None = None,
) -> FineTunedTarget | None:
    """The fine-tune of ``model`` these settings name, or ``None`` for ``model`` itself."""
    if stage_path:
        if job_id or checkpoint_id:
            raise ValueError("name the target by target_job_id and target_checkpoint_id, or by target_stage_path")
        return FineTunedTarget({"external_stage_path": stage_path})
    if not (job_id or checkpoint_id):
        return None
    if not (job_id and checkpoint_id):
        raise ValueError("pass target_job_id and target_checkpoint_id together")
    info = source_checkpoint_info(job_id, checkpoint_id) or {}
    base = trained_model(client, job_id)
    if base != model:
        # model picks the draft architecture and the tokenizer, so it has to be the base of the fine-tune.
        raise ValueError(f"job {job_id} fine-tuned {base!r}, not model={model!r}: pass model={base}")
    files = checkpoint_files(client, job_id, checkpoint_id)
    if not has_file(files, "config.json"):
        raise ValueError(
            f"checkpoint {checkpoint_id} of job {job_id} is not a weights-only checkpoint: it has no config.json. "
            "Save the fine-tune with checkpoint_type='weights-only'."
        )
    return FineTunedTarget(info, lora=has_file(files, f"{ADAPTER_DIR}/adapter_config.json"))
