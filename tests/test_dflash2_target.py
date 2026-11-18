# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Naming the fine-tune a DFlash2 draft is trained for, and reading the checkpoints that carry it."""

from __future__ import annotations

import io
import json

import pytest

from recipes.dflash2 import evaluate
from recipes.dflash2.target import FineTunedTarget
from recipes.dflash2.target import find_files
from recipes.dflash2.target import fine_tuned_target
from recipes.dflash2.target import has_file

BASE = "Qwen/Qwen3.5-4B"

# The export of a checkpoint lists its files flat, or all under one tag folder with the job's metadata beside it.
layouts = pytest.mark.parametrize("folder", ["", "global_step5/"], ids=["flat", "tag folder"])


def export(folder: str, *names: str) -> list[str]:
    return [folder + name for name in names] + (["metadata.json"] if folder else [])


class FakeClient:
    """The two calls the recipe makes: ``get_job`` of the fine-tuning job and ``export_checkpoint`` of its save."""

    def __init__(self, names: list[str], trained: str = BASE):
        self.names, self.trained = names, trained

    def get_job(self, job_id):
        return {"job_id": job_id, "sub_jobs": [{"job_type": "training", "model_name": self.trained}]}

    def export_checkpoint(self, job_id, checkpoint_id):
        return {"files": [{"filename": name, "url": f"https://presigned.example/{name}"} for name in self.names]}


def files(names: list[str]) -> list[dict]:
    return FakeClient(names).export_checkpoint("job", "cp")["files"]


def target_of(names: list[str], **kwargs):
    return fine_tuned_target(FakeClient(names, **kwargs), BASE, job_id="sft-job", checkpoint_id="cp_1")


# ---- the fine-tune a draft is trained for ----------------------------------------------------------------------


@layouts
def test_a_full_fine_tune_is_its_checkpoint_without_an_adapter(folder):
    target = target_of(export(folder, "config.json", "model.safetensors", "tokenizer.json"))

    assert target == FineTunedTarget({"checkpoint_id": "cp_1", "source_job_id": "sft-job"}, lora=False)


@layouts
def test_a_lora_fine_tune_is_its_checkpoint_with_an_adapter(folder):
    names = export(folder, "config.json", "model.safetensors", "default/adapter_config.json")

    assert target_of(names) == FineTunedTarget({"checkpoint_id": "cp_1", "source_job_id": "sft-job"}, lora=True)


def test_a_fine_tune_of_another_model_is_refused():
    with pytest.raises(ValueError, match="fine-tuned 'Qwen/Qwen3-8B', not model='Qwen/Qwen3.5-4B'"):
        target_of(["config.json", "model.safetensors"], trained="Qwen/Qwen3-8B")


@pytest.mark.parametrize("names", [dict(job_id="sft-job"), dict(checkpoint_id="cp_1")])
def test_a_saved_fine_tune_needs_both_its_job_and_its_checkpoint(names):
    with pytest.raises(ValueError, match="together"):
        fine_tuned_target(FakeClient(["config.json"]), BASE, **names)


@pytest.mark.parametrize(
    "job",
    [dict(job_id="sft-job"), dict(checkpoint_id="cp_1"), dict(job_id="sft-job", checkpoint_id="cp_1")],
    ids=["job only", "checkpoint only", "job and checkpoint"],
)
def test_a_stage_path_is_not_combined_with_the_job_that_saved_a_checkpoint(job):
    with pytest.raises(ValueError, match="or by target_stage_path"):
        fine_tuned_target(FakeClient(["config.json"]), BASE, stage_path="@DB.SCHEMA.STAGE/qwen-sft", **job)


# ---- a checkpoint whose own config.json is missing -------------------------------------------------------------


@layouts
def test_a_checkpoint_with_only_a_targets_config_is_not_weights_only(folder):
    names = export(folder, "model.safetensors", "target/config.json", "target/model.safetensors")

    assert find_files(files(names), "config.json") == []
    with pytest.raises(ValueError, match="not a weights-only checkpoint"):
        target_of(names)


@layouts
def test_a_draft_with_only_a_targets_config_has_no_block_size_to_read(folder, monkeypatch):
    def fetched(url, timeout):
        raise AssertionError(f"read {url}")

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", fetched)
    names = export(folder, "model.safetensors", "target/config.json", "target/model.safetensors")

    assert has_file(files(names), "target/config.json")
    with pytest.raises(ValueError, match="0 config.json files at its root"):
        evaluate.draft_block_size(files(names), "cp_9")


# ---- a draft that carries a target -----------------------------------------------------------------------------


@layouts
def test_the_block_size_is_read_from_the_drafts_own_config(folder, monkeypatch):
    configs = {
        f"https://presigned.example/{folder}config.json": {"dflash_config": {"block_size": 16}},
        f"https://presigned.example/{folder}target/config.json": {"model_type": "qwen3_5"},
    }
    monkeypatch.setattr(
        evaluate.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(json.dumps(configs[url]).encode())
    )
    names = export(folder, "config.json", "model.safetensors", "target/config.json")

    assert evaluate.draft_block_size(files(names), "cp_9") == 16


@layouts
def test_a_draft_carries_a_lora_target_when_its_target_has_an_adapter_pack(folder):
    lora = files(export(folder, "config.json", "target/config.json", "target/default/adapter_config.json"))
    full = files(export(folder, "config.json", "target/config.json"))
    plain = files(export(folder, "config.json", "model.safetensors"))

    assert [has_file(f, "target/config.json") for f in (lora, full, plain)] == [True, True, False]
    assert [has_file(f, "target/default/adapter_config.json") for f in (lora, full, plain)] == [True, False, False]


@layouts
def test_an_adapter_a_draft_carries_is_not_the_drafts_own(folder):
    # Used as a target by mistake, a draft is not a LoRA fine-tune because the target inside it is one.
    names = export(folder, "config.json", "model.safetensors", "target/config.json", "target/default/adapter_config.json")

    assert target_of(names).lora is False
