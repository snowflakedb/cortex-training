# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A DFlash2 draft for a fine-tune of the catalog model: naming the fine-tune, and the jobs each step submits."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import chz
import pytest

from recipes.dflash2 import evaluate
from recipes.dflash2 import train
from recipes.dflash2.target import FineTunedTarget
from recipes.dflash2.target import fine_tuned_target
from recipes.dflash2.target import has_file

BASE = "Qwen/Qwen3.5-4B"
WEIGHTS_ONLY = ["config.json", "model.safetensors", "tokenizer.json"]
LORA_WEIGHTS_ONLY = WEIGHTS_ONLY + ["default/adapter_config.json", "default/adapter_model.safetensors"]


class FakeClient:
    def __init__(self, files=WEIGHTS_ONLY, trained=BASE):
        self.files, self.trained = list(files), trained

    def get_job(self, job_id):
        return {"job_id": job_id, "sub_jobs": [{"job_type": "training", "model_name": self.trained}]}

    def export_checkpoint(self, job_id, checkpoint_id):
        return {"files": [{"filename": name, "url": f"https://presigned.example/{name}"} for name in self.files]}


def _target(client=None, **names):
    return fine_tuned_target(client or FakeClient(), BASE, **names)


# ---- naming the fine-tune --------------------------------------------------------------------------------------


def test_no_fine_tune_is_the_catalog_model():
    assert _target() is None


def test_a_saved_fine_tune_is_its_weights_only_checkpoint():
    assert _target(job_id="sft-job", checkpoint_id="cp_1") == FineTunedTarget(
        {"checkpoint_id": "cp_1", "source_job_id": "sft-job"}, lora=False
    )


def test_a_lora_fine_tune_is_recognized_by_its_adapter_pack():
    assert _target(FakeClient(LORA_WEIGHTS_ONLY), job_id="sft-job", checkpoint_id="cp_1").lora


def test_an_uploaded_fine_tune_is_its_stage_path():
    stage = "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft"

    assert _target(stage_path=stage) == FineTunedTarget({"external_stage_path": stage})


@pytest.mark.parametrize("names", [dict(job_id="sft-job"), dict(checkpoint_id="cp_1")])
def test_a_saved_fine_tune_needs_both_its_job_and_its_checkpoint(names):
    with pytest.raises(ValueError, match="together"):
        _target(**names)


def test_a_fine_tune_is_named_one_way():
    with pytest.raises(ValueError, match="or by target_stage_path"):
        _target(job_id="sft-job", checkpoint_id="cp_1", stage_path="@S/x")


def test_a_fine_tune_of_another_model_is_refused_before_any_job():
    with pytest.raises(ValueError, match="fine-tuned 'Qwen/Qwen3-8B', not model='Qwen/Qwen3.5-4B'"):
        _target(FakeClient(trained="Qwen/Qwen3-8B"), job_id="sft-job", checkpoint_id="cp_1")


def test_a_resumable_checkpoint_is_not_a_target():
    with pytest.raises(ValueError, match="not a weights-only checkpoint"):
        _target(FakeClient(["global_step3/mp_rank_00_model_states.pt"]), job_id="sft-job", checkpoint_id="cp_1")


# ---- the training job ------------------------------------------------------------------------------------------


def _config(**overrides) -> train.Config:
    return chz.Blueprint(train.Config).apply({"data_path": "data.jsonl", **overrides}).make()


def _training(body: dict) -> dict:
    return train.training_sub_job(body)


def test_without_a_fine_tune_the_training_job_has_no_source():
    assert "source_checkpoint_info" not in _training(train.job_body(_config()))


def test_a_saved_fine_tune_is_the_training_jobs_frozen_target():
    sub = _training(train.job_body(_config(), _target(job_id="sft-job", checkpoint_id="cp_1")))

    assert sub["source_checkpoint_info"] == {"checkpoint_id": "cp_1", "source_job_id": "sft-job"}
    assert sub["training_config"]["dflash2"]["target_from_source_checkpoint"] is True
    # The target layer adds to the draft's settings rather than replacing them.
    assert sub["training_config"]["dflash2"]["draft_config"]["architectures"] == ["DFlash2DraftModel"]
    assert sub["model_name"] == BASE


def test_an_uploaded_fine_tune_is_imported_without_the_flag():
    sub = _training(train.job_body(_config(), _target(stage_path="@S/qwen-sft")))

    assert sub["source_checkpoint_info"] == {"external_stage_path": "@S/qwen-sft"}
    assert "target_from_source_checkpoint" not in sub["training_config"]["dflash2"]


def test_a_job_config_file_still_trains_against_the_fine_tune(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(train.job_body(_config())))

    sub = _training(train.job_body(_config(job_config=str(path)), _target(job_id="sft-job", checkpoint_id="cp_1")))

    assert sub["source_checkpoint_info"]["checkpoint_id"] == "cp_1"
    assert sub["training_config"]["dflash2"]["target_from_source_checkpoint"] is True


# ---- the endpoint that serves the draft ------------------------------------------------------------------------


def _sampling(body: dict) -> dict:
    return next(sub for sub in body["sub_job_configs"] if sub["job_type"] == "sampling")


def _endpoint(**kwargs) -> dict:
    return evaluate.endpoint_body(
        model=BASE, draft_job_id="draft-job", draft_checkpoint_id="cp_9", num_speculative_tokens=15, **kwargs
    )


def test_an_endpoint_of_a_draft_with_a_lora_target_serves_its_adapter():
    sampling = _sampling(_endpoint(lora=True))

    assert sampling["inference_config"]["peft_config"]["peft_type"] == "Lora"
    assert sampling["inference_config"]["vllm_config"]["speculative_config"]["from_source_checkpoint"] is True
    assert sampling["source_checkpoint_info"] == {"checkpoint_id": "cp_9", "source_job_id": "draft-job"}


def test_an_endpoint_of_any_other_draft_serves_full_weights():
    assert "peft_config" not in _sampling(_endpoint())["inference_config"]


def test_the_block_size_is_the_drafts_own_when_it_carries_a_target(monkeypatch):
    configs = {
        "https://presigned.example/config.json": {"dflash_config": {"block_size": 16}},
        "https://presigned.example/target/config.json": {"model_type": "qwen3_5"},
    }
    monkeypatch.setattr(
        evaluate.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(json.dumps(configs[url]).encode())
    )
    files = FakeClient(["config.json", "model.safetensors", "target/config.json"]).export_checkpoint("j", "c")["files"]

    assert evaluate.draft_block_size(files, "cp_9") == 16


def test_the_export_lists_a_checkpoints_files_under_its_tag_folder(monkeypatch):
    # What the service returns: every file of the checkpoint sits under global_step<N>/.
    names = [
        "global_step2/config.json",
        "global_step2/model.safetensors",
        "global_step2/target/config.json",
        "global_step2/target/default/adapter_config.json",
        "metadata.json",
    ]
    configs = {
        "https://presigned.example/global_step2/config.json": {"dflash_config": {"block_size": 16}},
        "https://presigned.example/global_step2/target/config.json": {"model_type": "qwen3_5"},
    }
    monkeypatch.setattr(
        evaluate.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(json.dumps(configs[url]).encode())
    )
    files = FakeClient(names).export_checkpoint("j", "c")["files"]

    assert evaluate.draft_block_size(files, "cp_9") == 16
    assert has_file(files, "target/config.json")
    assert has_file(files, "target/default/adapter_config.json")
    assert not has_file(files, "default/adapter_config.json")


def test_a_tag_folder_checkpoint_of_a_lora_fine_tune_is_recognized():
    names = ["global_step5/config.json", "global_step5/default/adapter_config.json", "metadata.json"]

    assert _target(FakeClient(names), job_id="sft-job", checkpoint_id="cp_1").lora


def test_the_endpoint_detects_a_draft_that_carries_a_lora_target(monkeypatch, tmp_path):
    client = FakeClient(["config.json", "model.safetensors", "target/config.json", "target/default/adapter_config.json"])
    seen = {}
    monkeypatch.setattr(evaluate, "draft_block_size", lambda files, checkpoint_id: 16)
    monkeypatch.setattr(evaluate, "endpoint_body", lambda **kwargs: seen.update(kwargs) or {"sub_job_configs": []})
    monkeypatch.setattr(evaluate, "running_job", _stop_before_serving)

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("")

    with pytest.raises(_Stopped):
        evaluate.evaluate_draft(client, model=BASE, prompts=str(prompts), draft_job_id="d", draft_checkpoint_id="c")

    assert seen["lora"] is True
    assert seen["num_speculative_tokens"] == 15


class _Stopped(Exception):
    pass


def _stop_before_serving(*args, **kwargs):
    raise _Stopped


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    """evaluate_draft loads the tokenizer before it starts the endpoint; keep the tests off the network."""
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(apply_chat_template=None)
    )
