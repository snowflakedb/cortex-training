from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import smoke_test_model_catalog as smoke_catalog  # noqa: E402
from smoke_test_model_catalog import build_profile_request  # noqa: E402
from smoke_test_model_catalog import build_forward_backward_probe_spec  # noqa: E402
from smoke_test_model_catalog import run_data_plane_probes  # noqa: E402
from validate_model_catalog import (  # noqa: E402
    CatalogValidationError,
    load_catalog,
    load_and_validate,
    validate_catalog,
)

CONFIG_DIR = REPO_ROOT / "model-catalog"


def test_checked_in_catalog_is_valid():
    summary = load_and_validate(CONFIG_DIR, REPO_ROOT)

    assert summary == {
        "schemaVersion": 1,
        "lastReviewed": "2026-09-01",
        "models": 10,
        "profiles": 12,
    }


def test_catalog_context_limits_match_supported_profiles():
    models_doc, _ = load_catalog(CONFIG_DIR)
    expected_limits = {
        "Qwen/Qwen3-0.6B": 32768,
        "Qwen/Qwen3-1.7B": 32768,
        "Qwen/Qwen3-8B": 32768,
        "Qwen/Qwen3.5-4B": 262144,
        "Qwen/Qwen3.6-35B-A3B": 262144,
        "Qwen/Qwen3.8-27B": 262144,
        "deepseek-ai/DeepSeek-V4-Flash-0731": 1048576,
        "openai/gpt-oss-120b": 131072,
        "zai-org/GLM-5.2": 32768,
        "zai-org/GLM-5.2-FP8": 1048576,
    }

    for model in models_doc["models"]:
        expected = expected_limits[model["modelId"]]
        capabilities = model["capabilities"]
        assert capabilities["inference"]["maxContextTokens"] == expected
        if capabilities["training"]["supported"]:
            assert {
                profile["maxContextTokens"]
                for profile in capabilities["training"]["profiles"].values()
            } == {expected}


@pytest.mark.parametrize(
    ("model_id", "expected_profile_id", "expected_tp"),
    [
        ("Qwen/Qwen3-0.6B", "inference-8gpu", 1),
        ("Qwen/Qwen3-1.7B", "inference-8gpu", 1),
        ("Qwen/Qwen3-8B", "inference-8gpu", 1),
        ("Qwen/Qwen3.5-4B", "inference-8gpu", 1),
        ("Qwen/Qwen3.6-35B-A3B", "inference-8gpu", 1),
        ("Qwen/Qwen3.8-27B", "inference-8gpu", 1),
    ],
)
def test_qwen_inference_tensor_parallel_recommendations(
    model_id,
    expected_profile_id,
    expected_tp,
):
    models_doc, profiles = load_catalog(CONFIG_DIR)
    model = next(item for item in models_doc["models"] if item["modelId"] == model_id)
    recommendation = model["capabilities"]["inference"]
    profile = next(
        item
        for item in profiles
        if item["id"] == recommendation["recommendedProfileId"]
    )

    assert recommendation["recommendedProfileId"] == expected_profile_id
    assert (
        profile["subJobs"][0]["args"]["extra_sampling"]["vllm_config"][
            "tensor_parallel_size"
        ]
        == expected_tp
    )


def test_validation_dates_match_completed_live_smokes():
    models_doc, profiles = load_catalog(CONFIG_DIR)

    assert all("lastValidated" not in profile for profile in profiles)
    recommendations = {}
    for model in models_doc["models"]:
        capabilities = model["capabilities"]
        if capabilities["inference"]["supported"]:
            recommendations[(model["modelId"], "inference")] = capabilities["inference"]
        if capabilities["training"]["supported"]:
            recommendations.update(
                {
                    (model["modelId"], profile_key): recommendation
                    for profile_key, recommendation in capabilities["training"][
                        "profiles"
                    ].items()
                }
            )

    assert len(recommendations) == 34
    validated = {
        key: recommendation["lastValidated"]
        for key, recommendation in recommendations.items()
        if "lastValidated" in recommendation
    }
    assert validated == {
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "inference"): "2026-08-31",
        ("zai-org/GLM-5.2-FP8", "inference"): "2026-09-01",
    }
    assert all(
        "max_seq_len" not in sub_job["args"]
        for profile in profiles
        for sub_job in profile["subJobs"]
    )


def test_qwen_38_live_smoke_uses_catalog_rl_lora_profile():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    model = next(
        item for item in models_doc["models"] if item["modelId"] == "Qwen/Qwen3.8-27B"
    )
    recommendation = model["capabilities"]["training"]["profiles"]["rlLora"]
    profile = next(
        item
        for item in profiles
        if item["id"] == recommendation["recommendedProfileId"]
    )
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3.8-27B",
        "rlLora",
    )
    wire = request["sub_job_configs"]

    assert [sub_job["job_type"] for sub_job in wire] == ["training", "sampling"]
    assert {sub_job["model_name"] for sub_job in wire} == {"Qwen/Qwen3.8-27B"}
    assert [
        wire[0]["training_config"]["n_gpus"],
        wire[1]["inference_config"]["n_gpus"],
    ] == [8, 8]
    assert "peft_config" in wire[0]["training_config"]
    assert "peft_config" in wire[1]["inference_config"]
    assert recommendation["recommendedProfileId"] == "dense-rl-lora-eager-8x8"
    vllm_config = profile["subJobs"][1]["args"]["extra_sampling"]["vllm_config"]
    assert vllm_config["enforce_eager"] is True
    assert vllm_config["extra_engine_kwargs"]["disable_custom_all_reduce"] is True


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "openai/gpt-oss-120b",
    ],
)
def test_large_quantized_inference_uses_single_node_tp8_with_fp8_kv_cache(
    model_id,
):
    models_doc, profiles = load_catalog(CONFIG_DIR)
    model = next(item for item in models_doc["models"] if item["modelId"] == model_id)
    recommendation = model["capabilities"]["inference"]
    profile = next(
        item
        for item in profiles
        if item["id"] == recommendation["recommendedProfileId"]
    )
    sampling = profile["subJobs"][0]["args"]
    vllm_config = sampling["extra_sampling"]["vllm_config"]

    assert recommendation["recommendedProfileId"] == "inference-8gpu-tp8-fp8-kv"
    assert sampling["n_gpus"] == 8
    assert vllm_config["tensor_parallel_size"] == 8
    assert vllm_config["kv_cache_dtype"] == "fp8"


@pytest.mark.parametrize(
    "model_id",
    [
        "zai-org/GLM-5.2",
        "zai-org/GLM-5.2-FP8",
    ],
)
def test_glm52_inference_uses_multinode_tp16_with_hopper_mla_cache(model_id):
    models_doc, profiles = load_catalog(CONFIG_DIR)
    model = next(item for item in models_doc["models"] if item["modelId"] == model_id)
    recommendation = model["capabilities"]["inference"]
    profile = next(
        item
        for item in profiles
        if item["id"] == recommendation["recommendedProfileId"]
    )
    sampling = profile["subJobs"][0]["args"]
    vllm_config = sampling["extra_sampling"]["vllm_config"]

    assert recommendation["recommendedProfileId"] == "inference-16gpu-fp8-kv"
    assert sampling["n_gpus"] == 16
    assert vllm_config["tensor_parallel_size"] == 16
    assert vllm_config["kv_cache_dtype"] == "fp8_ds_mla"
    assert vllm_config["gpu_memory_utilization"] == 0.9


def test_glm52_fp8_api_example_advertises_the_supported_1m_profile():
    request = json.loads((REPO_ROOT / "examples/api/glm-sampling.json").read_text())
    inference = request["sub_job_configs"][0]["inference_config"]
    vllm_config = inference["vllm_config"]

    assert inference["max_seq_len"] == 1048576
    assert inference["n_gpus"] == 16
    assert inference["gpu_memory_utilization"] == 0.9
    assert vllm_config["tensor_parallel_size"] == 16
    assert vllm_config["kv_cache_dtype"] == "fp8_ds_mla"


def test_router_replay_sampling_uses_single_node_tensor_parallelism():
    _, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "moe-rl-full-16x16")
    sampling = profile["subJobs"][1]["args"]

    assert sampling["n_gpus"] == 16
    assert sampling["extra_sampling"]["router_replay"]["enabled"] is True
    assert sampling["extra_sampling"]["vllm_config"]["tensor_parallel_size"] == 8


@pytest.mark.parametrize(
    ("model_id", "profile_key", "expected_max_context"),
    [
        ("Qwen/Qwen3-0.6B", "inference", 32768),
        ("Qwen/Qwen3.8-27B", "sftFull", 262144),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "inference", 1048576),
        ("openai/gpt-oss-120b", "inference", 131072),
        ("zai-org/GLM-5.2", "inference", 32768),
    ],
)
def test_recommendation_uses_advertised_model_limit(
    model_id,
    profile_key,
    expected_max_context,
):
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        model_id,
        profile_key,
    )

    assert {
        (sub_job.get("training_config") or sub_job.get("inference_config"))[
            "max_seq_len"
        ]
        for sub_job in request["sub_job_configs"]
    } == {expected_max_context}


def test_every_recommendation_materializes_its_advertised_context_limit():
    models_doc, _ = load_catalog(CONFIG_DIR)

    for model in models_doc["models"]:
        capabilities = model["capabilities"]
        recommendations = []
        if capabilities["inference"]["supported"]:
            recommendations.append(("inference", capabilities["inference"]))
        if capabilities["training"]["supported"]:
            recommendations.extend(capabilities["training"]["profiles"].items())

        for profile_key, recommendation in recommendations:
            request = build_profile_request(
                CONFIG_DIR,
                REPO_ROOT,
                model["modelId"],
                profile_key,
            )
            assert {
                (sub_job.get("training_config") or sub_job.get("inference_config"))[
                    "max_seq_len"
                ]
                for sub_job in request["sub_job_configs"]
            } == {recommendation["maxContextTokens"]}


@pytest.mark.parametrize("profile_key", ["sftFull", "rlFull"])
def test_dense_full_recommendations_use_zero_stage_two(profile_key):
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3.8-27B",
        profile_key,
    )
    training = next(
        sub_job["training_config"]
        for sub_job in request["sub_job_configs"]
        if sub_job["job_type"] == "training"
    )

    assert training["ds_config"]["zero_optimization"]["stage"] == 2


def test_moe_rl_enables_router_replay_on_sampling_only():
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3.6-35B-A3B",
        "rlFull",
    )
    training = next(
        sub_job["training_config"]
        for sub_job in request["sub_job_configs"]
        if sub_job["job_type"] == "training"
    )
    sampling = next(
        sub_job["inference_config"]
        for sub_job in request["sub_job_configs"]
        if sub_job["job_type"] == "sampling"
    )

    assert sampling["router_replay"] == {"enabled": True}
    assert "router_replay" not in training
    assert training["prime_rl"] == {
        "fused_lm_head_token_chunk_size": 8192,
        "fused_cross_entropy": False,
    }


@pytest.mark.parametrize("profile_key", ["sftLora", "rlLora"])
def test_qwen36_lora_uses_prime_rl_attention_targets(profile_key):
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3.6-35B-A3B",
        profile_key,
    )

    assert {
        (sub_job.get("training_config") or sub_job.get("inference_config"))[
            "max_seq_len"
        ]
        for sub_job in request["sub_job_configs"]
    } == {262144}
    training = next(
        sub_job["training_config"]
        for sub_job in request["sub_job_configs"]
        if sub_job["job_type"] == "training"
    )
    assert training["model_provider"] == "prime_rl"
    assert {
        tuple(
            (sub_job.get("training_config") or sub_job.get("inference_config"))[
                "peft_config"
            ]["target_modules"]
        )
        for sub_job in request["sub_job_configs"]
    } == {("q_proj", "k_proj", "v_proj", "o_proj")}


def test_qwen36_lora_rejects_huggingface_mlp_target_names():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "moe-sft-lora-8gpu")
    profile["subJobs"][0]["args"]["extra_training"]["peft_config"][
        "target_modules"
    ].append("down_proj")

    with pytest.raises(CatalogValidationError, match="supports only attention"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_forward_backward_probe_matches_catalog_training_batch_size():
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3-0.6B",
        "sftLora",
    )

    spec = build_forward_backward_probe_spec(request, "sftLora")
    payload = spec["payload"]

    assert len(payload["input_ids"]) == 8
    assert payload["input_ids"][0] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert payload["position_ids"] == "arange"
    assert payload["labels"] == {
        "strategy": "next_token",
        "mask_padding": False,
    }


def test_rl_forward_backward_probe_uses_packing_aware_grpo_contract():
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3.6-35B-A3B",
        "rlFull",
    )

    spec = build_forward_backward_probe_spec(request, "rlFull")
    payload = spec["payload"]

    assert len(payload["input_ids"]) == 32
    assert payload["include_attention_mask"] is True
    assert payload["labels"] == "none"
    assert spec["processing"] == {
        "loss_fn": "grpo",
        "post": ["compute_logprobs"],
        "config": {
            "eps_clip": 0.2,
            "loss_agg_mode": "token-mean",
            "entropy_coeff": 0.0,
            "global_batch_size": 32,
        },
    }
    assert spec["context"]["advantages"]["dtype"] == "float32"
    assert spec["context"]["loss_mask"]["data"][0] == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]


@pytest.mark.parametrize(
    ("profile_key", "expected_operations"),
    [
        ("inference", ["generate"]),
        ("sftLora", ["forward-backward"]),
        ("rlFull", ["generate", "forward-backward"]),
    ],
)
def test_live_smoke_executes_workflow_data_plane_probes(
    monkeypatch,
    profile_key,
    expected_operations,
):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def generate(self, job_id, prompts, sampling_params):
            self.calls.append(("generate", job_id, prompts, sampling_params))
            return "generate-request"

        def forward_backward(self, job_id, payload):
            self.calls.append(("forward-backward", job_id, payload))
            return "forward-backward-request"

        def poll_request(self, job_id, request_id):
            self.calls.append(("poll", job_id, request_id))
            if request_id == "generate-request":
                return {"results": [{"text": "OK"}]}
            return {"avg_loss": 0.25}

    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3-0.6B",
        profile_key,
    )
    monkeypatch.setattr(
        smoke_catalog,
        "_build_forward_backward_probe_payload",
        lambda *_: b"forward-backward-payload",
    )
    client = FakeClient()

    probes = run_data_plane_probes(client, "job-1", profile_key, request)

    assert [probe["operation"] for probe in probes] == expected_operations
    if "generate" in expected_operations:
        assert (
            "generate",
            "job-1",
            ["Reply with OK."],
            {"max_tokens": 1, "temperature": 0.0},
        ) in client.calls
        assert ("poll", "job-1", "generate-request") in client.calls
    if "forward-backward" in expected_operations:
        assert (
            "forward-backward",
            "job-1",
            b"forward-backward-payload",
        ) in client.calls
        assert ("poll", "job-1", "forward-backward-request") in client.calls


def test_live_smoke_rejects_non_finite_forward_backward_loss(monkeypatch):
    class FakeClient:
        def forward_backward(self, job_id, payload):
            return "forward-backward-request"

        def poll_request(self, job_id, request_id):
            return {"avg_loss": float("nan")}

    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3-0.6B",
        "sftFull",
    )
    monkeypatch.setattr(
        smoke_catalog,
        "_build_forward_backward_probe_payload",
        lambda *_: b"forward-backward-payload",
    )

    with pytest.raises(RuntimeError, match="non-finite avg_loss"):
        run_data_plane_probes(FakeClient(), "job-1", "sftFull", request)


def test_duplicate_model_id_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    models_doc["models"].append(copy.deepcopy(models_doc["models"][0]))

    with pytest.raises(CatalogValidationError, match="duplicate model id"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_missing_recommended_profile_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    models_doc["models"][0]["capabilities"]["inference"][
        "recommendedProfileId"
    ] = "missing"

    with pytest.raises(CatalogValidationError, match="unknown profile missing"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_profile_cannot_override_capability_context_limit():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "dense-sft-full-8gpu")
    profile["subJobs"][0]["args"]["max_seq_len"] = 32769

    with pytest.raises(CatalogValidationError, match="max_seq_len is injected"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_profile_gpu_count_must_be_a_multiple_of_eight():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "inference-8gpu")
    profile["subJobs"][0]["args"]["n_gpus"] = 4

    with pytest.raises(CatalogValidationError, match="n_gpus must be a multiple of 8"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_sampling_tensor_parallel_size_must_divide_gpu_count():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "inference-8gpu")
    profile["subJobs"][0]["args"]["extra_sampling"]["vllm_config"][
        "tensor_parallel_size"
    ] = 3

    with pytest.raises(
        CatalogValidationError,
        match="tensor_parallel_size must be a divisor of n_gpus",
    ):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_recommendation_validation_date_must_be_iso_formatted():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    models_doc["models"][0]["capabilities"]["inference"][
        "lastValidated"
    ] = "August 20, 2026"

    with pytest.raises(CatalogValidationError, match="must be an ISO date"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_zero_stage_three_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "dense-sft-full-8gpu")
    zero_optimization = profile["subJobs"][0]["args"]["extra_training"]["ds_config"][
        "zero_optimization"
    ]
    zero_optimization["stage"] = 3

    with pytest.raises(CatalogValidationError, match="ZeRO stage 3 is unsupported"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_unsupported_capability_cannot_carry_configuration():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    capability = models_doc["models"][-1]["capabilities"]["training"]
    capability["profiles"] = copy.deepcopy(
        models_doc["models"][0]["capabilities"]["training"]["profiles"]
    )

    with pytest.raises(
        CatalogValidationError,
        match="unsupported capabilities cannot define profiles",
    ):
        validate_catalog(models_doc, profiles, REPO_ROOT)
