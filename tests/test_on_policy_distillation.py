# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the on-policy distillation example."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from examples.distillation import episodes as episodes_mod
from examples.distillation import teacher as teacher_mod
from examples.distillation import train as runner
from examples.distillation.tasks import wiki_search
from examples.utils import _forward_backward_payload
from examples.utils import collate

from cortex_training import wire


@dataclass
class _Config:
    config: str | None = "unused"
    connection: str | None = None
    job_id: str | None = None
    teacher_job_id: str | None = None
    judge_job_id: str | None = None
    student: str = "Qwen/Qwen3.5-2B"
    teacher: str = "Qwen/Qwen3.8-27B"
    judge: str | None = "Qwen/Qwen3-30B-A3B"
    context: str = "standard"
    teacher_context: str | None = None
    judge_context: str = "standard"
    hardware: str = "h200"
    lora: bool = False
    student_max_seq_len: int | None = 16384
    teacher_max_seq_len: int | None = 20480
    judge_max_seq_len: int | None = 20480
    teacher_gpus: int | None = 2
    judge_gpus: int | None = 2
    job_config: str | None = None
    teacher_job_config: str | None = None
    judge_job_config: str | None = None
    questions_per_step: int = 2
    group_size: int = 2
    max_turns: int = 3
    turn_max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    eval_temperature: float = 0.7
    eval_top_p: float = 0.8
    eval_top_k: int = 20
    max_steps: int = 2
    warmup_steps: int = 1
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    teacher_tau: float = 1.0
    teacher_clip: float = 3.0
    teacher_clip_negative: float | None = 1.0
    reward_coef: float = 0.0
    eps_clip: float = 0.2
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0
    eval_every: int = 1
    n_eval: int = 2
    eval_samples: int = 1
    evaluate_teacher: bool = True
    max_prompts_per_request: int | None = 2
    teacher_prompts_per_request: int | None = 8
    seed: int = 0
    weight_sync_bucket_size: int | None = None
    debug_image_tag: str | None = None
    log_path: str = "/tmp/unused"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def _position(token: int, logprob: float, key=int) -> dict:
    return {key(token): {"logprob": logprob, "rank": 1}}


# --- a tiny tokenizer that renders the Qwen chat format ------------------------

_SPECIAL = {"<|im_end|>": 1000, "<|endoftext|>": 1001}
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in _SPECIAL) + ")")


class _Tokenizer:
    """Character-level tokens plus special tokens, with the Qwen tool-message layout."""

    pad_token_id = None
    eos_token_id = _SPECIAL["<|endoftext|>"]

    def __init__(self, think_off_header: bool = True):
        self.think_off_header = think_off_header

    def get_vocab(self):
        return {"x": 0}

    def convert_tokens_to_ids(self, token):
        return _SPECIAL[token]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in _SPLIT.split(text):
            ids.extend([_SPECIAL[piece]] if piece in _SPECIAL else [ord(c) for c in piece])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        names = {v: k for k, v in _SPECIAL.items()}
        return "".join(
            ("" if skip_special_tokens else names[i]) if i in names else chr(i) for i in ids
        )

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, enable_thinking=False, tokenize=True
    ):
        text = "<|im_start|>system\ntools\n" if tools else ""
        for i, m in enumerate(messages):
            if m["role"] == "user":
                text += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            elif m["role"] == "assistant":
                calls = "".join(
                    f"<tool_call>\n{json.dumps(c['function'])}\n</tool_call>"
                    for c in m.get("tool_calls") or []
                )
                text += f"<|im_start|>assistant\n{m['content']}{calls}<|im_end|>\n"
            else:
                if i == 0 or messages[i - 1]["role"] != "tool":
                    text += "<|im_start|>user"
                text += f"\n<tool_response>\n{m['content']}\n</tool_response>"
                if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                    text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if self.think_off_header and not enable_thinking:
                text += "<think>\n\n</think>\n\n"
        return text if not tokenize else self.encode(text)


# --- teacher scoring ----------------------------------------------------------


@pytest.mark.parametrize("key", [int, str])
def test_teacher_completion_logprobs_reads_completion_positions(key):
    full = [11, 12, 13, 14, 15]
    result = {
        "prompt_logprobs": [None] + [_position(t, -0.1 * t, key) for t in full[1:]]
    }

    values = teacher_mod.teacher_completion_logprobs(result, full, prompt_len=3)

    assert values == pytest.approx([-1.4, -1.5])


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({}, "no 'prompt_logprobs'"),
        ({"prompt_logprobs": [None, _position(12, -0.1)]}, "prompt_logprobs for"),
        (
            {
                "prompt_logprobs": [
                    _position(11, -0.1),
                    _position(12, -0.1),
                    _position(13, -0.1),
                ]
            },
            "should be None",
        ),
        (
            {"prompt_logprobs": [None, _position(12, -0.1), _position(99, -0.1)]},
            "missing",
        ),
        (
            {"prompt_logprobs": [None, _position(12, -0.1), _position(13, math.nan)]},
            "not finite",
        ),
        (
            {"prompt_logprobs": [None, _position(12, -0.1), _position(13, 0.5)]},
            "not finite",
        ),
    ],
)
def test_teacher_completion_logprobs_fails_closed(result, message):
    with pytest.raises(ValueError, match=message):
        teacher_mod.teacher_completion_logprobs(result, [11, 12, 13], prompt_len=2)


def test_sampled_token_logprobs_requires_aligned_logprobs():
    result = {
        "token_ids": [5, 6],
        "logprobs": [_position(5, -0.2), _position(6, -0.3, str)],
    }
    assert teacher_mod.sampled_token_logprobs(result) == pytest.approx([-0.2, -0.3])

    with pytest.raises(ValueError, match="logprobs=0"):
        teacher_mod.sampled_token_logprobs({"token_ids": [5]})
    with pytest.raises(ValueError, match="logprob positions"):
        teacher_mod.sampled_token_logprobs(
            {"token_ids": [5, 6], "logprobs": [_position(5, -0.2)]}
        )


class _ScoringClient:
    def __init__(self):
        self.requests = []
        self.polled = []

    def generate(self, job_id, prompts, sampling_params):
        self.requests.append((job_id, prompts, sampling_params))
        return f"r{len(self.requests) - 1}"

    def poll_request(self, job_id, request_id):
        self.polled.append(request_id)
        _, prompts, _ = self.requests[int(request_id[1:])]
        return {
            "results": [
                {"prompt_logprobs": [None] + [_position(t, -1.0) for t in p[1:]], "token_ids": [0]}
                for p in prompts
            ]
        }


def test_score_with_teacher_sends_exact_token_ids():
    client = _ScoringClient()

    scores = teacher_mod.score_with_teacher(client, "teacher", [[1, 2], [3]], [[7, 8, 9], [4]])

    assert [(job, prompts) for job, prompts, _ in client.requests] == [
        ("teacher", [[1, 2, 7, 8, 9], [3, 4]])
    ]
    assert client.requests[0][2] == {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0}
    assert scores == [[-1.0, -1.0, -1.0], [-1.0]]


def test_score_with_teacher_chunks_requests_and_keeps_their_order():
    client = _ScoringClient()
    prompts = [[1, t] for t in range(5)]

    scores = teacher_mod.score_with_teacher(
        client, "teacher", prompts, [[9]] * 5, max_prompts_per_request=2
    )

    assert [len(p) for _, p, _ in client.requests] == [2, 2, 1]
    assert client.polled == ["r0", "r1", "r2"]
    assert [p for _, chunk, _ in client.requests for p in chunk] == [p + [9] for p in prompts]
    assert scores == [[-1.0]] * 5


def test_check_shared_tokenizer():
    same = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    other = SimpleNamespace(get_vocab=lambda: {"a": 0, "c": 1})
    teacher_mod.check_shared_tokenizer(same, same)
    with pytest.raises(ValueError, match="share a tokenizer"):
        teacher_mod.check_shared_tokenizer(same, other)


def test_reverse_kl_estimates():
    kl = teacher_mod.reverse_kl_estimates([[-0.5, -1.0]], [[-1.5, -1.0]])
    assert kl["kl/k1_per_token"] == pytest.approx(0.5)
    assert kl["kl/k3_per_token"] == pytest.approx((math.exp(-1.0) + 1.0 - 1.0) / 2)


def test_clipped_fraction_counts_tokens_outside_the_clip():
    student = [[0.0] * 8]
    teacher = [[-10.0] + [1.0] * 7]

    assert teacher_mod.clipped_fraction(
        student, teacher, clip=3.0, clip_negative=3.0
    ) == pytest.approx(1 / 8)
    assert teacher_mod.clipped_fraction(
        student, teacher, clip=0.5, clip_negative=20.0
    ) == pytest.approx(7 / 8)
    assert teacher_mod.clipped_fraction([], [], clip=3.0, clip_negative=3.0) is None


# --- episodes -----------------------------------------------------------------


def test_parse_tool_calls_reads_json_and_xml_formats():
    json_call = '<tool_call>\n{"name": "wiki_search_pages", "arguments": {"query": "x"}}\n</tool_call>'
    xml_call = (
        "<tool_call>\n<function=wiki_read_section>\n<parameter=section_id>\npage:sec\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    assert episodes_mod.parse_tool_calls("I'll search. " + json_call) == [
        ("wiki_search_pages", {"query": "x"})
    ]
    assert episodes_mod.parse_tool_calls(xml_call) == [("wiki_read_section", {"section_id": "page:sec"})]
    assert episodes_mod.parse_tool_calls("<tool_call>{bad</tool_call>") == [
        (episodes_mod.INVALID_TOOL_CALL, {})
    ]
    assert episodes_mod.parse_tool_calls("The answer is 42.") == []


@pytest.mark.parametrize("think_off_header", [True, False])
def test_chat_format_appends_tool_turns_like_the_chat_template(think_off_header):
    fmt = episodes_mod.ChatFormat(_Tokenizer(think_off_header))

    assert fmt.header_suffix == ("<think>\n\n</think>\n\n" if think_off_header else "")
    assert fmt.check_tool_turn(wiki_search.TOOLS)
    assert fmt.stop_ids == [1000, 1001]


_TEMPLATES = Path(__file__).resolve().parent / "fixtures" / "chat_templates"


@pytest.mark.parametrize(
    "template",
    # From Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B, and Qwen/Qwen3.8-27B, which Qwen3.8-Flash-Next shares.
    ["qwen3.5-2b.jinja", "qwen3.5-4b.jinja", "qwen3.8.jinja"],
)
def test_chat_format_matches_vendored_qwen_templates(template):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<|im_end|>": 0, "<|endoftext|>": 1, "[UNK]": 2}
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]")),
        eos_token="<|im_end|>",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = (_TEMPLATES / template).read_text()

    fmt = episodes_mod.ChatFormat(tokenizer)

    assert fmt.header_suffix == "<think>\n\n</think>\n\n"
    assert fmt.check_tool_turn(wiki_search.TOOLS)


def test_chat_format_matches_real_qwen_templates():
    get_tokenizer = pytest.importorskip("tinker_cookbook.tokenizer_utils").get_tokenizer
    for model in ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.8-27B"):
        try:
            tokenizer = get_tokenizer(model)
        except Exception as exc:  # no network or no cached tokenizer
            pytest.skip(f"cannot load {model}: {exc}")
        assert episodes_mod.ChatFormat(tokenizer).check_tool_turn(wiki_search.TOOLS)


def _two_turn_episode():
    student, teacher = episodes_mod.ChatFormat(_Tokenizer(False)), episodes_mod.ChatFormat(_Tokenizer())
    episode = episodes_mod.Episode(
        problem="q", ids=[student.first_prompt("q", []), teacher.first_prompt("q", [])], spans=[[], []]
    )
    turns = [student.tok.encode("call") + [1000], student.tok.encode("done") + [1000]]
    for i, tokens in enumerate(turns):
        for k, fmt in enumerate((student, teacher)):
            episode.spans[k].append((len(episode.ids[k]), len(tokens)))
            episode.ids[k].extend(tokens)
            if i == 0:
                episode.ids[k].extend(fmt.tool_turn(["RESULT"]))
    return episode, turns


def test_train_sequences_train_only_agent_tokens_with_aligned_teacher_scores():
    episode, turns = _two_turn_episode()
    prompts, rests = episodes_mod.teacher_scoring_inputs([episode])
    first = episode.spans[1][0][0]
    # A fake teacher score per teacher token that encodes its position.
    scores = [[-(first + i) / 1000.0 for i in range(len(rests[0]))]]
    agent_lps = episodes_mod.agent_teacher_logprobs([episode], scores)

    sequences, shifted = episodes_mod.train_sequences([episode], [0.0], agent_lps)

    seq = sequences[0]
    trained = [p for p, m in enumerate(seq.loss_mask) if m]
    assert [seq.input_ids[p + 1] for p in trained] == turns[0] + turns[1]
    teacher_positions = [s + j for s, n in episode.spans[1] for j in range(n)]
    assert [shifted[0][p] for p in trained] == pytest.approx([-t / 1000.0 for t in teacher_positions])
    assert prompts == [episode.ids[1][:first]]
    tool_result = range(episode.spans[0][0][0] + len(turns[0]), episode.spans[0][1][0] - 1)
    assert not any(seq.loss_mask[p] for p in tool_result)
    with pytest.raises(ValueError, match="teacher scores"):
        episodes_mod.train_sequences([episode], [0.0], [agent_lps[0][:-1]])


def test_forward_backward_payload_carries_teacher_logprobs():
    episode, _ = _two_turn_episode()
    agent_lps = [[-0.5] * episode.agent_tokens]
    sequences, shifted = episodes_mod.train_sequences([episode], [0.0], agent_lps)
    kwargs, context = collate(sequences, pad_token_id=0, max_seq_len=4096, temperature=1.0)
    context["teacher_log_probs_shifted"] = episodes_mod.shifted_tensor(shifted, context["loss_mask"])

    frame = wire.loads(
        _forward_backward_payload(kwargs, context=context, processing=runner.processing_block(_Config(), 1))
    )

    assert torch.equal(frame["context"]["teacher_log_probs_shifted"] != 0, context["loss_mask"] > 0)
    assert torch.equal(frame["context"]["advantages"], torch.zeros_like(context["loss_mask"]))
    assert frame["processing"]["config"]["teacher_clip_negative"] == 1.0


# --- wiki-search task ---------------------------------------------------------


class _Collection:
    def query(self, query_texts, n_results):
        return {"ids": [["alpha", "beta"]], "metadatas": [[{"title": "Alpha"}, {"title": "Beta"}]]}


def _wiki_task():
    index = wiki_search.WikiIndex(
        title={"alpha": "Alpha", "beta": "Beta"},
        content={"alpha": "# Alpha\nintro\n## Early life\nborn here\n## Career\nworked", "beta": "no headings"},
        collection=_Collection(),
    )
    return wiki_search.WikiSearchTask(index=index)


def test_wiki_tools_search_view_and_read():
    task = _wiki_task()

    assert json.loads(task.call_tool("wiki_search_pages", {"query": "a"}))["pages"][0] == {
        "page_id": "alpha",
        "title": "Alpha",
    }
    sections = json.loads(task.call_tool("wiki_view_sections", {"page_id": "alpha"}))["sections"]
    assert [s["section_id"] for s in sections] == ["alpha:alpha", "alpha:early_life", "alpha:career"]
    assert task.call_tool("wiki_read_section", {"section_id": "alpha:early_life"}) == "## Early life\nborn here"
    assert json.loads(task.call_tool("wiki_view_sections", {"page_id": "beta"}))["sections"] == [
        {"section_id": "beta:full", "section_name": "Full Page"}
    ]


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("wiki_read_section", {"section_id": "alpha"}, "Invalid section_id"),
        ("wiki_read_section", {"section_id": "alpha:nope"}, "Section not found"),
        ("wiki_view_sections", {"page_id": "gamma"}, "unknown id"),
        ("wiki_search_pages", {"q": "x"}, "requires the argument 'query'"),
        ("delete_everything", {}, "Unknown tool"),
    ],
)
def test_wiki_tool_errors_come_back_as_text(name, arguments, message):
    assert message in _wiki_task().call_tool(name, arguments)


def test_wiki_tool_results_are_truncated():
    task = _wiki_task()
    task.index.content["alpha"] = "x" * (wiki_search.TOOL_RESULT_MAX_CHARS + 50)

    text = task.call_tool("wiki_read_section", {"section_id": "alpha:full"})

    assert len(text) == wiki_search.TOOL_RESULT_MAX_CHARS + len("\n...[truncated]")


def test_wiki_judge_prompt_and_verdict():
    task = _wiki_task()
    question = wiki_search.Question(0, "Who?", "Homer Jones")

    prompt = task.judge_prompt(question, "It was Homer Jones.")

    assert "Who?" in prompt and "Homer Jones" in prompt and "It was Homer Jones." in prompt
    assert task.judge_verdict("Yes") == 1.0
    assert task.judge_verdict("no") == 0.0


class _FakeDatasets:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def load_dataset(self, name, split, revision=None):
        self.calls.append((name, split, revision))
        return self.rows[name]


def test_wiki_datasets_load_at_pinned_revisions(monkeypatch, tmp_path):
    import sys

    fake = _FakeDatasets(
        {
            wiki_search.QUESTIONS_DATASET: [{"question": f"q{i}", "answer": "a"} for i in range(5)],
            wiki_search.CORPUS_DATASET: [{"id": "alpha", "title": "Alpha", "content": "# Alpha"}],
        }
    )
    clients = []

    class _Client:
        def __init__(self, path):
            clients.append(path)

        def get_or_create_collection(self, name):
            return SimpleNamespace(count=lambda: 1, upsert=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake.load_dataset))
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=_Client))

    problems = wiki_search.WikiSearchTask(index_dir=tmp_path).load(seed=0, n_eval=2)
    wiki_search.WikiIndex.load(tmp_path)

    assert (len(problems.train), len(problems.eval)) == (3, 2)
    assert fake.calls == [
        (wiki_search.QUESTIONS_DATASET, "train", wiki_search.QUESTIONS_REVISION),
        (wiki_search.CORPUS_DATASET, "train", wiki_search.CORPUS_REVISION),
    ]
    assert clients == [str(tmp_path / wiki_search.CORPUS_REVISION)]


def test_example_metadata_records_the_pinned_dataset_revisions():
    metadata = (runner._EXAMPLE_DIR / "example.yaml").read_text()

    assert f"{wiki_search.QUESTIONS_DATASET}: {wiki_search.QUESTIONS_REVISION}" in metadata
    assert f"{wiki_search.CORPUS_DATASET}: {wiki_search.CORPUS_REVISION}" in metadata


def test_wiki_index_rejects_page_ids_containing_a_colon(monkeypatch, tmp_path):
    import sys

    fake = _FakeDatasets({wiki_search.CORPUS_DATASET: [{"id": "a:b", "title": "A", "content": ""}]})
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake.load_dataset))
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=None))

    with pytest.raises(ValueError, match="must not contain ':'"):
        wiki_search.WikiIndex.load(tmp_path)


# --- loss configuration -------------------------------------------------------


def test_processing_block_enables_the_grpo_teacher_term():
    block = runner.processing_block(_Config(teacher_clip_negative=1.5), global_batch_size=4)

    assert block["loss_fn"] == "grpo"
    assert block["config"]["teacher_tau"] == 1.0
    assert block["config"]["teacher_clip"] == 3.0
    assert block["config"]["teacher_clip_negative"] == 1.5
    assert block["config"]["global_batch_size"] == 4


def test_processing_block_omits_the_teacher_term_for_the_rl_comparison():
    block = runner.processing_block(_Config(teacher_tau=0.0, reward_coef=1.0), global_batch_size=4)

    assert not {"teacher_tau", "teacher_clip", "teacher_clip_negative"} & set(block["config"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"teacher_tau": -1.0}, "teacher_tau"),
        ({"teacher_clip": 0.0}, "teacher_clip"),
        ({"teacher_clip_negative": -1.0}, "teacher_clip_negative"),
        ({"teacher_tau": 0.0, "reward_coef": 0.0}, "no training signal"),
        ({"teacher_tau": 0.0, "reward_coef": 1.0, "judge": None}, "needs a judge"),
        ({"teacher_gpus": 0}, "teacher_gpus"),
        ({"lr_schedule": "linear"}, "lr_schedule"),
        ({"max_turns": 0}, "max_turns"),
        ({"eval_samples": 0}, "eval_samples"),
    ],
)
def test_validate_config_rejects_unusable_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        runner.validate_config(_Config(**overrides))


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    config = _Config(warmup_steps=5)
    values = [runner.lr_at(step, 1e-5, config, total_steps=100) for step in (0, 4, 5, 100)]
    assert values == pytest.approx([2e-6, 1e-5, 1e-5, 1e-6])
    assert runner.lr_at(50, 1e-5, _Config(lr_schedule="constant"), total_steps=100) == 1e-5


def _sub(body, job_type):
    return next(sub for sub in body["sub_job_configs"] if sub["job_type"] == job_type)


def test_job_bodies_pair_a_student_with_sampling_only_teacher_and_judge():
    bodies = runner.job_bodies(_Config())

    assert {sub["job_type"] for sub in bodies["student"]["sub_job_configs"]} == {"training", "sampling"}
    assert [sub["job_type"] for sub in bodies["teacher"]["sub_job_configs"]] == ["sampling"]
    assert [sub["job_type"] for sub in bodies["judge"]["sub_job_configs"]] == ["sampling"]
    assert "teacher" not in runner.job_bodies(_Config(teacher_tau=0.0, reward_coef=1.0))
    assert "judge" not in runner.job_bodies(_Config(judge=None))


def test_job_bodies_build_the_validated_footprint_from_the_catalog():
    bodies = runner.job_bodies(_Config())
    training = _sub(bodies["student"], "training")["training_config"]
    sampling = _sub(bodies["student"], "sampling")["inference_config"]
    teacher = _sub(bodies["teacher"], "sampling")
    judge = _sub(bodies["judge"], "sampling")

    # examples/distillation/overrides.json and the sizing flags give the validated job.
    assert (training["n_gpus"], sampling["n_gpus"], training["train_batch_size"]) == (1, 1, 1)
    assert training["max_seq_len"] == sampling["max_seq_len"] == 16384
    assert training["optimizer"]["lr"] == 1e-6
    assert training["mb_spec"]["max_tokens_per_mb"] == 16384
    assert training["ds_config"]["zero_optimization"]["reduce_scatter"] is True
    # On-policy distillation replays no routing.
    assert "router_replay" not in training and "router_replay" not in sampling
    assert teacher["model_name"] == "Qwen/Qwen3.8-27B"
    assert teacher["inference_config"]["max_seq_len"] == 20480
    assert teacher["inference_config"]["n_gpus"] == 2
    assert teacher["inference_config"]["vllm_config"]["tensor_parallel_size"] == 2
    assert judge["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert (judge["inference_config"]["n_gpus"], judge["inference_config"]["max_seq_len"]) == (2, 20480)


def test_teacher_and_judge_gpus_none_keep_the_catalog_footprint():
    bodies = runner.job_bodies(_Config(teacher_gpus=None, judge_gpus=None))

    assert _sub(bodies["teacher"], "sampling")["inference_config"]["n_gpus"] == 8
    assert _sub(bodies["judge"], "sampling")["inference_config"]["n_gpus"] == 4


def test_job_bodies_enforce_a_teacher_window_above_the_student():
    with pytest.raises(ValueError, match="must exceed the student"):
        runner.job_bodies(_Config(student_max_seq_len=8192, teacher_max_seq_len=4096))


def _sampler_file(tmp_path, name, model, samplers=1):
    path = tmp_path / f"{name}.json"
    sub = {"job_type": "sampling", "model_name": model, "inference_config": {"max_seq_len": 999, "n_gpus": 1}}
    path.write_text(json.dumps({"hardware": "H200", "sub_job_configs": [sub] * samplers}))
    return path


def _student_file(tmp_path, training_model="org/S", sampling_model="org/S"):
    path = tmp_path / "student.json"
    path.write_text(
        json.dumps(
            {
                "hardware": "H200",
                "sub_job_configs": [
                    {"job_type": "training", "model_name": training_model, "training_config": {"max_seq_len": 512, "n_gpus": 1}},
                    {"job_type": "sampling", "model_name": sampling_model, "inference_config": {"max_seq_len": 512, "n_gpus": 1}},
                ],
            }
        )
    )
    return path


def test_job_config_files_replace_the_catalog_bodies(tmp_path):
    bodies = runner.job_bodies(
        _Config(
            job_config=str(_student_file(tmp_path)),
            teacher_job_config=str(_sampler_file(tmp_path, "teacher", "org/T")),
            judge_job_config=str(_sampler_file(tmp_path, "judge", "org/J")),
        )
    )

    assert _sub(bodies["student"], "training")["model_name"] == "org/S"
    assert _sub(bodies["teacher"], "sampling")["model_name"] == "org/T"
    assert _sub(bodies["judge"], "sampling")["model_name"] == "org/J"


def test_a_manual_teacher_or_judge_does_not_build_the_catalog_model(tmp_path):
    # Models that are not in the catalog: building them from the catalog would raise.
    bodies = runner.job_bodies(
        _Config(
            teacher="org/NotInCatalog",
            teacher_job_config=str(_sampler_file(tmp_path, "teacher", "org/T")),
            judge="org/AlsoNotInCatalog",
            judge_job_config=str(_sampler_file(tmp_path, "judge", "org/J")),
        )
    )

    assert {sub["model_name"] for sub in bodies["student"]["sub_job_configs"]} == {"Qwen/Qwen3.5-2B"}
    assert _sub(bodies["teacher"], "sampling")["model_name"] == "org/T"


def test_a_manual_student_does_not_build_the_catalog_student(tmp_path):
    bodies = runner.job_bodies(_Config(student="org/NotInCatalog", job_config=str(_student_file(tmp_path))))

    assert {sub["model_name"] for sub in bodies["student"]["sub_job_configs"]} == {"org/S"}
    assert _sub(bodies["teacher"], "sampling")["model_name"] == "Qwen/Qwen3.8-27B"


def test_a_student_job_config_with_mismatched_models_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must name the same model"):
        runner.job_bodies(_Config(job_config=str(_student_file(tmp_path, "org/S", "org/Other"))))


@pytest.mark.parametrize("role", ["teacher", "judge"])
def test_a_sampler_job_config_with_duplicate_samplers_is_rejected(tmp_path, role):
    path = _sampler_file(tmp_path, role, "org/T", samplers=2)

    with pytest.raises(ValueError, match=f"the {role} job .* exactly one sampling sub-job"):
        runner.job_bodies(_Config(**{f"{role}_job_config": str(path)}))


def test_write_job_records_writes_student_teacher_and_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CONFIGS_DIR", tmp_path)
    bodies = {
        "student": {"sub_job_configs": [{"job_type": "training"}, {"job_type": "sampling"}]},
        "teacher": {"sub_job_configs": [{"job_type": "sampling"}]},
        "judge": {"sub_job_configs": [{"job_type": "sampling"}]},
    }

    paths = runner.write_job_records(_Config(), bodies)

    assert {path.name for path in paths} == {
        "qwen3.5-2b-standard-h200-full-student.json",
        "qwen3.8-27b-teacher.json",
        "qwen3-30b-a3b-judge.json",
    }
    # Attached or job_config= jobs write no record.
    assert runner.write_job_records(_Config(job_id="s", teacher_job_id="t", judge_job_id="j"), bodies) == []


# --- end to end with a fake client --------------------------------------------


@dataclass
class _Problems:
    train: list
    eval: list


class _Task:
    name = "toy"
    tools = [wiki_search.TOOLS[0]]

    def load(self, *, seed, n_eval):
        questions = [wiki_search.Question(i, f"q{i}", "1") for i in range(6)]
        return _Problems(train=questions[n_eval:], eval=questions[:n_eval])

    def user_message(self, problem):
        return problem.question

    def call_tool(self, name, arguments):
        return "RESULT"

    def judge_prompt(self, problem, response):
        return f"JUDGE gold={problem.answer} response={response}"

    def judge_verdict(self, text):
        return float(text == "yes")


_CALL = '<tool_call>\n{"name": "wiki_search_pages", "arguments": {"query": "q"}}\n</tool_call>'


@dataclass
class _FakeClient:
    token_count_shortfall: int = 0
    created: list = field(default_factory=list)
    cancelled: list = field(default_factory=list)
    generated: list = field(default_factory=list)
    synced: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    _requests: dict = field(default_factory=dict)

    def create_job_from_body(self, body):
        model = body["sub_job_configs"][0]["model_name"]
        if len(body["sub_job_configs"]) == 2:
            job_id = "student-job"
        elif "Qwen3.8" in model:
            job_id = "teacher-job"
        else:
            job_id = "judge-job"
        self.created.append(job_id)
        return {"job_id": job_id}

    def wait_for_job(self, job_id):
        return None

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)

    def _request(self, value):
        request_id = f"r{len(self._requests)}"
        self._requests[request_id] = value
        return request_id

    def poll_request(self, job_id, request_id):
        return self._requests[request_id]

    def generate(self, job_id, prompts, sampling_params):
        self.generated.append((job_id, prompts, sampling_params))
        tok = _Tokenizer()
        if sampling_params.get("prompt_logprobs") == 0:
            results = [
                {"prompt_logprobs": [None] + [_position(t, -1.0) for t in p[1:]], "token_ids": [0]}
                for p in prompts
            ]
            return self._request({"results": results})
        results = []
        for prompt in prompts:
            text = tok.decode(prompt)
            if "JUDGE" in text:
                reply = "yes" if "response=1<|im_end|>" in text else "no"
            else:
                reply = "1" if "<tool_response>" in text else _CALL
            tokens = tok.encode(reply) + [1000]
            result = {"token_ids": tokens, "finish_reason": "stop"}
            if "logprobs" in sampling_params:
                result["logprobs"] = [_position(t, -0.5) for t in tokens]
            results.append(result)
        return self._request({"results": results})

    def forward_backward(self, job_id, payload):
        frame = wire.loads(payload)
        self.frames.append(frame)
        tokens = float(frame["context"]["loss_mask"].sum().item())
        return self._request(
            {
                "avg_loss": 0.1,
                "metrics": {
                    "teacher_term_token_count": tokens - self.token_count_shortfall,
                    "teacher_log_ratio_sum": -tokens,
                },
            }
        )

    def step(self, job_id, learning_rate=None):
        self.steps.append(job_id)
        return self._request({"global_steps": 1, "grad_norm": 1.0})

    def weight_sync(self, job_id, *, source_sub_job_id, target_sub_job_ids, weight_format=None, bucket_size=None):
        self.synced.append((job_id, tuple(target_sub_job_ids), weight_format))
        return self._request({"status": "ok"})

    def save(self, job_id, checkpoint_type=None):
        return self._request({"stage_path": "s3://stage/checkpoints/cp_abc/global_step2/"})


def _patch_runner(monkeypatch, client, tokenizer_for=lambda model: _Tokenizer()) -> list:
    """Route ``_train`` to ``client`` with fake tokenizers; return the logged metrics."""
    logged = []
    # The jobs are built from the catalog; keep each run's record out of the tree.
    monkeypatch.setattr(runner, "write_job_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "make_client", lambda path, connection=None, **kw: client)
    monkeypatch.setattr(
        runner,
        "setup_logging",
        lambda config, client=None, job_id=None: SimpleNamespace(
            log_metrics=lambda metrics, step: logged.append((step, metrics)),
            close=lambda: None,
        ),
    )
    pytest.importorskip("tinker_cookbook")
    monkeypatch.setattr("tinker_cookbook.tokenizer_utils.get_tokenizer", tokenizer_for)
    return logged


def test_end_to_end_with_a_fake_client(monkeypatch, tmp_path):
    client = _FakeClient()
    logged = _patch_runner(monkeypatch, client)

    runner._train(_Config(log_path=str(tmp_path)), _Task())

    assert client.created == ["student-job", "teacher-job", "judge-job"]
    assert sorted(client.cancelled) == ["judge-job", "student-job", "teacher-job"]
    assert client.steps == ["student-job", "student-job"]
    assert client.synced == [("student-job", ("student-job:sampling:0",), None)] * 2
    assert len(client.frames) == 2
    assert client.frames[0]["processing"]["config"]["teacher_clip_negative"] == 1.0
    # Each episode is one sequence: a tool call, the tool result as context, then the answer.
    assert client.frames[0]["kwargs"]["input_ids"].shape[0] == 4
    evals = [metrics for _, metrics in logged if "test/correct" in metrics]
    assert len(evals) == 3
    assert evals[0]["test/correct"] == 1.0
    assert evals[0]["teacher/test/correct"] == 1.0
    assert evals[0]["test/turns"] == 2.0
    trained = [metrics for _, metrics in logged if "train/kl/k1_per_token" in metrics]
    assert trained and all(m["rollout/end_answer"] == 1.0 for m in trained)
    rows = [json.loads(line) for line in (tmp_path / "train_episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and "<tool_response>" in rows[0]["transcript"]


def test_token_count_mismatch_stops_before_the_optimizer_step(monkeypatch, tmp_path):
    client = _FakeClient(token_count_shortfall=1)
    _patch_runner(monkeypatch, client)

    with pytest.raises(RuntimeError, match="stopped before the optimizer step"):
        runner._train(_Config(log_path=str(tmp_path)), _Task())

    assert len(client.frames) == 1
    assert client.steps == []
    assert sorted(client.cancelled) == ["judge-job", "student-job", "teacher-job"]


def test_attached_jobs_are_left_running(monkeypatch, tmp_path):
    client = _FakeClient()
    _patch_runner(monkeypatch, client)

    runner._train(
        _Config(
            job_id="student-job", teacher_job_id="teacher-job", judge_job_id="judge-job", log_path=str(tmp_path)
        ),
        _Task(),
    )

    assert client.created == []
    assert client.cancelled == []
    assert client.steps == ["student-job", "student-job"]


def test_a_lora_student_syncs_only_its_adapters(monkeypatch, tmp_path):
    client = _FakeClient()
    _patch_runner(monkeypatch, client)

    runner._train(_Config(lora=True, log_path=str(tmp_path)), _Task())

    assert {weight_format for _, _, weight_format in client.synced} == {"lora"}


def test_teacher_scoring_uses_the_configured_request_size(monkeypatch, tmp_path):
    client = _FakeClient()
    _patch_runner(monkeypatch, client)

    runner._train(_Config(teacher_prompts_per_request=1, log_path=str(tmp_path)), _Task())

    scoring = [p for job, p, params in client.generated if params.get("prompt_logprobs") == 0]
    assert scoring and all(len(prompts) == 1 for prompts in scoring)


def _without_think_header_for_the_student(model):
    return _Tokenizer(think_off_header="Qwen3.5" not in model)


def test_teacher_window_must_cover_the_teachers_longer_turn_headers(monkeypatch, tmp_path):
    client = _FakeClient()
    _patch_runner(monkeypatch, client, tokenizer_for=_without_think_header_for_the_student)
    body = runner.job_bodies(_Config())["teacher"]
    body["sub_job_configs"][0]["inference_config"]["max_seq_len"] = 16385
    teacher = tmp_path / "teacher.json"
    teacher.write_text(json.dumps(body))

    with pytest.raises(ValueError, match="teacher max_seq_len 16385 must exceed 16441"):
        runner._train(_Config(teacher_job_config=str(teacher), log_path=str(tmp_path)), _Task())
    assert client.created == []

    # The shipped teacher window leaves room, and every scored sequence fits it.
    runner._train(_Config(log_path=str(tmp_path)), _Task())
    assert client.steps == ["student-job", "student-job"]


def test_turn_max_tokens_must_leave_room_for_a_prompt(monkeypatch, tmp_path):
    client = _FakeClient()
    _patch_runner(monkeypatch, client)

    with pytest.raises(ValueError, match="turn_max_tokens 16384 leaves no room"):
        runner._train(_Config(turn_max_tokens=16384, log_path=str(tmp_path)), _Task())
    assert client.created == []


def test_entrypoint_runs_the_wiki_search_task(monkeypatch):
    pytest.importorskip("chz")
    from examples.distillation.on_policy import train as entry

    captured = {}
    monkeypatch.setattr(entry, "train", lambda config, task: captured.update(config=config, task=task))
    config = entry.Config(config="unused")
    entry.main(config)

    assert type(captured["task"]).__name__ == "WikiSearchTask"
    assert (
        config.teacher_tau,
        config.teacher_clip,
        config.teacher_clip_negative,
        config.lr_schedule,
        config.reward_coef,
        config.max_steps,
    ) == (1.0, 3.0, 1.0, "cosine", 0.0, 40)
    assert (config.student, config.teacher, config.judge) == (
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3-30B-A3B",
    )
    assert (config.student_max_seq_len, config.teacher_max_seq_len, config.teacher_gpus) == (16384, 20480, 2)
    assert runner.job_bodies(config)["student"]["sub_job_configs"][0]["model_name"] == "Qwen/Qwen3.5-4B"
