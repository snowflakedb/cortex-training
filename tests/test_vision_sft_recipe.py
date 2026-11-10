"""CPU checks for the vision SFT recipe: its synthetic data, the rows it sends, and the job it builds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
for _dependency in ("chz", "datasets", "tinker_cookbook"):
    pytest.importorskip(_dependency)

from recipes import utils  # noqa: E402
from recipes.vision_sft import task  # noqa: E402
from recipes.vision_sft import train  # noqa: E402

IMAGE = 99
END = 98
PROMPT = [5, 6, IMAGE, IMAGE, IMAGE, IMAGE, 7, 8]


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = END

    def encode(self, text, add_special_tokens=False):
        return [10 + len(word) for word in text.split()]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return END


class _Processor:
    """Stands in for the Hugging Face processor: a fixed prompt with four image tokens per image."""

    tokenizer = _Tokenizer()

    def __init__(self):
        self.images = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        assert add_generation_prompt and not tokenize and not enable_thinking
        assert messages[0]["content"][0] == {"type": "image"}
        return "prompt"

    def __call__(self, text, images, return_tensors):
        self.images.append(images[0])
        return {
            "input_ids": torch.tensor([PROMPT]),
            "pixel_values": torch.full((16, 6), float(len(self.images))),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


_PNG = task.spatial_examples(1, seed=0)[0].png


def _example(answer="top left"):
    return task.VisionExample("spatial", "where", "q", answer, _PNG)


def test_spatial_answers_name_what_is_drawn():
    for example in task.spatial_examples(40, seed=3):
        image = example.image()
        cell = image.width // 3
        if example.kind == "where":
            row, column = divmod(task.CELLS.index(example.answer), 3)
            color = example.question.split("the ", 1)[1].split()[0]
        else:
            cell_name = example.question.split("in the ", 1)[1].split(" cell")[0]
            row, column = divmod(task.CELLS.index(cell_name), 3)
            color = example.answer.split()[0]
        center = (column * cell + cell // 2, row * cell + cell // 2)
        assert image.getpixel(center) == task.COLORS[color], example


def test_spatial_examples_are_reproducible_and_alternate_questions():
    first, second = task.spatial_examples(6, seed=1), task.spatial_examples(6, seed=1)
    assert [e.png for e in first] == [e.png for e in second]
    assert [e.kind for e in first] == ["where", "what"] * 3
    assert {e.png for e in task.spatial_examples(6, seed=2)}.isdisjoint({e.png for e in first})


def test_rows_weight_only_the_answer_and_carry_their_images():
    processor = _Processor()
    row = task.build_sequence(_example("bottom center"), processor, max_seq_len=64)

    answer = [10 + len("bottom"), 10 + len("center"), END]
    assert row.sequence.input_ids == PROMPT + answer
    # loss_mask[i] weights the prediction of token i + 1.
    weighted = [row.sequence.input_ids[i + 1] for i, w in enumerate(row.sequence.loss_mask) if w > 0]
    assert weighted == answer
    assert row.images["pixel_values"].dtype == torch.bfloat16
    assert row.images["image_grid_thw"].tolist() == [[1, 4, 4]]


def test_the_blank_control_keeps_the_tokens_and_drops_the_picture():
    processor = _Processor()
    real = task.build_sequence(_example(), processor, max_seq_len=64)
    blank = task.build_sequence(_example(), processor, max_seq_len=64, blank_image=True)

    assert blank.sequence == real.sequence
    picture = processor.images[1]
    assert picture.size == processor.images[0].size
    assert picture.getextrema() == ((128, 128), (128, 128), (128, 128))


def test_rows_longer_than_the_window_are_refused():
    with pytest.raises(ValueError, match="max_image_pixels"):
        task.build_sequence(_example(), _Processor(), max_seq_len=8)


def test_collate_sends_one_image_entry_per_row_in_row_order():
    processor = _Processor()
    rows = [task.build_sequence(_example(answer), processor, max_seq_len=64) for answer in ("top left", "a b c d")]
    kwargs, context = task.collate_vision(rows, pad_token_id=0, max_seq_len=64)

    assert kwargs["input_ids"].shape == (2, len(PROMPT) + 5)
    assert [entry["pixel_values"][0, 0].item() for entry in kwargs["multi_modal_inputs"]] == [1.0, 2.0]
    assert context["loss_mask"].sum().item() == 3 + 5


def test_load_examples_rejects_unknown_datasets():
    with pytest.raises(ValueError, match="dataset must be one of"):
        task.load_examples("coco", n_train=2, n_eval=2)


def test_spatial_only_runs_need_no_download():
    train_rows, held_out = task.load_examples("spatial", n_train=8, n_eval=4)
    assert len(train_rows) == 8 and len(held_out) == 4
    assert {e.png for e in train_rows}.isdisjoint({e.png for e in held_out})


def test_the_default_job_is_the_catalog_lora_file():
    body = train.job_body(train.Config(config="unused"))
    assert body == utils.make_config("Qwen/Qwen3.6-35B-A3B", "standard", lora=True)
    (training,) = [sub for sub in body["sub_job_configs"] if sub["job_type"] == "training"]
    assert training["training_config"].get("sp_size", 1) == 1


def test_the_overrides_file_is_comment_only():
    data = json.loads((Path(task.__file__).parent / "overrides.json").read_text())
    assert set(data) == {"_comment"}


def test_heldout_nll_is_token_weighted_and_split_by_source(monkeypatch):
    processor = _Processor()
    examples = [
        task.VisionExample("chartqa", "chart", "q", "a b", _PNG),
        task.VisionExample("spatial", "where", "q", "top left", _PNG),
    ]
    rows = [task.build_sequence(example, processor, max_seq_len=64) for example in examples]

    def fake_forward(client, job_id, kwargs):
        assert len(kwargs["multi_modal_inputs"]) == len(kwargs["input_ids"])
        # -1 per labelled token in the first row, -2 in the second.
        scale = torch.tensor([[1.0], [2.0]])[: len(kwargs["input_ids"])]
        return {"logprobs": -(kwargs["labels"] != -100).float() * scale}

    monkeypatch.setattr(train, "forward", fake_forward)
    metrics = train.heldout_nll(None, "job", rows, examples, batch_size=2, pad_token_id=0, max_seq_len=64)

    assert metrics == {"valid_nll": pytest.approx(1.5), "valid_nll_chartqa": 1.0, "valid_nll_spatial": 2.0}
