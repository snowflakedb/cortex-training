# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Image question answering data for the vision SFT recipe, and the rows it sends.

Two sources, mixed by default:

* ``chartqa``: real charts from ``HuggingFaceM4/ChartQA`` with short answers. Train rows come from its
  ``train`` split and held-out rows from ``test``.
* ``spatial``: synthetic 3x3 grids holding three coloured shapes, asking where a shape is ("top left") or
  what a cell holds ("red circle"). Train and held-out rows use different seeds.

Tokenizing happens here, on the client, with the model's own Hugging Face processor: the chat template puts
one image placeholder in the prompt, and the processor expands it into one image token per merged patch and
returns the patches as ``pixel_values`` with their ``image_grid_thw``. A row is those token ids with a loss
mask on the answer, plus the row's images, which travel beside the tokens as one ``multi_modal_inputs``
entry per row.
"""

from __future__ import annotations

import io
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from recipes.utils import TrainSequence
from recipes.utils import collate

CHARTQA = "HuggingFaceM4/ChartQA"
CHARTQA_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
DATASETS = ("mixed", "chartqa", "spatial")

ROWS = ("top", "middle", "bottom")
COLUMNS = ("left", "center", "right")
CELLS = tuple(f"{row} {column}" for row in ROWS for column in COLUMNS)
COLORS = {
    "red": (220, 30, 30),
    "green": (30, 160, 40),
    "blue": (30, 60, 220),
    "yellow": (230, 200, 20),
    "purple": (140, 40, 180),
}
SHAPES = ("circle", "square", "triangle")


@dataclass(frozen=True)
class VisionExample:
    """One image, one question about it and the reference answer."""

    source: str
    kind: str
    question: str
    answer: str
    png: bytes

    def image(self):
        from PIL import Image

        return Image.open(io.BytesIO(self.png)).convert("RGB")


@dataclass(frozen=True)
class VisionSequence:
    """Token ids with a loss mask on the answer, and the row's images for ``multi_modal_inputs``."""

    sequence: TrainSequence
    images: dict[str, torch.Tensor]


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def spatial_image(objects: Sequence[tuple[str, str, int]], size: int = 224):
    """A white ``size`` x ``size`` image with each ``(color, shape, cell)`` drawn in its 3x3 cell."""
    from PIL import Image
    from PIL import ImageDraw

    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    cell = size // 3
    for color, shape, index in objects:
        row, column = divmod(index, 3)
        x0, y0 = column * cell + cell // 5, row * cell + cell // 5
        x1, y1 = (column + 1) * cell - cell // 5, (row + 1) * cell - cell // 5
        fill = COLORS[color]
        if shape == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=fill)
        elif shape == "square":
            draw.rectangle([x0, y0, x1, y1], fill=fill)
        else:
            draw.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=fill)
    return image


def spatial_examples(count: int, seed: int) -> list[VisionExample]:
    """Alternating "where is X?" and "what is in cell Y?" questions over fresh random grids."""
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        cells = rng.sample(range(9), 3)
        colors = rng.sample(tuple(COLORS), 3)
        objects = [(color, rng.choice(SHAPES), cell) for color, cell in zip(colors, cells)]
        color, shape, cell = rng.choice(objects)
        if index % 2 == 0:
            kind, answer = "where", CELLS[cell]
            question = (
                f"Where is the {color} {shape}? Answer with the row (top, middle or bottom) and the column "
                "(left, center or right), for example 'top left'."
            )
        else:
            kind, answer = "what", f"{color} {shape}"
            question = f"Which shape is in the {CELLS[cell]} cell? Answer with its color and shape, like 'red circle'."
        examples.append(VisionExample("spatial", kind, question, answer, _png(spatial_image(objects))))
    return examples


def chartqa_examples(split: str, count: int, seed: int) -> list[VisionExample]:
    """``count`` ChartQA rows from ``split`` with answers of at most 40 characters, in a seeded shuffle."""
    import datasets

    stream = datasets.load_dataset(CHARTQA, split=split, revision=CHARTQA_REVISION, streaming=True)
    examples = []
    for row in stream.shuffle(seed=seed, buffer_size=2000):
        answer = str(row["label"][0]).strip()
        if not answer or len(answer) > 40:
            continue
        question = f"{row['query']} Answer briefly."
        examples.append(VisionExample("chartqa", "chart", question, answer, _png(row["image"].convert("RGB"))))
        if len(examples) == count:
            return examples
    raise ValueError(f"{CHARTQA} {split} has fewer than {count} rows with a short answer")


def load_examples(dataset: str, *, n_train: int, n_eval: int, seed: int = 0) -> tuple[list, list]:
    """Train and held-out examples. ``mixed`` takes half of each from ChartQA and half from the spatial set."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    sources = ("chartqa", "spatial") if dataset == "mixed" else (dataset,)
    train: list[VisionExample] = []
    held_out: list[VisionExample] = []
    for source in sources:
        n_source_train = n_train // len(sources)
        n_source_eval = n_eval // len(sources)
        if source == "chartqa":
            train += chartqa_examples("train", n_source_train, seed)
            held_out += chartqa_examples("test", n_source_eval, seed)
        else:
            train += spatial_examples(n_source_train, seed + 1)
            held_out += spatial_examples(n_source_eval, seed + 2)
    random.Random(seed).shuffle(train)
    return train, held_out


def load_processor(model_name: str, *, max_image_pixels: int):
    """The model's Hugging Face processor, with images resized to at most ``max_image_pixels`` pixels."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    processor.image_processor.size = {"shortest_edge": 64 * 64, "longest_edge": int(max_image_pixels)}
    return processor


def build_sequence(
    example: VisionExample,
    processor: Any,
    *,
    max_seq_len: int,
    blank_image: bool = False,
) -> VisionSequence:
    """Chat prompt with the image, then the answer and the end-of-turn token, which carry the loss.

    ``blank_image`` swaps the picture for a grey one of the same size: the same tokens and the same number of
    image tokens, but nothing to look at, which is the control for how much the answer depends on the image.
    """
    from PIL import Image

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example.question}]}]
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    image = example.image()
    if blank_image:
        image = Image.new("RGB", image.size, (128, 128, 128))
    encoded = processor(text=[prompt_text], images=[image], return_tensors="pt")
    prompt = [int(token) for token in encoded["input_ids"][0].tolist()]
    tokenizer = processor.tokenizer
    answer = tokenizer.encode(example.answer, add_special_tokens=False)
    answer.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    tokens = prompt + answer
    if len(tokens) > max_seq_len:
        raise ValueError(
            f"a {example.source} row is {len(tokens)} tokens but max_seq_len is {max_seq_len}; "
            "lower max_image_pixels or raise max_seq_len"
        )
    # loss_mask[i] weights the prediction of token i + 1, so the answer is weighted from the prompt's last token.
    loss_mask = [0.0] * (len(prompt) - 1) + [1.0] * len(answer) + [0.0]
    images = {
        # The vision tower runs in bf16; casting here halves the request without changing what it computes.
        "pixel_values": encoded["pixel_values"].to(torch.bfloat16),
        "image_grid_thw": encoded["image_grid_thw"],
    }
    return VisionSequence(TrainSequence(input_ids=tokens, loss_mask=loss_mask), images)


def collate_vision(
    rows: Sequence[VisionSequence],
    *,
    pad_token_id: int,
    max_seq_len: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """:func:`recipes.utils.collate`, plus one ``multi_modal_inputs`` entry per row, in row order."""
    kwargs, context = collate([row.sequence for row in rows], pad_token_id=pad_token_id, max_seq_len=max_seq_len)
    kwargs["multi_modal_inputs"] = [row.images for row in rows]
    return kwargs, context
