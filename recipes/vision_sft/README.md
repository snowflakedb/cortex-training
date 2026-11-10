# Vision-language SFT

LoRA fine-tuning of a vision-language model on image question answering. Each training row is a chat prompt
holding one image and a question, followed by the answer, and the loss is on the answer only.

> **Experimental.** This recipe needs image support in the training service, which is not released yet:
> the trainer has to route each row's images to the model call that holds its tokens and give image tokens
> their 2-D rotary positions (M-RoPE). Until those changes ship, run it against a dev image built with them
> by passing `debug_image_tag=` (internal only).

## Quick start

From the repository root, after the [recipe prerequisites](../README.md#prerequisites):

```bash
uv pip install torchvision pillow   # the Qwen3.6 image processor needs torchvision
python -m recipes.vision_sft.train debug_image_tag=<dev image tag>
```

The default run trains `Qwen/Qwen3.6-35B-A3B` with the catalog's standard LoRA job (8 H200s, expert parallel 4,
sequence parallel 1) for one epoch over 384 rows, 48 steps of 8. It reports held-out answer NLL before and
after training, and once with blank images as a control. It then saves a weights-only checkpoint and cancels
the job.

## Data

`dataset=mixed` (the default) takes half of each split from two sources:

| Source | Train | Held out | Answers |
|---|---|---|---|
| [`HuggingFaceM4/ChartQA`](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | `train` split | `test` split | Short values from the chart, such as `51` or `Yes` |
| `spatial` (synthetic) | seed 1 | seed 2 | Three coloured shapes in a 3x3 grid: "where is the blue triangle?" → `top left`, "what is in the middle center cell?" → `red circle` |

`dataset=chartqa` or `dataset=spatial` uses one source. `spatial` needs no download.

## How a row is built

Tokenizing happens on the client with the model's own Hugging Face processor (`recipes/vision_sft/task.py`):

1. The chat template puts one image placeholder in the prompt (`<|vision_start|><|image_pad|><|vision_end|>` for
   Qwen3.6), with thinking off.
2. The processor resizes the image to at most `max_image_pixels` (default 448x448) and cuts it into 16x16
   patches. It returns them as `pixel_values` with the patch grid `image_grid_thw`, and expands the placeholder
   into one image token per 2x2 group of patches. A 448x448 image is 784 patches and 196 image tokens.
3. The answer and `<|im_end|>` are appended and carry the loss.

The request sends the tokens as usual and the images as `multi_modal_inputs`, one
`{pixel_values, image_grid_thw}` entry per row (`None` for a row without an image). The trainer encodes the
images with the model's vision tower and writes the embeddings over the image tokens. It checks that the
number of image tokens matches the images, so the processor has to be the one for the served checkpoint.

## Settings

| Setting | Default | |
|---|---|---|
| `model`, `context`, `hardware`, `lora` | `Qwen/Qwen3.6-35B-A3B`, `standard`, `h200`, `true` | Picks the catalog training file |
| `dataset` | `mixed` | `mixed`, `chartqa` or `spatial` |
| `n_train`, `n_eval`, `epochs` | `384`, `64`, `1` | Rows per run; held-out rows are trimmed to whole batches |
| `eval_every` | `0` | Also evaluate every N steps; 0 evaluates before and after only |
| `max_image_pixels` | `200704` (448x448) | Larger images are resized down to this |
| `blank_image_control` | `true` | Also score the held-out rows with grey images once |
| `save_checkpoint` | `true` | Save a weights-only checkpoint at the end |
| `lr`, `train_batch_size`, `max_seq_len` | catalog | Override the catalog values |
| `job_id` | | Attach to a running job instead of creating one |

## Results

The default run on 2026-11-10 (Qwen3.6-35B-A3B, standard LoRA, 48 steps of 8, dev image with the trainer
changes), held-out answer NLL per token:

| Held-out set | Before | Blank images | After 1 epoch |
|---|---|---|---|
| All 64 rows | 0.645 | 2.222 | **0.172** |
| ChartQA (32) | 0.665 | 2.301 | **0.281** |
| Spatial (32) | 0.618 | 2.113 | **0.020** |

Training NLL averaged 0.43 over the first 8 steps and 0.17 over the last 8, at about 4.5 s per step. The whole
run took 14 minutes, including 4 minutes for the job to start. Blank images raise the NLL more than threefold,
so the answers depend on the pictures. Part of the spatial gain is learning the answer format (`top left`).

## Limits

- Sequence parallel must be 1: a packed model call's images can't be split across sequence ranks.
- Only the Qwen3.5-MoE family (Qwen3.6-35B-A3B) builds its vision tower in training today. Dense Qwen3.5
  models load text-only.
- LoRA targets the attention projections of the language model, so the vision tower stays frozen.
- One image per row; video is refused.
- The saved checkpoint is served text-only: sampling does not take images yet.
