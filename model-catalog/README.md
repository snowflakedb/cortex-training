# Cortex Training Model Catalog

The supported models, and a tuned job file for each model, context, GPU type and
mode. The examples and `python -m recipes.utils` build their jobs from these
files unless you pass your own job file; see
[How an example builds its job](../recipes/README.md#how-an-example-builds-its-job).

## Layout

```text
model-catalog/
├── models/<slug>/                 # one folder per model
│   ├── model.json                 # model ID, license, architecture, validation dates
│   ├── standard/<hw>-<kind>-<mode>.json   # the model's smaller max_seq_len tier
│   └── long/<hw>-<kind>-<mode>.json       # the model's larger max_seq_len tier
├── rl-pairings.json               # pairings the lifecycle sweep runs (see RL pairings)
├── schema.json                    # schema for model.json
├── validate_catalog.py
└── smoke_test_catalog.py
```

- `<slug>` is the last part of the model ID, in lowercase:
  `Qwen/Qwen3.6-35B-A3B` → `qwen3.6-35b-a3b`.
- `<hw>` is `h200`, `b200` or `b300`, and matches the file's `hardware` field.
  All files are H200 today.
- `<kind>` is `training` (one training sub-job) or `inference` (one sampling
  sub-job).
- `<mode>` is `lora` (the file has a `peft_config`) or `full`.

Each file is a complete create-job body (see the
[create-job schema](../docs/reference/rest-api.md#8-create-job-schemas)), and its
existence is the claim of support. The catalog has no RL files: an RL job
combines a model's training and inference files of the same context, hardware
and mode.

`model.json` holds:

- `model_id`
- `license` (a name and URL)
- `architecture`: `dense` or `moe`. MoE models also record
  `moe: {num_moe_layers, topk, num_experts}`.
- `last_validated`: the date of the last successful live run for each job
  file, or `null`

## Updating the catalog

1. Add or change job files in the model's folder. For a new model, add a
   folder with a `model.json`. No example code changes.
2. In `model.json`, set the file's entry in the `last_validated` map to `null`
   until a live smoke test (below) passes, then to the test date.
3. Keep [`docs/_data/models.yaml`](../docs/_data/models.yaml) (its header
   explains each field) and the table in
   [model compatibility](../docs/reference/model-compatibility.md) in sync.
4. Validate:

   ```bash
   python model-catalog/validate_catalog.py
   python -m pytest tests/test_model_catalog.py -q
   ```

The validator checks:

- file names and `hardware`, `model_name` and LoRA mode are consistent with
  `model.json`;
- each file passes `cortex-training submit --dry-run`;
- GPU counts are 1, 2, 4 or multiples of 8, and `sp_size` and
  `vllm_config.tensor_parallel_size` divide them;
- long-context files have a larger `max_seq_len` than standard ones;
- every training and inference pair builds a valid RL job, and the default
  distillation student and teacher jobs build.

## H200 SFT throughput files

These training files are measured H200 SFT throughput finals: for each file, the
job with the most tokens per GPU found without GDRCopy, on the fewest nodes it
fits. Every full-weight file keeps FP32 master weights and FP32 Adam moments.
The files marked "fork" below run on DeepSpeed `jaelee/v3/zero-memory`
(`61c11b39`, on DeepSpeed master `0fe80cb9`): its ZeRO-1/2
`optimizer_state_swap` keeps the FP32 optimizer state in pinned host memory and
streams it through the step. Every other file runs on stock DeepSpeed.
Standard files keep their tier's `max_seq_len`. Long files, full and LoRA, train
at 262,144 tokens, or at the model's own maximum when its context is shorter:
Qwen3's 40,960 and Gemma 3's 131,072.

| Model | Files |
|---|---|
| `Qwen/Qwen3-0.6B` | `standard/h200-training-full` (1 GPU), `standard/h200-training-lora` (1 GPU), `long/h200-training-full` (1 GPU), `long/h200-training-lora` |
| `Qwen/Qwen3-1.7B` | `standard/h200-training-full`, `long/h200-training-full`, `long/h200-training-lora` |
| `Qwen/Qwen3-4B-Instruct-2507` | `standard/h200-training-full` (1 GPU), `long/h200-training-full`, `long/h200-training-lora` |
| `Qwen/Qwen3-8B` | `standard/h200-training-full` (4 GPUs), `standard/h200-training-lora` (4 GPUs), `long/h200-training-full` (4 GPUs), `long/h200-training-lora` |
| `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B` | `standard/h200-training-full`, `long/h200-training-full`, `long/h200-training-lora` |
| `Qwen/Qwen3.5-9B` | `standard/h200-training-full` (4 GPUs), `long/h200-training-full`, `long/h200-training-lora` |
| `Qwen/Qwen3.8-27B` | `standard/h200-training-full`, `long/h200-training-full`, `long/h200-training-lora` |
| `Qwen/Qwen3-30B-A3B` | `standard/h200-training-full`, `standard/h200-training-lora` |
| `Qwen/Qwen3.6-35B-A3B` | `standard/h200-training-full`, `long/h200-training-full`, `long/h200-training-lora` |
| `google/gemma-3-4b-it`, `google/medgemma-4b-it`, `google/medgemma-1.5-4b-it` | `standard/h200-training-full` (1 GPU), `long/h200-training-full`, `long/h200-training-lora` |
| `google/gemma-4-31B-it` | `standard/h200-training-full`, `long/h200-training-full`, `long/h200-training-lora` |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `standard/h200-training-full`, `long/h200-training-full`, `long/h200-training-lora` |
| `zai-org/GLM-5.3-Flash-BF16` | `standard/h200-training-full` (32 GPUs, fork), `long/h200-training-full` (32 GPUs, fork) |

The other files in the table use 8 GPUs. The MedGemma files are Gemma 3 4B's
bodies under their own model ID (same architecture; not measured on the
MedGemma checkpoints).

The GLM-5.3-Flash-BF16 and Qwen3.8-Flash-Next long full files train
262,144-token rows with UCCL-EP expert dispatch. They need a UCCL build with
upstream's 18b588a ("Keep the D2H handle table in device memory", #1085). With
UCCL a3d520e alone, an allocation that finds the GPU full moves the EP handle
table to host memory, and the next expert-parallel kernel faults (Xid 31, an
illegal memory access).

The MoE LoRA files train routed-expert LoRA through `target_parameters`, with
one exception: the Qwen3.6-35B-A3B long LoRA file keeps the attention-only
adapter its long LoRA sampler serves.

Every training file in the table has an inference file of its context and mode
with the same `max_seq_len`, so each pair composes an RL job; MoE router replay
needs the two windows to match. The samplers added for this follow the model's
other inference files: a long sampler takes the long full-weight sampler's
layout, a LoRA sampler the full-weight sampler's plus the training adapter, and
a Qwen3 long LoRA sampler (40,960 tokens) the standard LoRA sampler's engine
settings on one GPU. Their `last_validated` entries are `null`.

Some LoRA samplers serve a narrower adapter than their trainer's: no catalog
sampler serves expert LoRA, so the samplers of the Nemotron long LoRA files and
of Qwen3-30B-A3B's standard LoRA file leave out the trainers'
`target_parameters` (routed and shared experts). Gemma 4's lists the
language-model projections its trainer selects with a regex. These pairs build
RL jobs, but a follow-up command finds no catalog sampler that loads their
whole checkpoints.

The Gemma 4 31B and Qwen3.8-27B full files train on one node with
`shard_dense`: each trainable weight is sharded across the node's eight GPUs
and gathered when its layer runs, so a GPU holds an eighth of the weights
instead of all of them (ZeRO-2 shards only the optimizer state and
gradients). The room goes to keeping the optimizer on the GPU (Gemma 4) and to
a 64K sequence per GPU (Qwen3.8-27B's standard file, its checkpointed
activations offloaded to host memory), and the long full files train 256K
rows on eight GPUs: Gemma 4's used 16 before and Qwen3.8-27B's 32.

They need server changes that are not yet in a released image, the stacked
branches:

- Arctic-Platform `jaelee/v3/stack/15-gemma-4` (on main `3e2ba2bf`)
- dss-platform `jaelee/v3/stack/15-gemma-4` (on main `7731f9c`)

The files marked "fork" need that DeepSpeed branch in the image; the others run
on the stock release and set no DeepSpeed fork field. An
image without these branches drops or rejects fields such as `numa_bind`,
`shard_dense`, `checkpoint_keep_attention`, `fused_qk_norm_rope`, `ac_config.core_replay`
and the `prime_rl` loader options. The
Qwen3.5 files were measured with `flash-qla` installed, and the Nemotron long
LoRA files need a CUDA build of `mamba-ssm`. Every `last_validated` entry for
these files is `null`: the live smoke test below has not run on them.

Each file is its measured job body with only these changes: `hardware` added,
`step_peak_memory_log` (a diagnostic) removed, and `model_provider` written as
`arctic` (the client sends `prime_rl`).

## Smoke testing

Print the smoke-test plan for a catalog file without submitting it: a JSON
list with one `{"job", "request"}` entry per `--job`, where `request` is the
create-job body:

```bash
python model-catalog/smoke_test_catalog.py --model-id Qwen/Qwen3.8-27B --context long --job inference-full
```

`--job` is `<kind>-<mode>` (`training-lora`, `inference-full`, ...), or
`rl-lora` / `rl-full` for the composed RL job. You can repeat it. `--context`
defaults to `standard`, and `--hardware` to `h200`.

To run a job live, add `--submit`. The job runs a minimal probe once it's
running, and is cancelled afterwards, even if a probe fails. The probe is one
generate for inference, one forward-backward for training, and both for RL. The
PAT is read only from the environment, so it stays out of shell history:

```bash
export SNOWFLAKE_HOST='ACCOUNT.snowflakecomputing.com'
export SNOWFLAKE_DATABASE='CORTEX_TRAINING_DB'
export SNOWFLAKE_SCHEMA='PUBLIC'
read -rs SNOWFLAKE_PAT; export SNOWFLAKE_PAT

python model-catalog/smoke_test_catalog.py --model-id Qwen/Qwen3.8-27B \
  --context long --job inference-full --full-context-prefill --submit

unset SNOWFLAKE_PAT
```

The default probes check only startup and basic execution. Validating a
context-length claim takes the full-context flags:

| Flag | Effect |
|---|---|
| `--full-context-prefill` | Inference: prefill `max_seq_len - 1` tokens, then generate one |
| `--full-context-training` | Training: forward-backward at `max_seq_len` for the full batch |
| `--training-step` | Also run an optimizer step |
| `--checkpoint-round-trip` | Step, save a resumable checkpoint, load it back, then delete it |
| `--memory-telemetry` | Report peak GPU and CPU memory per step |
| `--training-sp-size N` | Try a sequence-parallel size before changing a file. Adjusts `train_batch_size` to `n_gpus / N` |
| `--debug-image-tag TAG` | Use a specific backend build |

## RL pairings

`rl-pairings.json` lists the RL pairings the lifecycle sweep runs, as
`MODEL--CONTEXT--MODE` IDs such as `qwen3.6-35b-a3b--standard--lora`. Each ID uses
`models/MODEL/CONTEXT/<hw>-training-MODE.json` and the matching inference file, where `<hw>`
is the file's `hardware` in lowercase. A model is swept only once its pairings are listed here.

The sweep runs each pairing through one RL job lifecycle: create the job, sample, train two
steps, sync weights, train two more, save resumable and weights-only checkpoints, cancel,
restore a new job from those checkpoints, sample, train steps 5 and 6, cancel, and delete the
checkpoints. MoE models also run with strict router replay.
