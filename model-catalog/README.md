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

## Checkpoint restart suite

Run the same lifecycle checks for any training/inference config pair:

```bash
python model-catalog/job_lifecycle_suite.py \
  --training-config /path/to/training.json \
  --inference-config /path/to/inference.json \
  --connection-config /path/to/connection.json --submit
```

Without `--submit`, this prints the derived RL request without allocating GPUs.
The suite probes sampling, runs two forward-backward/step pairs, syncs weights,
probes sampling again, runs two more training steps, and saves resumable and
weights-only checkpoints. It cancels the original job, creates a restored
training/sampling job, probes sampling, checks two more steps continue at 5 and
6, and cancels the restored job.

Training rows use the full configured `max_seq_len` and `train_batch_size`.
Sampling uses `max_seq_len - 1` prompt tokens and exactly one generated token.
Every training loss must be finite before its optimizer step.

MoE jobs enable strict router replay, bootstrap it on both jobs, and capture
fresh full-length rollouts before every training step. Training uses the
captured tokens, sample IDs, and sampling sub-job metadata. Unused standalone
sampling captures are discarded. Matching training/sampling context limits
are required.

The base files are not edited. The derived MoE RL request disables scalar-only
fused cross-entropy so GRPO can obtain per-token logprobs: with EP > 1 it sets
`model_options.lm_head.cross_entropy=false` and aligns existing legacy flags;
with EP1 it uses `prime_rl.fused_cross_entropy=false`.

On success or failure, jobs are stopped before the suite deletes its recorded
checkpoints and verifies their absence. Checkpoints are retained if a job cannot
be stopped; cleanup errors fail the run without hiding the original error.
Reports and failure execution logs are stored under `results/<timestamp>/`;
`--output-dir` selects a new report directory. `--poll-timeout` overrides the
operation/startup timeout, and `--weight-sync-bucket-size` sets transfer bucket
bytes.

Local regression tests create no jobs:

```bash
python -m pytest tests/test_job_lifecycle.py -q
```
<<<<<<< Updated upstream
=======

## Reproducible lifecycle sweep

From this directory, run every config in `rl-pairings.json` with three explicit
inputs:

```bash
python3 run_lifecycle_sweep.py \
  --image-tag BASE_IMAGE_TAG \
  --max-parallel-gpus 128 \
  --connection-config /path/to/connection.json
```

This submits live jobs. Add `--dry-run` to print the complete GPU plan without
connecting or allocating resources. `--output-dir results/RUN_NAME` selects a
new output directory. `--previous-run results/PREVIOUS_RUN` adds a comparison;
the previous and new runs must be sibling directories.

Add `--model MODEL` to run only that exact catalog model name; repeat the flag
to select multiple models, for example `--model gemma-3-4b-it --model medgemma-4b-it`.
All context/mode pairings for those models retain their catalog order. Unknown
or empty model names are rejected before an output directory or live connection
is created. Omitting the flag still runs the full matrix. The manifest records
`selected_models` in catalog order; repeat those same `--model` flags when
replaying a subset from archived source.

Use `--config MODEL--CONTEXT--MODE` instead of `--model` to select individual
pairings, for example `--config gemma-4-31b-it--standard--full`. Repeat the
flag for multiple configs. Unknown or empty IDs are rejected before allocation.
The manifest records `selected_configs` in catalog order; use those same
`--config` flags when replaying an exact config subset.

Add `--budget-scope sweep` when the GPU limit applies only to this sweep,
not unrelated jobs in the account. In that mode, all training and sampling
reservations count against the limit, and each backfill is additionally capped
by available capacity in the requested hardware pool. Less available capacity
means fewer concurrent configs; configs that do not fit remain queued until
capacity becomes available. The chosen scope is archived in the manifest and
ledger and must be retained when replaying the command.

Training and sampling GPUs are reserved together, including during checkpoint
restore. The default `--budget-scope account` also accounts conservatively for account jobs
already active at startup. Capacity is sampled before backfilling, and the
external GPU charge never drops below its confirmed high-water floor during
the sweep. Higher measured usage is charged immediately; ambiguous ownership
or over-budget samples pause admission. Known retired sweep allocations are
discounted only when confirming the floor, not when admitting jobs, so stale
capacity cannot permanently inflate it. Measured usage, the charged floor,
reservations, and observed peaks are recorded separately in the ledger.
Do not submit unrelated jobs to the same account
during a sweep. Only one sweep command can run for an operating-system user at
a time, including archived replays.
Configs are not tuned or retried. The base image tag is pinned on both original
and restored requests, and its resolved Hopper variant is verified before
lifecycle operations.

The output includes exact composed requests, input/source snapshots and SHA256
fingerprints, Python/package versions, a GPU reservation ledger, per-config
phase reports and failure logs, and an offline `index.html`. Connection contents
are never copied; only a fingerprint is recorded. The connection file must be
outside archived source/config files, and connection environment variables do
not override it. Source or connection changes
during a sweep stop further admission, and each child verifies its saved request
and connection fingerprints before submitting. Archived source fingerprints are
also checked before admitting each child. Cleanup must be confirmed before
reservations are released. An unrecorded create or unconfirmed cleanup stops further admission
and retains its reservation for manual reconciliation. Ctrl-C or SIGTERM stops
admission and waits for active suites to finish and clean up. Children execute
the archived helper, SDK, suite, and config utilities, not live source files.

Rebuild the compact, searchable red/green report without credentials or jobs:

```bash
python3 render_report.py results/RUN_NAME
```

The HTML embeds styles, scripts, phase details, and failure excerpts. Keep the
results directory for linked raw reports and full execution logs. Reusing the
same inputs reproduces the requests and test procedure, not deterministic
service timings or outcomes. To replay the saved source/configs rather than
the current checkout, run the snapshotted runner with a new output directory
and the same three inputs, budget scope, and model selection:

```bash
python3 results/RUN_NAME/inputs/model-catalog/run_lifecycle_sweep.py \
  --image-tag BASE_IMAGE_TAG --max-parallel-gpus 128 \
  --connection-config /path/to/connection.json \
  --output-dir results/REPLAY_NAME
```

Use the Python/package versions recorded in the manifest. The snapshotted
`render_report.py` can also rebuild the report. Local sweep/report tests
allocate no GPUs:

```bash
python3 -m pytest ../tests/test_lifecycle_sweep.py ../tests/test_lifecycle_report.py -q
```

Optional browser regression tests use synthetic results and allocate no GPUs.
Install Playwright and Chromium once, then test the offline report at desktop
and mobile sizes:

```bash
uv pip install playwright
python3 -m playwright install chromium
python3 -m pytest ../tests/test_lifecycle_report_browser.py -q
```

Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to use a preinstalled Chromium binary.
These tests exercise tile geometry, status colors, accessible labels, filters,
keyboard navigation, and failure details without contacting the service.
>>>>>>> Stashed changes
