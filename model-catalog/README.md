# Cortex Training model catalog

This directory is the customer-facing source of truth for Cortex Training
model availability, context limits, and recommended starting configurations.

## Files

- `models.json` declares inference and training support for every model.
  Supported training models provide SFT and RL starting profiles for LoRA and
  full-parameter training.
- `profiles/*.json` contains reusable configurations expressed as arguments to
  `SubJobConfig.training_job` and `SubJobConfig.sampling_job`. The selected
  model capability supplies `model_name` and injects its `maxContextTokens` as
  `max_seq_len`; profiles must not hard-code either model-specific value.
- `schema.json` documents the public catalog format.
- `scripts/smoke_test_model_catalog.py` builds a selected recommendation from
  the catalog and can optionally submit it as a real PAT-authenticated job.

Inference support provides one maximum context length and recommended profile.
Training support requires all four SFT/RL LoRA/full recommendations. Generated
recommendations use the longest sequence declared for that model and workflow;
adjust batch and distributed settings for the workload before submission.
Every catalog sub-job must request GPUs in multiples of eight.

Qwen3.6-35B-A3B LoRA training uses the `prime_rl` model provider so checkpoints
can be saved and loaded. PrimeRL currently supports LoRA on the attention
projections (`q_proj`, `k_proj`, `v_proj`, and `o_proj`) for this model. Routed
expert LoRA is not yet supported, and Hugging Face MLP names such as
`gate_proj`, `up_proj`, and `down_proj` are no-ops on the PrimeRL model.

## Updating the catalog

1. Update the model capability and the referenced profile in the same pull
   request.
2. Include checked-in evidence paths for every changed profile. Remove
   `lastValidated` from each affected model recommendation until that exact
   model and profile completes a live smoke test, then set it to the test date.
   For a maximum-context inference claim, use `--full-context-prefill`; a short
   prompt validates startup and basic generation but not the context limit.
3. Run:

   ```bash
   python scripts/validate_model_catalog.py
   python -m pytest tests/test_model_catalog.py -q
   ```

Snowflake product documentation vendors a pinned snapshot of this directory;
it does not download mutable data during a documentation build.

## Live job validation

Dry-run the exact request body generated from a catalog recommendation:

```bash
python scripts/smoke_test_model_catalog.py \
  --model-id Qwen/Qwen3.8-27B \
  --profile inference
```

To execute a live smoke test, set the connection values through environment
variables and add `--submit`. After the job reaches `running`, inference
profiles execute and poll a one-token `generate` request, SFT profiles execute
and poll a minimal `forward-backward` request, and RL profiles execute both.
The job is then cancelled. The PAT is intentionally not accepted as a
command-line argument, which keeps it out of shell history and process
listings.

The default inference probe uses a short text prompt and validates engine
startup plus basic generation. Add `--full-context-prefill` to send exactly
`max_seq_len - 1` pre-tokenized tokens and generate one token. Only the latter
validates execution at the configured context limit.

```bash
export SNOWFLAKE_HOST='ACCOUNT.snowflakecomputing.com'
export SNOWFLAKE_DATABASE='CORTEX_TRAINING_DB'
export SNOWFLAKE_SCHEMA='PUBLIC'
read -rs SNOWFLAKE_PAT
export SNOWFLAKE_PAT

python scripts/smoke_test_model_catalog.py \
  --model-id Qwen/Qwen3.8-27B \
  --profile inference \
  --full-context-prefill \
  --submit

unset SNOWFLAKE_PAT
```

Repeat `--profile` to validate multiple recommendations serially. Supported
values are `inference`, `sftLora`, `sftFull`, `rlLora`, and `rlFull`. Each
submitted job is cancelled after its data-plane probes complete, including
when a probe or later profile fails.
