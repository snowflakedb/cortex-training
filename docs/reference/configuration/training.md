# Training Configuration

The recipes take two separate things:

- `config=` — *(optional)* the Snowflake connection file (account host, PAT, database,
  schema). If omitted, recipes fall back to environment variables or your
  default `connections.toml` profile. See [Set up the client](../../getting-started/setup.md).
- `job_config=` — a JSON create-job body holding the whole training
  configuration. Each recipe ships examples under its own `configs/`
  directory.

Everything that shapes the run lives in the job-config JSON:

- Model, precision, and provider (`model_name`, `dtype`, `model_provider`)
- GPU count and parallelism (`n_gpus`, `ep_size`)
- Sequence length and batch shape (`max_seq_len`, `train_batch_size`, `ds_config`)
- Optimizer and gradient clipping (`optimizer`, `gradient_clipping`)
- LoRA or full-parameter method (presence of `peft_config`)

The remaining `name=value` command-line overrides are recipe-loop settings only
-- dataset, step count, evaluation cadence, logging and Weights & Biases. Run a
recipe module with no arguments to see its full list, or read its `Config` class.

The job-config object is posted as the create-job body with one exception: each
`peft_config` it carries is validated and rewritten into its canonical form
first, and an invalid one fails before anything is submitted. Everything else,
`optimizer` included, is posted unchanged, so
[REST API section 8](../rest-api.md#8-create-job-schemas) is the authoritative
schema for its fields. The
[conversational SFT README](../../../recipes/sft/README.md)
documents the shape with every field named.
