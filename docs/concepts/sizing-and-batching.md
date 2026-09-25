# Sizing and Batching

Training capacity depends on model architecture, parameter count, precision,
sequence length, micro-batch size, optimizer state, activation memory, and
parallelism strategy.

For the current recipes, all of these live in the job-config JSON rather than on
the command line. The fields that interact:

| Field | Where |
|---|---|
| `n_gpus` | `training_config` |
| `max_seq_len` | `training_config` |
| `train_batch_size` | `training_config` |
| `max_tokens_per_mb` | `training_config.mb_spec` |
| `sp_size` | `training_config`, long context only |
| `ep_size` | `training_config`, MoE only |

DSS derives DeepSpeed's internal batch fields from the runtime topology:

```text
ds_config.train_batch_size
ds_config.train_micro_batch_size_per_gpu
ds_config.gradient_accumulation_steps
```

Don't set those fields in job configs. DSS token-budget packing splits each
request into one or more physical `[1, packed_tokens]` model calls and
token-weights their gradients before the optimizer step. Configuring DeepSpeed
accumulation greater than one would scale those already-normalized gradients a
second time, so the client rejects it.

Use `train_batch_size` for the logical training batch and
`mb_spec.max_tokens_per_mb` for the maximum token budget of each packed model
call. For MoE training, `n_gpus` must also be divisible by `ep_size`.

There is no validated sizing guide or GPU requirement matrix yet, so treat the
shipped configs as starting points and measure your own.
