# Model Compatibility

The [model catalog](../../model-catalog/models.json) is the source of truth for
inference and training support. It records maximum context limits and points to
the reusable SFT, RL, and inference profiles under
[`model-catalog/profiles/`](../../model-catalog/profiles/).

When a recommendation is rendered, its `maxContextTokens` value is injected as
`max_seq_len`. The generated configuration therefore shows the longest context
declared for that model and workflow. Treat GPU counts and optimization values
as starting points that might need adjustment for a specific workload.

For `Qwen/Qwen3.6-35B-A3B`, LoRA training and checkpoint save/load require the
`prime_rl` model provider. Current PrimeRL LoRA support for this MoE model is
limited to `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Routed expert LoRA is not
yet supported.

Validate catalog changes with:

```bash
python scripts/validate_model_catalog.py
python -m pytest tests/test_model_catalog.py -q
```
