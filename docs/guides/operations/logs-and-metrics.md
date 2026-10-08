# Logs and Metrics

Download complete execution logs:

```bash
cortex-training download-log JOB_ID --output-dir /path/to/logs
```

Download persisted stdout/stderr reconstructed as one file per sub-job:

```bash
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs --resume
```

`--resume` continues a stdout download already in that directory. It does not
apply to execution logs.

Download reconstructed GPU metrics:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

Metrics are written to `<output_dir>/<sub_job_id>/gpu.jsonl`.

GPU scrape lines use `kind: "gpu"` when `kind` is present; mixed streams may
also contain other kinds. `sample_ts` is an RFC3339 UTC timestamp,
`global_gpu_index` identifies a GPU within its sub-job, `gpu_util_pct` is a
gauge from 0–100, and `fb_used_bytes` / `fb_free_bytes` report framebuffer.
JSON `null` means unmeasured; it is different from a real `0.0`. Group by both
`sub_job_id` and `global_gpu_index`, not `local_gpu_index`, which repeats on
different workers.

Open a job dashboard in the terminal:

```bash
cortex-training tui JOB_ID
```

The **Logs** tab tails a running job or loads its saved console after it
finishes. The **Resource** tab plots GPU utilization and lists each GPU. Use
`1`–`5` for 5 minutes, 15 minutes, 1 hour, 6 hours, or all history; use `r` to
refresh. Resource refresh is manual and resumes from chunks already in the TUI
cache. CPU and memory are not available in this view yet.

Recipe-level metrics are also written under each recipe's `log_path`.

Snowflake profile and PAT clients also emit best-effort client-side operation
metrics over Snowflake's OTLP endpoint. Failures of essential SDK methods are
emitted automatically; successful outcomes require
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1`. See
[Client metrics](../../reference/python-sdk.md#client-metrics).
