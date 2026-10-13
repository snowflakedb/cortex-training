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

View a running job's live logs, or a finished job's saved console, in the terminal:

```bash
cortex-training tui JOB_ID
```

Recipe-level metrics are also written under each recipe's `log_path`.

Snowflake profile and PAT clients also emit best-effort client-side operation
metrics over Snowflake's OTLP endpoint. Failures of essential SDK methods are
emitted automatically; successful outcomes require
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1`. See
[Client metrics](../../reference/python-sdk.md#client-metrics).
