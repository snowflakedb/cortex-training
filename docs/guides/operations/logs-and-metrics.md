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

`--resume` continues a stdout download already in that directory, including
one that was interrupted, and fetches only what is missing. It does not
apply to execution logs. On a terminal, stdout and metrics progress is
one stderr line that updates in place. A redirected stderr gets one final line
per sub-job. The JSON result stays on stdout.

Download the reconstructed runtime-metrics stream:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

Metrics are written to `<output_dir>/<sub_job_id>/metrics.jsonl`. The JSON
result's `filename` and `saved_path` use that name. This replaces `gpu.jsonl`
and is a breaking change: a `gpu.jsonl` already in the directory is left
untouched and is stale after you upgrade. The file mixes GPU rows, host rows,
and any later row kinds; download keeps unknown `kind` values as stored. The
Resource tab still plots GPU rows. It reads a committed `metrics.jsonl` when
one is present, and otherwise a committed `gpu.jsonl` from an earlier
download. It does not combine or delete those files.

Open a job dashboard in the terminal:

```bash
cortex-training tui JOB_ID
```

The **Logs** tab tails a running job or loads its saved console after it
finishes; the saved console is cached locally, so reopening the job only
fetches what is new. The **Resource** tab plots GPU utilization and lists each GPU. Use
`1`–`5` for 5 minutes, 15 minutes, 1 hour, 6 hours, or all history; use `r` to
refresh.

Recipe-level metrics are also written under each recipe's `log_path`.

Snowflake profile and PAT clients also emit best-effort client-side operation
metrics over Snowflake's OTLP endpoint. Failures of essential SDK methods, and
the outcome and timing of every download, are emitted automatically; other
successful outcomes require `CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1`. See
[Client metrics](../../reference/python-sdk.md#client-metrics).
