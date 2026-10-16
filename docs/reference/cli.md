# Cortex Training CLI and Client Reference

The `cortex-training` command and `cortex_training` Python package provide the
supported command-line and SDK interfaces for Cortex Training.

`ct` is installed as an alias for `cortex-training` and supports the same
commands and flags. You can replace `cortex-training` with `ct` in any example
below, such as `ct login config.json` or `ct --job JOB_ID step`.

## Installation

Requires Python 3.10+. Installing the package gives you the `cortex-training`
CLI, the `cortex-training tui` job dashboard, and the `cortex_training` Python SDK.

This project uses [uv](https://docs.astral.sh/uv/); `pip` works in place of
`uv pip` throughout if you prefer.

Install straight from the repository:

```bash
uv pip install git+https://github.com/snowflakedb/cortex-training.git
```

Or from a local checkout:

```bash
git clone https://github.com/snowflakedb/cortex-training.git
cd cortex-training
uv pip install .            # add -e for an editable/dev install
```

Verify the install:

```bash
cortex-training --help
cortex-training tui --help
```

## Jobs And Training Loops

A job is a server-side lifecycle resource for remote model workers, not a
training script. Its sub-jobs define the model, GPU requirements, and training
or sampling configuration. Submitting a job starts those workers; once the
job is running, you send batches, optimizer steps, or generation requests to
its job ID.

Jobs separate worker setup and GPU allocation from the code driving the
experiment. You can reuse running workers across requests, inspect status and
logs from another CLI process, and cancel the job to release capacity. A
combined training and sampling job also lets an RL loop synchronize weights
between its sub-jobs.

The CLI and `CortexTrainingClient` operate on the same jobs; they are not
alternative execution models. Use the CLI for setup, inspection, and individual
operations. Use a Python loop or a [recipe](../../recipes/README.md) for dataset
iteration, batching, rewards, evaluation, and repeated calls. The client loop
creates a job or uses an existing job ID, submits operations, and polls their
results. `submit --wait` waits for workers to be running; it does not upload or
execute your Python loop.

See [jobs and sub-jobs](../concepts/jobs-and-subjobs.md) and the
[Python SDK reference](python-sdk.md#job-lifecycle) for details.

## Quick Reference

Global flags such as `--config`, `--job`, and `--compact` go **before** the
subcommand. `login` also accepts its own `--config` after the subcommand.
`--job-id` is an alias for `--job`:

```bash
cortex-training --config config.json list
cortex-training --job JOB_ID step
cortex-training checkpoints JOB_ID
```

The data-plane actions `fwd-bwd`, `step`, `load`, `generate`, and `weight-sync`
require the global `--job JOB_ID` option. Their `--help` usage lines show its
placement. Management commands such as `get` and `checkpoints` instead take a
positional `JOB_ID` after the subcommand.

### [Connection](#connection-config)

```bash
cortex-training --connection training list    # Use a Snowflake profile
cortex-training list                          # Use the configured default profile
cortex-training login config.json             # Remember config for future commands
cortex-training login --config config.json    # Equivalent login syntax
cortex-training --config config.json list     # Use config for one command
```

### [Submit Jobs](#submit-a-job)

```bash
cortex-training submit examples/api/training.json
cortex-training submit examples/api/sampling.json
cortex-training submit job.json --dry-run     # Validate without submitting
cortex-training submit job.json --wait        # Wait until running, not finished
cortex-training submit job.json --comment "qwen3-8b lora sweep #3"  # Name the job
cortex-training submit - < job.json           # Read JSON from stdin
```

### [Manage Jobs](#manage-existing-jobs)

```bash
cortex-training list
cortex-training list --status running
cortex-training list --status active,failed --user JDOE --since 2026-10-01 --limit 20
cortex-training list --all                    # Every schema you can see
cortex-training get JOB_ID
cortex-training wait JOB_ID                   # Wait until running, not finished
cortex-training cancel JOB_ID
cortex-training checkpoints JOB_ID
cortex-training capacity                      # Every GPU type your account supports
cortex-training capacity --hardware H200
```

### [Training And Generation](#run-a-forward-backward-smoke-test)

```bash
cortex-training --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
cortex-training --job JOB_ID step             # Default learning rate: 1e-4
cortex-training --job JOB_ID step --lr 2e-5
cortex-training --job JOB_ID generate examples/api/generate.json
cortex-training --job JOB_ID weight-sync
cortex-training --job JOB_ID weight-sync --weight-format lora
```

See also [generation payloads](#run-a-generate-smoke-test) and
[weight-sync routing](#sync-training-weights).

### [Load Checkpoints](#load-a-checkpoint-into-a-running-job)

```bash
cortex-training --job JOB_ID load CHECKPOINT_ID
cortex-training --job JOB_ID load CHECKPOINT_ID --source-job-id SOURCE_JOB_ID
cortex-training --job JOB_ID load CHECKPOINT_ID --target-sub-job-id JOB_ID:training:0
cortex-training --job JOB_ID load CHECKPOINT_ID --no-poll
```

### [Logs And Metrics](#job-tui)

```bash
cortex-training tui                          # Open job list (a: all schemas)
cortex-training tui --schema-only            # Job list of the configured schema only, no a
cortex-training tui JOB_ID                   # Open job logs
cortex-training tui --debug                  # Synthetic jobs, no connection
cortex-training download-log JOB_ID --output-dir ./logs
cortex-training download-log JOB_ID --log-type stdout --output-dir ./logs
cortex-training download-metrics JOB_ID --output-dir ./metrics
```

See [execution logs](#download-execution-logs),
[persisted stdout](#download-persisted-stdout), and [GPU metrics](#download-gpu-metrics)
for download paths and output fields.

### [Output And Help](#json-output-and-help)

```bash
cortex-training --compact list
cortex-training get JOB_ID | jq '.sub_jobs'
cortex-training --help
cortex-training fwd-bwd --help
```

### Defaults And Waiting

| Command | Default behavior | Alternative |
|---------|------------------|-------------|
| `submit` | Return after submission | `--wait` waits until running; `--dry-run` validates without submitting |
| `wait JOB_ID` | Wait until running, not until training finishes | Use `get JOB_ID` to inspect current status |
| `fwd-bwd`, `generate` | Poll the submitted request until completion | Set top-level `"poll": false` in the input JSON |
| `step` | Poll until completion; learning rate `1e-4` | Set `--lr`; polling cannot be disabled |
| `load`, `weight-sync` | Poll the submitted request until completion | `--no-poll` returns without waiting for the result |
| `capacity` | Query each GPU type the account supports | Select one with `--hardware` |
| `download-log`, `download-metrics` | Write under the current directory | Set `--output-dir` |

## Detailed Reference

- [Connection config](#connection-config), [login](#login), and [environment variables](#environment-variables)
- [Submit](#submit-a-job), [manage jobs](#manage-existing-jobs), and [GPU capacity](#show-current-gpu-capacity)
- [Forward-backward and optimizer steps](#run-a-forward-backward-smoke-test)
- [Load checkpoints](#load-a-checkpoint-into-a-running-job) and [initialize sampling](#start-sampling-from-a-training-checkpoint)
- [Generate](#run-a-generate-smoke-test) and [sync weights](#sync-training-weights)
- [Download logs](#download-execution-logs), [stdout](#download-persisted-stdout), [metrics](#download-gpu-metrics), and [roles and access](#roles-and-access)
- [Job TUI](#job-tui), [JSON output and help](#json-output-and-help), and [troubleshooting](#troubleshooting)

`cortex-training` submits and manages Cortex Training jobs through the Cortex
Training REST endpoint.
The Snowflake-native workflow uses the same connection profiles as the Python
Connector and Snowflake CLI:

```toml
# ~/.snowflake/connections.toml
[training]
account = "ORG-ACCOUNT"
host = "ACCOUNT.snowflakecomputing.com"
user = "USER"
authenticator = "programmatic_access_token"
token = "YOUR_PROGRAMMATIC_ACCESS_TOKEN"
database = "CORTEX_TRAINING_DB"
schema = "PUBLIC"
```

Protect files containing credentials, then select the named profile:

```bash
chmod 600 ~/.snowflake/connections.toml
cortex-training --connection training list
```

`cortex-training list` with no connection arguments uses the
Connector-configured default profile. Set
`SNOWFLAKE_DEFAULT_CONNECTION_NAME=training`, configure
`default_connection_name = "training"` in Snowflake's `config.toml`, or name
the connection `[default]`.

Profile lookup is a fallback. An explicit or remembered legacy JSON config, or
a complete direct `--base-url` / `--host` + `--pat` connection, continues to
win. This preserves existing scripts.

### Legacy Connection Config

For Snowflake PAT auth, use `host` for the account hostname. Do not use
`base_url` for Snowflake PAT auth.

```json
{
  "host": "ACCOUNT.snowflakecomputing.com",
  "pat": "YOUR_PROGRAMMATIC_ACCESS_TOKEN",
  "database": "CORTEX_TRAINING_DB",
  "schema": "PUBLIC",
  "endpoint": "cortex-training",
  "poll_interval": 0.5,
  "poll_timeout": 1800.0,
  "verify_ssl": true
}
```

If you prefer not to store the PAT in the file, omit `pat` and set it in the
shell instead:

```bash
export CORTEX_TRAINING_PAT='YOUR_PROGRAMMATIC_ACCESS_TOKEN'
```

To target a local or otherwise compatible server instead of a Snowflake
account, use `base_url` with an explicit scheme. This skips PAT auth:

```json
{
  "base_url": "http://localhost:8084",
  "database": "MY_DB",
  "schema": "PUBLIC",
  "endpoint": "cortex-training"
}
```

### Legacy Login

Login validates the config and stores only the config path, not the config
contents:

```bash
cortex-training login config.json
cortex-training login --config config.json
```

Provide exactly one config path, either positionally or with `--config` after
`login`. Both forms validate and remember the same file. Login requires an
explicit path even when a global `--config`, `CORTEX_TRAINING_CONFIG`, or a
previous login is available.

The login state is written to `~/.config/cortex-training/login.json` by default,
or `$XDG_CONFIG_HOME/cortex-training/login.json` when `XDG_CONFIG_HOME` is set.

You can bypass login for one command with:

```bash
cortex-training --config config.json list
```

or by setting:

```bash
export CORTEX_TRAINING_CONFIG=/path/to/config.json
```

Explicit CLI flags override config values.

### Manage Existing Jobs

```bash
cortex-training list
cortex-training list --status running
cortex-training list --status active,failed --user JDOE --since 2026-10-01 --limit 20
cortex-training list --all
cortex-training get JOB_ID
cortex-training checkpoints JOB_ID
cortex-training cancel JOB_ID
cortex-training wait JOB_ID
```

`list` takes these filters:

| Flag | Takes | Notes |
|---|---|---|
| `--all` | nothing | lists jobs across every schema in the account that you can see, not just the configured one; see below for when it works |
| `--status` | comma list of `pending`, `placing`, `initializing`, `running`, `failed`, `cancelled`, `terminated`, `active`, `terminal`, `done` | `active` is `pending` through `running`; `terminal` is `cancelled`, `terminated`, and `failed`; `done` is `cancelled` and `terminated`, plus, without `--all`, the `done` that older jobs show; `pending` does not include `placing`. Without `--all`, matched against a row's status in any form the server sends, including `JOB_STATE_RUNNING` and aliases such as `canceled`, `completed`, `succeeded`, `error`, `queued`, `creating` |
| `--user` | user name | SQL name rules: unquoted means its uppercase form, `"quoted"` means exactly that name. Without `--all`, an unknown user lists no jobs; with `--all`, a name that matches no active user in the account is an error |
| `--since` | `YYYY-MM-DD` | a UTC day; keeps jobs created on or after it |
| `--until` | `YYYY-MM-DD` | a UTC day; keeps jobs created on or before it, so `--since D --until D` means "on day D" |
| `--limit` | number | prints at most this many of the newest matching jobs by `created_at`; a job listed without one counts as oldest |

`list` prints jobs oldest first by `created_at`, so the newest job that matches
is printed last, at the bottom of the terminal:
`cortex-training list --status terminal | jq '.jobs[-1]'` is the most recently
created job that has ended. A job listed without a `created_at` counts as the
oldest and is printed first. `--limit N` prints the last N jobs of that order.

Without `--all`, the CLI fetches the configured schema's jobs and applies the
filters itself, leaving out a job with no `created_at` when `--since` or
`--until` is set. With `--all`, the server applies them across the account
([REST API reference section 5.3](rest-api.md#across-the-account---get-apiv2cortex-training)).
The server no longer takes `done`, so the CLI sends `cancelled,terminated` for
it, and the `done` of older jobs matches no `--status` there. Either way, a
job whose status can't be read (`UNKNOWN`) matches no `--status` word.

`list --all` works once the service has account-level listing and Snowflake
has enabled it for the account. Until then it exits with status 1 and says why:

```text
error: account-level job listing is not available on this service yet: GET /api/v2/cortex-training answered 404 with no error body, so the route is not deployed
error: account-level job listing is not enabled for this account (HTTP 404); Snowflake enables it per account
```

A 404 with some other body, such as a proxy's page, says `answered 404
without the service's not-enabled error; the route may not be deployed yet, or
a proxy in front of the service answered` instead. Without `--all`, `list`
still lists the configured schema in these cases.

The service can stop accepting a page token partway through a long listing,
for example after it rotated the key that seals its tokens. `list --all` then
starts over from the first page once by itself; if the token is refused again,
it exits with status 1 and says so (`could not open page_token ... a second
time`). Run the command again.

Where Cortex Training or its REST API is off for the account, which stops every
Cortex Training command, or where something refuses the request, `list --all`
exits with one of these:

```text
error: account-level job listing was refused (HTTP 403): <the service's reason>
error: account-level job listing was refused (HTTP 403) with no error message: a network policy or a proxy may have blocked the request, or Cortex Training is not enabled for this account or its trial has ended
error: account-level job listing is unavailable: the Cortex Training REST API is turned off for this account (HTTP 405)
```

The service's reason for a 403 is that Cortex Training is not enabled for the
account, that its trial has ended, or, with `--status`, that the service was
refused a status lookup. Each message starts with
`[snowflake request id: ...]` when the service sent one, and the service's
error body, if any, follows on a `response body:` line.

With `--status`, the service fails a page only after waiting up to 300 seconds
on a status lookup that is failing, so `list --all --status ...` retries a
failing page twice instead of ten times. It can still take about 15 minutes to
fail; press Ctrl-C to stop it sooner.

With `--all`, a `--user` that matches no active user in the account, because
the name is mistyped, in the wrong case, or the user was dropped, fails too:

```text
error: [snowflake request id: ...] --user jdoe: no active user in this account has that name (HTTP 400: submitted_by user not found or not authorized); an unquoted name means JDOE, so quote a lowercase or mixed-case name to match it exactly: --user '"jdoe"'
response body: {"message": "submitted_by user not found or not authorized", ...}
```

A bad `--status`, `--since`, `--until`, or `--limit`, or an empty `--user`,
exits with status 2 before connecting.

`get` fetches current job details; `checkpoints` lists saved checkpoints.
`wait` waits for the job to reach **running**, not for training to finish.
See [Manage Jobs](../guides/operations/manage-jobs.md) for the operational workflow.

### Show Current GPU Capacity

Print the caller account's reserved GPU capacity and current usage, separated
by hardware:

```bash
cortex-training capacity
cortex-training capacity --hardware H200
```

The default command queries `H200`, `B200`, and `B300` independently and
prints a `capacity_by_hardware` map with one entry per GPU type your account
supports. Each entry includes `has_reservation`, `max_total_gpus`,
`reserved_gpus`, `in_use_gpus`, `pending_gpus`, and `available_gpus`. Types the
service rejects as unsupported for the account (error code `517604`) are listed
in `unsupported_hardware` instead, which is empty when every type is supported:

```json
{
  "capacity_by_hardware": {
    "B300": {"available_gpus": 8, "has_reservation": true, "max_total_gpus": 8, "...": "..."},
    "H200": {"available_gpus": 16, "has_reservation": true, "max_total_gpus": 16, "...": "..."}
  },
  "unsupported_hardware": ["B200"]
}
```

Any other error still fails the command, and so does an account that supports
none of the types.

Each hardware lookup is attempted once with at most a 10-second connect timeout
and a 30-second read timeout; a shorter SDK timeout still wins. A silent
network, proxy, or service hop therefore fails instead of leaving the command
blocked indefinitely. The three lookups run serially; use `--hardware` when
only one type is needed.

`--hardware` keeps the single-capacity response shape for one GPU type. Asking
for a type your account does not support returns the service's error, which
names the supported types.

`max_total_gpus` is the canonical ceiling and supersedes the deprecated
`reserved_gpus`. `in_use_gpus` counts only GPUs the account holds; queued work
is reported separately in `pending_gpus`. See
[REST API reference section 5.4](rest-api.md#54-capacity---get-capacity) and
[GPU hardware](../concepts/hardware.md).

#### Capacity Breakdown

`--breakdown` splits `in_use_gpus` and `pending_gpus` by sub-job type
(`job-type`) or by submitting user (`submitted-by`). Repeat it for both:

```bash
cortex-training capacity --hardware H200 --breakdown job-type --breakdown submitted-by
```

Each capacity object then carries a `breakdown`, as
`get_capacity_breakdown()` returns it:

```bash
cortex-training capacity --hardware H200 --breakdown submitted-by \
  | jq -c '.breakdown.by_submitted_by.groups[]'
```

```json
{"kind":"USER","submitted_by":"YOUR_USER","is_caller":true,"in_use_gpus":24,"pending_gpus":0,"job_count":2}
{"kind":"USER","submitted_by":"SOME_USER","is_caller":false,"in_use_gpus":8,"pending_gpus":16,"job_count":2}
{"kind":"OTHER","submitted_by":null,"is_caller":false,"in_use_gpus":8,"pending_gpus":0,"job_count":1}
{"kind":"UNATTRIBUTED","submitted_by":null,"is_caller":false,"in_use_gpus":4,"pending_gpus":0,"job_count":1}
```

Until the breakdown is enabled for your account, each dimension reports
`DISABLED`. See [REST API reference section 5.4](rest-api.md#breakdown---breakdown)
for every field and status.

### Submit A Job

The submit command expects a CreateJob JSON body:

```json
{
  "hardware": "H200",
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": "gpt2",
      "inference_config": {
        "max_seq_len": 128,
        "n_gpus": 1
      }
    }
  ]
}
```

Optional `hardware` is `H200`, `B200`, or `B300`. Omit it to use H200. Every
sub-job in the job uses that type. See [GPU hardware](../concepts/hardware.md).

Submit it:

```bash
cortex-training submit job.json
cortex-training submit job.json --wait
cortex-training submit job.json --dry-run
cortex-training submit job.json --comment "qwen3-8b lora sweep #3"
```

Without `--wait`, submission returns without waiting for the job to run.
`--wait` waits until **running**, not until training finishes. `--dry-run`
validates and prints the request body without sending it, applying the same
checks as a live submit — so the printed body already shows any `peft_config`
in its canonical form. A `log_probability` or `log_prob` sub-job is rejected
before send, including on `--dry-run`, with
`log_probability sub-jobs are not currently supported`.

`--comment TEXT` names the job. It sets the body's `comment`, replacing one the
JSON already has. A blank comment (empty, or only whitespace), from TEXT or
from the JSON, names nothing and is not sent, so `--comment ''` removes the
JSON's. The text is stored with the job, returned by `get` and `list`, and
shown as the job's name in `cortex-training tui`. It holds at most 1,024 bytes
of UTF-8 and can't be changed after the job is created. A longer comment, or
TEXT holding bytes that aren't UTF-8, fails before anything is sent, including
on `--dry-run`. A model deployment is submitted without the comment, with a
warning on stderr, because the server rejects one there;
[REST API section 5.1](rest-api.md#comment) says which requests count as one.

The repo includes a Prime-RL/Qwen3.6 training example:

```bash
cortex-training submit examples/api/training.json
cortex-training submit examples/api/sampling.json
```

That file creates a training sub-job for `Qwen/Qwen3.6-35B-A3B` with
`training_config.model_provider` set to `prime_rl`.

### Run A Forward-Backward Smoke Test

After the training job is running, send one tokenized training batch:

```bash
cortex-training --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
cortex-training --job-id JOB_ID step
```

The fwd-bwd JSON is human-readable: it contains text samples, tokenizer
settings, batch size, sequence length, `position_ids`, and label generation
settings. The CLI tokenizes the text, builds tensor kwargs, serializes
`{"args": (), "kwargs": ...}` as a DSSST1 safetensors frame (see
[REST API reference section 9](rest-api.md#9-dssst1-binary-wire-protocol)),
submits `forward_backward`, and polls the request by default. Set `"poll": false` in
the JSON to print only the submitted `request_id`.

Text payloads require `transformers` in the client environment. You can also
provide pre-tokenized tensor data directly under `payload.kwargs` for fully
offline use.

Run an optimizer step after fwd-bwd with:

```bash
cortex-training --job-id JOB_ID step
cortex-training --job-id JOB_ID step --lr 2e-5
```

When omitted, `--lr` defaults to `1e-4`.

### Load A Checkpoint Into A Running Job

After a job has already been created and reached running, load a checkpoint into
that existing job with:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID
```

To load from another job's checkpoint store:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID --source-job-id SOURCE_JOB_ID
```

To load into a specific training sub-job (useful for multi-sub-job sessions):

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID --target-sub-job-id JOB_ID:training:0
```

When `--target-sub-job-id` is omitted, the service routes the load to the
session's training sub-job. Sampling sub-jobs are not valid targets.

#### Discovering Sub-Job IDs

To find the training sub-job and its GPU count:

```bash
cortex-training get JOB_ID | jq '.sub_jobs[] | select(.job_type=="training") | {sub_job_id, n_gpus: .training_config.n_gpus}'
```

`get` takes the job id as a positional argument, so `--job-id` is not used here.
The global `--job-id` option is only for the data-plane subcommands that have no
positional job id (`fwd-bwd`, `step`, `load`, `generate`, `weight-sync`).
For Python sub-job discovery, see the
[runtime load API reference](rest-api.md#64-runtime-load---post-job_idload).

#### When to Use load --target-sub-job-id

A job has at most one training sub-job, so `load --target-sub-job-id` can be
omitted. Use it when you want explicit control over which sub-job receives the
checkpoint rather than relying on the server's default resolution; it must name a
training sub-job. `weight-sync` takes its own `--target-sub-job-id`, which names
sampling sub-jobs and can be repeated — see
[Sync Training Weights](#sync-training-weights).

#### DP Size Compatibility

When changing `n_gpus` from the checkpoint's source job, create the target
training sub-job with `"load_optimizer_states": false` in its `training_config`.
This cannot be changed at load time. See
[DP size compatibility](rest-api.md#dp-size-compatibility) for the constraint.

This is the runtime load path. Create-time initialization from a checkpoint
saved by Cortex Training still uses
[`source_checkpoint_info`](rest-api.md#65-create-time-checkpoint-initialization)
in the submitted sub-job JSON. To start a job from weights you trained
elsewhere, see [Start a Job From Your Own Weights](../guides/training/external-weights.md).

`load` polls until the request completes by default. Pass `--no-poll` to return
the request metadata without waiting for the result.

### Start Sampling From A Training Checkpoint

Sampling requires a `weights-only` checkpoint and a new sampling job with
`source_checkpoint_info` in its submitted JSON; `load` targets existing training
jobs, not sampling jobs. Resumable checkpoints are not directly loadable by the
sampling runtime.

See [Serve a Training Checkpoint](../guides/inference/serve-checkpoint.md)
for the recipe workflow, or
[Start sampling from saved weights](rest-api.md#134-start-sampling-from-saved-weights)
for the Python save-and-create example.

### Run A Generate Smoke Test

After a sampling job is running, send readable prompts with sampling
parameters:

```bash
cortex-training --job-id JOB_ID generate examples/api/generate.json
```

The generate JSON contains `prompts`, optional `sampling_params`, and optional
`routing_key` / `strict` fields. `sampling_params` may be one object applied to
all prompts or a list of objects/nulls aligned with `prompts`. A flat list of
integers such as `"prompts": [1, 2, 3]` is one pre-tokenized prompt, not three;
use the nested form `[[1, 2], [3, 4]]` for a batch. The CLI submits `generate`
and polls the request by default. Set `"poll": false` to print only the
submitted `request_id`.

### Sync Training Weights

For an RL-style job with one training and one sampling sub-job, sync training
weights into sampling with:

```bash
cortex-training --job-id JOB_ID weight-sync
```

By default this syncs from `JOB_ID:training:0` to `JOB_ID:sampling:0`, routes
the operation through `JOB_ID:training:0`, and polls for completion. Override
sub-job ids when needed:

```bash
cortex-training --job-id JOB_ID weight-sync \
  --source-sub-job-id JOB_ID:training:0 \
  --target-sub-job-id JOB_ID:sampling:0 \
  --target-sub-job-id JOB_ID:sampling:1
```

If a backend needs a different operation routing hint, pass
`--operation-sub-job-id` or `--operation-sub-job-type`.

Use `--weight-format lora` for adapter-only sync; `vllm` and `hf` are also
accepted formats. Pass `--no-poll` to return the request metadata without waiting
for synchronization to complete.

### Download Execution Logs

Pull every log file the job's experiment run produced. Each sub-job's
`_logs/` artifact directory may contain multiple files (e.g.
`execution.jsonl`, `server.log`); all of them are downloaded:

```bash
cortex-training download-log JOB_ID --output-dir /path/to/dir
```

Files are written as `<output_dir>/<sub_job_id>/<filename>` so siblings
do not collide. When `--output-dir` is omitted, the current working
directory is used instead. The CLI also prints a JSON summary listing
each `saved_path`.

Programmatic access is `CortexTrainingClient.fetch_execution_logs(job_id)`,
which returns a list of `{sub_job_id, filename, artifact_uri, content}` dicts.

### Download Persisted Stdout

Reconstruct each sub-job's persisted stdout/stderr chunks into
`<output_dir>/<sub_job_id>/stdout.log`:

```bash
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs --resume
```

The current working directory is used when `--output-dir` is omitted.
`--resume` continues a stdout download already in that directory. It does not
apply to execution logs, including `download-log` without `--log-type stdout`.
On a terminal, progress is one line on stderr that updates in place. A
redirected stderr gets one final line per sub-job. Progress is not part of
the JSON on stdout.
A large download may open additional Snowflake sessions for one sub-job, and
each session can prompt for authentication. Two downloads of the same
directory at once are unsupported.

### Download GPU Metrics

Reconstruct each sub-job's GPU metric chunks into
`<output_dir>/<sub_job_id>/gpu.jsonl`:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

The command prints the saved path, chunk count, and first/last logical artifact
URIs for each reconstructed file. `--resume` continues a metrics download
already in that directory. On a terminal, progress is one line on stderr
that updates in place. A redirected stderr gets one final line per sub-job.
Progress is not part of the JSON on stdout. A large download may open
additional Snowflake sessions for one sub-job, and each session can prompt
for authentication. Two downloads of the same directory at once are
unsupported.

### Roles And Access

`checkpoints` is a REST call: a role with only
`MONITOR` on the account can list a job's checkpoints. The
checkpoint files are stored on the job's Experiment too, but the service reads
them there for you. Saved logs and GPU metrics are read with SQL on the job's
Experiment instead (`download-log`, `download-metrics`, and the TUI's saved
console and Resources tab), so they need privileges on that Experiment, and a
`MONITOR`-only role can't read them.

When the service can't resolve the job's Experiment for your role, it answers
500, in one of two wordings. The experiment-run lookup that `download-log` and
`download-metrics` start with, before any SQL runs, says `Linked experiment is
not found for job JOB_ID`: the role can't see the Experiment, or it no longer
exists. `checkpoints` says `Linked experiment is not
found or not authorized for the training job`. The client retries neither, so
the command fails at once and adds `hint: this role can't read the job's
Experiment ...` (with `or it no longer exists` for the first wording). Ask for
access to the Experiment, or use the role that created the job.

A role the service lets through that has no privileges on the Experiment, such
as a `MONITOR`-only one, gets Snowflake's refusal of the SQL instead:
`Experiment '...' does not exist or not authorized`, or `Insufficient
privileges to operate on experiment '...'`. `download-log` and
`download-metrics` print Snowflake's message after `error:`, without a
traceback, add `hint: this role has no privileges on the job's Experiment
...`, and exit with status 1; `checkpoints` still
works. See
[REST API reference section 2.5](rest-api.md#25-roles-and-access).

The TUI explains a refused read in one line on the view that needed it,
instead of showing the raw error, and words the same refusal the same way on
every view: the Logs tab's saved console and execution log, and the Resources tab
(`GPU metrics unavailable: ...`). The service's answer above says the role
can't read the job's Experiment; the experiment-run lookup's wording,
`Linked experiment is not found for job ...`, adds that the Experiment may no
longer exist. When Snowflake refuses the SQL on the Experiment (`Insufficient
privileges to operate on experiment ...`, `Experiment ... does not exist or
not authorized`), as it does for a `MONITOR`-only role, the line says that
saved logs and GPU metrics live there and that `checkpoints` still works (the
service lists them). A refusal about another object, a warehouse or a schema,
keeps its own text. An HTTP 403 whose message names a privilege says the role
isn't allowed to read that view's data. A 403 with no such message (a proxy
or network gate can send an empty one) keeps the standard error text.

### Job TUI

`cortex-training tui` is a read-only terminal UI for finding a job and
inspecting its logs and GPU usage. It uses the same
connection handling and fallback order as the CLI, including named and
configured-default Snowflake profiles. Like `checkpoints`
and `list --all`, it never creates the configured database or schema, which
the other commands do when the service answers that they are not found: the
TUI shows that error instead, as it does when this role can't use them.

Every request the TUI sends is retried at most twice, with a 10-second connect
and a 45-second read timeout, where the other commands use the SDK's defaults
(ten retries, 30 and 600 seconds; [REST API reference section
3.5](rest-api.md#35-retries)). So a job id that isn't found (404), or a
service that keeps failing, shows its error within a few seconds rather than
after about a minute. Pressing `r` while a refresh is still reading says so.

```bash
cortex-training tui                 # opens the job list of the configured database and schema
cortex-training tui --schema-only   # the same, without a to switch to every database and schema
cortex-training tui JOB_ID          # opens that job's Logs tab directly
cortex-training tui --debug         # synthetic jobs, no connection needed
```

Without login state, pass connection details the same way as the CLI:

```bash
cortex-training tui JOB_ID --connection training
cortex-training tui JOB_ID --config config.json
cortex-training tui JOB_ID --host ACCOUNT.snowflakecomputing.com --pat YOUR_PAT \
  --database CORTEX_TRAINING_DB --schema PUBLIC --endpoint cortex-training
cortex-training tui JOB_ID --base-url http://localhost:8084   # local/mock
```

Keep the PAT out of your shell history by exporting it instead (omit `JOB_ID`
to open the job list):

```bash
export CORTEX_TRAINING_PAT='YOUR_PROGRAMMATIC_ACCESS_TOKEN'
cortex-training tui \
  --host ACCOUNT.snowflakecomputing.com \
  --database CORTEX_TRAINING_DB --schema PUBLIC
```

Pass `--sub-job-id JOB_ID:training:0` to scope the job details to one matching
sub-job; without `JOB_ID`, it opens that sub-job's job. An unmatched id is
shown as an error, with the job's sub-job ids, rather than falling back to a
different sub-job. `--debug` runs against generated jobs and telemetry, and
keeps its names and caches in a temporary directory; the
title says `(debug)`. Its jobs are in three locations: the configured
`CORTEX_TRAINING_DB.PUBLIC`, `TEAM_DB.EXPERIMENTS`, and a lowercase
`"sandbox"."dev"`. It runs on the wall clock: live jobs keep logging (in the
service's line format) and reporting GPU samples every 5 seconds, with four
GPUs to a node and each node's CPU and memory beside them (one running
job reports its GPUs only); the placing job starts up and runs within two minutes (its
sampling sub-job waits until then, as one the service isn't routing to yet);
the team's running job is cancelled five minutes in; and a debug run's tail
gets stuck behind a 300 KiB line a minute in, and again every ten minutes,
and restarts from the end. Among its other jobs are a job with a `log_prob`
sub-job, jobs whose status is `SUCCEEDED`, `DONE` or `JOB_STATE_CANCELLED`, an
`UNKNOWN` job its control plane no longer knows, a 25,000-line saved console,
and two jobs whose Experiment this role
can't read: one gone, and one that SQL refuses. Set
`CORTEX_TRAINING_TUI_DEBUG_ACCOUNT_LIST` to `not_deployed`,
`disabled`, `forbidden` or `rest_disabled` to preview `a` where the
account-level list is unavailable, or to `unavailable` for a list that fails
(503); `--debug --schema-only` works too.

#### Job list

The job list shows the jobs created in the last six months (183 days) that the
current role can see in the configured database and schema (`list_jobs`).
Press `a`, or click `a All schemas` in the footer, to list the jobs in every
database and schema the role can see instead, from the account-level list
([REST API reference section 5.3](rest-api.md#across-the-account---get-apiv2cortex-training));
the footer then offers `a Current schema`, which goes back. With `--schema-only`
the list never leaves the configured schema and the footer has no `a`. The
header line starts with when the list was last refreshed (`refreshing…` while
a refresh reads), then which list is shown:

| Header | The list holds |
|---|---|
| `Current schema: DB.SCHEMA` | the jobs of the configured database and schema (`list_jobs`), as the list opens |
| `all schemas` | after `a`: the jobs in every database and schema the role can see, from the account-level list; `all schemas (all in DB.SCHEMA)` when every one of them is in the same location |
| `newest 2,000 + every active job (more exist)` | the same, cut at the newest 2,000 because more exist, plus the active jobs created before those |
| `first 20 pages: N jobs + every active job (more may exist)` | the same, cut after 20 pages (see below), plus the active jobs |
| `Current schema only: DB.SCHEMA (reason)` | the jobs of the configured database and schema: with `--schema-only`, or in place of the account-level list where it turned unavailable while shown (see below) |

The configured schema's list is read with two retries and a 45-second read
timeout. After `a`, each refresh reads the account-level list newest first,
500 jobs a page with the same bounds per page. It stops after 2,000 jobs,
or after 20 pages, because the service examines a bounded number of jobs per
call, so a page can be short or empty and still have more after it: a refresh
never walks the whole history. It asks for no status, so jobs whose status the
service can't read (`unknown`) are listed too. When it stops with jobs left, it
also reads the active jobs (`pending`, `placing`, `initializing` and `running`,
through the same list with `status=active` and the same bounds) and adds those
created before the newest ones, so a long-running job stays under live. The
line under the header counts them, for example `newest 2,000 jobs + 3 older
active`. An `unknown` job older than the newest 2,000 is not listed, as a
status filter leaves those out, and if reading the active jobs fails or is cut
short too, the header says `older active jobs may be missing`.
A walk that stops at 20 pages with no job to show says that more may exist and
how to list one schema, rather than that there are none.

Search filters the jobs the list holds: when more exist, a search can't reach
the older ones, and the message for a search without results says so. To
search all of the configured schema's jobs, press `a` to go back to its list;
for another schema, start the TUI with that schema's `--database` and
`--schema` (and `--schema-only`). If the service can no longer open a page
token partway (400 `Invalid page_token`), the walk starts again from the first
page, once.

If the account-level list is unavailable when `a` asks for it, the list stays
on the configured schema (the footer offers `a All schemas` again) and a notice
says why, for example `Can't show all schemas (no account-wide list yet); the
list stays on CORTEX_TRAINING_DB.PUBLIC`. The reason is `no account-wide list
yet` while the service doesn't have the route, `account-wide list not enabled`
until Snowflake enables it for the account (both answer 404), `account-wide
list: HTTP 403` for a 403 without the service's own error message (a network
policy or a proxy in front of the account-level route alone can send one), or
the error itself: the service's refusal where Cortex Training or its REST API
is off for the account (403, or 405), or a failed load (a timeout, a 503).
After a 404, or a 403 without the service's message, `a` within a minute
answers the same without asking the service again. A later `a` asks again,
and the notice gives what the service answers then, also when that differs
from the last answer (a 503 after a 404, say).

Once every location is shown, a refresh whose account-level list answers 404
(not there, or not enabled) shows the configured database and schema in its
place, with the reason in the header: `Current schema only:
CORTEX_TRAINING_DB.PUBLIC (no account-wide list yet)` or `(account-wide list
not enabled)`. It reads that schema's list with the same bounds. It asks the
account-level list again at most every 10 minutes, or when you press `r` once a
minute has passed since it last asked, and lists every location once it
answers. If asking again is answered 403 or 405, the configured schema's list
stays, and the header says `(account-wide list: HTTP 403)`. A refresh the
service refuses (403 with its own message, or 405) is a failed refresh (below);
where no job was listed, the error is shown in place of the list and adds that
`a` lists the configured database and schema. After a 403 without the service's
own message, the list asks the configured schema once, and where that answers,
shows its jobs with `(account-wide list: HTTP 403)` in the header and asks the
account-level list again every 10 minutes. `--schema-only` lists only the
configured database and schema and never asks the account-level list: the
header says `Current schema only: DB.SCHEMA (--schema-only)`.

If a refresh fails, the list on screen stays, and a red line above it says
when it failed, why, and since when the list is from, for example `refresh
failed 14:06:12: the account-wide job list answered HTTP 503 Service
Unavailable: Control Plane unavailable; showing the list from 14:05`. The header
starts with `refresh failed 14:06:12 (list from 14:05)` in place of `last
refreshed`. The next good refresh clears both. A first load that fails shows the
whole error in place of the list, and the header starts with `load failed
14:06:12`; `r` tries again.

While a job's preview or details are open, the list doesn't refresh. Back on
the list, it reads the list again at once when the last read is more than 60
seconds old, and a status the preview read in the meantime shows straight away.

When the list holds jobs from more than one database and schema, a
**Location** column after GPUs shows each job's `DB.SCHEMA`, each name
double-quoted when it is not a plain uppercase identifier, as SQL needs it
(`TEAM_DB.EXPERIMENTS`, `"ml"."dev"`). It comes after the columns that fill a
140-column terminal: on one narrower than about 170 columns, scroll right to see
it. The job preview shows
the location too, and the job's details line names it, after the models, when
it isn't the configured location. The column is hidden while every job shares
one location, as the configured schema's own list does. Opening a job reads
everything about it from its own database and schema: its record and status,
the Logs tab's live tail, saved console and execution log, and GPU metrics.
The commands it shows name them too: the Logs tab's `download-log` hints, for example
`cortex-training --database TEAM_DB --schema EXPERIMENTS download-log JOB_ID`
(with `--connection NAME` for a connection profile, and otherwise the TUI's
`--config FILE`, `--host HOST` or `--base-url URL`). `tui JOB_ID` reads the job
from the configured database and schema; when it isn't there (404), the Logs
tab says where it looked and to pass the job's `--database` and `--schema`.

The TUI asks the account-level list only for jobs created in the last six
months. The configured schema's list has them all, so the list hides those
itself and counts them in the header (`1 older hidden`). Live jobs (`pending`,
`queued`, `placing`, `initializing`, `creating` and `running`, with a green `●`;
and `unknown`, with a yellow `?`, which means the service could not read the
job's live state, so it may still be running) are grouped first. Finished jobs
(`failed`, `cancelled`, `terminated`, `done`, and any other status) follow under
history with `○`, and so does an `unknown` job the service says the control
plane has no record of (`the job was not found in the Control Plane`). Sorted
by status with ended jobs first (`status ↑ ended first`), history comes first.
The `LIVE` and `HISTORY` rows give their count and what they hold; where the Name
column is too narrow for that, it goes on in the columns after it, so it is
never cut. Each row shows display name, short job id, status, type, created
time and owner, then model, requested GPUs, location (when the list spans more
than one) and updated time. Sorted by updated on a terminal too narrow for the
Updated column to be in view, the two times swap places, so the sorted time
shows beside the type. The owner is
the submitting user; a job that records none shows its owning role, dimmed, as
`role ROLE`. Every column has its full width when they all fit; otherwise the
dates drop the year (`10-15 14:05`: the list spans six months), and the Name
column narrows, to 16 cells at least, until name, id, status, type, the time
after it and owner fit (100 columns is enough). The rest scroll into view with
`→`/`←` (on a row with sub-jobs to unfold or fold, those come first), the
marker and the name staying in place. The Name column takes the
width a wide terminal has spare once every column fits, and a name longer than
the column ends in `…`. A sub-job row's Job ID is the sub-job id after the
job's (`training:0`). The GPUs column is what the job requested (see the job preview
below): `shared` for a job whose sub-jobs all run on another job's GPUs and
on a sub-job row multiplexed onto another job's sub-job (it requests none of
its own, whatever its config's `n_gpus` says, as the preview's GPUs line and
the Resources tab count it too), and `0` for a `log_probability` sub-job. The
list and the preview
call such a sub-job `log_prob #N`, as its sub-job id spells it, and a job with
training and log_prob sub-jobs `train+logprob` (`train+sample` for training and
sampling). A job with sub-jobs shows `▸` before its name while folded and
`▾` while its sub-job rows show under it. Live jobs start unfolded. A click on
the `▸`/`▾` unfolds or folds the job (a click anywhere else on a row
highlights it, and a second click opens it); so do `→` and `←` on its row, and
`←` on a sub-job row folds its job and moves to it. `e` unfolds or folds every
job. When the list is empty it says so: "No jobs created in the last 6 months that
this role can see, in any database and schema", or, for the configured
schema's list, for example "No jobs created in the last 6 months in
CORTEX_TRAINING_DB.PUBLIC that this role can see", followed, where `a`
switches to every location, by "a lists every database and schema". A walk
cut short with no job to show says "The account-wide list returned no jobs to
show in its first 20 pages; more may exist" instead.

A job's display name is its local name if you set one, else the first line of
its server `comment` (see `submit --comment`; trimmed, and cut with `…` when it
is long or has more lines), else its job id. Press `n` on a job to give it a
local name (up to 80 characters); the prompt starts from the local name, or
empty with the comment's first line shown as a hint, and saving that line (or
nothing) keeps the job named by its comment. Local names persist in `aliases.json` in the TUI cache
directory, are shared by every TUI session on that machine, and are not
visible to anyone else. A job with both shows the local name first and its
comment after it, dimmed. Names and comments are measured in terminal cells,
so wide (CJK, emoji) text is cut and wrapped to fit too.

A job whose status is `unknown` has no live state: the service could not reach
the control plane. Its row shows no times (the account-level list keeps its
creation time, and a job that turns `unknown` keeps the one it was listed with,
and its place in the order; a row of the configured schema's list that was
`unknown` from the start has none, so it stays in the list, sorted last), and
a line above the list gives
the service's sentence once, with the number of such jobs (for example
`? Job status unavailable: Control Plane is unavailable · 3 jobs`). Expanding
the job shows "? status unavailable" and the sub-jobs it was created with, their
ids derived from the job id as `JOB_ID:TYPE:INDEX` (only for an `unknown` row;
a config without a type is listed as `config #N`). `tui JOB_ID --sub-job-id
ID` on such a job tries that sub-job even when it can't be listed.

Search with `/`. Plain words match display name, the whole comment (also when
a local name is shown), job id (partial), sub-job id, status, owner, GPU type,
job type (as the Type column shows it, `train+sample`, and as the API spells
it, `log_probability`) and model, and all words must match. Matching ignores
case and Unicode spelling: an accented letter typed either way (`é` as one
character or as `e` plus an accent) and fullwidth letters (`ｗｉｄｅ`) find the
same jobs. Words also
match the start of a job's database or schema name in the account-wide list
(`team`, `experiments`, `team_db.exp`, `ml.dev`, or `"ml"."dev"` as the
Location column shows it), but not the middle of one, so `training` doesn't
match every job in `CORTEX_TRAINING_DB`. A word in quotes is searched as it
is, so `"training:0"` finds sub-job ids rather than reading `training` as a
field. `field<op>value` filters narrow further:

| Field | Meaning |
|-------|---------|
| `name`, `id`, `status`, `owner`, `type` | display name or comment, job id, status, user or role, `training`/`sampling`/`log_prob` |
| `model`, `method`, `dtype` | model name (`=` takes it whole, `Qwen/Qwen3-8B`, or as the Model column shows it, `Qwen3-8B`), `full` or `lora`, dtype |
| `gpu`, `gpus` | GPU type (`H200`, `B200`, `B300`), requested GPU count |
| `optimizer`, `lr`, `batch`, `seq` | optimizer name, learning rate, train batch size, max sequence length |
| `lora_r`, `lora_alpha` | LoRA rank and alpha |
| `db`, `schema` | database and schema name; a job of the configured schema's own list is in the configured ones. `:` matches part of the name in any case (`db:team`). `=` and `!=` take the name as stored or as SQL resolves it, a bare name being its uppercase form: `db=team_db` and `db=TEAM_DB` match `TEAM_DB`, `db=ml` and `db='"ml"'` match a lowercase `"ml"`, and `db=ML` doesn't |

Operators are `:` (contains, or equals for numbers), `=`, `!=`, `>`, `>=`,
`<`, and `<=`. Any other key reads that member of the job record, of a sub-job
(`seed`, `dtype`, `global_batch_size`) or of a sub-job's config (`ep_size`), so
`seed=42` and `ep_size>1` work. A dotted key reads a nested member of a
sub-job's config, for example `training_config.optimizer.lr<1e-4` or
`inference_config.n_gpus=2`. A member that holds a list matches when one of its
elements does:
`training_config.peft_config.target_modules:q_proj`,
`training_config.optimizer.betas:0.9`; with `!=`, a job matches when none of
them does. Examples:

```text
sweep
model:qwen3-8b method:lora lr<=1e-4
gpu:b200 gpus>=8 lora_r=32
training_config.optimizer.name=adamw status:cancelled
training_config.peft_config.target_modules:q_proj seed=42
db:team_db status:running
```

A search with no results says how many jobs were searched and how the words
and filters are applied, and, when more jobs exist than the list holds, what
the list holds and that the search covers only those. On one schema's list it
names that schema, and on the configured schema's (the default) it adds that
`a` searches every database and schema, for example `No jobs match “sweep”
among 171 job(s) in CORTEX_TRAINING_DB.PUBLIC in the last 6 months.` and then
`a searches every database and schema.` A key that is neither a
field nor a member of any listed job matches nothing: the message names it,
with the dotted key that reads it where only a nested member has that name
(`did you mean training_config.peft_config.target_modules?`), and lists the
fields, unless the search help under the box, which lists them, is showing.
When jobs older than six months, which the list hides, match the
search, it says how many and the `cortex-training tui JOB_ID` that opens the
newest. A message taller than the room a short terminal leaves scrolls (the
mouse wheel, or `Tab` to it and the arrow keys). The line under the header
gives the number of matches, then the search; it wraps rather than cutting
either. On a terminal shorter than 30 rows the
field help under the search box shows only while the box has focus.

| Key | Action |
|-----|--------|
| `Enter` | Open the job preview (or, on a sub-job row, that sub-job's details, on the Logs tab; on a "status unavailable" line, its job's preview); in the search box, show the matches of all that was typed and move to the list, where `Enter` opens the highlighted match |
| `/` / `esc` | Search / clear the search. Keys are taken in the order typed, also when they come faster than the list redraws: what is typed before `Enter` or `esc` is in the search they apply or clear, and the keys after `/`, `Enter` and `esc` go to the search box or the list |
| `s` / `o` | Sort by created, updated, or status (`succeeded` and `completed` sort with `done`, `canceled` with `cancelled`) / reverse the order. The line above the list says which way it runs: `created ↓ newest first` or `↑ oldest first` (the same for updated; jobs without that time, shown as `—`, come last either way), `status ↓ running first` or `↑ ended first` (history first). The cursor goes to the first job |
| `Home` / `End` | First / last job |
| `→` / `←` | Unfold the highlighted job's sub-jobs / fold them; on a sub-job row, `←` folds its job and moves to it. The job's row stays where it is on screen, and the list scrolls only as far as an unfolded job's sub-jobs need to show. Where there is nothing to unfold or fold, they scroll the columns sideways |
| click `▸` / `▾` | Unfold or fold that job, as `→` / `←` do |
| `e` | Expand or collapse every job's sub-jobs |
| `a` | All schemas / current schema: switch to the account-level list of every database and schema the role can see (the footer's `a All schemas`), or back to the configured one (`a Current schema`); hidden with `--schema-only` |
| `n` | Name the highlighted job |
| `ctrl+c` / right-click | Copy what is highlighted (text, or rows a drag picked, with their whole job ids), else every job in the list; the same in the preview |
| `r` | Refresh (the list also refreshes every 60 seconds while it is showing); while it shows the configured schema in place of the account-level list (`Current schema only: DB.SCHEMA (reason)`), `r` asks that list again once a minute has passed since the last ask |
| `?` | Show or close the keys panel |
| `q` | Quit |

Every screen's footer starts with Back (`b`) and Quit (`q`), then, in the job
details, the tab keys, then `?`: a narrow terminal cuts the footer's end, and
`?` opens a panel that lists every key that works where you are, including the
ones the footer has no room for. `esc` closes that panel first, before it
clears a search or goes back.

#### Job preview

Opening a job shows its one-line name (and where it comes from: the local
name, the job's comment, or none yet), its comment wrapped to the screen, id,
status (live, historical, or `? status unavailable` for an `unknown` job),
model, type, owner (`role ROLE` when no submitting user is recorded), requested
GPUs, creation time, elapsed time, and last update. Every field shows when the preview opens: the
comment is shown whole when the screen has the rows for it next to everything
else, and otherwise its first rows (at least three) with a note such as
`(+16 rows · e shows the whole comment)`; any other value stops after three
rows with `…` and a note such as `(+62 rows · e shows the whole value)`. `e`
shows every value whole, and the sub-job table with every column and cell
whole; `e` again shortens them. The footer offers it too while something is
cut, when it has the room.

- **GPUs** is what the job requested, by the control plane's rule: each
  sub-job's `n_gpus`, except that a `log_probability` sub-job uses none and a
  sub-job multiplexed onto another job's sub-job (`multiplex_job_id` in its
  config) adds none, because it runs on that sub-job's GPUs. The job record
  does not carry the allocation, which can round up to whole nodes.
- **Elapsed** runs from creation, so it includes time spent queued and placing.
  An `unknown` row of the configured schema's list carries no creation time,
  so a job that was already `unknown` when that list loaded shows `—` for both
  Created and Elapsed; the account-level list keeps the creation time. A job
  that turns `unknown` after the list loaded keeps the creation time from the
  list, and its Elapsed runs to now, since the job's end can't be read.
- **Last update** is the job's `updated_at` and how long ago that was. A time
  up to 5 minutes ahead of this computer's clock reads `just now`, and one
  further ahead says how far ahead it is; a record without one shows `—`.
- **Comment** is left out when the name already shows all of it (a one-line
  comment that fits).

Below the fields, a line says where the job stands: running; waiting
(`pending`, `placing` or `initializing`, each with what it waits for unless the
service says more); cancelled, terminated or completed (`succeeded`, `done`);
or failed, with where to look. The service's `reason` and `reason_details`
follow, the same way for every state (for example `reason: quota_filled — needs
8 GPUs; 64 currently active`, or what happened to an evacuated node), and
disappear once the service stops sending them. An `unknown` job says "Job
status unavailable" with the service's sentence, and that it may still be
running unless the service says the job was not found.
Long values (the comment, a GPU line that names another job, the reason)
wrap under where they start. A character carrying more than four combining
marks keeps the first four (so-called zalgo text would otherwise draw over the
rows around it). Below
that are the job's sub-jobs with their type, status, model, GPUs and key
config. A multiplexed sub-job's config
ends with the sub-job whose GPUs it runs on, for example `runs on
0a1b2c3d…:training:0`. The sub-job table fits the screen. When its cells are
wider, a model name over 36 cells and then the end of each config (down to 24
cells) are cut with `…`. On a narrower screen, such as 80 columns, the Type
column is left out (each sub-job's label already names its type, and the
Type field above has the job's), and the model and the config are cut further,
down to 10 and 12 cells. The line above the table then adds `e whole table`;
with `e` the table has every column and cell whole and, when that is wider
than the screen, scrolls sideways (`←` / `→`, `Home` / `End` for either edge)
with `← → scroll` on that line. A screen too narrow even for the cut table
scrolls it sideways the same way. A live job's preview refreshes every 60
seconds, and when you come back to it from the details; `r` refreshes it now.
Just under the fields, a dim line says when the copy on screen was read from
the service (`read from the service at 12:05:01`), that a read is in flight (with
how long it has taken, once that is 5 seconds or more: the client retries a
failing answer for about a minute), or, after a read fails, the error and which
copy is still shown, for example `couldn't refresh: HTTP 500 Internal Server
Error: ...; showing the copy read at 12:04:30` (or the job list's copy, when no
read has succeeded yet).
A click on the highlighted row (the "entire job" row or a sub-job row), or
`Enter`, opens the details for that scope, its telemetry dashboard, on the
Logs tab, as the line above the table says (`Click a job to open its
telemetry dashboard`); a click on another row highlights it first.
`l` and `g` open the Logs and Resources tabs
directly, and do the same in the details.

#### Job details

The line under the title names the job and its scope, and repeats the
service's `reason` and `reason_details` (or the `unknown` sentence) while it
sends them. A sub-job's details name it by its id once (`sub-job
JOB_ID:training:0`). The line keeps to one row when cutting the name and the
models with `…` is enough for that, and otherwise wraps between its parts,
with the reason whole. Switch tabs with `2` (Logs) and `3` (Resources), or `t`
for the next tab.

- **Logs** is the live and historical log view described below. Each line is
  tagged with the sub-job being tailed, for example `training:0 │ …`, and its
  wrapped rows line up under its text. Saved logs (`s`) and whole-log copies
  (a copy with nothing highlighted) stay untagged and uncoloured. In the pane,
  the tag has its sub-job type's colour (training cyan, sampling violet,
  log_prob amber), as has the dot before each sub-job in the list beside it
  and its name in the Resources header; the time is muted; the level is
  coloured by severity (`INFO` green, `WARNING` amber, `ERROR` red), and a
  warning's or an error's message is tinted to match, on each row it wraps
  to; the logger name is
  blue, the keys of `key=value` pairs are muted, exception names and
  tracebacks are red, the TUI's own notes (`— live —`) are in italics, and
  what the filter matches is marked in amber.
- **Resources** shows telemetry (`download_metrics`) for one sub-job at a
  time: the one selected on the Logs tab, until `Space` shows the next one
  (`<` and `>` go back and forth). The tab's header names it, and stays at the top while the
  stats and charts scroll under it; with two or more sub-jobs it also names
  its place among them (`training #0 (1 of 2 · space switches)`).
  - A panel on the left, sized to its labels as the Logs tab's sub-job list
    is, lists the two views, `GPU Resources` and `System Resources` (the one
    shown is marked `●`), and under them the shown view's units: `ALL`, the
    mean across them, then each GPU (`GPU 0`, numbered as the rows number
    them) or node (`node 1`, counting from 1), each with its newest GPU util %
    or CPU util %. A GPU or node that stopped reporting is dimmed and marked
    `(stale)`. A view the sub-job has no rows of reads `(none)`, dimmed, once
    the download has found none: most deployments collect no CPU or memory
    metrics, so `System Resources (none)` is the usual case, and choosing it
    says `CPU and memory aren't collected for this sub-job`. A sub-job with
    system rows and no GPU rows opens on the System view.
  - The panel has the focus when the tab is shown. Moving its highlight with
    `↑` `↓` (or a click) shows that entry at once: a view's entry switches to
    that view and its mean (`ALL`), and a GPU's or node's entry redraws every
    stat and chart for it alone. `↓` on `GPU Resources` goes to its `ALL` (`↑`
    from `ALL` reaches `System Resources`). `<` and `>` keep the view and go
    back to `ALL`; `w` keeps the GPU or node.
    `PgUp`, `PgDn`, `Home` and `End` scroll the stats and charts, also while
    the panel has the focus; so does the mouse wheel over them.
  - The header names the job's GPU type (`hardware`) and compares the GPUs the
    sub-job requested with the GPUs reporting in the newest samples. Requested
    follows the control plane's rule: `n_gpus` in the sub-job's config, but 0
    for a `log_probability` sub-job, and 0 for a sub-job that sets
    `multiplex_job_id`. A multiplexed sub-job runs on the GPUs of the sub-job it
    names, and only that sub-job reports GPU metrics, covering everything that
    runs on them, so the multiplexed sub-job's own tab says where they are.
    A GPU whose newest sample is more than two minutes (or three sampling
    intervals, if longer) older than the newest one counts as seen but not
    reporting, and is marked stale. In the System view the header names the
    nodes instead: `nodes: 2 reporting  ·  88 CPUs  ·  860 GiB each`, from each
    node's newest `cpu_count` and `mem_total_bytes` (a range when the nodes
    differ). The header also names the window (`last 10m`, `whole job`).
  - It also shows how often the sampler scraped, derived from the data
    (every 5 seconds is typical), and how old the newest sample is. DCGM can
    refresh a field less often than that, so consecutive samples may repeat a
    value; when the GPU util values change less often than every two scrapes,
    the header says how often they do (`scraped every 5 s (values change about
    every 1m)`), since the newest value can be that much older than the newest
    row. The age keeps counting between refreshes; samples reach the stage in
    batches, so it is often 30 seconds or more. When a running sub-job's newest
    sample is more than three minutes old (or six sampling intervals, if
    longer), the header says so in the warning color (`no new GPU samples for
    41m 56s; the windows end at 06:25:06`, or `no new system samples …`): no
    GPU or node counts as reporting, every one is marked stale, and the
    windows and the values are measured back from that sample. Each view is
    measured back from its own newest sample. Rows that can't be read are
    counted in the header (`3 rows skipped (malformed or unsupported)`):
    malformed JSON, a record version or sub-job this client doesn't read, or
    an unusable GPU index, node number (`worker_num`) or timestamp. Rows of a
    kind this client doesn't know are passed over. A row stamped more than 10
    minutes ahead of this computer's clock is counted too, so one skewed
    timestamp can't become the newest sample. A row without a usable
    `sample_interval_s` (missing, or not above 0) counts for the time since
    its GPU's (or node's) previous row, up to five minutes.
  - The header is always two lines, so the stats and charts under it never
    move. On a terminal too narrow for all of it, each line drops detail
    instead of wrapping. On the first, the sub-job's place goes short
    (`1 of 2 · space switches` becomes `1/2`), then the window to its key's name
    (`10m`), then away, then the place, then the GPU type (in the System
    view, the nodes' CPUs and memory). On the second, the why of the skipped
    rows, the values-change note and the newest sample's time go first, and a
    state's longer words: `cached · refresh failed: …` becomes `refresh
    failed: …`, and `older samples failed to load: …` becomes `older samples
    failed: …`.
    Then the skipped rows and the scrape interval go. An error keeps room for
    its first 40 characters where it can. The second line keeps room for
    `cached · refreshing…` and for the age to grow, so a refresh starting, or
    the age passing a minute, changes nothing else on it.
  - Every numeric field of a row is a metric, charted without a list of
    names, so a field a newer service adds gets its chart and stats row with
    no change here. A field that identifies or counts rows (a name ending in
    `_index`, `_num`, `_version` or `_interval_s`, and `seq`) is not one, nor
    is text, nor a field that is null in every row (the SM gauges today). Each
    view starts with its utilization and a derived memory use: **GPU util %**
    (`gpu_util_pct`, the share of time a kernel was running on the GPU, which
    says how busy it was, not how saturated) and **GPU memory used %**
    (`fb_used / (fb_used + fb_free)`); **CPU util %** (`cpu_util_pct`) and
    **memory used %** (`mem_used / mem_total`, at most 100%). The others follow
    in the order the file first has them, such as `GPU memory used (GiB)`,
    `GPU memory free (GiB)`, `power (W)`, `GPU temp (°C)`, `tensor active %`
    and `DRAM active %`, or `memory used (GiB)`. A node's `cpu_count` and
    `mem_total_bytes` stay the same through a job, so the header names them
    (above) and they get no chart or stats row. A name comes from its field: the unit suffix (`_pct`, `_bytes`,
    `_watts`, `_c`) becomes `%`, `GiB`, `W` or `°C`, and `gpu`, `cpu`, `sm`
    and `dram` are capitalized.
  - The stats block has a row per metric, in that order: the min, average,
    max and latest over the selected window and over the whole job, in the
    metric's unit, for the GPU or node shown or for the mean across them. The
    titles say which (`mean of 8 GPUs`, `GPU 3`, `node 2`); on a terminal too
    narrow for that layout, the metrics' short names (`GPU mem used %`,
    `power`) come first, then a row per metric and scope (`10m`, `job`), and
    the first line names what the values are.
  - Under it, a chart per metric, in the same order: two to a row in a pane
    at least 90 cells wide, one per row otherwise. The charts grow with the
    terminal: 12 rows of bars on a terminal of 48 rows or more, 10 from 40
    rows, 8 from 34, 6 from 28 and 4 below that, always an even number, so
    the middle label's row tops out at exactly half the scale; a dotted line
    marks that row where the bars stay under it. A percentage is charted from
    0 to 100% (labeled `100%`, `50%` and `0%`); any other metric from 0 to the
    smallest 1, 2, 2.5 or 5 × 10^k at or above the sub-job's highest sample of
    it, on any GPU or node (`1000 W`, `500 W`, `0 W`), so its axis doesn't move
    with the window or the unit shown. Each title names the metric, the window
    and what the values are (`power (W) · last 10m · mean of 8 GPUs`), and
    ends, at the chart's right edge, with the metric's latest value; on a
    terminal too narrow for that, every title leaves out what the values are
    (the stats name it), then the window, and a name that still doesn't fit
    ends in `…`, so the charts of a row stay level. A metric with no samples
    in the window says so in its chart's place.
  - Each chart has its own colour, by its place in the order (the first, GPU
    or CPU util %, is green; memory used % cyan; then blue, violet, amber,
    rose and so on): its bars (lighter toward the top), its title and latest
    value, and its name in the stats block. The frame, the axis labels and
    the times are muted, and the stats' latest values are bold. In the panel,
    each GPU's or node's value has a one-cell bar of it before it, in the
    first chart's colour (the value itself stays plain, readable under the
    highlight).
  - For the mean across GPUs or nodes, min and max are the lowest and highest
    average across them at one sampling interval. Each worker scrapes and
    uploads its samples on its own schedule, so a GPU or node without a sample
    at some interval counts with its last value while it is still reporting. A
    node that misses one scrape is carried across it, so a single miss leaves
    no blank column; a longer hole stays a gap. Samples older than seven hours
    are dropped, but only after their averages at each interval are folded
    into the whole-job chart's bins, so the whole job's min and max stay
    exact. A row that arrives after its time was folded (a chunk uploaded
    hours late) is known only as part of its bin's average: when one of its
    samples is lower than the min as shown, the min reads `≤`, since the true
    value may be lower, and a higher one makes the max read `≥`. The average
    weights GPU-time (or node-time): each sample counts for its sampling
    interval, so a GPU that reported for only part of the window counts for
    that part (for GPU util %, it is the share of the GPUs' time a kernel was
    running). Latest is the newest capture. For one GPU or node, min and max
    are its own lowest and highest samples, also in the whole-job view, the
    average is its own over the window, and latest its newest sample. A chart
    spanning a day or more dates its axis labels, and one spanning less than
    two minutes shows seconds, so no two labels read the same.
  - `w` cycles the window through 10 minutes (the default, for every job),
    30 minutes, 1 hour, 6 hours, and the whole job.
  - The key help stays at the bottom of the tab, with only the keys that do
    something there: `w window 10m · 30m · 1h · 6h · all   r refresh   space
    sub-job   ↑↓ GPU` (`space` with two or more sub-jobs, `↑↓` with more than
    one GPU or node). On a line too narrow for all of it, `w window` leaves
    out the windows' names, then whole entries are left out from the end and
    the line ends in `…`. The footer shows `space Sub-job` after `w Window`;
    on a terminal too narrow for every key, the footer leaves out the keys
    that don't fit whole, from the right (`?` lists every key).
  - Telemetry downloads when the tab is first shown; `r` refreshes it, and a
    live job's telemetry refreshes every 30 seconds while the tab is showing,
    once the selected sub-job runs. Before that (the job or the sub-job is
    pending, placing or initializing) the tab says `no GPU metrics yet:
    training #0 is PLACING (the first samples arrive about a minute after it
    starts running)` and doesn't poll the stage until the status check, every
    60 seconds, sees it running. A running sub-job whose first samples haven't
    arrived yet says `no GPU metrics yet: training #0 is RUNNING; the first
    samples arrive about a minute after it starts`, and the tab keeps polling
    for them. A `log_probability` sub-job, which runs on no
    GPUs of its own, says `no GPU metrics for this sub-job`. A refresh that
    fails keeps what is cached and says `cached · refresh failed: …` in the
    warning color, with the error cut to fit the line; the TUI's error log has
    it in full. A job with no sub-jobs says so instead.
    It is cached under `jobs/<job>/metrics/` in the TUI cache directory, for
    one job at a time: opening a job's details removes every other job's
    cached telemetry, which downloads again when that job's Resources tab is
    next shown (logs and saved consoles stay). The
    selected sub-job downloads first, and each sub-job shows as soon as its
    chunks are in. The first open of a job on a machine downloads the selected
    sub-job's chunks for the window on screen (plus three minutes) first and
    shows them, with the whole job's stats `loading older samples…`, while the
    rest of its history downloads; later opens download only the new chunks.
    If that download fails, the header says `older samples failed to load: …`
    in the warning color, and the stats that need the older samples read
    `older samples not loaded` (so do the chart titles of a window that needs
    them) until a refresh gets them: `r`, or on a live job the 30-second
    refresh.
    Each refresh reads only what the download appended, and the first read of
    a long job's cache shows the selected sub-job's newest samples before the
    rest of its history.

#### Logs

The Logs tab lists the job's sub-jobs, training first, in a panel as wide as
their labels; select one to tail its logs (the zone-manager pod is the Ray
head, so a sub-job's worker output is included). `r` reloads the list and
stays on the selected sub-job. The tail
polls only while the Logs tab is on screen: on another tab
it stops and later resumes from where it left
off, so no lines are lost. While a job runs, its status is also re-read every
60 seconds, so the view switches to the saved console when the job ends.
Logs are cached locally while the TUI runs, so reopening a job in the same
session replays instantly without re-fetching from the server: under
`~/.cache/cortex-training/jobs/<job>/` (or `$XDG_CACHE_HOME`), overridable with
`CORTEX_TRAINING_TUI_CACHE_DIR`. The same directory holds display names and
telemetry. When the TUI ends (`q`, a crash, `kill`, or a closed terminal), it
removes everything it downloaded: every job's logs, saved consoles, telemetry
and tail cursors (`jobs/`); display names stay. While another TUI still runs
on the same cache, the one that ends leaves the cache to it, and the last one
to end removes it. A TUI killed outright (`kill -9`) can't clean up, so the
next one to start removes what it left. `--debug` uses a temporary cache of its
own, removed when it ends.

The pane draws each line as text only: escape sequences in a line (colours,
cursor moves, clipboard or title writes) are removed and other control
characters show as `�`, so nothing in a log acts on your terminal; a line that
rewrites itself with carriage returns (a progress bar) shows its last state.
A leading timestamp shows in local time, as `YYYY-MM-DD HH:MM:SS`, like every
other time in the TUI. A JSON log record, after the timestamp and any Ray
`(Actor pid=...)` prefix, shows as `time LEVEL logger: message`, with the
message's line breaks kept and an error or exception chain on the lines under
it; its other fields (host, ids) are left out. Rows are cut by terminal cells,
so wide (CJK, emoji) text stays inside the pane. `s` saves the lines as they
came.

The header's subtitle starts with the job's status. On the Logs tab it adds
`last line HH:MM:SS`, the newest live line's own timestamp in local time
(`last line received` with the time it arrived, for a line without one), and
any filter, level, or pause; the job's `reason` and `reason_details` are on
the summary line under it.

The line under the log pane is the Logs tab's status line. It appears when the
job read is slow, when the tail is waiting, paused, or restarted, when the
filter or level hides every line (`no lines match …`), after `s` (for 15
seconds), or when the pane doesn't hold a sub-job's oldest lines:

- **Reading the job.** The Logs tab reads the job first, for its sub-jobs and
  status. Until that read answers, the pane says `— reading job JOB_ID —`, and
  after two seconds the status line counts the wait, for example
  `reading the job: no answer yet after 52 s (up to 3 tries of 45 s each)`: a
  service that doesn't answer is asked again after each 45-second read timeout.
  `r` meanwhile says the job is still being read rather than asking again. If
  the read fails and the job can't be shown from the list's copy or the cache,
  the pane says why, for example `[error] couldn't read job JOB_ID: no answer
  from the service within 45 s (ReadTimeout) (3 tries)` (the Resources tab
  gives the same reason), and `r` reads it again.
- **Waiting to start.** While the job or the selected sub-job is `pending`,
  `placing`, or `initializing`, the status line says it is waiting, with the
  service's `reason` and `reason_details` (for example `quota_filled` and the
  GPUs it needs), and that the status is checked every 60 seconds (`r` checks
  now), instead of showing an error. A pending job isn't tailed at
  all, since the service refuses `tail-logs` until the job is placed; nor is a
  pending sub-job of a placed job, which waits for its own placement. A placing
  or initializing sub-job is tailed, and its first lines end the wait. When the
  60-second status check sees the job start (or, for that sub-job, run), the
  tail starts by itself; so does a tail that failed while the job started up
  when the TUI couldn't read the job to tell why. The service's answer counts
  too: a 409 "is pending placement" waits for placement whatever the status
  says (for example `UNKNOWN`, when the service couldn't read the job's state),
  and the tail starts once the status changes. A sub-job the service isn't
  routing to yet (a 400 "does not belong to session" while its job starts)
  also waits; its tail is tried again when its status or the job's changes,
  and, once both are running, at each status check while someone is using the
  view. `r` tries it at once. An error once the job runs ends the wait and
  stays on screen.
- **The job ends.** Once a job stops running, `tail-logs` answers 409, and 404
  once the job's session is gone. Neither is retried: the TUI reads the job once
  and, if it has ended, switches to the saved console at once.
- **The tail stops on an error.** Any other error while the job is live stays
  in the pane, and the status line says `tail stopped: <the service's answer>
  (r retries; ...)`, in place of any waiting note. `r` tries again at once, and
  so does each 60-second status check until the job ends: while the job is
  placing or initializing, and otherwise (it runs, or its status is
  unavailable) while someone has used the view in the last 15 minutes. While
  one request (with the client's own retries) has had no answer for 20
  seconds, the status line says so.
- **Status unavailable.** While the job's status is `UNKNOWN` (the service
  couldn't read it), the status line says its log may not answer; the tail is
  tried all the same, and the note goes once lines arrive or the status is back.
  If the tail stops on an error meanwhile, the status line says so instead
  (`... so does each status check while the job's status is unavailable`), and
  each status check tries the tail again, as above.
- **A refused position.** The tail resumes from the position (cursor) it last
  saved. If the service refuses that cursor with a 400 (one saved under an
  older service version, say), the TUI drops it, reads the end of the log
  instead, and marks the restart; lines in between may be missing.
- **A log ahead of the view.** While pages come back full (the log is written
  faster than one page a poll), the next read goes out after 0.2 seconds
  instead of the poll interval until the tail catches up; when it falls 5
  seconds or more behind, the status line says how old the newest line shown is.
- **A stuck tail.** On some service versions a log line of 256 KiB or more is
  never delivered, and the tail stops moving while the job keeps logging. When
  a running job's tail hasn't moved for 45 seconds, the TUI reads the end of
  the log. If that has lines the tail hasn't shown while the tail still brings
  nothing, the tail continues from the end, leaves out the lines already shown,
  and marks where it restarted; lines between the two may be missing, and
  `download-log --log-type stdout` has them once their chunks are uploaded. If
  the tail does bring lines, the log was only quiet, and they show in order.
  A read of the end that still reaches back to the stuck line brings nothing
  new, so getting past it can take a few of these reads (a minute or two).
- **No input for 15 minutes.** A `tail-logs` call on a running job currently
  counts as activity on the job, so a Logs tab left open would keep an idle job
  alive, and billing, past its idle timeout. After 15 minutes with no key
  press, click, or scroll while Logs is showing, the live tail pauses and says
  `paused: no input for 15 min (press p to resume)`. `p` resumes it from where
  it stopped, as does choosing another sub-job or `r`. The 15 minutes count only
  while the job runs: reads while it is placed or starts up don't count as
  activity. Once the job ends, the saved console replaces the paused tail.
- **Scrollback.** The pane keeps a sub-job's newest 20,000 lines (fewer when
  long lines wrap to many rows: it holds at most 200,000 rows). A line that
  would take more than 20 rows shows its first and last rows, around one that
  says how many are left out (`s` saves the whole line); in a narrow pane that
  row is shortened to fit, keeping the count. Opening a running
  job reads only its log's last 2,000 lines (a shorter log is read whole, from
  its first line, however fast it writes), and a cached tail is replayed up
  to its newest 2,000 lines. Whenever the pane doesn't hold the log's first
  lines, for one of these reasons or because it dropped older ones, the status
  line says how many log lines it holds, for example
  `last 2,000 lines · cortex-training download-log JOB_ID --log-type stdout for the full log`.
  With a filter or level set, it counts the newest matching lines it holds
  instead, such as `last 625 matching lines`; older matches may have left the
  view.

A finished job loads its saved console (the whole job's output, which can
take minutes for a long job) when its Logs tab is first shown: the selected
sub-job's first, shown as soon as it is in, then the others' in the
background, with the chunk count on the status line meanwhile. The pane says
which sub-job's console comes first, and, for a sub-job you select before its
console is in, that it comes with the others'. A job that ended more than 10
minutes earlier keeps that console in the cache (`jobs/<job>/console/`) for
the rest of the session, so reopening it is instant (the cache is emptied when
the TUI ends). The
kept consoles of all jobs take at most 1 GiB together: past that, the ones
opened least recently are removed (and downloaded again if opened). A
console fetched sooner, for example right after the job ended while the view
was open, may still lack its last lines, which are often the error: the pane
says so under it, and `r` downloads the chunks uploaded since. Until then, the
live tail's lines that came after the console's last one stay on screen after
it, under `— N newer lines from the live tail, not in the saved console yet —`,
so the end of the job doesn't take away lines you saw. A sub-job with no
console yet says instead that its saved logs may not all be uploaded yet; `r`
checks again for its console and its execution log.
A job that never got GPUs has no logs at all: one that failed with reason
`placement_timeout`, `placement_too_large`, `placement_failed` (any
`placement_*` reason), `hardware_unavailable` or `pending_timeout`, or one that
ended while every sub-job was still waiting to be placed (`pending`, `queued`,
`creating` or `placing`). Its pane says so
with the reason, for example "No log output: the job never got GPUs (reason:
placement_timeout), so no sub-job started or wrote logs.", and its preview says
the same instead of pointing to Logs.
Jobs that fail while they are created or start up may have no saved console.
For a failed job's sub-job without one, the pane shows the error records from
that sub-job's execution log (`_logs/<sub_job_id>/execution.jsonl` on the job's
Experiment) under a header, each with its error message and exception chain;
when the log has no error records, its last 20 records. If the role can't read
the job's Experiment, where saved consoles, execution logs, and GPU metrics
live, the pane says so in one line instead of showing a download error. When
the service can't find the Experiment for the role at all (`Linked experiment
is not found for job ...`), the line adds that it may no longer exist.

The filter (`/`) and the level key apply to the lines the view already has:
changing either repaints the pane and never restarts the tail or sends a
request. A saved console is searched whole, off the screen's thread, so keys
keep working while a long one repaints. The level key hides lines below the
chosen level, reading the level from a structured entry or from the line
itself (a JSON `level` field, or a word such as `INFO` or `ERROR`); lines with
no recognizable level always stay, and a record that carries an error or an
exception counts as `ERROR` whatever its level says. Both apply to live
lines, saved consoles, and the execution-log records shown for a failed job
without a console, the latter a whole record at a time (a match in its
exception chain keeps the record); their header stays.

The TUI also writes two files into your home directory: saved logs from the `s`
key (`~/cortex-training-<job8>-<type>-<index>.log`, such as
`~/cortex-training-<job8>-training-0.log`, where `<job8>` is the first eight
characters of the job id) and its own error log
(`~/.cortex-training-errors.log`). `s` saves the sub-job's saved console, else
the lines its tail cached, else the execution log the pane shows; with nothing
to save it says so and writes no file.

Copying works on every screen and tab. Highlighting is copying: drag the
mouse over any text and it is on the clipboard when you let go, so Cmd-V (or
the terminal's paste) pastes it; a double-click (a block) or a triple-click
(everything around it) highlights and copies too. A drag that starts in the
log, a list or a scrolling pane stays in it: held past its top or bottom, it
scrolls that area and the highlight grows with it. Text can't be highlighted
in a table, so a drag over table rows (the job list, the preview's sub-jobs)
picks whole rows instead: they show highlighted and are copied, aligned under
the column names, with a last column for each whole job or sub-job id (the Job
ID column cuts it short); held past the table's top or bottom, it scrolls the
table. A click drops a highlight; a drag is never a click (it doesn't fold a
job, pick a list entry or run a footer key), and a footer key runs when it is
clicked, not as it is pressed. Changing what a view shows (another tab,
sub-job, GPU, node or window) drops its highlight.

`ctrl+c` and a right-click copy the same way: what is highlighted
(again, without clearing it), else everything the view holds, including what
is scrolled out of view: on Logs the whole log (its newest lines up to 1 MiB);
on Resources the panel (each GPU's or node's value), the header, every stats
row and every chart; on the job list every job; in the preview all of it.

The TUI takes the mouse, so the terminal's own selection and its copy key
(Cmd-C) see nothing; instead, a copy goes to this computer's clipboard
(`pbcopy` on macOS; `wl-copy`, `xclip` or `xsel` on Linux) and to the
terminal's (OSC 52, which reaches your own computer's clipboard over SSH,
where some terminals ignore it). Escape sequences and control characters are
removed, and a one-line note at the right end of the title row says what was
copied (`copied 12 lines`), for three seconds; when only the terminal's
clipboard could take it, the note says so, with a tip, for eight. To select
with the terminal itself instead, hold a key while dragging: Option in iTerm2,
Fn in macOS Terminal, Option in VS Code's terminal once
`terminal.integrated.macOptionClickForcesSelection` is on. With the keys
panel open, a copy takes its list of keys too.

Keys in the job details view:

| Key | Action |
|-----|--------|
| `2` `3` | Logs / Resources tab |
| `l` `g` | The same tabs, as in the preview, except `l` on Logs (below) |
| `t` | Next tab |
| `w` | Cycle the window (Resources) |
| `Space` | Next sub-job, round to the first after the last, when the job has two or more; the view stays, back to `ALL`. `<` / `>` go back / forward (Resources) |
| `↑` `↓` | Pick GPU Resources or System Resources, `ALL` or one GPU or node in the panel; the charts follow the highlight (Resources) |
| `/` | Filter the current source (Logs) |
| `l` | Cycle minimum log level (INFO / WARNING / ERROR) (Logs; `L` works too); elsewhere the Logs tab |
| `p` | Pause / resume auto-scroll; scroll back freely while paused. Resumes a tail paused for no input (Logs) |
| `s` | Save the current source to `~/cortex-training-<job8>-<type>-<index>.log` (Logs) |
| `ctrl+c` / right-click | Copy what is highlighted, else everything the tab holds: the log (its newest lines, up to 1 MiB) on Logs; the panel, header, stats and every chart on Resources (any tab) |
| `[` / `]` | Narrow / widen the sources panel, which is otherwise sized to its labels; the log keeps at least 40 columns (Logs) |
| `PgUp` `PgDn` `Home` `End` | Scroll the log, also while the sources panel has the focus (Logs); scroll the stats and charts, also while the panel has the focus (Resources) |
| `r` | Refresh the visible tab (on Logs, the sub-job list and its tail) |
| `b` / `esc` | Back, or to the job list for a job opened directly (`esc` closes the keys panel first, and on Logs clears an active filter first) |
| `?` | Show or close the keys panel |
| `q` | Quit |

The `--poll-interval` flag is the minimum interval between log polls per
source, biasing toward server reliability over freshness (except while the
tail catches up with full pages, above). Without the flag it
is the config file's `poll_interval` when the config sets one, else `1.0s`. The
same config value is the CLI's result-polling interval, so the example
config's `0.5` makes the TUI poll a live log twice a second, and each
`tail-logs` call on a running job counts as activity on it; pass
`--poll-interval 1` to keep the TUI at one poll a second.

### JSON Output And Help

Commands other than the TUI write JSON to stdout, pretty-printed by default.
Use the global `--compact` flag for compact JSON, or pipe output to `jq`:

```bash
cortex-training --compact list
cortex-training get JOB_ID | jq '.sub_jobs'
```

Text is printed as UTF-8, so a comment or name in any script reads as written.
Control and invisible formatting characters (such as ESC, a bidi override, or
a zero-width space) stay `\u` escapes, so printing a job can't change your
terminal. When stdout's encoding isn't UTF-8 (a Windows pipe, or
`PYTHONIOENCODING=ascii`), every non-ASCII character is printed as an escape
instead. Either way, `jq` and other JSON parsers read the same values.

Use `cortex-training --help` to list commands and global flags, or
`cortex-training COMMAND --help` for command-specific arguments. Job-scoped
data-plane help includes the required global option:

```text
usage: cortex-training --job JOB_ID fwd-bwd [-h] json_file
```

### Environment Variables

Connection values can also come from:

```bash
CORTEX_TRAINING_CONNECTION
SNOWFLAKE_DEFAULT_CONNECTION_NAME
CORTEX_TRAINING_CONFIG
CORTEX_TRAINING_BASE_URL
CORTEX_TRAINING_HOST
SNOWFLAKE_HOST
CORTEX_TRAINING_PAT
SNOWFLAKE_PAT
CORTEX_TRAINING_DATABASE
SNOWFLAKE_DATABASE
CORTEX_TRAINING_SCHEMA
SNOWFLAKE_SCHEMA
CORTEX_TRAINING_ENDPOINT
```

`CORTEX_TRAINING_DISABLE_TELEMETRY` (truthy) skips OTLP client metrics on
Snowflake profile and PAT clients. `CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY`
(truthy) also emits successful outcomes for essential operations; failures
are emitted by default. See the [Python SDK reference](python-sdk.md#client-metrics).

`CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS` (truthy) sends pre-tokenized prompts
as JSON lists inside the request frame instead of as tensors. The request body
stays a DSSST1 frame either way.

`CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES` is the per-request JSON body
ceiling for byte `forward()` `/operation` posts (integer bytes, default 16 MiB,
minimum 16 KiB). Frames larger than that are split into request chunks.

`CORTEX_TRAINING_TUI_DEBUG_ACCOUNT_LIST` (`not_deployed`, `disabled`,
`forbidden` or `rest_disabled`; `ok` or unset is the default) makes the
account-level list of `cortex-training tui --debug` answer as an unavailable one
does, to preview what the job list's `a` (all schemas) says then; `unavailable`
makes it fail (503).
See [Job TUI](#job-tui).

### Troubleshooting

If you see `provide --base-url for local/mock use, or both --host and --pat`,
the CLI found a `host` but no PAT. Add `"pat": "..."` to `config.json` or set
`CORTEX_TRAINING_PAT`.

If no legacy config or complete direct connection is present, the CLI falls
back to the Snowflake Connector's configured default. If your profile is not
named `default`, pass `--connection NAME` or set
`SNOWFLAKE_DEFAULT_CONNECTION_NAME`.

If Snowflake rejects your PAT, the CLI keeps the original error and adds next
steps: `394400 (08001) ... Programmatic access token is invalid` from a
connection profile, or a `401` on a request that sent the PAT directly
(`config.json`, `--pat`, or `CORTEX_TRAINING_PAT`). The most common cause is a
user with no network policy; see
[Network policy requirement](../getting-started/authentication.md#network-policy-requirement).
A `401` on a connection-profile request gets no hint, because the PAT was
already accepted at login.

If you see `Invalid URL ... No scheme supplied`, the config is using a bare
Snowflake hostname as `base_url`. Use `host` for Snowflake PAT auth, or use a
full local/mock URL such as `http://localhost:8084` for `base_url`.

For server errors, the CLI prints any Snowflake request id and response body
returned by the service. Include those details when debugging a `500`.
Everything the CLI writes to stderr, error bodies and warnings included,
shows control and invisible formatting characters as escapes such as `\x1b`,
so an answer can't move your terminal's cursor, retitle its window, or write
its clipboard.
