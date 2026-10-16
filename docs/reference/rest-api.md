# Cortex Training REST API Reference

> **Status.** This page is a repository snapshot for the client version in this
> branch. The service's own OpenAPI specification remains the source of truth
> for the wire schema.
>
> **Scope.** This document describes the customer-facing Cortex Training REST
> surface used by this repository's `CortexTrainingClient`. It covers REST paths,
> request framing, asynchronous polling, client-visible schemas, checkpoints,
> generation, operations, and logs.
>
> **Sources used for this snapshot.** The client and wire implementation in
> `src/cortex_training/client.py` and `src/cortex_training/wire.py`, the
> command-line interface in `src/cortex_training/_cli.py`, their unit tests,
> and their unit tests are the local evidence for this document. The service's
> own OpenAPI specification remains the source of truth when it is available.
>
> **Important distinction.** `CortexTrainingClient` is a low-level transport client.
> ArcticTraining may put additional keys such as `context` and `processing`
> inside a forward/backward batch, but those keys are backend contracts, not
> additional REST fields.

---

## 1. Core concepts

### 1.1 Jobs and sub-jobs

A job is the top-level lifecycle object. One job owns one or more typed
sub-jobs:

- `training`: model training, optimizer steps, saves, and runtime loads.
- `sampling`: generation and sampling-side operations.
- `log_probability`: a log-probability worker configuration.

Only `training` and `sampling` sub-jobs can be submitted. Both the Python
client and the CLI reject a `log_probability` sub-job before sending the
request, with the location-prefixed message
`log_probability sub-jobs are not currently supported`; the short alias
`log_prob` is rejected the same way.

The common RL layout is one training sub-job and one sampling sub-job in the
same job.

There is no dedicated public log-probability request method in the current
client. `fwd()` and `fwd_no_grad()` are aliases for the generic `forward`
operation; they are not separate REST endpoints.

### 1.2 Sub-job identifiers

Internal sub-job ids use:

```text
{job_id}:{sub_job_type}:{index}
```

`sub_job_type` is `training`, `sampling`, or `log_prob`, and `index` is the
zero-based occurrence of that type. Examples:

```text
b1fcb345:training:0
b1fcb345:sampling:0
b1fcb345:log_prob:0
```

The create-job `job_type` is `log_probability`, while its internal id segment
is `log_prob`.

### 1.3 Control and data planes

- Control-plane calls return their result synchronously: create, get, list,
  capacity, cancel, experiment metadata, and checkpoint metadata/export.
- Data-plane calls normally return a `request_id`: forward/backward, step,
  save, load, generate, weight sync, and generic operations that schedule work.
  Poll the request until it reaches a terminal state.
- Some generic operations are synchronous and return their result directly.

### 1.4 Supported models

The machine-readable source of truth for model availability, workflow support,
context limits, and recommended configurations is
[`model-catalog/models.json`](../../model-catalog/models.json). Referenced
profiles live under [`model-catalog/profiles/`](../../model-catalog/profiles/).

This API reference does not duplicate the model table. Run
`python scripts/validate_model_catalog.py` after changing the catalog or a
profile.

---

## 2. Connection, authentication, and headers

### 2.1 Base URL

Paths in this document are relative to:

```text
https://{account_host}/api/v2/databases/{database}/schemas/{schema}/{endpoint}
```

`CortexTrainingClient` defaults `endpoint` to `cortex-training`.

```python
client = CortexTrainingClient.from_connection_name("training")
```

The SQL statements API used by execution-log download is outside this prefix:

```text
https://{account_host}/api/v2/statements
```

So is the account-level job list
([section 5.3](#across-the-account---get-apiv2cortex-training)), at a fixed
path that `endpoint` does not change:

```text
https://{account_host}/api/v2/cortex-training
```

If you point the client at a server that exposes the endpoint under a
different name, pass it explicitly: `endpoint="my-endpoint-name"`.

### 2.2 Authentication

`CortexTrainingClient.from_connection_name(...)` uses the Snowflake
Connector's resolved REST origin and live session token:

```http
Authorization: Snowflake Token="<session-token>"
```

The client reconnects once and replays the request when Snowflake explicitly
reports that the session token expired. Named profiles and the configured
default profile are supported.

Direct PAT compatibility is available through
`CortexTrainingClient.from_pat(...)`, which sends:

```http
Authorization: Bearer <PAT>
X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN
```

TLS verification is enabled unless `verify_ssl=False` is passed.

### 2.3 Content types and framing

| Call | Request content type | Body |
|---|---|---|
| Control-plane calls, `step`, `save`, `load`, `operation` | `application/json` | JSON |
| `forward-backward` | `application/octet-stream` | DSSST1 safetensors frame, optionally split into DSSST1 request chunks |
| `generate` | `application/octet-stream` | DSSST1 safetensors frame, optionally split into DSSST1 request chunks |
| `generate-stream` | `application/octet-stream` | DSSST1 safetensors frame, sent in one request and capped at 60 MiB |

Raw `torch.save`/pickle is not the current binary protocol. See
[section 9](#9-dssst1-binary-wire-protocol).

### 2.4 Snowflake request id

Responses may include `x-snowflake-request-id`. Include it when diagnosing an
HTTP failure. When the response carries it, the Python client prefixes
raised HTTP errors with `[snowflake request id: <id>]`.

### 2.5 Roles and access

Listing a job's checkpoints ([section 5.7](#57-list-checkpoints---get-job_idcheckpoints))
and exporting them ([section 5.8](#58-export-checkpoint)) are REST calls: a
role with only `MONITOR` on the account can make them. The checkpoint files
are stored on the job's Experiment, but these routes read them for the caller.

Saved logs and GPU metrics need more. `fetch_execution_logs()`,
`download_stdout_logs()` and `download_metrics()` resolve the job's experiment
run ([section 5.6](#56-experiment-run---get-job_idexperiment-run)) and then run
SQL `LIST` and `GET` on its `snow://experiment/...` artifacts
([section 12](#12-logs-and-events)). That SQL needs privileges on the job's
Experiment, so a `MONITOR`-only role can't read saved logs or GPU metrics.

When the service can't resolve the job's Experiment for the caller's role, it
answers `500`, worded per route:

| Route | Message |
|---|---|
| Experiment run ([section 5.6](#56-experiment-run---get-job_idexperiment-run)), which saved logs and GPU metrics start with, before any SQL runs | `Linked experiment is not found for job <job id>`: the role can't see the Experiment, or it no longer exists |
| List and export checkpoints ([section 5.7](#57-list-checkpoints---get-job_idcheckpoints), [section 5.8](#58-export-checkpoint)) | `Linked experiment is not found or not authorized for the training job` |

Both start with `Linked experiment is not found`. A retry gets the same
answer, so the client raises either at once ([section 3.5](#35-retries)). Ask
for access to the Experiment, or use the role that created the job.

A role that the service answers but that has no privileges on the Experiment
gets past the experiment-run lookup, and Snowflake then refuses the SQL `LIST`
with `Experiment '<name>' does not exist or not authorized` (or
`Insufficient privileges to operate on experiment '<name>'`), raised as the
Snowflake Connector's `ProgrammingError` ([section 12.2](#122-full-execution-log-download)).
The checkpoint routes still work for that role.

---

## 3. Status, polling, retries, and errors

### 3.1 Asynchronous request model

A scheduling call usually returns:

```json
{
  "request_id": "request-id",
  "job_id": "job-id"
}
```

Poll:

```text
GET /{job_id}/requests/{request_id}
```

`CortexTrainingClient.poll_request(job_id, request_id)` handles status normalization,
backoff, DSSST1 result decoding, and cursor-addressed result chunks.

### 3.2 Job states

Customer-facing job states are:

| State | Meaning |
|---|---|
| `pending` | Waiting for placement |
| `placing` | Infrastructure is starting (waiting for GPUs) |
| `initializing` | Placed; Cortex Training is warming up (loading the model, starting engines) |
| `running` | Ready for data-plane calls |
| `cancelled` | Cancelled by the caller |
| `terminated` | Platform teardown completed |
| `failed` | Terminal failure; inspect `reason` |

`initializing` is split out from `placing` so callers can distinguish
"waiting for GPUs" from "warming up"; it is customer-visible and not
collapsed by the server. `wait_for_job()` treats it like `placing` — neither
`running` nor terminal — so it keeps polling.

Legacy or internal responses can contain `done`, `unknown`, or enum names such
as `JOB_STATE_RUNNING`. `wait_for_job()` lowercases and removes the
`JOB_STATE_` prefix internally. `unknown` is what Get job and List jobs return
when the service can't reach the control plane for a job; see
[section 5.2](#52-get-job---get-job_id) for that row's shape.

Current client caveat: `wait_for_job()` raises early for `failed`, `done`,
`cancelled`, and `canceled`, but not `terminated`; a terminated job therefore
waits until the polling timeout.

### 3.3 Request states

| State | Class |
|---|---|
| `pending` | In flight |
| `running` | In flight |
| `streaming` | In flight on a streaming backend |
| `done`, `completed`, `succeeded` | Success |
| `failed` | Failure |
| `cancelled`, `canceled` | Cancelled |

Raw enum names such as `REQUEST_STATE_DONE` can also appear.
`poll_request()` normalizes them internally.

### 3.4 Poll timing

Defaults:

- Initial interval: `0.5` seconds.
- Backoff multiplier: `1.25`.
- Maximum interval: `6` seconds.
- Overall timeout: `1800` seconds.

All values are configurable on `CortexTrainingClient(...)`.

### 3.5 Retries

Every HTTP request has a default 30-second connect timeout and 600-second read
timeout. Callers can configure a shorter positive scalar or a `(connect, read)`
pair with `CortexTrainingClient(request_timeout=...)`.

The general request path retries connection/time-out failures and HTTP:

```text
307, 308, 404, 409, 429, 500, 502, 503, 504
```

Redirects are never followed; a 307 or 308 is retried on the original URL.

An error whose body says `Linked experiment is not found or not authorized` is
not retried, whatever its status (the service sends it as a `500`): the
caller's role can't resolve the job's Experiment, and a retry gets the same
answer ([section 2.5](#25-roles-and-access)). It is raised on the first attempt
instead of after about a minute of retries.

Capacity is the exception: it is attempted once with the shorter bounds
documented in [section 5.4](#54-capacity---get-capacity).

`tail-logs` doesn't retry 404 or 409: from it they mean the job isn't running
(still pending placement, or ended) or that its session is gone, and neither
clears on a retry. `get_job()` tells which; see [section 7.8](#78-tail-logs).
The experiment-run lookup doesn't retry a 500 that says `Linked experiment is
not found`; see [section 5.6](#56-experiment-run---get-job_idexperiment-run).

The default is ten retries after the first attempt, with exponential jitter.

Create-job uses the narrower connection-establishment retry predicate so an
ambiguous response does not accidentally create a second server-assigned job.

### 3.6 Errors

Non-2xx responses are raised through `requests.Response.raise_for_status()`.
Because that call ignores 3xx, any 3xx response is raised first as an
`Unexpected redirect <status> for url: <url>` HTTP error. Bodies may contain:

```json
{"message": "description", "code": 409}
```

or validation details:

```json
{
  "detail": [
    {"loc": ["body", "field"], "msg": "description", "type": "value_error"}
  ]
}
```

Most data-plane calls and request polls require a `running` job. `tail-logs` is
the exception allowed during `placing`.

---

## 4. Endpoint index

Paths are relative to the prefix in [section 2.1](#21-base-url).

| REST path | HTTP | Client method | Purpose |
|---|---|---|---|
| `/` | `POST` | `create_job`, `create_job_from_body` | Create a job |
| `/` | `GET` | `list_jobs` | List jobs, optionally filtered by status |
| `/capacity` | `GET` | `get_capacity`, `get_capacity_breakdown` | Account reservation and GPU usage, optionally broken down |
| `/{job_id}` | `GET` | `get_job`, `wait_for_job` | Job and sub-job status |
| `/{job_id}:cancel` | `POST` | `cancel_job` | Cancel a job |
| `/{job_id}/experiment-run` | `GET` | `get_experiment_run` | Resolve experiment/run metadata |
| `/{job_id}/checkpoints` | `GET` | `list_checkpoints` | List checkpoints |
| `/{job_id}/checkpoints/{checkpoint_id}:export` | `POST` | `export_checkpoint` | Export checkpoint file links |
| `/{job_id}/checkpoints/{checkpoint_id}` | `DELETE` | `delete_checkpoint` | Delete a checkpoint |
| `/{job_id}/forward-backward` | `POST` | `forward_backward` | Submit forward plus backward |
| `/{job_id}/step` | `POST` | `step` | Submit an optimizer step |
| `/{job_id}/save` | `POST` | `save` | Save a checkpoint |
| `/{job_id}/load` | `POST` | `load` | Load a checkpoint into a running training job |
| `/{job_id}/generate` | `POST` | `generate` | Batch generation |
| `/{job_id}/generate-stream` | `POST` | `generate_stream` | Start polling-based streaming generation |
| `/{job_id}/operation` | `POST` | operation helpers | Generic routed operation |
| `/{job_id}/requests/{request_id}` | `GET` | `get_request_status`, `poll_request` | Poll async work |

The operation endpoint supports the operation types in
[section 7](#7-generic-operation-envelope).

These calls are outside the Cortex Training prefix:

| REST path | HTTP | Client use | Purpose |
|---|---|---|---|
| `/api/v2/cortex-training` | `GET` | `list_jobs_page`, `iter_jobs` | List jobs across the account ([section 5.3](#across-the-account---get-apiv2cortex-training)) |
| `/api/v2/statements` | `POST` | artifact download methods | Resolve the current Snowflake identity |

---

## 5. Control-plane endpoints

### 5.1 Create job - `POST /`

Typed Python call:

```python
job_id = client.create_job(
    sub_jobs=[training_sub_job, sampling_sub_job],
    experiment_name=None,
    hardware="B200",
    idle_timeout_seconds=1800,
    pending_timeout_seconds=86400,
    comment="qwen3-1.7b lora sweep #3",
)
```

REST body:

```json
{
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": "Qwen/Qwen3-1.7B",
      "inference_config": {
        "max_seq_len": 2048,
        "n_gpus": 1
      }
    }
  ],
  "experiment_name": "optional-experiment",
  "hardware": "B200",
  "idle_timeout_seconds": 1800,
  "pending_timeout_seconds": 86400,
  "comment": "qwen3-1.7b lora sweep #3"
}
```

`sub_job_configs` must be a non-empty list carrying **zero or one** `training`
sub-job and any number of `sampling` sub-jobs. A second training sub-job is
rejected with `at most one training sub-job is supported per job`; the server
enforces the same rule for every caller. A `log_probability` (or `log_prob`)
sub-job is rejected client-side with
`sub_job_configs[INDEX].job_type: log_probability sub-jobs are not currently
supported`.

The typed path validates each `SubJobConfig`; `create_job_from_body()` checks the
outer body, the non-empty list, the training-sub-job count, and each sub-job's
`job_type` for the unsupported log-probability spellings before forwarding it.
It also normalizes every `peft_config` the body carries, rewriting it in
place and rejecting an invalid one before the request is sent (see
[section 8.2](#82-trainingconfig)). Everything else in a raw body, `optimizer`
included, is forwarded exactly as given. `cortex-training submit` repeats those
checks so `--dry-run` fails the same way without a connection.

#### GPU hardware - `hardware`

Optional. One of `H200`, `B200`, or `B300`; every sub-job in the job runs on
that GPU type. Omitted means `H200`. Any other value is rejected — the typed
`create_job()` path raises `ValueError` before sending, and the server rejects
an unknown value on the raw `create_job_from_body()` path. The typed path also
accepts the `Hardware` enum (`Hardware.B200`) in place of the string.

See [GPU hardware](../concepts/hardware.md) for the job-level model and how
capacity is scoped per type.

#### `idle_timeout_seconds`

Optional. Bounds how long the job may sit idle before the server reclaims it.

| Value | Behavior |
|---|---|
| Omitted | Server default applies, currently 30 minutes. |
| `0` | Disables reclamation for this job. Not a zero-second timeout — the job is not reclaimed for being idle, though it can still be cancelled or terminated. |
| `300` through `604800` | Use that many seconds. |

Negative values, `1`-`299`, values above `604800`, and non-integers are rejected.

#### `pending_timeout_seconds`

Optional. Bounds how long the job may stay `pending` waiting for capacity before
it fails with reason `pending_timeout`.

| Value | Behavior |
|---|---|
| Omitted | Server default applies, currently 24 hours. |
| `300` through `604800` | Use that many seconds. |

There is no disable value: `0` is rejected, along with negatives, `1`-`299`,
values above `604800`, and non-integers.

When either field is set, Get job and List jobs echo it back; when omitted it may
be absent from those responses.

#### `comment`

Optional free text that names the job, at most **1,024 bytes** of UTF-8 (bytes,
not characters: 512 two-byte characters fill it). The service stores it with
the job and returns it unchanged in `comment` from Get job and List jobs, or
`null` when none was set. It is set at creation only; no route changes it
later. The job TUI shows its first line as the job's name.

The server rejects a comment on a model deployment: a create request in which
any sub-job sets `inference_config.deployment_class`. The client never sends
one there: `create_job(comment=...)`, `create_job_from_body()` and
`cortex-training submit --comment` leave it out of such a request, unchecked,
and log a warning. For any other job, a comment over 1,024 bytes, one that is
not a string, or one that is not valid UTF-8 text (an unpaired surrogate, as a
command-line byte that isn't UTF-8 decodes to) raises `ValueError` before
anything is sent (`submit` fails the same way, including on `--dry-run`). A
blank comment (empty, or only whitespace) names nothing, so none is sent:
`create_job()`, `create_job_from_body()` and `submit` all leave it out, and
`submit --dry-run` prints the body without it. `submit --comment ''` therefore
removes the comment from the JSON.

Response:

```json
{"job_id": "job-id"}
```

### 5.2 Get job - `GET /{job_id}`

The response shape consumed by this repository is flat at each sub-job:

```json
{
  "job_id": "job-id",
  "hardware": "H200",
  "status": "running",
  "reason": "",
  "created_at": "2026-07-20T18:00:00Z",
  "updated_at": "2026-07-20T18:01:00Z",
  "state_changed_at": "2026-07-20T18:01:00Z",
  "submitted_by": "SOME_USER",
  "owner_role": "SOME_ROLE",
  "comment": "qwen3-1.7b lora sweep #3",
  "sub_jobs": [
    {
      "sub_job_id": "job-id:training:0",
      "job_type": "training",
      "status": "running",
      "model_name": "Qwen/Qwen3-1.7B",
      "training_config": {
        "optimizer": {"name": "AdamW", "lr": 0.0001},
        "max_seq_len": 2048,
        "train_batch_size": 8,
        "n_gpus": 8
      }
    }
  ]
}
```

`wait_for_job()` repeatedly calls this endpoint until `running`.

`state_changed_at` is when the job entered its current status. The TUI uses it
to tell how long ago a finished job ended.

`hardware` is the GPU type every sub-job runs on (see
[GPU hardware](#gpu-hardware---hardware)); a job created without one reports the
default, `H200`. List jobs carries it too. The TUI's Resources tab names it in
its header.

`submitted_by` (submitting user's name) and `owner_role` (owning role's name)
are best-effort and may be absent, e.g. if the user/role has since been
dropped or the requesting role no longer has access to it. Both fields also
appear on each job entry returned by `list_jobs()`
([section 5.3](#53-list-jobs---get-)).

`comment` is the text given at creation ([section 5.1](#comment)), unchanged,
or `null` when none was set. It is always present, holds at most 1,024 bytes of
UTF-8, can't be edited later, and is never set on a model deployment, which
rejects one.

`reason` is a stable machine token whose meaning depends on the status. While
the job is `pending` it says why it waits: `quota_filled` (your reserved quota
is held by your own other jobs), `awaiting_fleet_capacity` (shared
contention), or `infrastructure_error`. On a terminal status it is the cause,
for example `cancelled_by_client`, `idle_timeout` or `sub_job_failed`. It is
empty for `placing` and `initializing`. Switch on `reason`, not on the
sentence below. The exception is status `unknown`, where `reason` is itself a
sentence (see the fallback rows below), so match on the status there, not on
the sentence's wording.

`reason_details` is an optional human-readable sentence beside `reason`, in two
cases: while the job is `pending` (for example
`needs 8 GPUs; 64 currently active`), and on a job that failed with
`node_evacuated`, where it says what happened and whether the job can resume
from a checkpoint. It is absent otherwise, and its wording may change between
releases, so show it but don't parse it. A later Get job that no longer
carries `reason` or `reason_details` means they no longer apply; the job TUI
replaces both, with the status, the update times and the sub-jobs, from each
fresh answer (it keeps `created_at`, which never changes).

When the service can't reach the control plane for a job, Get job and List
jobs return that job with status `unknown` instead of an error. Such a row
carries `job_id`, a `reason` sentence that starts with `Job status
unavailable:` (for example `Job status unavailable: Control Plane is
unavailable`), `comment`, `owner_role` and `submitted_by`. Its `sub_jobs` are
the configs the job was created with, each with status `unknown` and no
`sub_job_id`, and there are no `created_at`, `updated_at`, `state_changed_at`
or `hardware` fields. The sentence names the cause: unless it says the job was
not found, the job may still be running. The job TUI shows the sentence as
"status unavailable", tells a not-found job apart by the words `not found` in
it (best-effort: the sentence's wording is not a contract), and derives the
missing sub-job ids, for such a row only,
from the rule in [section 1.2](#12-sub-job-identifiers)
(`{job_id}:{type}:{index}`, in submission order).

### 5.3 List jobs - `GET /`

Optional query:

```text
?status=running
```

REST response:

```json
{"jobs": [{"job_id": "job-id", "status": "running", "sub_jobs": []}]}
```

`CortexTrainingClient.list_jobs()` returns only the `jobs` list, not the outer object.
The client forwards the status string without validating an enum. Each entry
has the Get job shape ([section 5.2](#52-get-job---get-job_id)), including
`comment`, `reason`, `reason_details` and the `unknown` fallback rows. Like
the account-level list's pages, it takes `max_retries` and `request_timeout`
to bound the call, for example `list_jobs(max_retries=2, request_timeout=(10,
45))`; without them it retries and times out as in [section 3.5](#35-retries).

This route reads `status` differently from the account-level list below. It
takes one state, which the service matches itself, with no groups or comma
lists; `pending` also returns `placing` jobs; an unknown word returns an empty
list instead of 400; and `done` is accepted but can leave out finished jobs.
The CLI's `list` without `--all` therefore sends no `status` and filters the
rows itself, with the account-level meanings
([CLI reference](cli.md#manage-existing-jobs)).

#### Across the account - `GET /api/v2/cortex-training`

Lists the jobs the caller can see in every database and schema of the account,
newest first, one page at a time. The path is fixed, whatever the client's
`endpoint` is, and outside the per-schema prefix:
`https://{account_host}/api/v2/cortex-training`. A job is listed when the
caller can see the job, its schema and its database; jobs in dropped databases
or schemas are not listed. Every other job call stays on the per-schema routes.

The route answers only where the service has it and has enabled it for the
account. Until then it returns 404; see
[Account-level list errors](#account-level-list-errors).

Optional query parameters; an empty value counts as absent:

| Parameter | Value |
|---|---|
| `database` | Only jobs in this database. SQL identifier rules apply: an unquoted name is case-insensitive, a `"quoted"` one exact. A database the caller cannot see returns 400. |
| `schema` | Only jobs in this schema of `database`, by the same rules. Requires `database`: a schema name alone does not name one schema, and sending it without `database` returns 400. A schema the caller cannot see returns 400. |
| `status` | Comma-separated and case-insensitive: `pending`, `placing`, `initializing`, `running`, `failed`, `cancelled`, `terminated`, and the groups `active` (`pending`, `placing`, `initializing`, `running`) and `terminal` (`cancelled`, `terminated`, `failed`). `pending` does not include `placing`. Any other value returns 400, `done` included: `terminal` replaced it. A job whose live status cannot be read never matches. |
| `submitted_by` | Only jobs submitted by this user, by SQL identifier rules. Jobs with no recorded submitter never match. A name that matches no active user in the account, for example a mistyped or dropped user, returns 400 `submitted_by user not found or not authorized`. |
| `created_after` | Inclusive lower bound on creation time: an RFC 3339 timestamp with a `T` and an offset, such as `2026-10-01T08:00:00Z`, or `YYYY-MM-DD` for midnight UTC. A space instead of the `T`, or no offset, returns 400. Must be earlier than `created_before`. |
| `created_before` | Exclusive upper bound on creation time, in the same forms. |
| `limit` | Page size, 1 to 1000. Defaults to 100. |
| `page_token` | The previous page's `next_page_token`. Send it with the same filters; `limit` may change. |

The creation bounds compare with the time the service recorded the job, which
can differ slightly from the job's `created_at`.

REST response:

```json
{
  "jobs": [
    {
      "job_id": "job-id",
      "database_name": "CORTEX_TRAINING_DB",
      "schema_name": "PUBLIC",
      "status": "RUNNING",
      "created_at": "2026-10-01T08:00:00Z",
      "submitted_by": "SOME_USER",
      "sub_jobs": [{"sub_job_id": "job-id:training:0", "status": "RUNNING"}]
    },
    {
      "job_id": "other-job-id",
      "database_name": "ml",
      "schema_name": "Exp 1",
      "status": "UNKNOWN",
      "reason": "Job status unavailable: ...",
      "created_at": "2026-09-30T21:00:00Z",
      "sub_jobs": [{"status": "UNKNOWN"}]
    }
  ],
  "next_page_token": "opaque-token"
}
```

Each job carries the per-schema list's fields plus `database_name` and
`schema_name`, unquoted as stored: double-quote them to address the job unless
they are plain uppercase identifiers, as `for_job(job)` does. Jobs come newest
first by the service's job id, which follows creation order to within clock
skew.

Pages and tokens:

- Each call examines at most 5,000 jobs, so a page can hold fewer than `limit`
  jobs, or none, and still carry `next_page_token`. Keep requesting until the
  token is absent. Nothing is indexed by status, submitter or creation time, so
  a selective filter over a long history takes many calls; with
  `created_after`, the walk ends soon after it passes that time.
- `next_page_token` is opaque and sealed by the service: pass it back
  unchanged, with the same filters. A token sent with other filters returns 400
  `page_token does not match the request filters`, so compute a moving bound
  such as `created_after` once per walk. A token the service can no longer
  open, for example after it rotated the key that seals tokens, returns 400
  `Invalid page_token`; start again from the first page.

Jobs whose live status the service cannot read:

- With no `status` filter, such a job comes back with `status` `UNKNOWN`, a
  `reason` that starts `Job status unavailable`, no sub-job ids, and its
  `created_at`.
- With any `status` filter, it is left out and not counted, so a filtered page
  is not a count of what matched: an `active` list can miss jobs. If the status
  lookup itself fails, the whole page fails, usually with 503, 504 or 500,
  after the service has waited up to 300 seconds for it.

##### Account-level list errors

| Answer | Meaning |
|---|---|
| 400 | A bad filter or page token. The message says which, for example `status must be a comma-separated list of ...`, `Invalid database or schema scope`, `submitted_by user not found or not authorized`, or `Invalid page_token`. |
| 403 | Cortex Training is not enabled for the account, or its trial has ended; the per-schema routes answer the same. Under a `status` filter, a 403 can also come from a status lookup the service was refused. The service's 403s carry a JSON error that says which; a 403 without one did not come from these checks, and a network policy or a proxy in front of the service can send one. |
| 404 with an empty body | The service does not have this route yet. |
| 404 with JSON `Account-level job listing is not enabled for this account` | The route is there but not enabled for the account. Snowflake enables it per account. |
| 405 with an empty body | The Cortex Training REST API is turned off for the account; the per-schema routes answer the same. |
| 409, 429, 500, 502, 503, 504 | Transient, or a failed status lookup under a `status` filter. |

##### Client methods

`CortexTrainingClient.list_jobs_page(...)` reads one page and returns
`{"jobs": [...], "next_page_token": str | None}`; pass the token back as
`page_token`, with the same filters, for the next page. `iter_jobs(...)` takes
the same filters without `page_token` and yields jobs across pages as you
iterate. Both take `page_size` for `limit`, check their arguments on the call,
treat `None` and `""` alike, and never create the configured database:

- `page_size` must be an `int` from 1 to 1000; a `bool` or a `float` raises
  `ValueError`.
- `status` is a comma-separated string or a list. `done`, which the service no
  longer takes, is sent as `cancelled,terminated`, its older meaning. Legacy
  `DONE` rows match no filter here, so they show up only in unfiltered lists.
- `created_after` and `created_before` take a `date` (midnight UTC), a
  `datetime`, or a string: `YYYY-MM-DD`, or `YYYY-MM-DDTHH:MM[:SS[.fraction]]`
  (a space may replace the `T`) with an optional `Z` or `+HH[:MM]` /
  `-HH[:MM]` offset. A naive `datetime`, or a string without an offset, means
  UTC. They are sent as RFC 3339 in UTC with `Z`, such as
  `2026-10-01T06:30:00Z`.

403, 404 and 405 raise `AccountJobListUnavailable` at once, without a retry.
It subclasses `requests.HTTPError`, and its `reason` is `not_deployed` (a 404
without the not-enabled error, normally with an empty body; with some other
body, such as a proxy's page, the message says a proxy may have answered),
`disabled` (the JSON 404), `forbidden` (403) or `rest_disabled` (405). Its
message is one line, and for a 403 it quotes the service's reason when the
answer carries one. A 400 `Invalid page_token` raises `InvalidPageToken`, also
an `HTTPError`: `iter_jobs` then restarts from the first page once by itself,
skipping the jobs it already yielded by `job_id`, and raises on a second one,
whose message says the token failed again after the restart.
After such a restart, jobs created since the walk began come after older jobs
already yielded, so sort by `created_at` when order matters.

Other failed answers, other 400s included, raise `requests.HTTPError`.
Connection errors and timeouts raise `requests.ConnectionError` or
`requests.Timeout` once retries are spent. A page body that is not a job list
raises `ValueError`, and so do bad arguments, which are checked before any
request. Catch `requests.RequestException` (an `OSError`) and `ValueError` to
handle them all.

Each page is retried and timed out like the client's other calls: by default
up to ten retries after the first attempt, on connection errors, timeouts, 307,
308, 409, 429, 500, 502, 503 and 504, with a 30-second connect and a 600-second
read timeout ([section 3.5](#35-retries)). One failing page under a `status`
filter can therefore take many minutes. Both methods take `max_retries`
(retries after the first attempt; `0` means one attempt) and `request_timeout`
(seconds, or a `(connect, read)` pair) to bound each page, for example
`list_jobs_page(max_retries=2, request_timeout=(10, 45))`. The CLI's
`list --all --status` reads each page with `max_retries=2`. The job TUI's list
reads the configured schema (`list_jobs` with those bounds) until `a` asks for
every location. It then reads pages of 500 with those bounds, `created_after`
six months back and no `status`, at most 2,000 jobs or 20 pages per refresh;
when it stops with jobs left, it also reads `status=active` with the same
bounds, so active jobs created before the newest ones are listed. If this list
is unavailable when `a` asks for it, the TUI stays on the configured schema and
says why. On `not_deployed` or `disabled` after it was shown, the TUI lists the
configured schema in its place and asks again at most every 10 minutes
([CLI reference](cli.md#job-list)).

`for_job(job)` returns a client bound to a listed job's `database_name` and
`schema_name`; it borrows the parent's session and creates no database either.

### 5.4 Capacity - `GET /capacity`

This account-scoped endpoint takes no account id from the caller. The server
resolves the account from the authenticated session. Besides `hardware`
(below), an optional `breakdown` query parameter splits the usage numbers; see
[Breakdown](#breakdown---breakdown).

Optional query:

```text
?hardware=B200
```

`hardware` scopes the numbers to one GPU type and takes the same values as the
create-job field: `H200`, `B200`, or `B300`, defaulting to `H200` when omitted.
`get_capacity()` sends the parameter only when you pass one, and rejects an
unknown value client-side. The CLI `capacity` command, with no `--hardware`
flag, calls this endpoint once per type and wraps the results in
`capacity_by_hardware`.

A known type that the account does not support is rejected with `409` and code
`517604`, for example:

```json
{
  "code": "517604",
  "message": "Cortex training getCapacity failed: unsupported hardware \"B200\": supported hardware is H200, B300"
}
```

The CLI lists such types in `unsupported_hardware` instead of failing, and
fails on any other error.

The SDK attempts each capacity lookup once with at most a 10-second connect
timeout and a 30-second read timeout; a shorter configured request timeout still
wins. This endpoint is a fast control-plane read, so a silent network, proxy, or
service hop fails rather than blocking indefinitely.

```json
{
  "has_reservation": true,
  "max_total_gpus": 64,
  "reserved_gpus": 64,
  "in_use_gpus": 8,
  "pending_gpus": 16,
  "available_gpus": 40
}
```

- `has_reservation`: whether `max_total_gpus` is a *guaranteed* commitment
  rather than a best-effort per-account cap. When false, placement draws on the
  shared pool, so `available_gpus` is an upper bound.
- `max_total_gpus`: the account's GPU ceiling — its commitment when
  `has_reservation`, else its per-account cap. `-1` means no ceiling, `0` a real
  quota of zero, `> 0` the limit.
- `reserved_gpus` **(deprecated)**: use `max_total_gpus` with
  `has_reservation` instead.
- `in_use_gpus`: GPUs the account holds — `placing`, `initializing`, and
  `running` jobs. Queued work is not counted here; see `pending_gpus`.
- `pending_gpus`: GPUs requested by jobs still waiting for capacity
  (`pending`). Claims quota, so it reduces `available_gpus`.
- `available_gpus`: what a new job could start with now — ceiling headroom
  capped by currently schedulable capacity. Zero does not mean blocked: a
  submit within the ceiling is still accepted and queued.

Proto3 JSON may omit false or zero fields, so an account holding nothing under a
zero ceiling responds with literally `{}`. `get_capacity()` fills in the
documented defaults for all six keys.

#### Breakdown - `breakdown`

Optional. A comma-separated list of dimensions that split `in_use_gpus` and
`pending_gpus`:

| Dimension | Splits the usage by |
|---|---|
| `job_type` | Sub-job type |
| `submitted_by` | The user who submitted each job |

Names are case-insensitive; blanks and repeats are ignored. More than 8
names, or a name not matching `[a-z_]{1,32}`, is HTTP `400`. A well-formed
name the server does not know is listed in `unsupported_dimensions`. Without
the parameter the response has no `breakdown` key.

The breakdown is enabled per account. Until it is enabled for yours, each
requested dimension reports `DISABLED`. Example for `?breakdown=job_type,submitted_by`:

```json
{
  "has_reservation": true,
  "max_total_gpus": 64,
  "reserved_gpus": 64,
  "in_use_gpus": 44,
  "pending_gpus": 16,
  "available_gpus": 4,
  "breakdown": {
    "unsupported_dimensions": [],
    "job_count": 6,
    "truncated": false,
    "by_job_type": {
      "status": "OK",
      "groups": [
        {"job_type": "TRAINING", "in_use_gpus": 28, "pending_gpus": 16,
         "sub_job_count": 5},
        {"job_type": "SAMPLING", "in_use_gpus": 16, "pending_gpus": 0,
         "sub_job_count": 2},
        {"job_type": "LOG_PROBABILITY", "in_use_gpus": 0, "pending_gpus": 0,
         "sub_job_count": 0}
      ]
    },
    "by_submitted_by": {
      "status": "OK",
      "groups": [
        {"kind": "USER", "submitted_by": "YOUR_USER", "is_caller": true,
         "in_use_gpus": 24, "pending_gpus": 0, "job_count": 2},
        {"kind": "USER", "submitted_by": "SOME_USER", "is_caller": false,
         "in_use_gpus": 8, "pending_gpus": 16, "job_count": 2},
        {"kind": "OTHER", "in_use_gpus": 8, "pending_gpus": 0,
         "job_count": 1},
        {"kind": "UNATTRIBUTED", "in_use_gpus": 4, "pending_gpus": 0,
         "job_count": 1}
      ]
    }
  }
}
```

- `unsupported_dimensions`: always present, possibly empty.
- `job_count`: jobs behind `in_use_gpus` and `pending_gpus`, including those
  holding no GPUs. Present only when a dimension is `OK`.
- `truncated`: the account had more jobs than the server attributes one by
  one; the rest count as `UNATTRIBUTED`. Present only when a dimension is `OK`.
- `by_job_type` / `by_submitted_by`: present only when requested.

Each dimension has a `status` and `groups`. Every status comes with HTTP
`200` and the six totals unchanged:

| `status` | Meaning |
|---|---|
| `OK` | `groups` adds up to `in_use_gpus` and `pending_gpus`. |
| `DISABLED` | Not enabled for the account. `groups` is empty. |
| `UNAVAILABLE` | Cannot be computed for this request, including when the service does not support it yet. `groups` is empty. |

Read the split from `groups`; a newer service may add statuses.

`by_job_type.groups[]`: `TRAINING`, `SAMPLING` and `LOG_PROBABILITY` first,
then any other type (`UNSPECIFIED` or `UNKNOWN` when the service cannot name
it). A job counts under each of its sub-jobs' types. `log_probability`
sub-jobs and sub-jobs that set `multiplex_job_id` add 0 GPUs but count in
`sub_job_count`.

`by_submitted_by` names the submitter of every job the caller can see. Each
job counts in exactly one group:

| `kind` | Jobs |
|---|---|
| `USER` | Submitted by `submitted_by`, and visible to the caller. `is_caller` marks the caller's row. |
| `OTHER` | Not visible to the caller. |
| `UNATTRIBUTED` | No known submitter: none was recorded, the user no longer exists, or the job is past the `truncated` limit. |

`USER` rows come first, most GPUs (`in_use_gpus + pending_gpus`) first and
then by name, then `OTHER`, then `UNATTRIBUTED`. Find the caller's row by
`is_caller`, not position. The breakdown never contains job ids or database
or schema names.

`get_capacity_breakdown(hardware=None, *, dimensions=("job_type",
"submitted_by"))` sends this parameter. It keeps `groups` whenever the server
sends them and passes `status` through as sent. A missing `status` reads as
`OK` with groups and `UNAVAILABLE` without, and a dimension missing from the
response (including from a server without this parameter) is `UNAVAILABLE`.
It fills missing counts with `0` and always lists the three known job types
when there are groups. The CLI exposes it as
[`capacity --breakdown`](cli.md#capacity-breakdown).

### 5.5 Cancel job - `POST /{job_id}:cancel`

No body. Pending, placing, and running jobs enter cancellation. Repeating a
cancel while a job is cancelling or already cancelled is an idempotent success.
Terminated or failed jobs return a precondition/conflict-style error.

`cancel_job()` returns `None`.

### 5.6 Experiment run - `GET /{job_id}/experiment-run`

```json
{
  "experiment_name": "DB.SCHEMA.EXPERIMENT",
  "experiment_run_name": "RUN_NAME"
}
```

`fetch_execution_logs()` uses these values to locate the run's stage.

When the service can't resolve the job's Experiment for the caller, this route
answers 500 `Linked experiment is not found for job <job id>`: the caller's
role can't see the Experiment, or it no longer exists. (The checkpoint routes
word the same failure `Linked experiment is not found or not authorized for
the training job`.) A retry can't change either answer, so
`get_experiment_run()` raises a 500 whose message starts `Linked experiment is
not found` at once. Other failures keep the retries of
[section 3.5](#35-retries).

### 5.7 List checkpoints - `GET /{job_id}/checkpoints`

```json
{
  "checkpoints": [
    {
      "checkpoint_id": "CHECKPOINT_ID",
      "created_at": "2026-07-20T18:05:00Z",
      "checkpoint_type": "resumable"
    }
  ]
}
```

Each record carries only these three fields: no step or loss. `checkpoint_id`
is the id the service assigned at the save, `cp_<uuid>` in current releases: a
save request's own `checkpoint_id` is not forwarded
([section 6.3](#63-save-checkpoint---post-job_idsave)). Treat it as opaque and
pass it back as listed. `created_at` is the save time. `checkpoint_type` is
absent when the save didn't set one; such a checkpoint is resumable, so read
the field with that default, as the examples in
[section 13.3](#133-save-and-runtime-load) do. A role with only `MONITOR` on
the account can list checkpoints ([section 2.5](#25-roles-and-access)).
`list_checkpoints()` never creates the configured database: when it doesn't
exist, the service's error is raised.

`list_checkpoints()` returns only the `checkpoints` list. The CLI exposes this
method with a required job id and restores the server-shaped JSON envelope:

```bash
cortex-training checkpoints JOB_ID
```

The command prints `{"checkpoints": [...]}`. As with other commands, compact
output is selected with the global option:

```bash
cortex-training --compact checkpoints JOB_ID
```

### 5.8 Export checkpoint

```text
POST /{job_id}/checkpoints/{checkpoint_id}:export
```

No body.

```json
{
  "checkpoint_id": "CHECKPOINT_ID",
  "files": [
    {
      "filename": "model.safetensors",
      "url": "https://presigned.example/...",
      "size_bytes": 1073741824
    }
  ],
  "expires_at": "2026-07-20T23:10:00Z"
}
```

Each `url` is a presigned download link that lasts 5 hours from the export;
`expires_at` is when the links stop working. Export again for fresh links.
After that, the storage service refuses a link: Google Cloud Storage answers
`400` (`ExpiredToken`), S3 and Azure `403`. `filename` is relative and may
include a folder, and `size_bytes` is the file's size. The links carry their
own authorization: fetch them without the Snowflake `Authorization` header,
which must not reach the storage service. A role with only `MONITOR` on the
account can export ([section 2.5](#25-roles-and-access)).

`export_checkpoint()` returns this object unchanged, and like
`list_checkpoints()` never creates the configured database. A `404` (a wrong
job or checkpoint id) is retried like any other before it is raised
([section 3.5](#35-retries)), and so is a `409`
`Checkpoint <checkpoint id> directory is empty or has no metadata.json`: the
checkpoint's folder holds no finished save (one still being written, or one
that never completed), and the list in
[section 5.7](#57-list-checkpoints---get-job_idcheckpoints) leaves such a
checkpoint out. Either ends the call after about a minute of retries.

### 5.9 Delete checkpoint

```text
DELETE /{job_id}/checkpoints/{checkpoint_id}
```

No body, and none returned: `200`/`204` on success, so `delete_checkpoint()`
returns `None`. Unlike `:export`, this is a plain `DELETE` verb on the checkpoint
resource. A missing checkpoint is not treated as success — the server's status
(e.g. `404`) propagates as a raised error.

---

## 6. Data-plane endpoints

### 6.1 Forward/backward - `POST /{job_id}/forward-backward`

The body is a DSSST1 frame. The standard object is:

```python
{
    "args": (),
    "kwargs": {
        "input_ids": input_ids,
        "labels": labels,
    },
}
```

ArcticTraining can add backend-owned `context` and `processing` keys.

Build a basic frame with:

```python
payload = serialize_forward_backward_args(
    args=(),
    kwargs={"input_ids": input_ids, "labels": labels},
)
request_id = client.forward_backward(job_id, payload)
result = client.poll_request(job_id, request_id)
```

Or serialize an extended batch directly:

```python
from cortex_training import wire

payload = wire.dumps(
    batch,
    metadata={
        "response_options": {
            "format": "dssst1",
            "delivery": "chunked",
        }
    },
)
```

`forward_backward()` **always** wraps the frame in a DSSST1 request-chunk
envelope, even when it fits in a single chunk, so every logical operation has a
caller-generated identity for idempotent replay. Frames larger than 60 MiB are
split across several chunks. Each chunk is posted to the same path; only the
last response may carry the `request_id`. See
[section 9.3](#93-request-chunking).

Typical polled result:

```json
{
  "job_id": "job-id",
  "avg_loss": 1.0237,
  "metrics": {},
  "post_process_outputs": {}
}
```

The exact metrics are backend/loss dependent. The current ArcticTraining
forward/backward response assembler intentionally returns
`post_process_outputs` as an empty object; callers must not assume that
`compute_logprobs` appears there.

When the training sub-job sets `router_replay.mode = "best_effort"` (see
[section 8.2](#82-trainingconfig)), `metrics` of a forward/backward request that
carries router-replay ids also reports how the request's rows were routed. Each row is counted once, however rows are sharded across
data-parallel and sequence-parallel ranks. A "fresh" row is one whose routing
no sampling worker held, so the trainer's own MoE gate routed it.

| Key | Meaning |
|---|---|
| `router_replay/rows_sent` | Rows in the request that carried a router-replay id |
| `router_replay/rows_replayed` | Rows routed with the sampler's captured routing |
| `router_replay/rows_fresh` | Rows routed by the trainer's own gate |
| `router_replay/replayed_fraction` | `rows_replayed / rows_sent` (`0.0` for a request with no rows) |
| `router_replay/tokens_replayed` | Valid tokens in replayed rows (attention-mask count; the padded length when the request has no attention mask) |
| `router_replay/tokens_fresh` | Valid tokens in fresh rows (attention-mask count; the padded length when the request has no attention mask) |

In the default `"strict"` mode these keys are absent.

### 6.2 Optimizer step - `POST /{job_id}/step`

```json
{"learning_rate": 0.00002}
```

The field is optional. `step(job_id)` sends `{}`.

Immediate response:

```json
{"request_id": "request-id", "job_id": "job-id"}
```

Typical result:

```json
{"global_steps": 12, "last_lr": 0.00002}
```

`last_lr` may be a scalar or list depending on the backend.

When per-step peak-memory reporting is enabled (see
[section 8.5](#85-memory-diagnostics-settings)), the result also carries a
`peak_memory` object of byte counts:

```json
{
  "global_steps": 12,
  "last_lr": 0.00002,
  "peak_memory": {"gpu_peak": 123456789, "cpu_peak": 987654321}
}
```

`gpu_peak` is the job-wide maximum GPU high-water mark across ranks for the
current `/forward-backward` plus `/step` cycle; `cpu_peak` is rank 0's current
process RSS. The key is absent when the setting is disabled.

### 6.3 Save checkpoint - `POST /{job_id}/save`

```json
{
  "checkpoint_id": "optional-tag",
  "checkpoint_type": "weights-only"
}
```

Both fields are optional in the Python client request. When `checkpoint_type`
is supplied, the client lowercases and validates it as:

- `resumable`: training weights plus optimizer/training state.
- `weights-only`: Hugging Face-style model assets suitable for sampling
  initialization.

The backend default is `resumable`.

Compatibility caveat: `checkpoint_id` is not represented in the server's
`SaveRequest`, so a caller-selected value is not forwarded. Use the
server-assigned id from the result's `stage_path` or the job's checkpoint list
(below), rather than relying on a caller-selected value.

Immediate response:

```json
{"request_id": "request-id", "job_id": "job-id"}
```

On `release_20260903_175601`, the polled `weights-only` save result observed on
2026-09-07 carries no `checkpoint_id` key:

```json
{
  "job_id": "job-id:training:0",
  "stage_path": "s3://<stage>/.../checkpoints/cp_00000000-0000-4000-8000-000000000001/global_step1/",
  "checkpoint_path": "/tmp/dss.ray_dss/ds/job-<job-id>:training:0/weights-only",
  "checkpoint_tag": "global_step1",
  "version": 1.0
}
```

The durable checkpoint id is the `cp_<uuid>` path segment of `stage_path`
(`/checkpoints/(cp_[0-9a-fA-F-]+)/`), and `GET /{job_id}/checkpoints` lists the
same id with its `checkpoint_type` and `created_at`. Resolve the id from either;
`checkpoint_path` is a mount inside the saving job's containers and does not
resolve from any other job.

### 6.4 Runtime load - `POST /{job_id}/load`

```json
{
  "checkpoint_id": "global_step12",
  "source_job_id": "optional-source-job",
  "target_sub_job_id": "optional-job-id:training:0"
}
```

`checkpoint_id` is required. `source_job_id` loads from another job's checkpoint
store. `target_sub_job_id` routes the load to a specific training sub-job of the
session. The route is asynchronous:

```python
request_id = client.load(
    job_id,
    checkpoint_id="global_step12",
    source_job_id=source_job_id,
    target_sub_job_id=f"{job_id}:training:0",  # optional
)
result = client.poll_request(job_id, request_id)
```

When `target_sub_job_id` is omitted, the service routes the load to the
session's training sub-job (the historical behavior). When supplied, the sub-job
must belong to this session — otherwise the request fails with `400` — and must
be a training sub-job; loading into a non-training (e.g. sampling) zone is
rejected with `501`.

#### Discovering sub-job IDs

Sub-job IDs follow the format `{job_id}:{job_type}:{index}`, for example:
`b1fcb345:training:0`. To discover available training sub-jobs:

```python
job = client.get_job(job_id)
training_sub_jobs = [
    sj for sj in job["sub_jobs"] if sj["job_type"] == "training"
]
for sj in training_sub_jobs:
    print(f"ID: {sj['sub_job_id']}, DP size: {sj['training_config']['n_gpus']}")
```

#### When to use target_sub_job_id (load)

A job has at most one training sub-job, so omit `target_sub_job_id` to use the
default. Use it when you want explicit routing control — for example naming the
sub-job in tooling that must not rely on the server's default resolution. It
selects a training sub-job; the sampling-side selector is `target_sub_job_ids` on
weight sync.

#### DP size compatibility

If the target sub-job has a different `n_gpus` than the checkpoint's source
sub-job (changing DP size), the target sub-job **must** have been created with
`load_optimizer_states: false` in its `SubJobConfig`. The optimizer states are
DP-sharded and cannot be resized. This is a **creation-time** setting; it cannot
be changed at runtime. The server will reject incompatible loads.

### 6.5 Create-time checkpoint initialization

Create-time initialization is not `resume_from_checkpoint` in the typed
`cortex-training` API. To initialize from a checkpoint saved by Cortex
Training, use:

```json
{
  "source_checkpoint_info": {
    "checkpoint_id": "checkpoint-id",
    "source_job_id": "source-job-id"
  }
}
```

This object is a field on `SubJobConfig`. The server stamps scoped `stage_info`
credentials; clients should not provide credentials themselves.

To initialize from a complete Hugging Face safetensors directory uploaded to a
Snowflake stage, use the alternative source field:

```json
{
  "source_checkpoint_info": {
    "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
  }
}
```

`external_stage_path` is mutually exclusive with both `checkpoint_id` and
`source_job_id`. The service resolves the stage and credentials under the
submitting role. The source must be an S3-backed stage containing a flat,
uncompressed Hugging Face safetensors model directory. See
[Start a Job From Your Own Weights](../guides/training/external-weights.md)
for the upload workflow and file requirements.

When initializing from a checkpoint saved by Cortex Training, sampling requires
a `weights-only` checkpoint. The source job no longer needs to be running after
the checkpoint has been saved.

For training initialization from a Cortex Training `resumable` checkpoint, set
`training_config.load_optimizer_states=false` (see
[section 8.2](#82-trainingconfig)) to restore weights only with a fresh
optimizer. This is required when the new job changes data-parallel size, since
the DP-sharded optimizer cannot be resized.

### 6.6 Generate - `POST /{job_id}/generate`

Python call:

```python
request_id = client.generate(
    job_id,
    prompts=["Hello", [1, 2, 3]],
    sampling_params={"max_tokens": 64, "temperature": 0.7},
    routing_key="conversation-1",
    strict=False,
)
```

Logical request object inside the DSSST1 frame:

```json
{
  "prompts": ["Hello", [1, 2, 3]],
  "sampling_params": {"max_tokens": 64, "temperature": 0.7},
  "routing_key": "conversation-1",
  "strict": false
}
```

Rules:

- `prompts` is a list of string prompts and/or token-id lists. A flat list of
  integers such as `[1, 2, 3]`, and a bare 1-D tensor, are each read as one
  pre-tokenized prompt rather than as a batch. For a batch of several prompts,
  use the nested form `[[1, 2, 3], [4, 5]]`.
- `sampling_params` may be one object or a list of objects/null values aligned
  with `prompts`.
- `routing_key` may be one string or an aligned list of strings/null values.
- `strict` controls strict routing-key affinity.

Pre-tokenized prompts travel in the frame's tensor section as `int32` tensors,
not as JSON numbers in the frame header. A token id that does not fit `int32`
falls back to `int64` for that prompt, which only doubles its wire size. String
prompts are not converted — the server tokenizes them. Setting
`CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS` to a truthy value puts the token ids
back in the header as JSON lists; the request body is a DSSST1 frame either
way.

For pre-tokenized prompts, the client fetches and caches the sampling sub-job's
`inference_config.max_seq_len`. It rejects a prompt when
`len(prompt) >= max_seq_len`, preserving room for at least one output token.
The check runs before the tensor conversion, so the environment variable above
does not disable it. String prompts are left for the server tokenizer to
validate.

Generate uses the same DSSST1 response options as forward/backward, but unlike
forward/backward it sends the frame unwrapped when it fits, and only splits it
into request chunks above 60 MiB. See [section 9.3](#93-request-chunking).

Typical polled result:

```json
{
  "job_id": "job-id",
  "results": [
    {
      "text": " generated text",
      "token_ids": [101, 102],
      "finish_reason": "stop"
    }
  ]
}
```

The backend can add fields such as log probabilities or action masks. For a
request submitted by the same `CortexTrainingClient` instance, `poll_request()`
converts tensor values under `results` back to Python lists.

#### Per-token log probabilities

On server versions that pass vLLM's `logprobs` and `prompt_logprobs` sampling
options through, two optional `sampling_params` keys add log probabilities to
each result:

| Key | Result field | Contents |
|---|---|---|
| `logprobs: 0` | `logprobs` | One entry per generated token, aligned with `token_ids` |
| `prompt_logprobs: 0` | `prompt_logprobs` | One entry per prompt token; the first entry is `null` because nothing precedes the first token |

Each entry maps a token id to an object that includes `logprob`: the
log probability of that token given everything before it. With `0`, the map
holds only the token that actually appears at that position; a positive value
`k` also includes the `k` most likely alternatives. The token id is a JSON
object key, so it can arrive as a string.

`prompt_logprobs` scores a known token sequence without decoding or
re-tokenizing it: send the sequence as a pre-tokenized prompt with
`max_tokens: 1` and read `prompt_logprobs[i]` as the log probability of prompt
token `i`. The prompt must stay shorter than the sampling sub-job's
`max_seq_len`. The
[on-policy distillation recipe](../../recipes/distillation/README.md) scores
student rollouts with a teacher this way.

### 6.7 Streaming generate - `POST /{job_id}/generate-stream`

`generate_stream()` accepts the same logical fields as `generate()` and encodes
them into the same DSSST1 frame, including the `int32` tensor encoding of
pre-tokenized prompts described in [section 6.6](#66-generate---post-job_idgenerate).

Unlike `generate`, this path is not request-chunked: the client sends the frame
in a single POST, and the encoded body must not exceed 60 MiB.

**Breaking change.** `POST /{job_id}/generate-stream` no longer accepts a
UTF-8 JSON body. There is no dual encoding and no negotiation — the frame is
the only accepted request form. `CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS`
changes how prompts are encoded *inside* the frame; it does not turn the body
back into JSON. The response is unchanged.

Immediate response:

```json
{
  "request_id": "request-id",
  "job_id": "job-id",
  "count": 2
}
```

Poll the unified request endpoint with `max_events`:

```python
status = client.get_request_status(
    job_id,
    request_id,
    max_events=64,
)
```

Events are opaque backend objects. Common event shapes are:

```json
{"type": "result", "index": 0, "result": {"text": "text", "token_ids": [1, 2]}}
```

```json
{"type": "error", "index": 1, "error": "description"}
```

```json
{"type": "done", "completed": 1, "failed": 1}
```

Streaming event delivery advances a server-side delivery cursor. There is no
client-supplied retry cursor for these stream events, so losing a successful
poll response can lose the events consumed by that response.

### 6.8 Poll request - `GET /{job_id}/requests/{request_id}`

Optional query parameters:

| Parameter | Meaning |
|---|---|
| `max_events` | Event/result-chunk count; server default is 16 and current cap is 512 |
| `cursor` | Retry cursor for cursor-addressed DSSST1 result chunks |

Example:

```json
{
  "request_id": "request-id",
  "status": "done",
  "created_at": "2026-07-20T18:00:00Z",
  "updated_at": "2026-07-20T18:00:01Z",
  "result": {},
  "error": "",
  "events": [],
  "next_cursor": ""
}
```

There are two distinct event-delivery modes:

1. Streaming-generation events are destructively drained by `max_events`.
2. Large DSSST1 results use `result_chunk` events and `next_cursor`.
   `poll_request()` echoes `next_cursor` as `cursor`, validates each chunk hash,
   reassembles the frame, and decodes the result.

`poll_request()` returns `{}` if a successful terminal status has no result. It
raises `RuntimeError` for failed/cancelled status and `TimeoutError` at the
configured deadline.

---

## 7. Generic operation envelope

```text
POST /{job_id}/operation
Content-Type: application/json
```

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {}
}
```

- `operation_type` is required.
- `sub_job_id` and `sub_job_type` are optional routing hints. If both are
  supplied, they must resolve to the same target.
- `payload` is operation-specific.
- Byte payloads passed to `CortexTrainingClient.forward()` are converted to
  `{"payload_b64": "...", "content_type": "application/octet-stream"}` inside
  the JSON envelope.
- Operation responses are opaque. If a response contains a `request_id`, poll
  it; otherwise consume it inline.

The service accepts these operation types:

| Operation type | Client method | Execution |
|---|---|---|
| `forward` | `forward`, `fwd`, `fwd_no_grad` | Async |
| `weight-sync` | `weight_sync` | Async |
| `bootstrap-router-replay` | `bootstrap_router_replay` | Inline; poll if a deployment returns a request id |
| `router-replay-discard` | `router_replay_discard` | Inline |
| `reset-prefix-cache` | `reset_prefix_cache` | Inline |
| `cancel-request` | `cancel_request` | Inline acknowledgement |
| `tail-logs` | `tail_logs`, `stream_logs` | Inline cursor page |

Only `tail-logs` is allowed while the job is `placing`; all other operations
require `running`.

### 7.1 Forward

```json
{
  "operation_type": "forward",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "payload_b64": "base64-data",
    "content_type": "application/octet-stream"
  }
}
```

`forward()`, `fwd()`, and `fwd_no_grad()` all send exactly
`operation_type="forward"`. The aliases do not add no-gradient semantics.

A byte payload is a DSSST1 frame posted as a JSON `/operation` body. A frame
that fits the JSON ceiling is one POST of the raw frame. A larger frame is
split with `wire.encode_byte_chunks` (`kind="request"`, `operation="forward"`)
and each chunk is its own POST whose `payload_b64` is that request chunk. Only
the last POST returns the pollable `request_id`.

The ceiling is `CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES` (integer
bytes, default 16 MiB, minimum 16 KiB). Dict payloads stay a single plain
`/operation` POST.

Do not treat this route as a portable log-probability API.

### 7.2 Weight sync

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:training:0",
    "target_sub_job_ids": ["job-id:sampling:0"],
    "weight_format": "vllm"
  }
}
```

`weight_sync()` defaults operation routing to the source training sub-job and
returns the `request_id` string.

Optional payload fields:

- `weight_format`: `"vllm"` (default on server) | `"hf"` for full-model
  weights. For adapter-only synchronization, see
  [section 7.3](#73-lora-adapter-sync).

On server versions that include this behavior, the default weight sync, which
pauses generation and keeps in-flight requests, also clears the sampling
sub-job's prefix cache. No request after the sync reads KV computed under the
old weights. In-flight requests are preempted and recomputed with the new
weights, and the next turn of each conversation recomputes its history, so
generation is somewhat slower right after a sync. Syncs that use another mode
are unchanged. A sampling sub-job can opt out with
`vllm_config.clear_cache_on_weight_sync` (see
[section 8.3](#83-inferenceconfig)).

Each sampling target in the result reports `clear_cache`, at
`targets[].recv.recv.clear_cache`. It is the value the sync sent to the sampling
workers' pause, so it shows what was requested and is not proof that the cache
was cleared. It is `false` when no pause was issued. A result without the key
comes from a server version that predates this behavior. If you set the opt-out
and the result still shows `clear_cache: true`, check the key's spelling and
that it is set under `vllm_config`.

### 7.3 LoRA adapter sync

LoRA synchronization uses the same `weight-sync` operation, but it requires
LoRA to be enabled on both sub-jobs when they are created. It is not sufficient
to change only `weight_format` at sync time:

- The training sub-job's `training_config.peft_config` applies PEFT and makes
  the LoRA adapter parameters trainable.
- The sampling sub-job's `inference_config.peft_config` enables the vLLM LoRA
  manager at engine startup.
- The sampling configuration must match the training adapter's `r`,
  `lora_alpha`, and `target_modules`. Reusing the same configuration object for
  both sub-jobs is the simplest way to keep them aligned.

The sync request transfers only the trained adapter tensors:

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:training:0",
    "target_sub_job_ids": ["job-id:sampling:0"],
    "weight_format": "lora"
  }
}
```

Subsequent syncs replace the resident adapter under the same stable adapter
identity.

CLI equivalent:

```bash
cortex-training --job-id JOB_ID weight-sync \
  --weight-format lora
```

See [section 13.5](#135-enable-lora-training-and-adapter-sync) for the complete
create, train, sync, and generate workflow.

### 7.4 Bootstrap router replay

```json
{
  "operation_type": "bootstrap-router-replay",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:sampling:0",
    "target_sub_job_id": "job-id:training:0",
    "max_cache_bytes": 4096
  }
}
```

The sampling sub-job is the routing source and the training sub-job is the
replay target/operation receiver. `max_cache_bytes` is optional.
For a mixed training/sampling job, pass `sub_job_id=target_sub_job_id` (or the
matching `sub_job_type`) because the low-level client does not infer the
operation receiver.

### 7.5 Router replay discard

```json
{
  "operation_type": "router-replay-discard",
  "sub_job_id": "job-id:sampling:0",
  "sub_job_type": "sampling",
  "payload": {
    "sample_ids": ["sample-1", "sample-2"]
  }
}
```

`router_replay_discard()` also accepts an `extra` payload for forward-compatible
fields. Explicit `extra["sample_ids"]` takes precedence over the method's
`sample_ids` argument.

### 7.6 Reset prefix cache

```json
{
  "operation_type": "reset-prefix-cache",
  "sub_job_id": "job-id:sampling:0",
  "sub_job_type": "sampling",
  "payload": {
    "drain": true,
    "timeout_s": 60.0,
    "retry_interval_s": 0.1
  }
}
```

The shown payload values are the client defaults. `extra` can supply new
backend fields and override `drain`.

### 7.7 Cancel request

```json
{
  "operation_type": "cancel-request",
  "payload": {
    "request_id": "request-id"
  }
}
```

With a self-describing request id, the server can recover the owning sub-job.
For legacy ids, omitting a sub-job hint causes server-side fan-out across the
job's sub-jobs.

### 7.8 Tail logs

```json
{
  "operation_type": "tail-logs",
  "sub_job_id": "job-id:training:0",
  "payload": {
    "cursor": "cursor-0",
    "max_lines": 50
  }
}
```

`cursor` is opaque: pass back the previous page's `next_cursor`. Without a
cursor (omitted or empty), `max_lines` tails the log's last lines; with
neither, the read starts at the beginning of the container log, which for a
long-running job is hours of output. With a cursor, `max_lines` caps the
entries on that page. The service currently bounds both: at most 2,000 lines
for `max_lines`, and at most 2,000 entries and 256 KiB of log per page. These
bounds may change; a client should follow `next_cursor` rather than rely on
them.

Response:

```json
{
  "entries": [],
  "next_cursor": "cursor-1",
  "eof": true
}
```

Each entry is `{"_raw": "<RFC 3339 timestamp, nanoseconds> <line>"}`. The
service currently cuts a line over 64 KiB, and that entry carries
`"_truncated": true`. `eof` is not set for a live pod, so a follower keeps
polling.

`stream_logs()` repeatedly calls this operation. With `follow=False`, it stops
at an empty EOF page; with `follow=True`, it keeps polling.

While the job can't serve logs, the operation answers:

| Answer | When |
|---|---|
| 200, no entries, same cursor | The job is `placing` or `initializing` and the sub-job's head pod isn't up yet |
| 400 `sub_job_id ... does not belong to session ...` | The service isn't routing to that sub-job yet, while its job starts |
| 409 `session ... is pending placement: <reason>` | The job is `pending` |
| 409 `session ... has been cancelled`, `has been terminated`, or `has failed: <reason>` | The job is no longer running |
| 404 | The job's session is gone, some time after the job ended |

The client raises 404 and 409 from this call at once, without the retries
other calls get ([section 3.5](#35-retries)).

On some service versions, a log line of 256 KiB or more is never delivered and
holds the cursor: later pages come back empty with the same `next_cursor` while
the job keeps logging. A read without a cursor and with a small `max_lines`
gets past the line once it is older than that window. A quiet log looks the
same until it logs again, so the TUI makes that read when a running job's tail
hasn't moved for 45 seconds, and continues from it only when it has lines the
tail hasn't shown and the old cursor still brings nothing. It then leaves out
the lines it already showed and marks the gap. A window that still reaches the
stuck line brings nothing new, so the TUI shrinks it (200, then 50, then 10
lines) on later reads; getting past the line can take a few of them.

The TUI saves each sub-job's last `next_cursor` and resumes from it. If the
service answers a read with a saved cursor with 400 (other than the "does not
belong to session" answer above), the TUI drops that cursor and reads the
log's end without one. While pages come back full (1,000 entries or 128 KiB
and more), it sends the next read after 0.2 seconds instead of its poll
interval, until it catches up.

On a running job, each `tail-logs` call currently counts as activity on the
job: it restarts the job's idle timer
([`idle_timeout_seconds`](#idle_timeout_seconds)) and moves its `updated_at`.
A viewer that keeps polling keeps an idle job alive. Calls while the job is
`placing` or `initializing` don't count. The TUI's live tail pauses after 15
minutes with no input while the job runs.

### 7.9 Unsupported zone-events client helper

`CortexTrainingClient.tail_events()` and `stream_events()` send:

```json
{"operation_type": "zmd-events"}
```

The server does not currently accept this operation type, and the methods are
unit-tested only for request construction. Treat them as unavailable.

---

## 8. Create-job schemas

### 8.1 `SubJobConfig`

Exactly one type-specific config is set.

| Field | Type | Required | Notes |
|---|---|---|---|
| `job_type` | `training`, `sampling`, `log_probability` | yes | Typed `JobType` enum. `log_probability` is a schema value only: the client and CLI reject it at submission |
| `model_name` | string | yes | Must be non-empty |
| `training_config` | object | for training | Produced from `TrainingConfig` |
| `inference_config` | object | for sampling/log probability | Produced from `InferenceConfig` |
| `global_batch_size` | integer | no | Top-level passthrough field |
| `dtype` | string | no | Example: `bfloat16` |
| `seed` | integer | no | |
| `model_post_init` | list of strings | no | Server maps to post-init hooks |
| `source_checkpoint_info` | object | no | Create-time initialization from a saved checkpoint or external weights |

Typed factories:

```python
SubJobConfig.training_job(...)
SubJobConfig.sampling_job(...)
```

The typed client has no `resume_from_checkpoint` field. Use
`source_checkpoint_info`.

### 8.2 `TrainingConfig`

The typed client requires:

| Field | Type | Client validation |
|---|---|---|
| `optimizer` | object | Canonicalized; empty only when `gradient_clipping` is set |
| `max_seq_len` | integer | Greater than zero |
| `train_batch_size` | integer | Greater than zero |
| `n_gpus` | integer | Greater than zero |

#### `optimizer` canonicalization

`optimizer` is no longer an opaque non-empty object. The typed `TrainingConfig`
rewrites it into the canonical flat form on construction and rejects anything it
cannot place. The canonical keys are `name`, `lr`, `weight_decay`, `betas`,
`eps`, and `fused`; the wire form is

```json
{"name": "AdamW", "lr": 1e-5, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8}
```

The rewrites are:

- `type` becomes `name`. Setting both to different values is an error.
- `beta1` / `beta2` become `betas`. Supplying only one fills its partner from
  `[0.9, 0.999]`. `betas` together with a disagreeing `beta1`/`beta2` is an
  error.
- A DeepSpeed-style `params` sub-object is flattened onto the top level. A key
  set in both places with different values is an error.
- `gradient_clipping` inside `optimizer` is hoisted onto the typed
  `gradient_clipping` field, which is where the server reads it. When both are
  set the typed field wins and the nested value is dropped. A block holding only
  `gradient_clipping` therefore canonicalizes to `{}`, which is valid.
- Scheduler names (`lr_scheduler`, `lr_scheduler_type`, `warmup_steps`,
  `warmup_ratio`, `warmup_steps_proportion`, `num_warmup_steps`,
  `training_horizon`) are rejected rather than accepted and ignored. Drive the
  learning rate per step from the client, or set a server-side DeepSpeed
  scheduler under `ds_config`.
- Any other key is rejected, as are non-numeric, non-finite, or out-of-range
  values.

Canonicalization never fills defaults — the server owns those, so an explicitly
submitted value stays distinguishable from a defaulted one. `betas` is the one
exception, since its canonical form is a pair.

This runs on the typed `TrainingConfig` path only. `create_job_from_body()`
forwards `optimizer` exactly as given and leaves the rewrite to the server.

Optional typed fields:

- `gradient_clipping`
- `multiplex_job_id`
- `load_optimizer_states`

`load_optimizer_states` controls checkpoint resume behavior for this training
sub-job. When `false`, resuming a `resumable` checkpoint (via
`source_checkpoint_info` or the runtime `/load` endpoint) restores model weights
only and starts the optimizer fresh. Set it `false` to change data-parallel size
while keeping expert parallelism, because the DP-sharded optimizer cannot be
resized; a `true`/default resume against a different DP/world size fails fast
with an actionable error. `None` (omitted) uses the server default (`true`).

`extra_training` is merged as open passthrough data, without overriding typed
keys. Examples include `model_provider`, `ep_size`, `ds_config`,
`activation_checkpointing`, `prime_rl`, `router_replay`, `peft_config`, the
memory-diagnostics settings in [section 8.5](#85-memory-diagnostics-settings),
and these newer long-context/memory knobs:

- `sp_size`: Ulysses sequence-parallel degree, sharding each sample's sequence
  across `sp_size` ranks. `1` (default) disables it; the server requires
  `world_size % sp_size == 0`.
- `cuda_allocator_conf`: per-job `PYTORCH_CUDA_ALLOC_CONF` string applied before
  PyTorch is imported, e.g. `"expandable_segments:True"` (or `"backend:native"`
  to disable). Omitted uses the server default.
- `ac_config`: activation-checkpointing config, including CPU activation offload
  via `offload_config.enabled=true`. Offload requires `mode="full"`.
- `router_replay`: `{"enabled", "max_cache_bytes", "mode"}`. `mode` is
  `"strict"` (default) or `"best_effort"`; values other than `"strict"` and
  `"best_effort"` are rejected. It applies to the training sub-job
  only: `"best_effort"` with `enabled` `false`, or any `mode` other than
  `"strict"` on a sampling sub-job, is rejected. `"strict"` fails a forward/backward
  request when any of its rows has no routing held by a sampling worker (for
  example, a row generated before a sampler restart); the request error is
  `router_replay_preflight_failed` with `stage` `"receive"` and `reason`
  `"receiver_error"`. `"best_effort"` routes such rows with the trainer's own MoE
  gate, replays the others, and reports the split in the
  `router_replay/rows_*` and `router_replay/tokens_*` metrics of
  [section 6.1](#61-forwardbackward---post-job_idforward-backward). A model
  whose MoE routers cannot route a row by their own gate, or that has no MoE
  routers, rejects `"best_effort"` when the training sub-job starts. With whole-block activation
  checkpointing, `"best_effort"` also requires `ac_config.router_replay_recompute`
  (default `true`); a training sub-job with it disabled rejects `"best_effort"`
  at startup. If the service does not yet support `"best_effort"`, router-replay
  bootstrap fails with a `router replay bootstrap rejected` error; use `"strict"`
  until the service is upgraded. If only some sampling workers lack support, a
  request with a missing row fails exactly as in `"strict"`.

`max_cache_bytes` is the per-worker cap in bytes. The service stores each
expert id as `uint8` when the model has at most 256 experts and as `uint16`
when it has 257 to 65536 experts. A `uint16` capture uses twice as many bytes
as a `uint8` capture for the same sequence length, MoE layer count, and top-k.
The service does not choose the cap. Derive both the sampling and training
caps with `cortex_training.router_replay_planner`, which selects that width
from the expert count and includes it in `CachePlan.max_cache_bytes`. The same
object can be set on a sampling sub-job through `extra_sampling["router_replay"]`.
See [python-sdk.md](python-sdk.md#router-replay-capacity-planning).

For LoRA training, set `extra_training["peft_config"]` to a PEFT
`LoraConfig`-compatible object. It is validated and rewritten client-side, so it
must specify `peft_type="Lora"` — exactly that spelling, not the `"LORA"` a
Hugging Face `adapter_config.json` writes — plus `target_modules`,
`target_parameters`, or both. `r`, `lora_alpha`, `lora_dropout`, and `bias` are
filled with their defaults (`8`, `8`, `0.0`, `"none"`) by the client when
omitted, although explicit values are recommended when the same configuration is
also used by sampling. Unsupported keys are rejected.

`peft_config` is validated the same way on all three submission paths: typed
training, typed sampling ([section 8.3](#83-inferenceconfig)), and the raw
`create_job_from_body()` body. A config with no `peft_config` stays valid, and
every other key in `extra_training` / `extra_sampling` remains open
passthrough.

The current client also rejects an effective PrimeRL config that combines an
enabled/default `fused_cross_entropy` with `fp32_lm_head=True` or an integer
`fused_lm_head_token_chunk_size`.

### 8.3 `InferenceConfig`

The typed client requires:

| Field | Type | Client validation |
|---|---|---|
| `max_seq_len` | integer | Greater than zero |
| `n_gpus` | integer | Greater than zero |

`multiplex_job_id` is optional. `extra_sampling` is an open passthrough object;
common values include `gpu_memory_utilization` and a nested `vllm_config`.

Enable speculative decoding on a sampling sub-job with
`vllm_config.speculative_config`. The client forwards that block to the
server. Typical keys are `method`, `model` (the draft model), and
`num_speculative_tokens`.

`speculative_config.model` uses the same convention as `model_name`: pass an
`<org>/<model>` Hub id. The server maps it to the baked model cache.
Explicit local paths are used as-is. Callers should not hard-code cache
locations.

```python
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3.8-27B",
    max_seq_len=16384,
    n_gpus=8,
    extra_sampling={
        "vllm_config": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "speculative_config": {
                "method": "dflash",
                "model": "z-lab/Qwen3.8-27B-DFlash2",
                "num_speculative_tokens": 7,
            },
        },
    },
)
```

For LoRA sampling, set `extra_sampling["peft_config"]` when creating the
sub-job. This enables the vLLM LoRA manager before the model starts. The
adapter's `r`, `lora_alpha`, and `target_modules` must match the training
configuration used for adapter synchronization. It is validated and rewritten by
the same rules as the training block (see
[section 8.2](#82-trainingconfig)).

`vllm_config.clear_cache_on_weight_sync` controls whether a weight sync clears
the sampling sub-job's prefix cache. It is a boolean, default `true`, and
applies only to the default weight sync (see
[section 7.2](#72-weight-sync)). Set it to `false` to opt out, for example
`extra_sampling={"vllm_config": {"clear_cache_on_weight_sync": False}}`. With
`false`, only a best-effort reset after the sync remains. That reset cannot clear
blocks held by running requests, so requests may read KV computed under the old
weights. The key is not validated: a misspelled key is silently ignored, and the
sync keeps clearing the cache. The `clear_cache` key in each sampling target's
sync result shows whether the opt-out was applied (see
[section 7.2](#72-weight-sync)).

For either config type, the server requires `multiplex_job_id` to be a complete
`{job_id}:{sub_job_type}:{index}` id outside the job being created.

### 8.4 `source_checkpoint_info`

```json
{
  "checkpoint_id": "checkpoint-id",
  "source_job_id": "source-job-id"
}
```

Exactly one source is required:

- `checkpoint_id` identifies a checkpoint saved by Cortex Training.
  `source_job_id` identifies the job that owns it and is required for cross-job
  initialization.
- `external_stage_path` identifies a complete Hugging Face safetensors
  directory in a stage the submitting role can read. It cannot be combined with
  `checkpoint_id` or `source_job_id`.

```json
{
  "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
}
```

The typed client forwards this object without validating its fields. The server
resolves stage credentials; clients must not provide `stage_info` themselves.

### 8.5 Memory-diagnostics settings

Training workers expose low-volume, customer-facing memory observability through
`training_config` keys. The typed client has no dedicated fields for these, so
pass them through `extra_training`. Each is an optional per-job override; when
omitted or `null`, the server system default applies.

| Key | Type | Effect |
|---|---|---|
| `step_peak_memory_log` | bool | Adds a `peak_memory` object (`gpu_peak`, `cpu_peak` byte counts) to each `/step` result. See [section 6.2](#62-optimizer-step---post-job_idstep). |
| `training_memory_telemetry` | bool | Emits structured allocator events (`fwd_bwd_end`, `step_start`, `step_end`) to the training event stream. |

```python
training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    extra_training={"step_peak_memory_log": True},
)
```

`step_peak_memory_log` costs one `all_reduce(MAX)` per step; both settings are
off by default. Failure snapshots on a failed `/forward-backward` or `/step` are
independent of these flags. Per-rank operator diagnostics configured by the
launcher are out of scope for this client-facing surface.

---

## 9. DSSST1 binary wire protocol

### 9.1 Why DSSST1

DSSST1 is a safetensors-based frame implemented by `cortex_training.wire`. It
serializes nested dict/list/tuple structures containing tensors and JSON-safe
values without pickle execution.

`wire.loads()` explicitly rejects legacy pickle and `torch.save` signatures.

### 9.2 Frame contents

A frame is one safetensors blob:

```text
u64 header length | safetensors JSON header | tensor bytes
```

Safetensors metadata contains:

- `dss`: the nested structure skeleton and wire version `DSSST1`.
- `op`: optional operation metadata such as response options, router replay,
  and request/result chunk descriptors.

Encode/decode:

```python
from cortex_training import wire

frame = wire.dumps(value, metadata=metadata)
value = wire.loads(frame)
metadata = wire.read_metadata(frame)
```

### 9.3 Request chunking

`forward_backward()`, `generate()`, and byte `forward()` call:

```python
wire.encode_byte_chunks(
    frame,
    kind="request",
    operation="fwd-bwd-or-generate-or-forward",
    max_bytes=...,
)
```

`forward_backward()` always sends a chunk envelope, including when the request
fits in one chunk, so every logical operation has a caller-generated identity.
`generate()` still sends the original frame unchanged when it fits, as
`application/octet-stream`. Byte `forward()` also sends the original frame
unchanged when it fits; it splits only when the encoded JSON body would exceed
`CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES` (default 16 MiB). Those
chunks travel inside the `/operation` JSON envelope (`payload_b64`) rather than
as octet-stream. Each DSSST1 request chunk contains:

- A `uint8` payload tensor.
- `chunk_idx` and `total_chunks`.
- `chunk_group_id`.
- Original frame size and SHA-256.
- Operation name.

For forward-backward, `chunk_group_id` is also the idempotency identity and
`frame_sha256` binds that identity to one exact frame:

- The server retains accepted chunks with the training job rather than the
  HTTP connection.
- An unknown group must start at chunk `0`. A later first chunk returns `409`
  with code `chunk_group_restart_required`; the client may replay once from
  chunk `0` using the same encoded chunks and group id.
- Repeating the same group id and frame hash returns or waits for the original
  execution result. It does not run forward-backward again.
- Reusing a group id with a different frame hash, chunk count, or chunk payload
  returns `409`.

The final chunk schedules work and returns the pollable `request_id`. Retrying a
final chunk after losing its response can return a new `request_id`, but both
request records resolve through the same at-most-once forward-backward
execution.

### 9.4 Encoded results

A non-chunked DSSST1 result can appear inside poll JSON as:

```json
{
  "content_type": "application/octet-stream",
  "encoding": "base64",
  "wire_format": "DSSST1",
  "payload_b64": "base64-frame"
}
```

`poll_request()` base64-decodes and passes the frame to `wire.loads()`. If the
JSON envelope also has a `metrics` object, those keys are shallow-merged into
the decoded result's `metrics`. Envelope keys win on collision. Payload
`metrics` are kept when the envelope has none.

A result with `payload_b64` but no `wire_format` is left encoded unless
`poll_request()` is completing a `forward()` request id minted by this client.
In that case the blob is decoded the same way and envelope `metrics` are merged.

### 9.5 Result chunks

Large results can arrive through poll events:

```json
{
  "type": "result_chunk",
  "payload_b64": "base64-chunk-frame",
  "payload_sha256": "sha256"
}
```

A poll can also include `next_cursor`. `poll_request()` drains all pages,
validates chunk SHA-256 values, uses `wire.decode_result_chunks()`, and returns
the reconstructed object. Terminal-page `result.metrics` are merged into that
object with the same envelope-wins rule as [section 9.4](#94-encoded-results).

---

## 10. Forward/backward batch contract

### 10.1 Transport-level object

The transport accepts a DSSST1 object. The minimal conventional shape is:

```python
batch = {
    "args": (),
    "kwargs": {
        "input_ids": input_ids,
        "labels": labels,
    },
}
```

Common model kwargs are `input_ids`, `attention_mask`, `position_ids`, and
`labels`, generally shaped `[batch, sequence]`.

### 10.2 Readable payload helper

`build_forward_backward_payload(spec)` is a CLI/readability helper. It builds
only `args` and `kwargs`, then calls `serialize_forward_backward_args()`.

Direct tensor-shaped JSON:

```json
{
  "payload": {
    "kwargs": {
      "input_ids": {"data": [[1, 2, 3]], "dtype": "long"},
      "labels": {"data": [[2, 3, -100]], "dtype": "long"}
    }
  }
}
```

It can also tokenize `texts` with a configured Transformers tokenizer.

When building labels, supported strategies are:

- `next_token` / `shifted_input_ids`
- `input_ids` / `self`
- `none`
- explicit tensor data

The default helper strategy is next-token labels. It rolls input ids left,
sets the last target to `ignore_index` (default `-100`), and can mask padding.

### 10.3 ArcticTraining extensions

ArcticTraining serializes a larger object directly with `wire.dumps()`:

```python
batch = {
    "args": (),
    "kwargs": {...},
    "context": {...},
    "processing": {
        "loss_fn": "grpo",
        "config": {...},
        "post": [...],
    },
}
```

`context` and `processing` are open backend contracts. Their accepted keys,
registered loss functions, and post-processors come from the deployed training
backend, not the REST schema. The optional ratio controls of the `grpo` loss and
its variants, and the `grpo` teacher term, both described below, are an
exception: their configuration keys, context tensors, and the metrics named
below, are listed here.

Do not use `build_forward_backward_payload()` for this extended shape: that
helper currently ignores `context` and `processing`.

#### Ratio controls for the CISPO loss

On server versions that include these controls, the `grpo` loss and its variants
`ap_grpo`, `grpo_echo_v1`, and `ap_grpo_echo_v1` accept optional ratio controls
in `processing.config`. All are off by default, and any active control requires
`use_cispo_loss` to be `true`. The masks drop tokens from the CISPO policy term
only; the loss normalizers stay as they were.

| Key | Meaning |
|---|---|
| `ratio_mask_bounds_pos`, `ratio_mask_bounds_neg` | Inclusive importance-ratio bands, chosen by the sign of the token's advantage (see below). A pair `[low, high]` of finite numbers or `null`, with `low < high` and `low >= 0`; a `null` end is unbounded |
| `prob_diff_mask_max_pos`, `prob_diff_mask_max_neg` | Finite, non-negative maximum directional probability gap between the current policy and the sampler |
| `seq_mask_stat` | The sequence statistic that the sequence gate tests: `mean_log_ratio` (default) or `mean_k3` |
| `seq_mask_bounds_pos`, `seq_mask_bounds_neg` | Inclusive bands for that statistic, chosen by advantage sign (see below). A pair of finite numbers or `null`, with `low < high`; a `null` end is unbounded |
| `ratio_m2_threshold` | Finite, positive M2PO filtering threshold for the current-policy versus sampler ratio |
| `log_ratio_sq_coef` | Finite, non-negative coefficient of a squared log-ratio penalty. Unlike the masks, the penalty is computed over the original loss mask whatever the masks drop. A positive value is rejected with `token-mean` aggregation when `sequence_loss_weights` is present |
| `ratio_stats` | Boolean. `true` reports the ratio metrics (histograms, mask counts, and the metrics below) without enabling a mask |
| `ratio_mask_rebalance` | Boolean. `true` rescales the kept advantages of each sign that keeps any mass so that its total returns to its pre-mask value (see below) |

A control is active as soon as its key is present with a valid value, even a
neutral one such as `log_ratio_sq_coef: 0.0`. An explicit `null` is rejected for
every ratio-control key, so omit a key to leave it off; inside a bounds pair, a
`null` end still means unbounded. An explicit `false` of `ratio_stats` or
`ratio_mask_rebalance` does not count: a request whose only ratio-control keys
are one or both of these set to `false` is inert and needs no `use_cispo_loss`.
The mixed losses `grpo_mixed_v1` and `ap_grpo_mixed_v1` reject every
ratio-control key, including an explicit `false`, so that policy-only gates
cannot affect NLL tokens.

The `_pos` and `_neg` gates split tokens by the sign of the token-level
advantage, before any sequence-level averaging and including the teacher term
when it is enabled. An advantage of exactly `0` counts as positive. With
`importance_sampling_level` set to `"sequence"`, the CISPO term uses each
sequence's averaged advantage while the gates keep the token-level sign, so the
two can differ. A sequence is tested against `seq_mask_bounds_pos` when it has
at least one positive-side token and against `seq_mask_bounds_neg` when it has
at least one negative-side token, and a band violation drops only that side's
tokens.

Whenever a ratio control is active, the `metrics` of the forward/backward result
carry the keys below. Keys ending in `_sum` are added up over workers. The
result also carries other ratio-control entries, such as per-drop counts and
further histogram bins, that are not listed here.

| Metric | Meaning |
|---|---|
| `ratio_mask_pos_pre_mass_sum`, `ratio_mask_neg_pre_mass_sum` | The advantage mass before masking: `\|A\|` summed over the policy tokens with advantage `A >= 0` (`pos`) or `A < 0` (`neg`). It goes through the loss's own normalization (for `token-mean`, divided by the number of policy tokens in the whole request), without the data-parallel factor and without importance, clip, `rollout_is_weights`, or `behav_imp_weight_cap` weights. `A` is the advantage the CISPO term uses: with `importance_sampling_level` set to `"sequence"`, each sequence's averaged advantage, so the sequence counts as `pos` or `neg` as a whole by the sign of its mean |
| `ratio_mask_pos_kept_mass_sum`, `ratio_mask_neg_kept_mass_sum` | The same sums over the tokens that survive every active drop (ratio band, probability gap, sequence gate, `ratio_m2_threshold`) |
| `ratio_mask_kept_k3_sum` | The sum over the kept tokens of `exp(d) - 1 - d`, where `d` is the trainer log probability minus the sampler log probability (`old_log_probs_shifted`), clamped to `[-20, 20]`. It is raw, not divided by the loss denominators, and is `0` when the request has no `old_log_probs_shifted` |
| `ratio_trainable_token_count`, `ratio_mask_dropped_token_count` | The policy tokens the ratio controls see, and how many of them some active drop removes |
| `seq_stat_bin_<i>_sequence_count`, `i` from `0` to `6` | The number of sequences whose log ratio (current policy minus sampler log probability), averaged over the sequence's policy tokens, falls in bucket `i` of the edges `-0.2, -0.05, -0.01, 0.01, 0.05, 0.2`. Each bucket includes its upper edge: bucket `0` holds values up to `-0.2` and bucket `6` values above `0.2`. It always bins the mean log ratio, also when `seq_mask_stat` is `mean_k3` |
| `ratio_mask_rebalance` | The flag as `0.0` or `1.0` |
| `ratio_rebalance_pos_post_mass_sum`, `ratio_rebalance_neg_post_mass_sum` | Only with `ratio_mask_rebalance` set to `true`. The kept mass of the sign after scaling: equal to its pre-mask mass, up to rounding, when the sign's kept mass is positive, and `0` when it is not |
| `ratio_rebalance_pos_unrestored_mass_sum`, `ratio_rebalance_neg_unrestored_mass_sum` | Only with `ratio_mask_rebalance` set to `true`. The pre-mask mass of a sign whose kept mass is `0`, and `0` otherwise |

The four mass sums equal their value for the unsplit batch, up to rounding, when
the loss normalization is request-global: `token-mean`; `prompt-mean` with
`prompt_group_ids` (the service derives the request-wide counts it needs) or with
client-supplied `sequence_loss_weights`; and the `seq-mean-token-sum`,
`seq-mean-token-sum-norm`, and `seq-mean-token-mean` values of `loss_agg_mode`
when the client sends a request-global `global_batch_size`. Otherwise each call
normalizes by its own counts, and so do these sums. Sequence-parallel shards
count once, and a worker with no policy token reports `0`. The `post` and
`unrestored` sums below are in the same normalized units and are also added up
over workers. The k3 sum and the two counts are not normalized: they add up across microbatches, workers, and
sequence-parallel ranks in every aggregation mode, apart from the placement
effects below.

`ratio_m2_threshold` ranks the tokens of one model call, so with it the kept
mass, the dropped count, and the k3 sum depend on how rows are placed on
workers, while the pre-mask mass does not. The baseline `m2_threshold` key,
which is not `ratio_m2_threshold`, acts on the tokens a worker holds in one call
(under sequence parallelism, on each rank's own window) and removes tokens
before the ratio controls act. Those tokens are in neither the pre-mask nor the
kept mass, and are excluded from `ratio_trainable_token_count`. With it the
masses, the k3 sum, and the counts depend on how rows are placed in calls and on
the sequence-parallel split, and with several calls per worker on
`max_tokens_per_mb`. With either key the restoration scale below depends on
placement as well.

The clamp of `d` applies to the reported k3 sum only: `seq_mask_stat` set to
`mean_k3` gates on the unclamped k3. The kept-token mean is
`ratio_mask_kept_k3_sum / (ratio_trainable_token_count -
ratio_mask_dropped_token_count)`. When the request carries
`old_log_probs_shifted`, the service also reports
`trainable_logprob_lowvar_all/mean`, the same estimator over every trainable
token before masking, with the same clamp. With no drop, no token removed by the
baseline `m2_threshold`, and finite log probabilities,
`ratio_mask_kept_k3_sum / ratio_trainable_token_count` equals it up to rounding.
The service computes `trainable_logprob_lowvar_all/mean` with `exp(d) - 1 - d` in
float32, which loses precision for `|d|` of about `1e-3` or less, so a gap there
comes from that metric. A trainer log probability that is not finite is replaced
by `0` before the loss, and its token still counts. A sampler log probability
that is not finite at a trained token makes the request fail, so no k3 sum is
returned.

With `ratio_mask_rebalance` set to `true`, once the masks have decided which
tokens to drop, the kept tokens of positive advantage are scaled by `P0/Pk` and
those of negative advantage by `N0/Nk`. `P0` and `N0` are
`ratio_mask_pos_pre_mass_sum` and `ratio_mask_neg_pre_mass_sum`; `Pk` and `Nk`
are `ratio_mask_pos_kept_mass_sum` and `ratio_mask_neg_kept_mass_sum`; all four
are summed over all workers. Every worker applies the same scale, and each sign
returns to its own pre-mask mass, not to the other sign's. Only advantage
magnitudes change: which tokens are dropped, the counts, and the k3 sum do not
depend on the scaling. The scale is computed for one forward/backward request,
so a step split across several requests gets one scale per request, and a sign
that loses all its mass in one part is not restored there. A sign whose kept mass
is `0` has nothing to scale, including a sign whose only surviving tokens have
advantage `0`: its scale is `1` and its pre-mask mass is reported as
unrestored. With nothing dropped, both scales are `1` and the objective is
unchanged, up to rounding. The scale is not capped and is not reported, but it
can be computed from the reported masses, and it is large when little of a
sign's mass survives (`100` when one of 100 equal-magnitude positive tokens is
kept). The CISPO ratio clip (`is_weight_clip_max`, when set) still applies
independently of the scaling, so the restoration does not equalize
ratio-weighted gradients.

`ratio_mask_rebalance` set to `true` and `ratio_m2_threshold` each require one
packed model call per worker. A request that would need more than one call per
worker is rejected before it is dispatched to the workers, with an error that
names the key (both keys when both are set) and the per-worker call counts. The
training sub-job's `training_config.mb_spec.max_tokens_per_mb` budget (default
`10240` tokens when `mb_spec` or the key is omitted) must therefore be large
enough to keep each worker's share in one call. Two rejections come later:
`ratio_m2_threshold` is also rejected under sequence parallelism, and an active
ratio control without `use_cispo_loss` is rejected. A worker raises both when it
resolves the loss, before any forward pass.

#### Teacher term for on-policy distillation

On server versions that include it, the `grpo` and `ap_grpo` losses accept a
teacher term that adds, at every trained token, a clipped teacher-minus-policy
log ratio to that token's advantage:

```text
advantage[i] += teacher_tau * clamp(teacher_log_probs_shifted[i] - log_prob[i],
                                    -teacher_clip_negative, teacher_clip)
```

`log_prob[i]` is the training policy's own log probability, detached. With
`context["advantages"]` all zero, this term is the whole signal and the update
is an on-policy distillation step on reverse KL to the teacher; with task
advantages, the two add.

| Key | Meaning |
|---|---|
| `teacher_tau` | Finite, non-negative weight of the term. `0` (the default) turns it off |
| `teacher_clip` | Finite, positive upper clip on the log ratio. Required when `teacher_tau > 0` |
| `teacher_clip_negative` | Optional finite, non-negative magnitude of the lower clip. Defaults to `teacher_clip` |

When `teacher_tau > 0`, `context` must carry `teacher_log_probs_shifted`: a
floating-point tensor with the same shape as the prediction-aligned
log probabilities, where position `i` holds the teacher's log probability of
`input_ids[:, i + 1]` (see [section 10.4](#104-shifted-log-probability-convention)).
Positions outside `loss_mask` are ignored, as are positions whose teacher value
is not finite. The term also requires the default token-level
`importance_sampling_level`, and with `token-mean` aggregation it rejects
`sequence_loss_weights`. Configuration keys are checked by name, so a misspelled
key fails the request instead of training without the term.

With the term enabled, the `metrics` of the forward/backward result carry:

| Metric | Meaning |
|---|---|
| `teacher_tau` | The weight that was applied |
| `teacher_term_token_count` | Tokens that received the term |
| `teacher_log_ratio_sum` | Sum of the unclipped teacher-minus-policy log ratios over those tokens |
| `teacher_clipped_log_ratio_sum` | The same sum after clipping |

`-teacher_log_ratio_sum / teacher_term_token_count` is a per-token estimate of
the reverse KL from the policy to the teacher on the trained tokens.

### 10.4 Shifted log-probability convention

ArcticTraining's RL adapter documents `_shifted` log-probability tensors as:

```text
tensor[:, i] is the log probability of input_ids[:, i + 1]
```

This is an ArcticTraining processing convention, not a REST-level field
requirement.

---

## 11. Result summary

These are conventional backend results, not closed REST schemas:

| Operation | Common result |
|---|---|
| `forward-backward` | `job_id`, `avg_loss`, `metrics`, `post_process_outputs`; `metrics` carries the ratio-control metrics when those controls are active (see [section 10.3](#103-arctictraining-extensions)) |
| `step` | `global_steps`, `last_lr`, optional `peak_memory` |
| `save` | `job_id`, `stage_path` (carries the `cp_<uuid>` id), `checkpoint_path`, `checkpoint_tag`, `version` — no `checkpoint_id` key in the observed release (see 6.3) |
| `load` | `checkpoint_id` and backend load metadata |
| `generate` | `job_id`, `results[]` |
| `weight-sync` | Completion/transfer metadata; each sampling target's result carries `clear_cache` (see [section 7.2](#72-weight-sync)) |
| `forward` | Backend-specific forward-only result |

Generic-operation responses and fields inside `metrics` or generation results
are intentionally open.

---

## 12. Logs and events

### 12.1 Live logs

Use `tail_logs()` for one cursor page or `stream_logs()` for an iterator. This
uses the `tail-logs` operation described in [section 7.8](#78-tail-logs).

The server reads the selected sub-job's zone-manager/head-pod stdout. It can
serve empty, non-EOF pages during placement while the pod is still appearing.
[Section 7.8](#78-tail-logs) lists what it answers while the job isn't
running, the cursor that stops on some service versions, and why a polling
viewer keeps a job active.

### 12.2 Full execution-log download

`fetch_execution_logs(job_id, *, sub_job_id=None, filename=None)`:

1. Calls `GET /{job_id}/experiment-run`.
2. For profile clients, opens an independent connection from the same
   Snowflake profile.
3. For direct PAT clients, resolves the current user and role through the SQL
   statements API and opens an equivalent Connector session.
4. Uses experiment artifact `LIST` and `GET` to download every object below a
   `/_logs/{sub_job_id}/` subtree.

`sub_job_id` and `filename`, when given, download only the matching files, for
example one sub-job's `execution.jsonl`. Its records are JSON lines with `ts`,
`level`, `logger` and `msg`; a failure record can also carry an `error`
message or a traceback. A job that fails while it is created or starts up may
have no saved console ([section 12.3](#123-persisted-stdout-download)); its
execution log still records the error, and the TUI shows those records
instead.

These reads, like the console and GPU-metrics downloads below, use SQL on the
job's Experiment. A role with no privilege on that Experiment gets
`does not exist or not authorized` from the `LIST`. When the service can't
resolve the Experiment at all, the experiment-run route answers 500
`Linked experiment is not found for job ...`, which the client raises without
retrying ([section 5.6](#56-experiment-run---get-job_idexperiment-run)).

Return:

```python
[
    {
        "sub_job_id": "job-id:training:0",
        "filename": "execution.jsonl",
        "artifact_uri": "snow://experiment/DB.SCHEMA.EXPERIMENT/versions/RUN/_logs/job-id:training:0/execution.jsonl",
        "content": "...",
    }
]
```

The client returns logical artifact URIs and does not expose physical storage
paths or object-store credentials.

### 12.3 Persisted stdout download

`download_stdout_logs(job_id, output_dir, *, resume=False, sub_job_ids=None, progress=None,
chunk_progress=None)` downloads gzip chunks under `_stdout/{sub_job_id}/`,
orders them by artifact name, and streams them into
`<output_dir>/<sub_job_id>/stdout.log`. A large download may open additional
Snowflake sessions to fetch one sub-job's chunks, and each of those sessions
can prompt for authentication. Each sub-job is committed before the next one
starts: every sub-job sorted by id, or only those in `sub_job_ids`, in that
order. `chunk_progress(sub_job_id, done, total)` is called after each chunk. A
failed chunk leaves that sub-job's committed bytes and manifest in place. A hidden
`.<filename>.manifest.json` beside the file records `version`, the sorted
relative chunk paths, and the committed byte length (and `"partial": true` while
a GPU-metrics file holds only its newest chunks, see
[section 12.4](#124-gpu-metrics-download)).

`resume=True` appends only when those paths are a prefix of the current sorted
list and the file is at least that many bytes. A longer file is truncated back
to the committed length. A missing or shorter file is rebuilt, as is a chunk
list that is no longer a prefix of the sorted paths.
Two downloads of the same output directory at once are unsupported.
The method stays silent unless `progress` is passed. On a terminal, the CLI
draws one updating line on stderr; a redirected stderr gets one final line per
sub-job.

### 12.4 GPU metrics download

`download_metrics(job_id, output_dir, *, resume=False, sub_job_ids=None,
progress=None, chunk_progress=None, priority=(), newest_first_seconds=None,
on_commit=None)` applies the same reconstruction to `gpu.YYYYMMDD-HHMMSS.UUID.gz`
chunks under `_metrics/{sub_job_id}/` and writes
`<output_dir>/<sub_job_id>/gpu.jsonl`, including `resume`, `sub_job_ids`,
`progress`, `chunk_progress`, and the manifest beside that file. A reader can
show a long job's recent samples before its whole history is downloaded:

- `priority` names sub-jobs to download first, in that order; the rest follow
  in id order. (`sub_job_ids`, when given, sets the order instead.)
- `newest_first_seconds`: a sub-job with no usable manifest first commits only
  the chunks whose name time is within that many seconds of its newest chunk's,
  plus the chunk before them (a name may give the time its rows start). Its
  manifest then also records `"partial": true`. The download then writes the
  whole file, copying those chunks' bytes instead of fetching them again, and
  drops the flag. A later `resume=True` download completes a file left
  partial.
- `on_commit(entry)` is called after every commit, on the downloading thread,
  with that sub-job's result entry plus `"complete"`: `False` after a partial
  commit, `True` once the file holds every listed chunk. An exception it raises
  stops the download; the files committed so far stay.

Any rebuild also reuses the committed bytes when the manifest's paths are one
contiguous run of the current list, for example when a chunk lands that sorts
before every chunk already downloaded.

The reconstructed file contains the JSONL records emitted by the deployed
training runtime. Version 2 GPU rows have no `kind` field and include:

| Field | Meaning |
|---|---|
| `record_version` | Row schema version; currently `2` |
| `sub_job_id` | Sub-job that produced the sample |
| `sample_ts` | RFC3339 UTC scrape time; fractional seconds may use nanosecond precision |
| `worker_num` | Worker ordinal within the sub-job |
| `local_gpu_index` | GPU index within one worker; it repeats across workers |
| `global_gpu_index` | GPU identity within one sub-job; group GPU series by this field |
| `gpu_uuid` | Device UUID reported by the GPU runtime |
| `gpu_util_pct` | DCGM GPU utilization: the percentage of the interval in which a kernel was running on the GPU. It says how busy the GPU was, not how saturated |
| `fb_used_bytes` / `fb_free_bytes` | Used and free framebuffer bytes; memory used is `fb_used / (fb_used + fb_free)` |
| `power_watts` | Board power gauge |
| `gpu_temp_c` | GPU temperature gauge |
| `sm_active_pct` / `sm_occupancy_pct` | Optional DCGM profiling gauges for SM activity; often `null` where profiling metrics aren't collected (`null` is not `0`) |
| `tensor_active_pct` / `dram_active_pct` | Optional DCGM profiling gauges for tensor and DRAM activity; often `null`, like the SM gauges |
| `seq` / `seq_epoch` | Sample sequence and its opaque generation |
| `sample_interval_s` | Sampling interval in seconds |

One row represents one GPU at one scrape. Gauge values are used as-is rather
than differenced. JSON `null` means unmeasured and is distinct from `0.0`.
`global_gpu_index` is unique only inside a sub-job, so consumers loading more
than one file also group by `sub_job_id`.

Deployments that collect CPU and memory also write version 3 system rows to the
same file, one per node per scrape, with these fields in this order:

| Field | Meaning |
|---|---|
| `kind` | `"system"` |
| `record_version` | Row schema version; currently `3` |
| `sub_job_id` | Sub-job that produced the sample |
| `sample_ts` | RFC3339 UTC scrape time, as for GPU rows |
| `worker_num` | The node: the same numbering as the GPU rows' `worker_num` |
| `seq` / `seq_epoch` | Sample sequence and its opaque generation |
| `sample_interval_s` | Sampling interval in seconds |
| `cpu_util_pct` | The node's CPU time as a percentage of its CPU limit (0 to 100) |
| `cpu_count` | The node's CPU limit |
| `mem_used_bytes` | Memory the node's processes use, not counting the page cache |
| `mem_total_bytes` | The node's memory limit |

A file can hold system rows and no GPU rows, and a node can miss a scrape's
system row while its GPU rows of that scrape are there. Rows without `kind` (or
with `kind: "gpu"`) are GPU rows; consumers should ignore kinds they do not
support.

### 12.5 Zone scheduling events

The client contains `tail_events()` and `stream_events()`, but the server does
not accept their operation type. See
[section 7.9](#79-unsupported-zone-events-client-helper).

---

## 13. End-to-end examples

### 13.1 Create training and sampling sub-jobs

```python
from cortex_training.client import CortexTrainingClient, SubJobConfig

client = CortexTrainingClient.from_pat(
    host=HOST,
    pat=PAT,
    database=DATABASE,
    schema=SCHEMA,
)

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    dtype="bfloat16",
)
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    dtype="bfloat16",
)

job_id = client.create_job(sub_jobs=[training, sampling])
client.wait_for_job(job_id)

training_id = f"{job_id}:training:0"
sampling_id = f"{job_id}:sampling:0"
```

### 13.2 Generate, train, step, and sync

```python
request_id = client.generate(
    job_id,
    prompts=["Write a short proof."],
    sampling_params={"max_tokens": 256, "temperature": 0.7},
)
rollouts = client.poll_request(job_id, request_id)["results"]

request_id = client.forward_backward(job_id, fwd_bwd_dssst1_frame)
train_result = client.poll_request(job_id, request_id)

request_id = client.step(job_id, learning_rate=1e-4)
step_result = client.poll_request(job_id, request_id)

request_id = client.weight_sync(
    job_id,
    source_sub_job_id=training_id,
    target_sub_job_ids=[sampling_id],
)
client.poll_request(job_id, request_id)
```

### 13.3 Save and runtime load

```python
from datetime import datetime

request_id = client.save(job_id, checkpoint_type="resumable")
client.poll_request(job_id, request_id)
# Select the latest resumable checkpoint without assuming list order. A
# checkpoint saved without a type has no checkpoint_type, and is resumable.
checkpoint_id = max(
    (cp for cp in client.list_checkpoints(job_id) if cp.get("checkpoint_type", "resumable") == "resumable"),
    key=lambda cp: datetime.fromisoformat(cp["created_at"].replace("Z", "+00:00")),
)["checkpoint_id"]

request_id = client.load(job_id, checkpoint_id=checkpoint_id)
client.poll_request(job_id, request_id)
```

### 13.4 Start sampling from saved weights

```python
from datetime import datetime

request_id = client.save(training_job_id, checkpoint_type="weights-only")
client.poll_request(training_job_id, request_id)
checkpoint_id = max(
    (cp for cp in client.list_checkpoints(training_job_id)
     if cp.get("checkpoint_type", "resumable") == "weights-only"),
    key=lambda cp: datetime.fromisoformat(cp["created_at"].replace("Z", "+00:00")),
)["checkpoint_id"]

sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    source_checkpoint_info={
        "checkpoint_id": checkpoint_id,
        "source_job_id": training_job_id,
    },
)
sampling_job_id = client.create_job(sub_jobs=[sampling])
```

### 13.5 Enable LoRA training and adapter sync

Configure the same adapter on both sub-jobs. The training configuration creates
the trainable PEFT parameters; the sampling configuration enables LoRA support
in vLLM before the first adapter sync.

```python
from cortex_training.client import CortexTrainingClient, SubJobConfig

client = CortexTrainingClient.from_pat(
    host=HOST,
    pat=PAT,
    database=DATABASE,
    schema=SCHEMA,
)

lora_config = {
    "peft_type": "Lora",
    "r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    dtype="bfloat16",
    extra_training={"peft_config": lora_config},
)
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    dtype="bfloat16",
    extra_sampling={"peft_config": lora_config},
)

job_id = client.create_job(sub_jobs=[training, sampling])
client.wait_for_job(job_id)

training_id = f"{job_id}:training:0"
sampling_id = f"{job_id}:sampling:0"

# Submit a DSSST1 training batch, then update the adapter parameters.
request_id = client.forward_backward(job_id, fwd_bwd_dssst1_frame)
client.poll_request(job_id, request_id)

request_id = client.step(job_id, learning_rate=1e-4)
client.poll_request(job_id, request_id)

# Broadcast only LoRA tensors. Later syncs update the same resident adapter.
request_id = client.weight_sync(
    job_id,
    source_sub_job_id=training_id,
    target_sub_job_ids=[sampling_id],
    weight_format="lora",
)
client.poll_request(job_id, request_id)

# Generation now uses the synchronized resident adapter.
request_id = client.generate(
    job_id,
    prompts=["Write a short proof."],
    sampling_params={"max_tokens": 256, "temperature": 0.7},
)
result = client.poll_request(job_id, request_id)
```

---

## 14. Known limitations

These are current gaps, not supported API behavior:

1. `save(checkpoint_id=...)` sends a field that is absent from the server's
   `SaveRequest`, so a caller-selected id is not honored. Use the server-assigned
   id from `stage_path` or the job's checkpoint list (section 6.3).
2. `tail_events()` and `stream_events()` send an operation type the server does
   not accept (see [section 7.9](#79-unsupported-zone-events-client-helper)).
3. `wait_for_job()` does not treat `terminated` as terminal, so a torn-down job
   polls until `poll_timeout` rather than failing immediately.
4. Generate prompt validation resolves `max_seq_len` from the first sub-job
   carrying an `inference_config` rather than matching `job_type="sampling"`. A
   `log_probability` sub-job listed first therefore supplies the wrong window.
