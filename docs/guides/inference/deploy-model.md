# Deploy a Model for Streaming Inference

A model deployment serves a base model or a training checkpoint as an
OpenAI-compatible chat endpoint, for real-time (streaming) inference: each
request gets its reply token by token as it is generated. Deployments are
managed with `cortex-training model-deployment`, and you call them with any
OpenAI-compatible client, `requests`, or `curl`.

For a runnable version of this page, see the
[model deployment example](../../../examples/model_deployment/README.md). To
generate through the Cortex Training generate API instead, see
[Serve a Training Checkpoint](serve-checkpoint.md).

Only `EVALUATION` deployments are supported today. A deployment holds its GPUs
until you drop it.

## 1. Write a spec

From a base model:

```json
{
  "deployment": {"class": "EVALUATION", "min_replicas": 1, "gpus_per_replica": 1, "gpu_type": "H200"},
  "source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
  "inference": {"max_seq_len": 8192}
}
```

From a training run, point `source` at a completed `weights-only` checkpoint.
It names no model; the client reads it from the training job:

```json
"source": {"type": "checkpoint", "job_id": "TRAINING_JOB_UUID", "checkpoint_id": "CHECKPOINT_ID"}
```

`gpus_per_replica` (1, 2, 4, or 8) is each replica's tensor-parallel size, and
`min_replicas * gpus_per_replica` must be at most 8 or a multiple of 8. Which
models can be deployed is decided by the server. Check a spec without creating
anything:

```bash
cortex-training model-deployment create --spec deployment.json --dry-run
```

## 2. Create the deployment

```bash
cortex-training model-deployment create --spec deployment.json --wait
```

`--wait` returns once the deployment is `READY`, which can take several
minutes on a cold start. It prints the deployment ID and the endpoint:

```json
{
  "base_url": "https://ACCOUNT.snowflakecomputing.com/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training/inference/v1",
  "deployment": "sf-ct-DEPLOYMENT_UUID",
  "status": "READY"
}
```

Without `--wait`, `create` returns at `PENDING`; run
`cortex-training model-deployment wait sf-ct-DEPLOYMENT_UUID` later. If waiting
fails, times out, or is interrupted, the deployment keeps running until you
drop it.

From Python:

```python
from cortex_training import CortexTrainingClient

client = CortexTrainingClient.from_pat(
    host="ACCOUNT.snowflakecomputing.com",
    pat=PAT,
    database="CORTEX_TRAINING_DB",
    schema="PUBLIC",
)
deployment_id = client.create_model_deployment(spec)["deployment"]
ready = client.wait_for_model_deployment(deployment_id)
base_url = ready["base_url"]
```

## 3. Stream a chat completion

Authenticate the same way you did for the SDK and pass the deployment ID as
`model`. `client.model_deployment_headers()` returns the right headers for
either credential: `Authorization: Bearer <PAT>` plus
`X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN` for a PAT, or
`Authorization: Snowflake Token="<session token>"` for a connection profile
(`CortexTrainingClient.from_connection_name`). A session token sent as a Bearer
token is rejected with HTTP 401. Inference is not yet available on all accounts;
until it is enabled for yours, requests return HTTP 404.

With the `openai` package:

```python
from openai import AuthenticationError, OpenAI


def stream_haiku(headers):
    openai = OpenAI(
        base_url=base_url,
        api_key="unused",  # the headers below replace the Authorization it sends
        default_headers=headers,
    )
    return openai.chat.completions.create(
        model="sf-ct-DEPLOYMENT_UUID",
        messages=[{"role": "user", "content": "Write a haiku about snow."}],
        max_completion_tokens=256,
        stream=True,
    )


headers = client.model_deployment_headers()
try:
    stream = stream_haiku(headers)
except AuthenticationError:  # HTTP 401
    if not headers.get("Authorization", "").startswith("Snowflake Token="):
        raise  # a PAT does not expire mid-run
    # The session expired. An OpenAI client keeps the headers it was built
    # with, so build a new one with the headers for a new session.
    stream = stream_haiku(client.model_deployment_headers(rejected=headers))
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

A session token expires with its session, so take fresh headers from
`client.model_deployment_headers()` for each `OpenAI` client rather than once
per process. After a 401, pass the rejected headers back as `rejected=`: the
client opens a new session only if no other caller already replaced the
rejected one, so concurrent callers share the new session. The
`examples/model_deployment` example also checks the 401's error code, and
retries only a session expiry.

`openai.models.list()` (`GET $BASE_URL/models` with the same headers) is meant
to list your `READY` deployments, but it can return an empty list while a
deployment is `READY`
([known limitation 6](../../reference/rest-api.md#15-known-limitations)). Use
`cortex-training model-deployment list` to find deployments.

With `curl` (`-N` turns off buffering so events print as they arrive). For a PAT:

```bash
curl -N "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $PAT" \
  -H "X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"model": "sf-ct-DEPLOYMENT_UUID", "stream": true, "max_completion_tokens": 256,
       "messages": [{"role": "user", "content": "Write a haiku about snow."}]}'
```

For a connection profile, replace the two PAT headers with
`-H "Authorization: Snowflake Token=\"$SESSION_TOKEN\""`.

The streamed response is server-sent events: one `data: {...}` line per
`chat.completion.chunk`, whose `choices[0].delta.content` holds the next piece
of text, and a final `data: [DONE]`. The first chunk's delta carries only
`"role": "assistant"`, and the last chunk before `[DONE]` has an empty delta
and the `finish_reason`. Add `"stream_options": {"include_usage": true}` to get
one more chunk, with empty `choices`, that carries the token counts. Leave out
`"stream": true` to get one `chat.completion` object instead.

Limit the reply length with `max_completion_tokens`. The endpoint rejects
`max_tokens` with HTTP 400 (`"param": "max_tokens"`,
`"code": "unsupported_parameter"`).

## 4. Inspect

```bash
cortex-training model-deployment list --status READY
cortex-training model-deployment describe sf-ct-DEPLOYMENT_UUID
cortex-training model-deployment download-log sf-ct-DEPLOYMENT_UUID --output-dir ./logs
cortex-training model-deployment download-metrics sf-ct-DEPLOYMENT_UUID --output-dir ./metrics
```

A `FAILED` deployment reports `failure: {code, message}`; its execution logs
say why.

## 5. Drop it

```bash
cortex-training model-deployment drop sf-ct-DEPLOYMENT_UUID
```

`drop` is safe to retry. A deployment that already failed or terminated holds
no GPUs and reports `"dropped": false` with its status.

## Reference

- [CLI reference: Model Deployments](../../reference/cli.md#model-deployments)
- [REST API reference section 14](../../reference/rest-api.md#14-model-deployments):
  spec rules, statuses, drop, downloads, and errors, and
  [section 14.8](../../reference/rest-api.md#148-calling-a-deployment) for the
  chat completions request, streaming format, and errors
