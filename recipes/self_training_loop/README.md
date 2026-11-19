# Self-Training Loop on Judged Agent Traces

A LangGraph agent is traced through Cortex AI Gateway with TruLens, LLM judges
score every run, and Qwen3-8B is fine-tuned on the runs that pass. The tuned
model then runs the same agent, and the next version trains on all runs kept so
far. Training data is selected by the judges alone, without reference answers.

This recipe ports the code from
[Snowflake-Labs/agent-self-training-loop](https://github.com/Snowflake-Labs/agent-self-training-loop),
which runs the loop as standalone scripts and ML Jobs and includes the
architecture diagrams and the cost analysis.

## The agent and its tasks

A ReAct agent (`langgraph.prebuilt.create_react_agent`) answers retail-finance
questions over a synthetic SQLite database with two tools, `sql_query` and
`calculator`. The business rules are stated only in the system prompt: only
shipped orders count, line values are net of discount, and returns are valued
at the discounted price. `tasks.py` builds the database and 395 tasks
deterministically:

| Tasks | Ids | Types | Training | Held out |
|---|---|---|---|---|
| Baseline | `h*` | 7, e.g. month-over-month growth, the second-highest customer | 186 | 49 |
| Drift | `n*` | 4, added after v1, e.g. a category's share of a region's revenue | 130 | 30 |

Each task has a reference answer computed in SQL, used only to measure
accuracy.

## The loop

1. **Seed.** glm-5.3 on the gateway answers the 186 baseline training tasks 3
   times each (`collect.py`).
2. **Store.** TruLens sends each run's spans to the gateway, which writes them
   to `AGENT_TRACE_TABLE` with its own chat spans and the judges' eval spans.
   Runs on a sampling job bypass the gateway, so an `agt.trajectory` span
   carries each run's messages.
3. **Judge and select.** claude-sonnet-4-6 scores every run with
   `answer_relevance` and four trace-level agent judges (`logical_consistency`,
   `execution_efficiency`, `tool_selection`, `tool_calling`). `select.py` builds
   `TRAJECTORIES` and the view `TRAINING_SET_JUDGED`: runs that finished and
   met every cutoff, the best 2 per task (`sql/trajectories.sql`).
4. **Train.** `train.py` trains a LoRA adapter on Qwen3-8B from the kept traces
   with the shared SFT runner.
5. **Serve and repeat.** `collect.py` serves the checkpoint on a sampling job and
   runs the agent 4 times on each of the 316 training tasks, as stand-in
   production traffic, and twice on each of the 79 held-out tasks. Each new
   version trains from base Qwen3-8B on the seed runs and every earlier
   version's training-task runs.

A version is promoted when its held-out judge pass rate beats glm-5.3's on both
baseline and drift tasks.

## Results

Each model answered every held-out task twice. Judge pass rate is the share of
runs that met every selection cutoff. Accuracy is against the reference
answers:

| Model | Judge pass: baseline | drift | overall | Accuracy: baseline | drift | overall |
|---|---|---|---|---|---|---|
| glm-5.3 | 69.4% | 71.7% | 70.3% | 100.0% | 96.1% | 98.5% |
| Qwen3-8B v1 | 62.2% | 13.3% | 43.7% | 89.6% | 34.5% | 68.8% |
| Qwen3-8B v2 | 72.4% | 45.0% | 62.0% | 88.5% | 76.7% | 84.0% |
| Qwen3-8B v3 | 90.8% | 76.7% | 85.4% | 89.8% | 93.3% | 91.1% |

The seed has no drift-task runs, so the drift tasks were learned only from the
model's own runs. v1 had a judge-approved run on 64 of the 130 drift training
tasks, and v2, trained on those, on 116. v3 is the first version to pass the
promotion gate, with 46 of 60 passing drift runs against glm-5.3's 43. Of the
runs kept for training, 97.5 to 99.7% were correct.

| Version | Trained on | Turn examples | Steps |
|---|---|---|---|
| v1 | seed | 1,154 | 289 |
| v2 | seed + v1 | 2,469 | 618 |
| v3 | seed + v1 + v2 | 4,166 | 1,042 |

These results come from the original project's runs. Those runs used the same
tasks, prompts, judges, cutoffs, rendering, and training and sampling settings
as this recipe. This code was tested on a fresh seed run (558 runs, 556 correct;
349 traces kept on 180 tasks, giving 1,192 turn examples and 298 steps), a
3-step training job, and sampling from that checkpoint. The full loop has not
been rerun with this code, so `last_validated` is unset.

## Setup

```bash
uv pip install -e . "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly" chz datasets
uv pip install trulens-core==2.16.0 trulens-feedback==2.16.0 trulens-apps-langgraph==2.16.0 \
  trulens-apps-langchain==2.16.0 trulens-providers-cortex==2.16.0 trulens-otel-semconv==2.16.0 \
  langgraph==1.2.19 langgraph-prebuilt==1.1.1 langchain-core==1.7.5 langchain-openai==1.7.3 \
  openai==3.45.0 opentelemetry-sdk==1.45.2 opentelemetry-exporter-otlp-proto-http==1.45.2 \
  snowflake-snowpark-python
```

The account needs Cortex AI Gateway, `claude-sonnet-4-6` and `glm-5.3` through
Cortex, and 4 free H200 GPUs. Training and sampling run as separate jobs on 4
GPUs each.

1. **Gateway logging.** `sql/gateway_spec.sql` turns on payload capture and
   client telemetry. It replaces the whole gateway specification; save the
   current one first and restore it afterwards.
2. **PAT.** The gateway's chat and telemetry endpoints need a programmatic
   access token: `export SNOWFLAKE_PAT=...`.
3. **Schema.** `schema=DATABASE.SCHEMA` (default
   `CORTEX_TRAINING_DB.SELF_TRAINING_LOOP`) holds `TRAJECTORIES` and
   `TRAINING_SET_JUDGED`. It is created if missing. Training jobs use the
   Cortex Training client's database and schema (`CORTEX_TRAINING_DATABASE`,
   `CORTEX_TRAINING_SCHEMA`).

## Run

```bash
C="connection=<profile> schema=<DB>.<SCHEMA>"

# Seed: glm-5.3 on the baseline training tasks, then select
python -m recipes.self_training_loop.collect $C run_tag=seed drift=false rollouts=3 workers=4
python -m recipes.self_training_loop.select  $C run_tags=seed

# v1: train, then collect its training-task and held-out runs
python -m recipes.self_training_loop.train   $C source_run_tags=seed
python -m recipes.self_training_loop.collect $C run_tag=v1-runs rollouts=4 \
    source_job_id=<JOB_ID> checkpoint_id=<CHECKPOINT_ID>
python -m recipes.self_training_loop.collect $C run_tag=v1-eval split=heldout rollouts=2 \
    source_job_id=<JOB_ID> checkpoint_id=<CHECKPOINT_ID>
python -m recipes.self_training_loop.select  $C run_tags=v1-runs,v1-eval

# v2 trains on source_run_tags=seed,v1-runs; v3 on seed,v1-runs,v2-runs.

# Judge pass rate and accuracy per run tag
python -m recipes.self_training_loop.select  $C run_tags=seed-eval,v1-eval,v2-eval,v3-eval report=true
```

For the glm-5.3 held-out row, run `collect.py` with no checkpoint and
`run_tag=seed-eval split=heldout rollouts=2`.

`train.py` prints the job and checkpoint ids. `collect.py` starts a 4-GPU
sampling job for the checkpoint (`sampling_gpus=`) and cancels it when the
runs finish, before judging. Judging takes the longest: claude-sonnet-4-6 is
rate limited per account, and 1,264 traces can take several hours.

## How the training data is built

Each assistant turn of a kept trace is one example, with that turn as its only
target. Qwen3's chat template drops the empty `<think></think>` block from
earlier assistant turns but adds it to the generation prompt, so a single
render of a whole trace differs from what the model saw at each step.
`train.py` renders turn *i* as the continuation of the generation prompt for
the messages before it. It uses the same pinned Qwen3-8B tokenizer, tool
schemas and `/no_think` system prompt as the sampling chat model in `agent.py`.
Training runs `ceil(epochs * examples / 8)` steps at a constant learning rate of
1e-4 and reshuffles the examples on each pass. Traces with a turn longer than
8,192 tokens are skipped.

## Files

| File | Purpose |
|---|---|
| `tasks.py` | Database, tasks, reference answers and tools |
| `agent.py` | The agent, the gateway chat model, and the sampling-job chat model |
| `tracing.py` | TruLens exporter to the gateway, the trajectory span, and the judges |
| `collect.py` | Runs the agent on tasks, records and judges each run |
| `select.py` | Builds `TRAJECTORIES` and `TRAINING_SET_JUDGED`, prints results |
| `train.py` | Turn examples from the kept traces, and the training entrypoint |
| `store.py` | Snowpark session and SQL-file helpers |
| `sql/` | Gateway specification, selection SQL and the results query |
| `overrides.json` | Training settings on top of the catalog file |
