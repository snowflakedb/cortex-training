# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the agent on tasks, trace every run with TruLens to Cortex AI Gateway, and judge it.

Seed runs use a gateway model; later runs serve a trained checkpoint from a Cortex Training
sampling job, started here and cancelled when the runs finish (before judging):

    export SNOWFLAKE_PAT=...   # a PAT for the gateway's chat and telemetry endpoints
    python -m recipes.self_training_loop.collect connection=<profile> run_tag=seed drift=false rollouts=3
    python -m recipes.self_training_loop.collect connection=<profile> run_tag=v1-train rollouts=4 \\
        source_job_id=<JOB_ID> checkpoint_id=<CHECKPOINT_ID>
    python -m recipes.self_training_loop.collect connection=<profile> run_tag=v1-eval split=heldout \\
        rollouts=2 source_job_id=<JOB_ID> checkpoint_id=<CHECKPOINT_ID>

Spans, trajectory spans and eval spans land in AGENT_TRACE_TABLE; select.py turns them into
TRAJECTORIES and the training set.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chz

from recipes.inference.endpoint import inference_endpoint_body
from recipes.self_training_loop.agent import (
    STUDENT_SYSTEM_PROMPT,
    build_agent,
    gateway_chat_model,
    load_tokenizer,
    sampling_chat_model,
    text_of,
    trajectory_messages,
)
from recipes.self_training_loop.store import snowpark_session
from recipes.self_training_loop.tasks import SYSTEM_PROMPT, is_correct, load_tasks
from recipes.self_training_loop.tracing import (
    APP_NAME,
    gateway_url,
    judges,
    tee_exporter,
    trajectory_recorder,
)
from recipes.utils import make_client, running_job

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


@chz.chz
class Config:
    # Snowflake connection profile name (connections.toml); the default profile when None.
    connection: str | None = None
    schema: str = "CORTEX_TRAINING_DB.SELF_TRAINING_LOOP"
    warehouse: str | None = None
    gateway: str = "SNOWFLAKE"
    # Environment variable that holds a PAT for the gateway.
    pat_env: str = "SNOWFLAKE_PAT"

    run_tag: str = "seed"
    split: str = "train"
    # Include the drift task types. The seed run answers only the baseline tasks.
    drift: bool = True
    # Comma-separated task-id prefixes to keep (h = baseline, n = drift); empty keeps all.
    kinds: str = ""
    rollouts: int = 3
    limit: int = 0
    workers: int = 8
    temperature: float = 0.7

    # Gateway model for seed runs.
    seed_model: str = "glm-5.3"
    # A trained checkpoint to serve from a sampling job instead.
    source_job_id: str | None = None
    checkpoint_id: str | None = None
    # Attach to an already running sampling job (left running afterwards).
    job_id: str | None = None
    model: str = "Qwen/Qwen3-8B"
    context: str = "standard"
    hardware: str = "h200"
    max_seq_len: int = 8192
    # Sampling GPUs, all in one tensor-parallel group. The catalog default is 2 GPUs at TP 1.
    sampling_gpus: int = 4


def choose_tasks(config: Config, tasks: list[dict]) -> list[dict]:
    chosen = [t for t in tasks if t["split"] == config.split]
    if config.kinds:
        chosen = [
            t for t in chosen if t["task_id"].startswith(tuple(config.kinds.split(",")))
        ]
    return chosen[: config.limit] if config.limit else chosen


def main(config: Config) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from trulens.apps.langgraph import TruGraph
    from trulens.core import TruSession
    from trulens.core.database.connector.default import DefaultDBConnector

    pat = os.environ.get(config.pat_env)
    if not pat:
        raise ValueError(f"set {config.pat_env} to a PAT for the gateway")
    session = snowpark_session(config.connection, config.schema, config.warehouse)
    url = gateway_url(session, config.gateway)

    tmp = tempfile.mkdtemp(prefix="trulens-")
    connector = DefaultDBConnector(database_url=f"sqlite:///{tmp}/trulens.sqlite")
    exporter = tee_exporter(connector, endpoint=url, api_key=pat)
    tru_session = TruSession(connector=connector, span_exporter=exporter)
    record_trajectory = trajectory_recorder(tru_session, app_version=config.run_tag)

    db, tasks = load_tasks(drift=config.drift)
    chosen = choose_tasks(config, tasks)
    feedbacks = judges(session, {t["question"]: t for t in tasks})
    logger.info(
        "%d tasks x %d rollouts, run_tag=%s",
        len(chosen),
        config.rollouts,
        config.run_tag,
    )

    # TruLens allows one active recording per app wrapper, so each worker thread gets its own.
    local = threading.local()
    lock = threading.Lock()
    wrappers: list = []

    # The trained model runs with Qwen3's /no_think switch on the system prompt, as in training.
    sampled = bool(config.source_job_id or config.job_id)
    system_prompt = STUDENT_SYSTEM_PROMPT if sampled else SYSTEM_PROMPT

    def recorder(llm):
        if not hasattr(local, "tru"):
            with lock:
                local.tru = TruGraph(
                    build_agent(db, llm, system_prompt=system_prompt),
                    app_name=APP_NAME,
                    app_version=config.run_tag,
                    feedbacks=feedbacks,
                )
                wrappers.append(local.tru)
        return local.tru

    def run_all(make_llm) -> list[dict]:
        def run_one(task, k):
            conversation_id = f"{config.run_tag}:{task['task_id']}:{k}"
            llm = make_llm(conversation_id)
            graph = build_agent(db, llm, system_prompt=system_prompt)
            meta = {
                "task_id": task["task_id"],
                "split": task["split"],
                "conversation_id": conversation_id,
                "run_tag": config.run_tag,
                "rollout": k,
            }
            result, error = None, None
            with recorder(llm):
                try:
                    result = graph.invoke(
                        {"messages": [{"role": "user", "content": task["question"]}]},
                        config={"metadata": meta, "recursion_limit": 24},
                    )
                except Exception as e:  # kept; select.py filters unfinished runs
                    error = f"{type(e).__name__}: {e}"[:300]
                    logger.warning("%s failed: %s", conversation_id, error)
            if (
                result is not None
            ):  # outside the recording, so it is not a TruLens record
                record_trajectory(
                    conversation_id, json.dumps(trajectory_messages(result["messages"]))
                )
            text = text_of(result["messages"][-1].content) if result else ""
            return {
                "task_id": task["task_id"],
                "correct": is_correct(text, task["answer"]),
                "error": error,
            }

        rows = []
        jobs = [(t, k) for t in chosen for k in range(config.rollouts)]
        started = time.time()
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            for future in as_completed([pool.submit(run_one, t, k) for t, k in jobs]):
                rows.append(future.result())
                if len(rows) % 25 == 0 or len(rows) == len(jobs):
                    logger.info(
                        "runs %d/%d correct=%d errors=%d %.0fs",
                        len(rows),
                        len(jobs),
                        sum(r["correct"] for r in rows),
                        sum(bool(r["error"]) for r in rows),
                        time.time() - started,
                    )
        return rows

    sampling_seconds = 0.0
    if sampled:
        client = make_client(None, config.connection)
        tokenizer = load_tokenizer()
        body, _ = inference_endpoint_body(
            model=config.model,
            context=config.context,
            hardware=config.hardware,
            lora=True,
            max_seq_len=config.max_seq_len,
            source_job_id=config.source_job_id,
            checkpoint_id=config.checkpoint_id,
        )
        for sub_job in body["sub_job_configs"]:
            if sub_job["job_type"] == "sampling":
                sub_job["inference_config"]["n_gpus"] = config.sampling_gpus
                sub_job["inference_config"]["vllm_config"]["tensor_parallel_size"] = (
                    config.sampling_gpus
                )
        with running_job(client, body, job_id=config.job_id) as job_id:
            started = time.time()
            llm = sampling_chat_model(
                client, job_id, tokenizer, temperature=config.temperature
            )
            rows = run_all(lambda _conversation_id: llm)
            sampling_seconds = time.time() - started
        # The sampling job is cancelled on exit (unless attached with job_id=), before judging.
    else:
        rows = run_all(
            lambda conversation_id: gateway_chat_model(
                gateway_url=url,
                api_key=pat,
                model=config.seed_model,
                conversation_id=conversation_id,
                temperature=config.temperature,
            )
        )

    tru_session.force_flush()
    time.sleep(5)
    events = connector.get_events(app_name=APP_NAME, app_version=config.run_tag)
    logger.info("computing judges on %d local spans", len(events))
    if wrappers:
        wrappers[0].compute_feedbacks(
            events=events, raise_error_on_no_feedbacks_computed=False
        )
    tru_session.force_flush()
    time.sleep(10)
    tru_session._experimental_tracer_provider.shutdown()
    with contextlib.suppress(OSError):
        shutil.rmtree(tmp)
    summary = {
        "run_tag": config.run_tag,
        "runs": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "sampling_seconds": round(sampling_seconds),
        "remote_spans": exporter.remote_spans,
        "remote_failures": exporter.remote_failures,
    }
    logger.info("done %s", summary)
    return summary


if __name__ == "__main__":
    chz.nested_entrypoint(main)
