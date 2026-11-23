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

"""TruLens tracing to Cortex AI Gateway, the trajectory span, and the judges.

TruLens records each agent run as OpenTelemetry spans. The exporter sends them both to
TruLens' local store, which computing the judges reads, and to the gateway's OTLP endpoint,
so app spans, eval spans and the gateway's own chat spans land in one ``AGENT_TRACE_TABLE``.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any

from recipes.self_training_loop.tasks import is_correct

JUDGE_MODEL = "claude-sonnet-4-6"
# Trace-level agent judges used to select training runs, next to answer_relevance.
# The planning judges are left out because this agent does not write a plan.
GPA_JUDGES = (
    "logical_consistency",
    "execution_efficiency",
    "tool_selection",
    "tool_calling",
)
APP_NAME = "retail_agent"


def gateway_url(snowpark_session, gateway: str) -> str:
    """The gateway's base URL on the ``<org>-<account>`` host.

    The ``<org>-<account_name>`` host with an underscore fails Python TLS hostname checks,
    so the account name's underscores become hyphens.
    """
    org, account = snowpark_session.sql(
        "SELECT CURRENT_ORGANIZATION_NAME(), CURRENT_ACCOUNT_NAME()"
    ).collect()[0]
    host = f"{org}-{account}".lower().replace("_", "-")
    return f"https://{host}.snowflakecomputing.com/api/v2/aigateways/{gateway.lower()}"


def tee_exporter(local_connector, *, endpoint: str, api_key: str):
    """A span exporter that writes to TruLens' local store and to the gateway's OTLP endpoint."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from trulens.experimental.otel_tracing.core.exporter.connector import (
        TruLensOtelSpanExporter,
    )

    class GatewayTeeExporter(SpanExporter):
        def __init__(self):
            self.local = TruLensOtelSpanExporter(local_connector)
            self.remote = OTLPSpanExporter(
                endpoint=f"{endpoint}/telemetry/v1/traces",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            self.remote_spans = 0
            self.remote_failures = 0

        def export(self, spans: Sequence) -> SpanExportResult:
            local = self.local.export(spans)
            if self.remote.export(spans) == SpanExportResult.SUCCESS:
                self.remote_spans += len(spans)
            else:
                self.remote_failures += 1
            return local

        def shutdown(self):
            self.local.shutdown()
            self.remote.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    return GatewayTeeExporter()


def trajectory_recorder(session, *, app_version: str):
    """Writes a run's messages as JSON on a plain OTel span (``agt.trajectory``).

    The sampling job is called directly rather than through the gateway, so the gateway has
    no payload for those runs; this span carries the conversation for every run instead. It
    is not a TruLens record, because TruLens judges every record it finds.
    """
    tracer = session._experimental_tracer_provider.get_tracer("agt.trajectory")

    def record(conversation_id: str, messages_json: str) -> None:
        with tracer.start_as_current_span("agt.trajectory") as span:
            span.set_attribute("ai.observability.app_name", APP_NAME)
            span.set_attribute("ai.observability.app_version", app_version)
            span.set_attribute("agt.conversation_id", conversation_id)
            span.set_attribute("agt.trajectory", messages_json)

    return record


def question_text(value: Any) -> str:
    """The user text from a recorded graph input (a dict, or its repr)."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip()
    if isinstance(value, dict):
        if "messages" in value:
            value = value["messages"][-1]
        return str(value.get("content", "")).strip()
    return str(value).strip()


def judges(snowpark_session, tasks_by_question: dict[str, dict]) -> list:
    """answer_relevance and the four trace-level judges, plus correctness.

    correctness compares the final answer with the task's reference answer. It is recorded
    only to measure the judges: production traffic has no reference answers, so it never
    selects training data or gates a version.
    """
    from trulens.core import Feedback, Metric, Selector
    from trulens.providers.cortex import Cortex

    provider = Cortex(snowpark_session=snowpark_session, model_engine=JUDGE_MODEL)

    def correctness(question, answer: str) -> float:
        task = tasks_by_question.get(question_text(question))
        return 1.0 if task and is_correct(answer, task["answer"]) else 0.0

    trace = {"trace": Selector(trace_level=True)}
    return [
        Feedback(correctness, name="correctness").on_input().on_output(),
        Feedback(provider.relevance_with_cot_reasons, name="answer_relevance")
        .on_input()
        .on_output(),
        *(
            Metric(
                implementation=getattr(provider, f"{name}_with_cot_reasons"),
                name=name,
                selectors=trace,
            )
            for name in GPA_JUDGES
        ),
    ]
