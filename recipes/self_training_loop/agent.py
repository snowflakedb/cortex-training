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

"""The LangGraph agent, on a gateway model or on a Cortex Training sampling job.

The sampling job takes token ids, so :func:`sampling_chat_model` renders each prompt
locally with Qwen3-8B's chat template (thinking off, the tool schemas in the system
turn) and parses ``<tool_call>`` blocks from the reply. train.py renders the training
examples with the same template, so a trained checkpoint sees prompts exactly as it
saw them in training.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from typing import Any

from recipes.inference.endpoint import generate_results
from recipes.self_training_loop.tasks import (
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    run_calculator,
    run_sql,
)

CONVERSATION_HEADER = "x-snowflake-ai-gateway-conversation-id"
MODEL = "Qwen/Qwen3-8B"
# The tokenizer and chat template revision the trainer and sampler both render with.
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# Qwen3's soft switch: the trained model answers without a reasoning block.
STUDENT_SYSTEM_PROMPT = SYSTEM_PROMPT + " /no_think"
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)


def render(
    tokenizer, messages: list[dict[str, Any]], *, generation_prompt: bool = False
) -> str:
    """Chat-template text for ``messages`` (system turn included), with the tool schemas."""
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOL_SCHEMAS,
        tokenize=False,
        add_generation_prompt=generation_prompt,
        enable_thinking=False,
    )


def text_of(content) -> str:
    """Text of a LangChain message's content (a string, or a list of content blocks)."""
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            if isinstance(b, dict) and b.get("type", "text") == "text"
            else ""
            for b in content
        )
    return content or ""


def tool_text(content) -> str:
    """A tool message's content as the model sees it."""
    return content if isinstance(content, str) else json.dumps(content)


def chat_message(m) -> dict[str, Any] | None:
    """One LangGraph message as a chat-template message; None for other message types."""
    kind = getattr(m, "type", None)
    if kind == "system":
        return {"role": "system", "content": text_of(m.content)}
    if kind == "human":
        return {"role": "user", "content": text_of(m.content)}
    if kind == "tool":
        return {"role": "tool", "content": tool_text(m.content)}
    if kind == "ai":
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": text_of(m.content).strip(),
        }
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["args"]},
                }
                for c in m.tool_calls
            ]
        return msg
    return None


def trajectory_messages(messages) -> list[dict[str, Any]]:
    """A run's messages without the system prompt (user, assistant with tool_calls, tool).

    train.py adds the system prompt back when it renders the run.
    """
    return [c for c in map(chat_message, messages) if c and c["role"] != "system"]


def parse_reply(text: str) -> tuple[str, list[dict[str, Any]]]:
    """``(content, tool_calls)`` from a generated reply. A malformed call stays in the text."""
    calls = []
    for i, block in enumerate(TOOL_CALL.findall(text)):
        try:
            call = json.loads(block)
            calls.append(
                {
                    "name": call["name"],
                    "args": call.get("arguments") or {},
                    "id": f"call_{uuid.uuid4().hex[:12]}_{i}",
                }
            )
        except (ValueError, KeyError):
            continue  # left in the text, so the run fails visibly
    return (TOOL_CALL.sub("", text).strip() if calls else text.strip()), calls


def sampling_chat_model(
    client: Any, job_id: str, tokenizer, *, temperature: float, max_tokens: int = 1024
):
    """A LangChain chat model served by a running Cortex Training sampling job."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    sampling_params = {"max_tokens": max_tokens, "temperature": temperature}
    # The fast tokenizer is not thread-safe ("Already borrowed"), and the agent runs in worker threads.
    tokenizer_lock = threading.Lock()

    class SamplingChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "cortex-training-sampling"

        def bind_tools(self, tools, **kwargs):
            return self  # the tool schemas are fixed and rendered into every prompt

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            chat = [c for c in map(chat_message, messages) if c]
            with tokenizer_lock:
                prompt = tokenizer(
                    render(tokenizer, chat, generation_prompt=True),
                    add_special_tokens=False,
                )["input_ids"]
            result = generate_results(
                client, job_id, [prompt], sampling_params, batch_size=1
            )[0]
            content, calls = parse_reply(result["text"])
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=content, tool_calls=calls))
                ]
            )

    return SamplingChatModel()


def gateway_chat_model(
    *,
    gateway_url: str,
    api_key: str,
    model: str,
    conversation_id: str,
    temperature: float,
):
    """A gateway model, with the conversation id the gateway's spans are joined on."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=f"{gateway_url}/v1",
        api_key=api_key,
        temperature=temperature,
        max_retries=8,  # gateway models are rate limited per account
        timeout=180,
        max_completion_tokens=4096,
        default_headers={CONVERSATION_HEADER: conversation_id},
    )


def build_agent(conn: sqlite3.Connection, llm, *, system_prompt: str = SYSTEM_PROMPT):
    """The ReAct agent over the retail database with the sql_query and calculator tools."""
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def sql_query(query: str) -> str:
        """Run a read-only SQLite query and return up to 50 rows as JSON."""
        return run_sql(conn, query)

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression with + - * / ( ) and decimals."""
        return run_calculator(expression)

    return create_react_agent(llm, [sql_query, calculator], prompt=system_prompt)
