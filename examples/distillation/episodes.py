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

"""Multi-turn tool-use episodes for on-policy distillation.

An episode is one attempt at one question: the agent calls tools until it gives
a final answer, runs out of turns, or fills its context. The conversation is
extended token by token instead of re-rendered each turn, so the training
sequence is exactly the tokens the agent was sampled on, and each agent turn
stays verbatim. ``ChatFormat.check_tool_turn`` confirms at startup that the
appended tool turns match the model's chat template.

Each model in an episode (the agent, and the teacher that scores it) keeps its
own running copy of the conversation in its own chat format. The models share a
tokenizer, so the agent's tokens are appended unchanged to every copy.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import torch

from examples.distillation.teacher import generate_results
from examples.distillation.teacher import sampled_token_logprobs
from examples.utils import TrainSequence

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Qwen3.5-family models write tool calls as XML; Qwen3 models write JSON.
XML_CALL = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
INVALID_TOOL_CALL = "invalid_tool_call"
EPISODE_ENDS = ("answer", "max_turns", "context", "length", "empty")
_THINK_OFF_HEADER = "<think>\n\n</think>\n\n"


class ChatFormat:
    """One model's chat format, extended incrementally with thinking off."""

    def __init__(self, tokenizer: Any):
        self.tok = tokenizer
        probe = self.render([{"role": "user", "content": "x"}], text=True)
        self.header_suffix = _THINK_OFF_HEADER if probe.endswith(_THINK_OFF_HEADER) else ""
        self.stop_ids = sorted(
            {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        )

    def render(
        self, messages: list[dict], tools: list | None = None, text: bool = False
    ) -> Any:
        out = self.tok.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=not text,
        )
        if text:
            return out
        return [int(t) for t in (out["input_ids"] if hasattr(out, "input_ids") else out)]

    def first_prompt(self, user_message: str, tools: list) -> list[int]:
        return self.render([{"role": "user", "content": user_message}], tools=tools)

    def tool_turn_text(self, results: Sequence[str]) -> str:
        body = "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
        return f"\n<|im_start|>user{body}<|im_end|>\n<|im_start|>assistant\n{self.header_suffix}"

    def tool_turn(self, results: Sequence[str]) -> list[int]:
        return self.tok.encode(self.tool_turn_text(results), add_special_tokens=False)

    def check_tool_turn(self, tools: list) -> bool:
        """The appended tool turn matches what the chat template renders for the same messages."""
        name = tools[0]["function"]["name"]
        call = {"type": "function", "function": {"name": name, "arguments": {}}}
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": "R1"},
            {"role": "tool", "content": "R2"},
        ]
        rendered = self.render(messages, tools=tools, text=True)
        return rendered.endswith(self.tool_turn_text(["R1", "R2"]))

    def decode_turn(self, tokens: Sequence[int]) -> str:
        text = self.tok.decode(list(tokens), skip_special_tokens=False)
        return text.replace("<|im_end|>", "").replace("<|endoftext|>", "")


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """``(name, arguments)`` for every tool call in an agent turn, in either Qwen format."""
    calls: list[tuple[str, dict]] = []
    for raw in TOOL_CALL.findall(text):
        xml = XML_CALL.fullmatch(raw)
        if xml:
            calls.append((xml.group(1), dict(XML_PARAM.findall(xml.group(2)))))
            continue
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", {})
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                raise TypeError("arguments must be an object")
            calls.append((str(obj.get("name")), args))
        except (json.JSONDecodeError, AttributeError, TypeError):
            calls.append((INVALID_TOOL_CALL, {}))
    return calls


@dataclasses.dataclass
class Episode:
    problem: Any
    # One running token sequence per format; ids[0] is the agent's.
    ids: list[list[int]]
    # Per format: (start, length) of each agent turn in ids[k].
    spans: list[list[tuple[int, int]]]
    logprobs: list[float] = dataclasses.field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    final: str = ""
    end: str = ""
    reward: float = 0.0

    @property
    def agent_tokens(self) -> int:
        return sum(length for _, length in self.spans[0])


def run_episodes(
    client: Any,
    job_id: str,
    formats: Sequence[ChatFormat],
    problems: Sequence[Any],
    *,
    user_message: Callable[[Any], str],
    tools: list,
    call_tool: Callable[[str, dict], str],
    repeats: int,
    sampling: dict,
    max_turns: int,
    turn_max_tokens: int,
    max_seq_len: int,
    want_logprobs: bool,
    max_prompts_per_request: int | None = None,
) -> list[Episode]:
    """Play ``repeats`` episodes per problem with ``formats[0]``'s model as the agent.

    Episodes advance in lockstep: each round sends one batched generate request
    for every episode still running, then runs the tool calls it produced.
    """
    episodes = [
        Episode(
            problem,
            [f.first_prompt(user_message(problem), tools) for f in formats],
            [[] for _ in formats],
        )
        for problem in problems
        for _ in range(repeats)
    ]
    agent = formats[0]
    for turn in range(max_turns):
        for e in episodes:
            if not e.end and len(e.ids[0]) + turn_max_tokens > max_seq_len:
                e.end = "context"
        active = [e for e in episodes if not e.end]
        if not active:
            break
        params = {"max_tokens": turn_max_tokens, "stop_token_ids": agent.stop_ids, **sampling}
        if want_logprobs:
            params["logprobs"] = 0
        results = generate_results(
            client, job_id, [e.ids[0] for e in active], params, max_prompts_per_request
        )
        for e, result in zip(active, results):
            tokens = [int(t) for t in result.get("token_ids") or []]
            if not tokens:
                e.end = "empty"
                continue
            for k in range(len(formats)):
                e.spans[k].append((len(e.ids[k]), len(tokens)))
                e.ids[k].extend(tokens)
            if want_logprobs:
                e.logprobs.extend(sampled_token_logprobs(result))
            e.turns += 1
            if result.get("finish_reason") == "length":
                e.end = "length"
                continue
            text = agent.decode_turn(tokens)
            calls = parse_tool_calls(text)
            if not calls:
                e.end, e.final = "answer", TOOL_CALL.sub("", text).strip()
                continue
            e.tool_calls += len(calls)
            if turn == max_turns - 1:
                e.end = "max_turns"
                continue
            outputs = [call_tool(name, args) for name, args in calls]
            for k, f in enumerate(formats):
                e.ids[k].extend(f.tool_turn(outputs))
    for e in episodes:
        e.end = e.end or "max_turns"
    return episodes


def teacher_scoring_inputs(episodes: Sequence[Episode]) -> tuple[list[list[int]], list[list[int]]]:
    """Per episode, the teacher's prompt and the rest of its conversation through the last agent turn."""
    prompts, rests = [], []
    for e in episodes:
        first = e.spans[1][0][0]
        last_start, last_len = e.spans[1][-1]
        prompts.append(e.ids[1][:first])
        rests.append(e.ids[1][first : last_start + last_len])
    return prompts, rests


def agent_teacher_logprobs(
    episodes: Sequence[Episode], teacher_scores: Sequence[Sequence[float]]
) -> list[list[float]]:
    """The teacher's logprob of each agent token, in the order the agent wrote them."""
    out = []
    for e, scores in zip(episodes, teacher_scores):
        first = e.spans[1][0][0]
        out.append(
            [lp for start, n in e.spans[1] for lp in scores[start - first : start - first + n]]
        )
    return out


def train_sequences(
    episodes: Sequence[Episode],
    advantages: Sequence[float],
    teacher_agent_lps: Sequence[Sequence[float]] | None = None,
) -> tuple[list[TrainSequence], list[list[float]] | None]:
    """One sequence per episode, with the loss on the agent's tokens only.

    Returns the sequences and, with teacher scores, each row's teacher logprobs
    on the prediction-aligned positions: position ``p`` predicts token ``p + 1``,
    so the teacher's logprob of agent token ``t`` lands at ``t - 1``.
    """
    sequences: list[TrainSequence] = []
    shifted: list[list[float]] | None = [] if teacher_agent_lps is not None else None
    for i, (e, advantage) in enumerate(zip(episodes, advantages)):
        if teacher_agent_lps is not None and len(teacher_agent_lps[i]) != e.agent_tokens:
            raise ValueError(
                f"episode {i}: {len(teacher_agent_lps[i])} teacher scores for "
                f"{e.agent_tokens} agent tokens"
            )
        last_start, last_len = e.spans[0][-1]
        ids = e.ids[0][: last_start + last_len]
        mask = [0.0] * len(ids)
        row = [0.0] * len(ids)
        k = 0
        for start, n in e.spans[0]:
            for j in range(n):
                mask[start - 1 + j] = 1.0
                if teacher_agent_lps is not None:
                    row[start - 1 + j] = float(teacher_agent_lps[i][k + j])
            k += n
        sequences.append(TrainSequence(input_ids=ids, loss_mask=mask, advantage=advantage))
        if shifted is not None:
            shifted.append(row)
    return sequences, shifted


def shifted_tensor(rows: Sequence[Sequence[float]], loss_mask: torch.Tensor) -> torch.Tensor:
    """Pad per-row teacher logprobs to the collated batch width."""
    out = torch.zeros(loss_mask.shape, dtype=torch.float32)
    for row, values in enumerate(rows):
        out[row, : len(values)] = torch.tensor(list(values), dtype=torch.float32)
    return out
