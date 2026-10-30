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

"""The verifiers ``reverse-text`` taskset that prime-rl trains on.

Single turn: reverse the user text character by character inside
``<reversed_text>`` tags. The reward is the ``difflib`` LCS ratio between the
tagged answer and the true reversal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

RL_DATASET = "PrimeIntellect/Reverse-Text-RL"
SFT_DATASET = "willcb/R1-reverse-wikipedia-paragraphs-v1-1000"
SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
_TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


@dataclass(frozen=True)
class ReverseTextTask:
    idx: int
    prompt: str
    answer: str


def load_tasks(dataset: str = RL_DATASET, split: str = "train") -> list[ReverseTextTask]:
    """Tasks in source order; the answer is the prompt reversed."""
    import datasets

    rows = datasets.load_dataset(dataset, split=split)
    return [ReverseTextTask(idx, row["prompt"], row["prompt"][::-1]) for idx, row in enumerate(rows)]


def task_messages(task: ReverseTextTask) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]


def reply_content(text: str) -> str:
    """The assistant reply without its reasoning, split at the first ``</think>``."""
    if "</think>" in text:
        _, _, text = text.partition("</think>")
    return text.strip()


def lcs_reward(text: str, answer: str) -> float:
    match = _TAG.search(reply_content(text))
    response = match.group(1).strip() if match else ""
    return SequenceMatcher(None, response, answer).ratio()
