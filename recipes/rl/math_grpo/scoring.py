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

"""Prompt construction and response scoring shared by GRPO and MATH-500 eval."""

from __future__ import annotations


FORMAT_COEF = 0.1


def build_prompt(question: str, renderer) -> list[int]:
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv

    conversation = [
        *MathEnv.standard_fewshot_prefix(),
        {"role": "user", "content": question + MathEnv.question_suffix()},
    ]
    return renderer.build_generation_prompt(conversation).to_ints()


def stopped_cleanly(result: dict, max_tokens: int | None) -> bool:
    finish_reason = result.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason != "length"
    if max_tokens is None:
        return True
    return len(result.get("token_ids") or []) < max_tokens


def score_response(
    response: str,
    answer: str,
    *,
    result: dict,
    max_tokens: int | None,
    format_coef: float = FORMAT_COEF,
) -> tuple[float, dict[str, float]]:
    from tinker_cookbook.recipes.math_rl.math_env import safe_grade
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    well_formed = stopped_cleanly(result, max_tokens)
    try:
        given = extract_boxed(response)
        format_ok = True
    except ValueError:
        given = None
        format_ok = False

    correct_format = float(well_formed and format_ok)
    correct_answer = float(safe_grade(given, answer)) if given is not None else 0.0
    reward = format_coef * (correct_format - 1.0) + correct_answer
    return reward, {"format": correct_format, "correct": correct_answer}
