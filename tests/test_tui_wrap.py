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

"""Wrapping and cutting wide (CJK, emoji) text to terminal cells (no ``textual`` needed)."""

import pytest
from rich.cells import cell_len

from cortex_training.tui.format import cut_cells
from cortex_training.tui.format import wrap_log_line
from cortex_training.tui.views import fit_cells
from cortex_training.tui.views import one_line
from cortex_training.tui.views import wrap_cells

CJK = "模型训练任务已经完成"  # two cells a character
EMOJI = "\N{GRINNING FACE}" * 5  # two cells each
MARKED = "e\N{COMBINING ACUTE ACCENT}" * 6  # one cell a pair


@pytest.mark.parametrize("width", range(1, 10))
@pytest.mark.parametrize("text", [CJK, EMOJI, f"ab{CJK}cd"])
def test_rows_hold_what_fits_and_lose_nothing(text, width):
    rows = wrap_log_line(text, width)
    assert "".join(rows) == text
    assert all(cell_len(row) <= max(width, 2) for row in rows)  # in a one-cell pane a wide character still shows
    assert all(cell_len(row) + 2 > width for row in rows[:-1])  # a row ends early only before a wide character


def test_a_wide_character_is_never_split_and_marks_stay_on_theirs():
    assert wrap_log_line("模型训练任务", 5) == ["模型", "训练", "任务"]
    assert wrap_log_line(EMOJI, 5) == ["\N{GRINNING FACE}" * 2] * 2 + ["\N{GRINNING FACE}"]
    assert wrap_log_line(MARKED, 4) == ["e\N{COMBINING ACUTE ACCENT}" * 4, "e\N{COMBINING ACUTE ACCENT}" * 2]


def test_breaks_fall_between_words_and_a_wide_prefix_counts_its_cells():
    assert wrap_log_line("hello 世界 wide 文字", 8) == ["hello", "世界", "wide", "文字"]
    rows = wrap_log_line("模型训练任务", 12, first="训练 │ ", rest="     │ ")  # both prefixes take 7 cells
    assert rows == ["训练 │ 模型", "     │ 训练", "     │ 任务"]


def test_cutting_to_cells_never_splits_a_wide_character():
    assert cut_cells("模型训练", 5) == "模型" and cut_cells("模型", 1) == ""
    assert fit_cells("模型训练", 5) == "模型…" and fit_cells("模型训练", 8) == "模型训练"
    assert one_line("模型训练任务\nsecond line", 9) == "模型训练…"  # the ellipsis also says more lines follow
    assert wrap_cells("模型 训练任务", 5) == ["模型", "训练", "任务"]
