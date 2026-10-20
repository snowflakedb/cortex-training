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

"""The Logs pane's display helpers: the day's rule and the compact time, the kinds a
traceback's lines take, and the search the error jumps (e, n) make."""

import pytest

from cortex_training.tui.format import compact_stamp
from cortex_training.tui.format import copied_text
from cortex_training.tui.format import day_rule
from cortex_training.tui.format import display_day
from cortex_training.tui.format import is_day_rule
from cortex_training.tui.format import wrap_lines

pytest.importorskip("textual")

from rich.text import Text  # noqa: E402

from cortex_training.tui.log_style import AFTER_EXCEPTION  # noqa: E402
from cortex_training.tui.log_style import DAY_KIND  # noqa: E402
from cortex_training.tui.log_style import ERROR_KINDS  # noqa: E402
from cortex_training.tui.log_style import IN_TRACEBACK  # noqa: E402
from cortex_training.tui.log_style import ISSUE_KINDS  # noqa: E402
from cortex_training.tui.log_style import NO_TRACEBACK  # noqa: E402
from cortex_training.tui.log_style import NOTE_KIND  # noqa: E402
from cortex_training.tui.log_style import TRACE_KIND  # noqa: E402
from cortex_training.tui.log_style import TRACEBACK_BACKGROUND  # noqa: E402
from cortex_training.tui.log_style import TRACEBACK_KIND  # noqa: E402
from cortex_training.tui.log_style import LogHighlighter  # noqa: E402
from cortex_training.tui.log_style import in_context  # noqa: E402
from cortex_training.tui.log_style import line_kind  # noqa: E402
from cortex_training.tui.log_style import next_line_of  # noqa: E402
from cortex_training.tui.log_style import place_of  # noqa: E402
from cortex_training.tui.log_style import traceback_kind  # noqa: E402


# ─── the day and the time ────────────────────────────────────────────────────
def test_a_display_line_shows_its_time_and_names_its_day():
    line = "2026-10-18 14:50:39 INFO  train: step 1"
    assert display_day(line) == "2026-10-18"
    assert compact_stamp(line) == "14:50:39 INFO  train: step 1"
    assert display_day("2026-10-18 14:50:39") == "2026-10-18"  # a stamp alone
    assert compact_stamp("2026-10-18 14:50:39") == "14:50:39"


@pytest.mark.parametrize("text", ["", "step 1 loss=0.3", "2026-10-18", "2026-10-18T14:50:39 x",
                                  "2026-10-18 14:50:39,123 x", "abcd-ef-gh ij:kl:mn x", "— reading job 7 —"])
def test_a_line_without_the_pane_s_stamp_is_left_as_it_is(text):
    assert display_day(text) is None
    assert compact_stamp(text) == text


def test_the_day_rule_spans_the_pane_and_reads_in_english():
    rule = day_rule("2026-10-18", 40)
    assert rule.startswith("── Sun 18 Oct 2026 ─") and len(rule) == 40
    assert is_day_rule(rule)
    assert is_day_rule(day_rule("2026-01-05", 0))  # not laid out yet: its words, and a little rule
    assert not is_day_rule("── Sun 18 Oct 2026 ── and then a log line")
    assert not is_day_rule("— reading job 7 —")


def test_wrap_lines_draws_the_time_alone():
    chunks, displays, cut = wrap_lines(["2026-10-18T14:50:39Z INFO x"], 80, max_lines=10, max_rows=10)
    assert not cut
    assert display_day(displays["2026-10-18T14:50:39Z INFO x"])  # the display text keeps the date
    assert display_day(chunks[0][1][0]) is None and chunks[0][1][0].endswith(" INFO x")


# ─── what a highlight copies ─────────────────────────────────────────────────
# Two lines as the pane draws them: the first with its day's rule, the second wrapped.
WHOLE = ["2026-10-17T05:00:00Z INFO step 1", "2026-10-17T05:00:01Z INFO step 2 loss=0.5 lr=1e-5"]
SHOWN = ["05:00:00 INFO step 1", "05:00:01 INFO step 2 loss=0.5 lr=1e-5"]
ROWS = [day_rule("2026-10-17", 24), "05:00:00 INFO step 1", "05:00:01 INFO step 2", "         loss=0.5 lr=1e-5"]
LINE_ROWS = [2, 2]


def _copy(start=None, end=None):
    return copied_text(ROWS, LINE_ROWS, WHOLE.__getitem__, SHOWN.__getitem__, start, end, indent=" " * 9,
                       rules={0})


def test_a_highlight_over_whole_lines_copies_them_as_logged():
    assert _copy() == "\n".join(WHOLE)  # everything: no rule, whole stamps, no indent
    assert _copy((0, 3), (3, len(ROWS[3]))) == "\n".join(WHOLE)  # from inside the rule: still whole
    assert _copy((2, 0), (3, len(ROWS[3]))) == WHOLE[1]  # a wrapped line, both its rows
    assert _copy((0, 0), (0, 10)) == ""  # the rule alone


def test_a_highlight_over_part_of_a_line_copies_that_part_as_its_text_has_it():
    assert _copy((1, 9), (1, 13)) == "INFO"
    assert _copy((2, 14), (3, 17)) == "step 2 loss=0.5"  # across the wrap: its space back, no indent
    assert _copy((3, 0), (3, 13)) == "loss"  # from the indent
    assert _copy((1, 9), (2, 13)) == WHOLE[0][21:] + "\n" + "05:00:01 INFO"  # a line's end, then the next's start


def test_a_log_row_that_reads_like_a_day_rule_is_copied_with_its_line():
    rule = day_rule("2026-10-17", 24)  # the text of a rule, logged: not one of the pane's rules
    rows = [rule, "         more"]
    copied = copied_text(rows, [2], lambda i: rule + "\nmore", lambda i: rule + "\nmore", indent=" " * 9)
    assert copied == rule + "\nmore"


def test_the_rows_a_long_line_leaves_out_come_back_and_its_note_is_copied_as_shown():
    rows = ["a b", "         … 9 rows left out …", "         z"]

    def copy(start, end):
        return copied_text(rows, [3], lambda i: "raw", lambda i: "a b\nmore\nz", start, end, indent=" " * 9)

    assert copy((0, 2), (2, 10)) == "b\nmore\nz"  # its text, the rows left out too
    assert copy((0, 2), (1, 12)) == "b\n… 9"  # ending on the note, which isn't in its text: as shown


# ─── context: a traceback's block, and the day's rule ───────────────────────
TRACEBACK = [
    "14:50:39 INFO  train: step 3",
    "Traceback (most recent call last):",
    '  File "train.py", line 12, in <module>',
    "    main()",
    "",
    "During handling of the above exception, another exception occurred:",
    "ValueError: bad shape",
    "14:50:40 INFO  zone: tearing down",
]


def _kinds(lines):
    kinds, block = [], NO_TRACEBACK
    for text in lines:
        kind, block = traceback_kind(text, line_kind(text), block)
        kinds.append(kind)
    return kinds


def test_a_console_s_traceback_is_one_block_from_its_header_to_its_exception():
    assert _kinds(TRACEBACK) == ["INFO", TRACEBACK_KIND, TRACE_KIND, TRACE_KIND, TRACE_KIND, TRACE_KIND,
                                 TRACE_KIND, "INFO"]


def test_a_traceback_s_lines_may_carry_the_service_s_stamp():
    stamped = ["2026-10-18 14:50:39 Traceback (most recent call last):",
               '2026-10-18 14:50:39   File "x.py", line 1, in f',
               "2026-10-18 14:50:39 KeyError: 'a'",
               "2026-10-18 14:50:39 next line"]
    assert _kinds(stamped) == [TRACEBACK_KIND, TRACE_KIND, TRACE_KIND, None]


def test_a_chained_traceback_and_one_under_an_error_are_one_stop():
    chained = ["Traceback (most recent call last):", '  File "a.py", line 1, in f', "OSError: gone", "",
               "The above exception was the direct cause of the following exception:", "",
               "Traceback (most recent call last):", '  File "b.py", line 2, in g', "RuntimeError: failed",
               "14:50:40 INFO  zone: tearing down"]
    kinds = _kinds(chained)
    assert kinds == [TRACEBACK_KIND, *[TRACE_KIND] * 8, "INFO"]
    assert sum(kind in ERROR_KINDS for kind in kinds) == 1
    under = ["14:50:39 ERROR train: step failed", "Traceback (most recent call last):", "KeyError: 'x'"]
    assert _kinds(under) == ["ERROR", TRACE_KIND, TRACE_KIND]  # the error's line is the stop
    indented = ["    Traceback (most recent call last):", "    OSError: gone", "",
                "    The above exception was the direct cause of the following exception:",
                "    Traceback (most recent call last):", "    RuntimeError: failed"]
    assert _kinds(indented) == [TRACEBACK_KIND, *[TRACE_KIND] * 5]  # as an execution log's record shows one


def test_two_tracebacks_back_to_back_are_two_stops():
    lines = ["Traceback (most recent call last):", '  File "a.py", line 1, in f', "ValueError: one",
             "Traceback (most recent call last):", '  File "b.py", line 2, in g', "KeyError: two"]
    kinds = _kinds(lines)
    assert kinds == [TRACEBACK_KIND, TRACE_KIND, TRACE_KIND, TRACEBACK_KIND, TRACE_KIND, TRACE_KIND]
    assert sum(kind in ERROR_KINDS for kind in kinds) == 2
    # A blank line between them doesn't chain them either: only the sentence does.
    assert _kinds([*lines[:3], "", *lines[3:]])[4] == TRACEBACK_KIND


def test_a_line_with_a_level_or_a_block_too_long_ends_a_traceback():
    cut_short = ["Traceback (most recent call last):", '  File "a.py", line 1, in f',
                 "14:50:41 INFO  train: step 4", '    {"a": 1}']
    assert _kinds(cut_short) == [TRACEBACK_KIND, TRACE_KIND, "INFO", None]  # the JSON after it isn't red
    long = ["Traceback (most recent call last):", *(['    {"key": "value"},'] * 400)]
    kinds = _kinds(long)
    assert kinds[:300] == [TRACEBACK_KIND, *[TRACE_KIND] * 299] and set(kinds[300:]) == {None}


def test_lines_outside_a_traceback_keep_their_level_and_notes_stay_notes():
    assert traceback_kind("  indented, but no traceback before it", None, NO_TRACEBACK) == (None, NO_TRACEBACK)
    assert traceback_kind("ValueError: on its own", None, NO_TRACEBACK) == (None, NO_TRACEBACK)
    assert traceback_kind("— live —", NOTE_KIND, (IN_TRACEBACK, 3)) == (NOTE_KIND, (IN_TRACEBACK, 3))  # in no way


def _chunk(text, rows=1):
    return text, [text] + [f"         row {n}" for n in range(1, rows)], rows


def test_in_context_puts_each_day_s_rule_above_its_first_line():
    texts = ["2026-10-17 23:59:59 INFO  a", "2026-10-17 23:59:59 INFO  b", "2026-10-18 00:00:01 INFO  c"]
    chunks, kinds, rows, day, block = in_context([_chunk(t) for t in texts], ["INFO"] * 3, texts, 30)
    assert [count for _, _, count in chunks] == [2, 1, 2]
    assert chunks[0][1][0] == day_rule("2026-10-17", 30) and chunks[2][1][0] == day_rule("2026-10-18", 30)
    assert rows == [DAY_KIND, "INFO", "INFO", DAY_KIND, "INFO"]
    assert kinds == ["INFO"] * 3 and day == "2026-10-18" and block == NO_TRACEBACK
    # Lines written after them, on the same day: no rule, as the context says where they left off.
    more, _, more_rows, _, _ = in_context([_chunk(texts[2])], ["INFO"], [texts[2]], 30, day=day)
    assert more[0][2] == 1 and more_rows == ["INFO"]


def test_in_context_skips_the_tui_s_notes_and_carries_a_traceback_across_pages():
    first = ["2026-10-18 10:00:00 Traceback (most recent call last):"]
    chunks, kinds, rows, day, block = in_context([_chunk(first[0]), _chunk("— live —")], [None, NOTE_KIND],
                                                 [first[0], None], 40)
    assert kinds == [TRACEBACK_KIND, NOTE_KIND] and rows == [DAY_KIND, TRACEBACK_KIND, NOTE_KIND]
    assert block[0] == IN_TRACEBACK
    later = ["2026-10-18 10:00:00   File \"x.py\", line 1, in f", "2026-10-18 10:00:00 OSError: gone"]
    _, kinds, _, _, block = in_context([_chunk(t) for t in later], [None, None], later, 40, day, block)
    assert kinds == [TRACE_KIND, TRACE_KIND] and block[0] == AFTER_EXCEPTION


def test_a_record_s_own_traceback_rows_are_a_block_under_its_error():
    text = "2026-10-18 10:00:00 ERROR runtime: failed\n    Traceback (most recent call last):\n    OSError: x"
    pieces = ["10:00:00 ERROR runtime: failed", "             Traceback (most recent call last):",
              "             OSError: x"]
    chunks, kinds, rows, _, _ = in_context([(text, pieces, 3)], ["ERROR"], [text], 60)
    assert kinds == ["ERROR"]
    assert rows == [DAY_KIND, "ERROR", TRACE_KIND, TRACE_KIND]


# ─── jumps ───────────────────────────────────────────────────────────────────
KINDS = ["INFO", "ERROR", None, "WARNING", TRACEBACK_KIND, TRACE_KIND, TRACE_KIND, "INFO", NOTE_KIND]


def test_next_line_of_goes_either_way_and_round_past_the_ends():
    assert next_line_of(KINDS, None, 1, ERROR_KINDS) == 1  # from the first line
    assert next_line_of(KINDS, None, -1, ERROR_KINDS) == 4  # from the last: the newest, a traceback's first line
    assert next_line_of(KINDS, 1, 1, ERROR_KINDS) == 4  # never a traceback's later lines
    assert next_line_of(KINDS, 4, 1, ERROR_KINDS) == 1  # round past the end
    assert next_line_of(KINDS, 1, -1, ERROR_KINDS) == 4  # and past the start
    assert next_line_of(KINDS, 1, 1, ISSUE_KINDS) == 3  # warnings too
    assert next_line_of(KINDS, 1, 1, frozenset({"ERROR"})) == 1  # the only one: back to it
    assert next_line_of(["INFO", None], None, 1, ERROR_KINDS) is None
    assert next_line_of([], None, 1, ERROR_KINDS) is None


def test_place_of_counts_the_stops():
    assert place_of(KINDS, 1, ERROR_KINDS) == (1, 2)
    assert place_of(KINDS, 4, ERROR_KINDS) == (2, 2)
    assert place_of(KINDS, 3, ISSUE_KINDS) == (2, 3)


# ─── how rows are drawn ──────────────────────────────────────────────────────
def _styles(text, kind, query=""):
    highlighter = LogHighlighter()
    highlighter.row_kind = kind
    highlighter.query = query
    row = Text(text)
    highlighter.highlight(row)
    return [(row.plain[span.start:span.end], str(span.style)) for span in row.spans]


def test_rows_are_styled_from_the_palette():
    from cortex_training.tui import theme

    styles = dict(_styles("14:50:39 ERROR train: it broke", "ERROR"))
    assert styles["14:50:39"] == theme.FAINT
    assert styles["ERROR"] == f"bold {theme.ERR}"
    assert styles["train:"] == theme.MUTED


def test_a_traceback_row_marks_its_file_and_its_exception():
    from cortex_training.tui import theme

    assert ('"train.py"', theme.INFO) in _styles('  File "train.py", line 12, in <module>', TRACE_KIND)
    assert ("ValueError: bad shape", f"bold {theme.ERR}") in _styles("ValueError: bad shape", TRACE_KIND)
    assert TRACEBACK_BACKGROUND.startswith("#")


def test_a_log_row_that_reads_like_a_note_is_drawn_as_a_log_row():
    from cortex_training.tui.log_style import NOTE
    from cortex_training.tui.log_style import UNKNOWN_KIND

    assert ("— the job ended —", NOTE) not in _styles("— the job ended —", None)  # the pane says: a log line
    assert ("— the job ended —", NOTE) in _styles("— the job ended —", NOTE_KIND)
    assert ("— the job ended —", NOTE) in _styles("— the job ended —", UNKNOWN_KIND)  # a plain Log's guess


def test_the_day_rule_and_the_filter_s_matches():
    assert any(style.startswith("bold") and text == "Sun 18 Oct 2026"
               for text, style in _styles(day_rule("2026-10-18", 40), DAY_KIND))
    assert any(text == "step" and " on " in style for text, style in _styles("14:50:39 INFO  step 1", "INFO", "STEP"))
