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

"""What the TUI draws but didn't write never acts on the terminal (no ``textual`` needed).

``printable`` (job records), ``safe_text`` (log lines) and ``one_line`` (one
row) share one rule; only how a log line shows carriage returns and tabs differs.
"""

import json

import pytest

from cortex_training.tui.format import display_line
from cortex_training.tui.format import one_line
from cortex_training.tui.format import printable
from cortex_training.tui.format import safe_text
from cortex_training.tui.views import printable_data

ESC, BEL, ST = "\x1b", "\x07", "\x1b\\"
FFFD = "\N{REPLACEMENT CHARACTER}"
MARK = "\N{COMBINING ACUTE ACCENT}"
CLEANERS = [printable, safe_text, one_line]

# Each escape sequence goes whole; the text around it stays.
ESCAPES = [
    (f"red {ESC}[31mtext{ESC}[0m end", "red text end"),  # CSI: colours
    (f"{ESC}[2J{ESC}[Hcleared", "cleared"),  # CSI: clear the screen, move the cursor
    (f"x{ESC}]52;c;ZXZpbA=={BEL}y", "xy"),  # OSC 52 (write the clipboard), ended by BEL
    (f"{ESC}]0;title{ST}ok", "ok"),  # OSC 0 (retitle the window), ended by ST
    (f"{ESC}Pq#0;2;0;0;0{ST}done", "done"),  # DCS (sixel graphics)
    (f"a{ESC}cb", "ab"),  # RIS: reset the terminal
    (f"a{ESC}(Bb", "ab"),  # a character set designation
    (f"cut {ESC}[", "cut "),  # cut short at the end
]


@pytest.mark.parametrize("clean", CLEANERS)
@pytest.mark.parametrize("text, shown", ESCAPES)
def test_escape_sequences_go_whole(clean, text, shown):
    assert clean(text) == shown


def test_an_unterminated_osc_ends_with_its_line():
    text = f"{ESC}]0;title\nnext line"
    assert printable(text) == safe_text(text) == "\nnext line"
    assert one_line(text) == " / next line"  # not "": the escape goes before the line breaks do


def test_an_esc_that_starts_no_sequence_shows_as_the_replacement_character():
    assert printable(f"a{ESC}") == f"a{FFFD}"
    assert printable(f"{ESC}{ESC}[1mz") == f"{FFFD}z"
    assert printable("\x9b31m") == f"{FFFD}31m"  # an 8-bit CSI: its C1 control is replaced, so nothing acts


@pytest.mark.parametrize("char", ["\x00", BEL, "\x7f", "\x85", "\N{RIGHT-TO-LEFT OVERRIDE}",
                                  "\N{LEFT-TO-RIGHT ISOLATE}", "\N{ZERO WIDTH SPACE}", "\N{WORD JOINER}",
                                  "\N{ZERO WIDTH NO-BREAK SPACE}"])
def test_controls_and_bidi_and_invisible_characters_show_as_the_replacement_character(char):
    for clean in CLEANERS:
        assert clean(f"a{char}b") == f"a{FFFD}b"


def test_nothing_a_terminal_acts_on_survives_any_cleaner():
    text = f"{ESC}[31m{ESC}]52;c;eA=={BEL}{ESC}\x9b{BEL}\r\N{RIGHT-TO-LEFT OVERRIDE}ok\t" * 3
    for clean in (*CLEANERS, display_line):
        shown = clean(text)
        assert not [ch for ch in shown if (ch < " " and ch != "\n") or "\x7f" <= ch <= "\x9f"
                    or ch == "\N{RIGHT-TO-LEFT OVERRIDE}"], (clean.__name__, shown)


def test_a_json_log_record_cannot_smuggle_an_escape():
    line = json.dumps({"level": "INFO", "msg": f"{ESC}]0;title{BEL}{ESC}[31mred{ESC}[0m"})  # "\u001b" in the JSON
    assert ESC in json.loads(line)["msg"]
    assert display_line(line) == "INFO  red"


def test_a_log_line_shows_carriage_returns_and_tabs_as_a_terminal_would():
    assert safe_text("10%\r50%\r100%") == "100%"  # a progress bar: its last state
    assert printable("10%\r50%") == f"10%{FFFD}50%"  # elsewhere nothing is hidden
    assert safe_text("a\tbb\tc") == "a   bb  c"  # to the next stop of four cells
    assert printable("a\tbb\tc") == "a    bb    c"


def test_emoji_joiners_and_stacked_marks_are_tamed_by_every_cleaner():
    coder = "\N{WOMAN}\N{ZERO WIDTH JOINER}\N{PERSONAL COMPUTER}"
    for clean in CLEANERS:
        assert clean(coder) == "\N{WOMAN}\N{PERSONAL COMPUTER}"  # the width counted, on every terminal
        assert clean("z" + MARK * 12) == "z" + MARK * 4
    ssa = "\N{DEVANAGARI LETTER KA}\N{DEVANAGARI SIGN VIRAMA}\N{ZERO WIDTH JOINER}\N{DEVANAGARI LETTER SSA}"
    assert printable(ssa) == ssa  # a joiner a script uses stays


def test_job_records_are_cleaned_dict_keys_too():
    record = {f"comment{ESC}[7m": [f"name{ESC}]0;x{BEL}", {"model": "z" + MARK * 9}], "n_gpus": 8, "lora": None}
    assert printable_data(record) == {"comment": ["name", {"model": "z" + MARK * 4}], "n_gpus": 8, "lora": None}
