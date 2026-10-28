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

"""The TUI looks the same in every terminal: each of its colors is the theme's, its text reads on
each surface in 24-bit and in 256 colors, a status differs by shape as well as color, and no
glyph it draws is also an emoji (which some terminals draw in color, two cells wide)."""

import ast
import re
from pathlib import Path

import pytest

pytest.importorskip("textual")

from rich.cells import cell_len  # noqa: E402
from rich.color import Color  # noqa: E402

from cortex_training.tui import theme  # noqa: E402

TUI = Path(__file__).resolve().parents[1] / "src" / "cortex_training" / "tui"
# A color as the palette writes one (#rrggbb, or #rrggbbaa); not "#12" or "PR #1234".
_COLOR = re.compile(r"(?<![\w&#])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6})(?![0-9a-zA-Z_])")
# The code points Unicode gives Emoji=Yes below U+1F000 (emoji-data.txt, besides ASCII digits,
# # and *), and U+FE0F, which asks for the emoji. Everything from U+1F000 up counts too.
_EMOJI_BMP = (
    (0x00A9, 0x00A9), (0x00AE, 0x00AE), (0x203C, 0x203C), (0x2049, 0x2049), (0x2122, 0x2122), (0x2139, 0x2139),
    (0x2194, 0x2199), (0x21A9, 0x21AA), (0x231A, 0x231B), (0x2328, 0x2328), (0x23CF, 0x23CF), (0x23E9, 0x23F3),
    (0x23F8, 0x23FA), (0x24C2, 0x24C2), (0x25AA, 0x25AB), (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE),
    (0x2600, 0x2604), (0x260E, 0x260E), (0x2611, 0x2611), (0x2614, 0x2615), (0x2618, 0x2618), (0x261D, 0x261D),
    (0x2620, 0x2620), (0x2622, 0x2623), (0x2626, 0x2626), (0x262A, 0x262A), (0x262E, 0x262F), (0x2638, 0x263A),
    (0x2640, 0x2640), (0x2642, 0x2642), (0x2648, 0x2653), (0x265F, 0x2660), (0x2663, 0x2663), (0x2665, 0x2666),
    (0x2668, 0x2668), (0x267B, 0x267B), (0x267E, 0x267F), (0x2692, 0x2697), (0x2699, 0x2699), (0x269B, 0x269C),
    (0x26A0, 0x26A1), (0x26A7, 0x26A7), (0x26AA, 0x26AB), (0x26B0, 0x26B1), (0x26BD, 0x26BE), (0x26C4, 0x26C5),
    (0x26C8, 0x26C8), (0x26CE, 0x26CF), (0x26D1, 0x26D1), (0x26D3, 0x26D4), (0x26E9, 0x26EA), (0x26F0, 0x26F5),
    (0x26F7, 0x26FA), (0x26FD, 0x26FD), (0x2702, 0x2702), (0x2705, 0x2705), (0x2708, 0x270D), (0x270F, 0x270F),
    (0x2712, 0x2712), (0x2714, 0x2714), (0x2716, 0x2716), (0x271D, 0x271D), (0x2721, 0x2721), (0x2728, 0x2728),
    (0x2733, 0x2734), (0x2744, 0x2744), (0x2747, 0x2747), (0x274C, 0x274C), (0x274E, 0x274E), (0x2753, 0x2755),
    (0x2757, 0x2757), (0x2763, 0x2764), (0x2795, 0x2797), (0x27A1, 0x27A1), (0x27B0, 0x27B0), (0x27BF, 0x27BF),
    (0x2934, 0x2935), (0x2B05, 0x2B07), (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x3030, 0x3030),
    (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299), (0xFE0F, 0xFE0F),
)
# The patterns that find emoji in a log's text (format.py), which draw none.
_FINDS_EMOJI = {("format.py", "_EMOJI"), ("format.py", "_JOINED_EMOJI")}


def _strings(path: Path):
    """``(line, text)`` for each string in ``path``'s code, docstrings included; the
    strings that make up an f-string one by one."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and (path.name, target.id) in _FINDS_EMOJI for target in node.targets):
            skip.update(id(inner) for inner in ast.walk(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node.lineno, node.value


def _is_emoji(ch: str) -> bool:
    point = ord(ch)
    return point >= 0x1F000 or any(low <= point <= high for low, high in _EMOJI_BMP)


def test_every_color_the_tui_draws_is_the_themes():
    # One palette: a color written elsewhere drifts from it, and isn't drawn for 256 colors with it.
    found = [f"{path.name}:{line}: {match.group()}" for path in sorted(TUI.glob("*.py")) if path.name != "theme.py"
             for line, text in _strings(path) for match in _COLOR.finditer(text)]
    assert not found, "colors belong in theme.py: " + ", ".join(found)


def test_no_glyph_the_tui_draws_is_also_an_emoji():
    found = [f"{path.name}:{line}: U+{ord(ch):04X} {ch}" for path in sorted(TUI.glob("*.py"))
             for line, text in _strings(path) for ch in sorted(set(text)) if _is_emoji(ch)]
    assert not found, "these are emoji too, in color and two cells wide in some terminals: " + ", ".join(found)


def test_the_emoji_check_knows_the_glyphs_it_replaced():
    assert all(_is_emoji(ch) for ch in "❄⏸⬇⚠✔✖▶◀\U0001F680")
    assert not any(_is_emoji(ch) for ch in "❆‖↓↑✓×−●◐◔○?▎·…─│█▏◂")


def _luminance(hex_color: str) -> float:
    def channel(value: int) -> float:
        value /= 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = Color.parse(hex_color).get_truecolor()
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def _drawn(hex_color: str, colors: str) -> str:
    """``hex_color`` as a terminal of ``colors`` draws it: itself in 24-bit, else theme.Xterm256's choice."""
    if colors == "truecolor":
        return hex_color
    red, green, blue = Color.from_ansi(theme.xterm256(*Color.parse(hex_color).get_truecolor())).get_truecolor()
    return f"#{red:02x}{green:02x}{blue:02x}"


def _contrast(fore: str, back: str, colors: str) -> float:
    lighter, darker = sorted((_luminance(_drawn(fore, colors)), _luminance(_drawn(back, colors))), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


# What each kind of text needs against what it is drawn on (WCAG's ratios: 7 and 4.5 for text,
# 3 for the least important: hints and rules).
_TEXT_ON = [
    (("TEXT",), ("BACKGROUND", "SURFACE", "PANEL"), 7.0),
    (("MUTED",), ("BACKGROUND", "SURFACE", "PANEL"), 4.5),
    (("FAINT",), ("BACKGROUND", "SURFACE", "PANEL"), 3.0),
    (("ACCENT", "OK", "INFO", "WARN", "ERR", "ENDED", "WARN_TEXT", "ERR_TEXT"), ("BACKGROUND", "SURFACE"), 4.5),
    (("ERR_TEXT", "TEXT"), ("ERR_SURFACE", "MARK_SURFACE"), 4.5),  # a traceback's rows, the line a jump landed on
    (("BACKGROUND",), ("ACCENT", "OK", "INFO", "WARN", "ERR", "ENDED"), 4.5),  # a status pill's word
    (("ACCENT", "TEXT"), ("HIGHLIGHT",), 4.5),  # a highlighted row: the accent on its band, and a cell's text
]


@pytest.mark.parametrize("colors", ["truecolor", "256"])
def test_the_palette_s_text_reads_on_what_it_is_drawn_on(colors):
    weak = [f"{fore} on {back}: {_contrast(getattr(theme, fore), getattr(theme, back), colors):.2f} < {need}"
            for fores, backs, need in _TEXT_ON for fore in fores for back in backs
            if _contrast(getattr(theme, fore), getattr(theme, back), colors) < need]
    assert not weak, weak


def test_the_highlighted_row_is_the_pair_the_contrast_check_reads():
    variables = theme.CORTEX_THEME.variables
    assert (variables["block-cursor-background"], variables["block-cursor-foreground"]) == (theme.HIGHLIGHT,
                                                                                            theme.ACCENT)


def test_a_status_differs_in_shape_not_only_in_color():
    # Gray (NO_COLOR), every status color is about the same: the glyph tells them apart.
    looks = set(theme.STATUS_LOOKS.values())
    assert len({look.glyph for look in looks}) == len(looks)
    assert all(cell_len(look.glyph) == 1 for look in looks)
    assert cell_len(theme.BRAND_ICON) == 1


def test_charts_and_sub_jobs_keep_their_colors_apart_on_256_colors():
    hues = [_drawn(hue, "256") for hue in theme.CHART_HUES]
    assert len(set(hues)) == len(hues)
    kinds = [theme.SUB_JOB_COLORS[kind] for kind in ("training", "sampling", "log_prob")] + [theme.SUB_JOB_OTHER]
    assert len({_drawn(color, "256") for color in kinds}) == len(kinds)


def test_every_palette_color_is_a_hex_color():
    names = [name for name, value in vars(theme).items() if name.isupper() and isinstance(value, str)
             and value.startswith("#")]
    assert {"TEXT", "BACKGROUND", "BORDER", "ACCENT"} <= set(names)
    for name in names:
        assert re.fullmatch(r"#[0-9a-f]{6}", getattr(theme, name)), name
    for hue in (*theme.CHART_HUES, *theme.SUB_JOB_COLORS.values(), theme.SUB_JOB_OTHER):
        assert re.fullmatch(r"#[0-9a-f]{6}", hue), hue
