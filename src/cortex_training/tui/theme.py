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

"""The TUI's look: one palette, one theme, and one way to draw a job status.

Every screen draws status, accents and muted text from here, so a job reads the
same in the list, its preview and its dashboard. Colors are hex values rather
than names: Textual's ``Content`` markup does not know Rich's color names (a
name it does not know is drawn in the default color), and hex reads the same in
both. The palette is tuned for dark terminals and stays legible on light ones.
On a terminal of 256 colors each is drawn as the nearest it has (:class:`Xterm256`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rich.color import Color as RichColor
from rich.color import ColorType
from rich.segment import Segment
from rich.style import Style
from textual.content import Content
from textual.filter import LineFilter
from textual.theme import Theme

# ─── Palette ──────────────────────────────────────────────────────────────────
ACCENT = "#90b5f6"  # titles, the selected tab, links, the footer's keys, a highlighted row's text
ACCENT_DIM = "#5999f8"  # the accent a step deeper: Textual's secondary, a scrollbar being dragged
OK = "#88e096"  # running, healthy
INFO = "#79cce5"  # warming up (initializing)
WARN = "#ed8d30"  # waiting (pending, placing), warnings, unknown
ERR = "#ff6c6c"  # failed, errors
MUTED = "#a2a8b5"  # labels, secondary text
ENDED = MUTED  # finished without failing (terminated, cancelled, done): as quiet as a label
FAINT = "#67738c"  # rules, hints, the least important text
BORDER = "#4a4a4a"  # the borders of cards and boxes
RULE = BORDER  # hairlines drawn across a table (the job list's section rows)
GRID = "#30363d"  # a chart's gridline, a meter's empty track: seen, never read as data
TEXT = "#e6edf3"

# The title bar's mark (its click opens the command palette, as Textual's own icon does).
# Not U+2744, which is also an emoji: some terminals draw that one in color, two cells wide.
BRAND_ICON = "❆"

BACKGROUND = "#0d1117"
SURFACE = "#161b22"
PANEL = "#21262d"
HIGHLIGHT = "#383c47"  # the band behind a highlighted row (Textual's block cursor), its text in ACCENT
# The Logs pane: a warning's and an error's message (in the level's own color, light enough
# to read as text), a traceback's rows, and the line a jump (e, n) landed on.
WARN_TEXT = WARN
ERR_TEXT = ERR
ERR_SURFACE = "#3d1a1a"
MARK_SURFACE = "#1c2a3a"
# Each chart's color, by its place in the metrics' order: GPU util % (or CPU util %)
# green, memory used % cyan, then on round the palette. Hues that read on the
# background, a step lighter than the status colors, so a chart's line never reads as
# a status; the same chart keeps its color across windows, units and refreshes.
CHART_HUES = ("#4ade80", "#22d3ee", "#60a5fa", "#a78bfa", "#fbbf24", "#fb7185", "#2dd4bf", "#f472b6", "#a3e635",
              "#fb923c")
# A sub-job's color, by its type: its name on the Resources tab.
SUB_JOB_COLORS = {"training": "#22d3ee", "sampling": "#c084fc", "log_prob": "#fbbf24", "logprob": "#fbbf24"}
SUB_JOB_OTHER = "#60a5fa"

CORTEX_THEME = Theme(
    name="cortex",
    primary=ACCENT,
    secondary=ACCENT_DIM,
    accent=ACCENT,
    warning=WARN,
    error=ERR,
    success=OK,
    foreground=TEXT,
    background=BACKGROUND,
    surface=SURFACE,
    panel=PANEL,
    dark=True,
    variables={
        "block-cursor-background": HIGHLIGHT,
        "block-cursor-foreground": ACCENT,
        "block-cursor-text-style": "bold",
        "block-cursor-blurred-background": f"{HIGHLIGHT}80",
        "block-hover-background": "#ffffff0d",
        "footer-key-foreground": ACCENT,
        "footer-description-foreground": MUTED,
        "footer-background": SURFACE,
        "input-selection-background": "#90b5f666",
        "scrollbar": PANEL,
        "scrollbar-hover": BORDER,
        "scrollbar-active": ACCENT_DIM,
        "scrollbar-background": BACKGROUND,
        "border": BORDER,
        "border-blurred": PANEL,
    },
)


# ─── Job status ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StatusLook:
    glyph: str  # one cell, drawn before the status everywhere it is listed
    color: str
    bold: bool = False

    @property
    def style(self) -> str:
        return f"bold {self.color}" if self.bold else self.color


_RUNNING = StatusLook("●", OK, bold=True)
_WARMING = StatusLook("◐", INFO)
_PLACING = StatusLook("◔", WARN)
_WAITING = StatusLook("○", WARN)
_FAILED = StatusLook("×", ERR, bold=True)
_CANCELLED = StatusLook("−", ENDED)
_ENDED = StatusLook("✓", ENDED)
_UNKNOWN = StatusLook("?", WARN)

STATUS_LOOKS: dict[str, StatusLook] = {
    "running": _RUNNING,
    "initializing": _WARMING,
    "placing": _PLACING,
    "creating": _PLACING,
    "pending": _WAITING,
    "queued": _WAITING,
    "failed": _FAILED,
    "error": _FAILED,
    "cancelled": _CANCELLED,
    "canceled": _CANCELLED,
    "terminated": _ENDED,
    "done": _ENDED,
    "succeeded": _ENDED,
    "completed": _ENDED,
    "unknown": _UNKNOWN,
}


def status_key(status) -> str:
    """``JOB_STATE_RUNNING`` / ``Running`` / ``running`` -> ``running``."""
    return str(status or "").strip().lower().removeprefix("job_state_") or "unknown"


def status_look(status) -> StatusLook:
    """How a status is drawn; a status this TUI does not know reads as unknown."""
    return STATUS_LOOKS.get(status_key(status), _UNKNOWN)


def status_style(status) -> str:
    """The style of a status word, or ``""`` for no status at all."""
    return status_look(status).style if str(status or "").strip() else ""


def status_badge(status, *, word: str | None = None) -> Content:
    """``● RUNNING``: the glyph and the upper-case word, in the status's color."""
    look = status_look(status)
    text = (word if word is not None else status_key(status)).upper()
    return Content.assemble((f"{look.glyph} ", look.color), (text, look.style))


def status_pill(status, *, word: str | None = None) -> Content:
    """`` ● RUNNING ``, reversed: the one status on a screen that should be seen first."""
    look = status_look(status)
    text = (word if word is not None else status_key(status)).upper()
    return Content.styled(f" {look.glyph} {text} ", f"bold {BACKGROUND} on {look.color}")


# ─── On a 256-color terminal ──────────────────────────────────────────────────
# xterm's 256 colors: 16-231 are a 6x6x6 cube of these levels, 232-255 grays from 8 to 238.
_CUBE_LEVELS = (0x00, 0x5F, 0x87, 0xAF, 0xD7, 0xFF)


def _cube_level(value: int) -> int:
    return 0 if value < 48 else 1 if value < 114 else (value - 35) // 40


def xterm256(red: int, green: int, blue: int) -> int:
    """The nearest of xterm's 256 colors to an RGB color: the cube's or the gray ramp's,
    whichever is nearer, as tmux chooses."""
    r, g, b = _cube_level(red), _cube_level(green), _cube_level(blue)
    cube = 16 + 36 * r + 6 * g + b
    cube_red, cube_green, cube_blue = _CUBE_LEVELS[r], _CUBE_LEVELS[g], _CUBE_LEVELS[b]
    if (cube_red, cube_green, cube_blue) == (red, green, blue):
        return cube
    average = (red + green + blue) // 3
    step = 23 if average > 238 else max(0, (average - 3) // 10)
    gray = 8 + 10 * step
    to_cube = (cube_red - red) ** 2 + (cube_green - green) ** 2 + (cube_blue - blue) ** 2
    to_gray = (gray - red) ** 2 + (gray - green) ** 2 + (gray - blue) ** 2
    return 232 + step if to_gray < to_cube else cube


@lru_cache(1024)
def _xterm256_color(color: RichColor) -> RichColor:
    if color.type is not ColorType.TRUECOLOR or color.triplet is None:
        return color
    return RichColor.from_ansi(xterm256(*color.triplet))


@lru_cache(4096)
def _xterm256_style(style: Style) -> Style:
    color, bgcolor = style.color, style.bgcolor
    if color is None and bgcolor is None:
        return style
    return style + Style.from_color(color and _xterm256_color(color), bgcolor and _xterm256_color(bgcolor))


class Xterm256(LineFilter):
    """Draws each color as the nearest of a 256-color terminal's (:func:`xterm256`).

    Rich puts a color on the gray ramp only below 15% saturation, else in the cube, whose
    levels jump from 0 to 0x5f. So the palette's blue-tinted grays (BACKGROUND, SURFACE,
    PANEL) all came out black, and a card, the header and the footer lost their edges;
    TEXT came out cyan. The nearer of the two keeps them apart, as grays, and leaves the
    palette's hues where Rich puts them.
    """

    def apply(self, segments: list[Segment], background) -> list[Segment]:
        style_of = _xterm256_style
        return [Segment(text, None if style is None else style_of(style), control)
                for text, style, control in segments]
