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
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.content import Content
from textual.theme import Theme

# ─── Palette ──────────────────────────────────────────────────────────────────
ACCENT = "#29b5e8"  # the brand blue: titles, the selected tab, the focused cursor
ACCENT_DIM = "#1b6f8f"
OK = "#3fb950"  # running, healthy
INFO = "#58a6ff"  # warming up (initializing)
WARN = "#d29922"  # waiting (pending, placing), warnings, unknown
ERR = "#f85149"  # failed, errors
ENDED = "#9e9e9e"  # finished without failing (terminated, cancelled, done)
MUTED = "#8b949e"  # labels, secondary text
FAINT = "#6e7681"  # rules, hints, the least important text
RULE = "#3d444d"  # hairlines drawn across a table (the job list's section rows)
TEXT = "#e6edf3"

# The title bar's mark (its click opens the command palette, as Textual's own icon does).
BRAND_ICON = "❄"

BACKGROUND = "#0d1117"
SURFACE = "#161b22"
PANEL = "#21262d"
# The Logs pane: a warning's and an error's message (WARN and ERR a step lighter, to read
# as text), a traceback's rows, and the line a jump (e, n) landed on.
WARN_TEXT = "#e3b341"
ERR_TEXT = "#ff7b72"
ERR_SURFACE = "#24161a"
MARK_SURFACE = "#1c2a3a"

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
        "block-cursor-background": "#1f6feb",
        "block-cursor-foreground": "#ffffff",
        "block-cursor-text-style": "bold",
        "block-cursor-blurred-background": "#1f6feb40",
        "block-hover-background": "#ffffff0d",
        "footer-key-foreground": ACCENT,
        "footer-description-foreground": MUTED,
        "footer-background": SURFACE,
        "input-selection-background": "#1f6feb66",
        "scrollbar": PANEL,
        "scrollbar-hover": "#30363d",
        "scrollbar-active": ACCENT_DIM,
        "scrollbar-background": BACKGROUND,
        "border": "#30363d",
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
_FAILED = StatusLook("✕", ERR, bold=True)
_CANCELLED = StatusLook("⊘", ENDED)
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
