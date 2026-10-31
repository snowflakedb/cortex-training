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

"""How the TUI colors its screens in the terminal it starts in: decided once, as it starts.

Textual reads the terminal by itself: its color depth from ``COLORTERM`` and ``TERM`` (through
Rich), and ``NO_COLOR``. Left to it, two setups draw badly. A terminal that names itself dumb
(``TERM=dumb``, which some tools set for the commands they run) gets no color system at all, so
nothing is styled, not even the highlighted row; and a ``NO_COLOR`` that is set but empty turns
color off, where https://no-color.org says only a non-empty one does. :func:`colors` decides
instead, from ``--color`` and the environment, and :func:`note` says why the screen looks as it
does when it can't show the palette, so a gray screen is never a mystery.

This is the one module that reads the terminal's environment; :mod:`cortex_training.tui.theme`
draws the palette for what it decides. It imports neither Textual nor Rich: a color system for
Textual has to be in the environment before Textual is imported, which reads it only then.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

COLOR_CHOICES = ("auto", "always", "never")
# The TERM values Rich gives no color system, so Textual draws plain characters: no color, and
# no bold or reverse either. A terminal that runs a full-screen TUI draws 256 colors.
_DUMB_TERMS = frozenset({"dumb", "unknown"})
_DUMB_COLOR_SYSTEM = "256"


@dataclass(frozen=True)
class Colors:
    """How the TUI colors its screens (see :func:`colors`)."""

    hues: bool = True  # False: every color drawn as its gray, as Textual draws NO_COLOR
    off_because: str | None = None  # what in the environment turned the hues off ("NO_COLOR is set")
    color_system: str | None = None  # the TEXTUAL_COLOR_SYSTEM to start Textual with; None: Textual's own


def colors(choice: str = "auto", environ: Mapping[str, str] | None = None) -> Colors:
    """How to color the TUI for ``--color choice`` in ``environ`` (the process's by default).

    ``always`` and ``never`` decide by themselves, as no-color.org asks of a flag. ``auto`` draws
    hues unless ``NO_COLOR`` is set and not empty, or the terminal is dumb. A dumb terminal is drawn
    with 256 colors whichever the choice (grays, unless ``always``), so its highlighted row shows;
    a ``TEXTUAL_COLOR_SYSTEM`` of the user's own stays theirs.
    """
    environ = os.environ if environ is None else environ
    term = environ.get("TERM", "")
    dumb = term.lower() in _DUMB_TERMS and environ.get("TEXTUAL_COLOR_SYSTEM", "auto") == "auto"
    color_system = _DUMB_COLOR_SYSTEM if dumb else None
    if choice == "always":
        return Colors(True, None, color_system)
    if choice == "never":
        return Colors(False, None, color_system)
    if environ.get("NO_COLOR"):
        return Colors(False, "NO_COLOR is set", color_system)
    if dumb:
        return Colors(False, f"TERM={term}", color_system)  # as it is set: dumb, or unknown
    return Colors()


def note(colors: Colors, color_system: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """The line the TUI shows as it starts when it can't draw its palette here, else None.

    ``color_system`` is the one Textual draws with (``Console.color_system``: ``"truecolor"``,
    ``"256"``, ``"standard"``, or None). A 256-color terminal needs no line: the theme draws each
    color as the nearest of its 256 (:class:`cortex_training.tui.theme.Xterm256`).
    """
    environ = os.environ if environ is None else environ
    if colors.off_because:
        return f"Color is off because {colors.off_because}: --color always turns it on."
    if colors.hues and color_system == "standard":
        term = environ.get("TERM") or "not set"
        return (f"This terminal reports 16 colors (TERM={term}), so shades are approximate: "
                "a TERM ending in -256color shows them all.")
    return None
