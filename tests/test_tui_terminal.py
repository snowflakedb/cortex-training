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

"""The TUI in different terminals: how it decides to color its screens (--color, NO_COLOR, a
dumb TERM), the line it shows when it can't draw its palette, and its palette on 256 colors."""

import asyncio
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from cortex_training.tui import terminal
from cortex_training.tui.terminal import Colors

pytest.importorskip("textual")

from rich.color import Color  # noqa: E402
from rich.color import ColorSystem  # noqa: E402
from rich.color import ColorType  # noqa: E402
from rich.segment import Segment  # noqa: E402
from rich.style import Style  # noqa: E402
from textual.filter import Monochrome  # noqa: E402
from textual.geometry import Region  # noqa: E402
from textual.widgets._toast import Toast  # noqa: E402

from cortex_training.tui import __main__ as tui_main  # noqa: E402
from cortex_training.tui import theme  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402
from cortex_training.tui.theme import Xterm256  # noqa: E402
from cortex_training.tui.theme import xterm256  # noqa: E402
from tests.tui_fake import FakeClient  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_files(tmp_path, monkeypatch):
    # The job screens these open keep caches (conftest puts them in tmp_path), an error log and staging.
    import cortex_training.tui.widgets as widgets_module

    monkeypatch.setattr(widgets_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    yield
    from cortex_training.tui.cache_cleanup import _release_pending

    _release_pending()  # a closed screen's lease, still in its grace, holds a lock file open


@pytest.mark.parametrize(
    ("environ", "choice", "expected"),
    [
        ({"TERM": "xterm-256color"}, "auto", Colors()),
        # no-color.org: present and not empty. Textual's own check takes an empty one too.
        ({"TERM": "xterm-256color", "NO_COLOR": ""}, "auto", Colors()),
        ({"TERM": "xterm-256color", "NO_COLOR": "1"}, "auto", Colors(False, "NO_COLOR is set")),
        ({"TERM": "xterm-256color", "NO_COLOR": "1"}, "always", Colors()),
        ({"TERM": "xterm-256color"}, "never", Colors(False)),
        # Rich gives a dumb terminal no color system, so nothing would be styled: drawn with 256 colors.
        ({"TERM": "dumb"}, "auto", Colors(False, "TERM=dumb", "256")),
        ({"TERM": "unknown"}, "auto", Colors(False, "TERM=unknown", "256")),  # the TERM as it is set
        ({"TERM": "dumb"}, "always", Colors(True, None, "256")),
        ({"TERM": "dumb"}, "never", Colors(False, None, "256")),
        # What Cursor's agent runner sets: NO_COLOR is the reason given.
        ({"TERM": "dumb", "NO_COLOR": "1", "FORCE_COLOR": "0"}, "auto", Colors(False, "NO_COLOR is set", "256")),
        # The user's own color system stays theirs.
        ({"TERM": "dumb", "TEXTUAL_COLOR_SYSTEM": "standard"}, "auto", Colors()),
        ({"TERM": "dumb", "TEXTUAL_COLOR_SYSTEM": "auto"}, "auto", Colors(False, "TERM=dumb", "256")),
        ({}, "auto", Colors()),
    ],
)
def test_colors_follow_the_flag_then_no_color_then_a_dumb_terminal(environ, choice, expected):
    assert terminal.colors(choice, environ) == expected


def test_the_note_says_why_color_is_off_or_short():
    assert terminal.note(Colors(False, "NO_COLOR is set"), "256", {}) == (
        "Color is off because NO_COLOR is set: --color always turns it on.")
    assert "TERM=dumb" in terminal.note(Colors(False, "TERM=dumb", "256"), "256", {})
    sixteen = terminal.note(Colors(), "standard", {"TERM": "xterm"})
    assert "16 colors (TERM=xterm)" in sixteen and "-256color" in sixteen
    assert "TERM=not set" in terminal.note(Colors(), "standard", {})
    # Nothing to explain: the palette as it is, the 256-color one, or gray by --color never.
    assert terminal.note(Colors(), "truecolor", {}) is None
    assert terminal.note(Colors(), "256", {}) is None
    assert terminal.note(Colors(False), "standard", {}) is None


@pytest.mark.parametrize(
    ("rgb", "number"),
    [
        ((0, 0, 0), 16),
        ((0xFF, 0xFF, 0xFF), 231),
        ((0x5F, 0x87, 0xAF), 67),  # exactly a cube color
        ((0x80, 0x80, 0x80), 244),  # a gray: the ramp's 128 (#808080)
        ((0x0D, 0x11, 0x17), 233),  # a blue-tinted near-black: the ramp, not the cube's black
        ((0xF8, 0x51, 0x49), 203),  # saturated: the cube
        ((1, 1, 1), 16),  # nearer black than the ramp's first gray (8)
        ((6, 6, 6), 232),
    ],
)
def test_xterm256_picks_the_nearer_of_cube_and_gray_ramp(rgb, number):
    assert xterm256(*rgb) == number


def _on_256(hex_color: str) -> int:
    return xterm256(*Color.parse(hex_color).get_truecolor())


def test_the_palette_keeps_its_surfaces_and_text_tiers_apart_on_256_colors():
    # Rich made all three black (16): a card, the header and the footer lost their edges.
    surfaces = [_on_256(c) for c in (theme.BACKGROUND, theme.SURFACE, theme.PANEL, theme.RULE)]
    assert surfaces == sorted(set(surfaces)) and surfaces[0] != 16, surfaces
    # The text tiers stay apart, brightest first, as drawn (not by their numbers in the 256).
    tiers = [sum(Color.from_ansi(_on_256(c)).get_truecolor()) for c in (theme.TEXT, theme.MUTED, theme.FAINT)]
    assert tiers == sorted(set(tiers), reverse=True), tiers
    assert _on_256(theme.TEXT) == 255  # gray, not Rich's cyan 195


def test_the_palette_hues_stay_where_rich_puts_them_on_256_colors():
    for name in ("ACCENT", "ACCENT_DIM", "OK", "INFO", "WARN", "ERR", "WARN_TEXT", "ERR_TEXT"):
        hex_color = getattr(theme, name)
        assert _on_256(hex_color) == Color.parse(hex_color).downgrade(ColorSystem.EIGHT_BIT).number, name


def test_xterm256_filter_redraws_truecolor_only_and_keeps_the_rest_of_a_style():
    styled = Style(color=theme.TEXT, bgcolor=theme.SURFACE, bold=True, link="https://example.com")
    plain, ansi = Style(bold=True), Style(color="red")
    out = Xterm256().apply([Segment("a", styled), Segment("b", None), Segment("c", plain), Segment("d", ansi)], None)
    assert [segment.text for segment in out] == ["a", "b", "c", "d"]
    first = out[0].style
    assert (first.color.number, first.bgcolor.number) == (255, 234)
    assert first.bold and first.link == "https://example.com"
    assert out[1].style is None and out[2].style == plain and out[3].style.color == Color.parse("red")


def _app(monkeypatch, **environ):
    for name in ("COLORTERM", "NO_COLOR", "TEXTUAL_COLOR_SYSTEM", "TERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    return CortexTrainingLogTUI(MagicMock())


def _filters(app) -> list[str]:
    return [type(f).__name__ for f in app.get_line_filters()]


def test_the_app_draws_256_colors_through_its_filter(monkeypatch):
    app = _app(monkeypatch, TERM="xterm-256color")
    assert app.console.color_system == "256"
    assert "Xterm256" in _filters(app) and "Monochrome" not in _filters(app) and not app.no_color
    truecolor = _app(monkeypatch, TERM="xterm-256color", COLORTERM="truecolor")
    assert "Xterm256" not in _filters(truecolor)


def test_the_app_decides_no_color_itself(monkeypatch):
    # An empty NO_COLOR keeps the hues; Textual alone would have drawn gray.
    assert "Monochrome" not in _filters(_app(monkeypatch, TERM="xterm-256color", COLORTERM="truecolor", NO_COLOR=""))
    gray = _app(monkeypatch, TERM="xterm-256color", NO_COLOR="1")
    assert gray.no_color and _filters(gray).count("Monochrome") == 1
    assert _filters(gray).index("Monochrome") < _filters(gray).index("Xterm256")  # gray first, then its 256-color gray
    gray.colors = terminal.colors("always", {"NO_COLOR": "1"})
    assert not gray.no_color and "Monochrome" not in _filters(gray)
    gray.colors = terminal.colors("never", {})
    assert gray.no_color and _filters(gray).count("Monochrome") == 1
    # One filter object for the app's life: a strip keeps what each filter made of it, by the filter.
    assert gray.get_line_filters()[-1] is gray.get_line_filters()[-1]
    assert all(isinstance(f, Monochrome) or f.enabled for f in gray.get_line_filters())


async def _notes(app) -> list[str]:
    """What the first screen shows as toasts (run_test leaves them out unless asked)."""
    async with app.run_test(notifications=True) as pilot:
        for _ in range(20):
            await pilot.pause()
            if app.screen.query(Toast):
                break
        said = [str(toast._notification.message) for toast in app.screen.query(Toast)]
        app.workers.cancel_all()
        await pilot.pause()
    return said


def test_the_app_says_once_why_its_screen_is_gray(monkeypatch):
    said = asyncio.run(_notes(_app(monkeypatch, TERM="xterm-256color", NO_COLOR="1")))
    assert said.count("Color is off because NO_COLOR is set: --color always turns it on.") == 1


def test_the_app_says_nothing_about_a_terminal_that_shows_its_palette(monkeypatch):
    said = asyncio.run(_notes(_app(monkeypatch, TERM="xterm-256color")))
    assert not [note for note in said if "color" in note.lower()]


def test_run_passes_color_to_the_app_and_refuses_an_unknown_choice(monkeypatch):
    seen = []

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            seen.append(self)

        def run(self):
            pass

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training.cli.commands.build_client", lambda args, cls: MagicMock())
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.delenv("TEXTUAL_COLOR_SYSTEM", raising=False)
    assert tui_main.run(["--base-url", "http://localhost:1", "--database", "DB", "--color", "always"]) == 0
    assert seen[-1].colors == Colors(True, None, "256")
    # Textual is imported already, so its color system can't change: run() leaves the environment alone.
    assert "TEXTUAL_COLOR_SYSTEM" not in os.environ
    assert tui_main.run(["--base-url", "http://localhost:1", "--database", "DB"]) == 0
    assert seen[-1].colors == Colors(False, "NO_COLOR is set", "256")
    with pytest.raises(SystemExit):
        tui_main.run(["--base-url", "http://localhost:1", "--database", "DB", "--color", "sometimes"])


# ─── The real screens, in each kind of terminal ──────────────────────────────


def _drawn_colors(app) -> list[Color]:
    """Every color the screen's widgets draw, as the terminal gets them (after the app's filters)."""
    found = []
    for widget in app.screen.walk_children(with_self=True):
        width, height = widget.outer_size
        if not widget.display or not width or not height:
            continue
        for strip in widget.render_lines(Region(0, 0, width, height)):
            for segment in strip:
                if segment.style is not None:
                    found += [c for c in (segment.style.color, segment.style.bgcolor) if c is not None]
    return found


async def _shoot_screens(app, size) -> dict[str, list[Color]]:
    """The job list's colors, then a running job's Logs and Resources tabs'."""
    drawn = {}
    async with app.run_test(size=size) as pilot:
        for _ in range(100):
            await pilot.pause()
            if getattr(app.screen, "_job_by_item", None):
                break
        drawn["list"] = _drawn_colors(app)
        running = next(job for job in app._client.list_jobs() if job["status"] == "RUNNING")
        app.push_screen(LogScreen(app._client, running["job_id"], poll_interval=0.05))
        await pilot.pause(0.5)
        drawn["logs"] = _drawn_colors(app)
        await pilot.press("g")
        await pilot.pause(1.0)
        drawn["resources"] = _drawn_colors(app)
        app.workers.cancel_all()
        await pilot.pause()
    return drawn


@pytest.mark.parametrize("size", [(80, 24), (165, 44)])
def test_a_256_color_terminal_gets_256_colors_and_its_surfaces_apart(monkeypatch, size):
    for name in ("COLORTERM", "NO_COLOR", "TEXTUAL_COLOR_SYSTEM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.05)
    for screen, colors in asyncio.run(_shoot_screens(app, size)).items():
        assert colors, screen
        assert all(color.type is ColorType.EIGHT_BIT for color in colors), screen  # nothing left to Rich's mapping
        numbers = {color.number for color in colors}
        assert 16 not in numbers, screen  # Rich's black for the blue-tinted grays
        assert {233, 234} <= numbers, (screen, sorted(numbers))  # BACKGROUND and SURFACE, apart


def test_a_no_color_terminal_gets_grays_on_every_screen(monkeypatch):
    for name in ("TEXTUAL_COLOR_SYSTEM",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setenv("NO_COLOR", "1")
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.05)
    for screen, colors in asyncio.run(_shoot_screens(app, (120, 40))).items():
        rgb = {tuple(color.get_truecolor()) for color in colors}
        assert rgb and all(red == green == blue for red, green, blue in rgb), screen
        assert len(rgb) > 3, screen  # shades, not one gray: the hierarchy reads in brightness


def test_a_24_bit_terminal_gets_the_palette_as_it_is(monkeypatch):
    for name in ("NO_COLOR", "TEXTUAL_COLOR_SYSTEM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.05)
    for screen, colors in asyncio.run(_shoot_screens(app, (120, 40))).items():
        hexes = {color.get_truecolor().hex for color in colors if color.type is ColorType.TRUECOLOR}
        assert {theme.BACKGROUND, theme.SURFACE} <= hexes, screen
