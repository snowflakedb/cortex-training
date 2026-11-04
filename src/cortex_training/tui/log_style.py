# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Draw-time colours for plain-text log rows (levels, tracebacks, day rules, notes,
matches), and the kinds of line the pane jumps between."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rich.highlighter import Highlighter
from rich.text import Text

from cortex_training.tui.format import day_rule
from cortex_training.tui.format import display_day
from cortex_training.tui.theme import BACKGROUND
from cortex_training.tui.theme import ERR
from cortex_training.tui.theme import ERR_SURFACE
from cortex_training.tui.theme import ERR_TEXT
from cortex_training.tui.theme import FAINT
from cortex_training.tui.theme import INFO
from cortex_training.tui.theme import MARK_SURFACE
from cortex_training.tui.theme import MUTED
from cortex_training.tui.theme import OK
from cortex_training.tui.theme import SUB_JOB_COLORS
from cortex_training.tui.theme import SUB_JOB_OTHER
from cortex_training.tui.theme import TEXT
from cortex_training.tui.theme import WARN
from cortex_training.tui.theme import WARN_TEXT

NOTE = f"italic {MUTED}"
TIME = FAINT
LOGGER = MUTED
KEY = FAINT
MATCH = f"bold {BACKGROUND} on {WARN}"
# Drawn across the pane's width: a traceback's rows, so the block reads as one, and
# the line a jump (e, n) landed on.
TRACEBACK_BACKGROUND = ERR_SURFACE
MARK_BACKGROUND = MARK_SURFACE
_LEVELS = {
    "TRACE": (TIME, None), "DEBUG": (TIME, None), "INFO": (OK, None), "NOTICE": (INFO, None),
    "WARN": (f"bold {WARN}", WARN_TEXT), "WARNING": (f"bold {WARN}", WARN_TEXT),
    "ERROR": (f"bold {ERR}", ERR_TEXT), "CRITICAL": (f"bold {ERR}", ERR_TEXT), "FATAL": (f"bold {ERR}", ERR_TEXT),
}

# A line's time: the whole stamp in the display text the kinds are worked out from,
# the time alone on the rows the pane draws (see format.compact_stamp).
_HEAD = re.compile(r"((?:\d{4}-\d{2}-\d{2}[ T])?\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)"
                   r"(?:\s+(TRACE|DEBUG|INFO|NOTICE|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\b(?:\s+([^\s:]{1,60}:)(?=\s))?)?")
_PAIR = re.compile(r"(?<![\w./=])([A-Za-z_][\w./\-]{0,40})=(?=[^\s=])")
_PROBLEM = re.compile(r"\bTraceback \(most recent call last\):|\b[A-Z]\w*(?:Error|Exception)\b")
_NOTE = re.compile(r"— .+ —")
_DAY = re.compile(r"(── )(.+?)( ─*)")
_TRACEBACK = "Traceback (most recent call last):"
# A traceback's frame ('File "x.py", line 3, in f'), its chained-exception sentences,
# and the exception line that ends it ("ValueError: bad", "KeyboardInterrupt").
_FRAME = re.compile(r'(\s*File )("[^"]*")(, line \d+)?(, in .*)?')
_CHAIN = ("During handling of the above exception", "The above exception was the direct cause")
_EXCEPTION = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Failure)\b(?::.*)?"
                        r"|[A-Za-z_][\w.]*: .*")

NOTE_KIND = "note"  # a row of one of the TUI's own notes (see LogHighlighter.row_kind)
# The row's kind isn't known (a plain Log, which doesn't keep them): a row that reads
# like one of the TUI's notes is drawn as one. The Logs pane always says.
UNKNOWN_KIND = "unknown"
DAY_KIND = "day"  # the rule above the first line of a day (see format.day_rule)
TRACEBACK_KIND = "traceback"  # the first line of a traceback ("Traceback (most recent call last):")
TRACE_KIND = "traceback-body"  # a traceback's later lines: frames, code, chained causes, its exception
# What e / E stop at, and n / N: a traceback is one stop, at its first line.
ERROR_KINDS = frozenset({"ERROR", "CRITICAL", "FATAL", TRACEBACK_KIND})
ISSUE_KINDS = ERROR_KINDS | {"WARN", "WARNING"}


def line_kind(text: str, *, note: bool = False) -> str | None:
    """What styles every row of a log line as its first: ``note`` for the TUI's own
    notes, else the level of a line that starts with a time and a level (``ERROR``)."""
    if note:
        return NOTE_KIND
    head = _HEAD.match(text)
    return head.group(2) if head else None


def _body(text: str) -> str:
    """A display line after its time, if it has one."""
    head = _HEAD.match(text)
    return text[head.end(1) + 1:] if head and not head.group(2) else text


# Where a traceback stands after a line (see traceback_kind), as (phase, its lines so
# far): no traceback, or the line was an error's (a traceback right under it is that
# error's), or inside one, or just after its exception (blank lines and a chained
# exception's sentence may follow), or after that sentence (the chained traceback follows).
NO_PHASE, AFTER_ERROR, IN_TRACEBACK, AFTER_EXCEPTION, AFTER_CHAIN = "", "error", "in", "after", "chain"
NO_TRACEBACK = (NO_PHASE, 0)  # the state before any line
# A block this long has lost its end (a log cut short, or text that only looks like
# one): the lines after it are drawn as they are.
_MAX_TRACEBACK_LINES = 300
_ERROR_LEVELS = frozenset({"ERROR", "CRITICAL", "FATAL"})


def traceback_kind(text: str, kind: str | None, state: tuple = NO_TRACEBACK) -> tuple[str | None, tuple]:
    """A display line's kind with the lines before it in mind, and where a traceback
    stands after it: ``(kind, state)``. A console prints a traceback a line at a time,
    each line alone saying little; together they are one block, from its
    ``Traceback (most recent call last):`` to the exception line that ends it, with
    the exceptions chained to it ("The above exception was the direct cause …").
    The block is one stop for e: its first line, unless it is chained to the one
    before or sits right under an error's line, which is then the stop. Only the
    chained-exception sentence chains the next traceback to the block: another one
    straight after an exception is a block, and a stop, of its own.

    A line with a level of its own (``14:50:39 INFO …``) ends a block, as does its
    _MAX_TRACEBACK_LINES-th line: a traceback cut short, then indented text (a JSON
    dump), would otherwise stay red."""
    if kind == NOTE_KIND:
        return kind, state  # the TUI's own: never part of the log's traceback, nor in its way
    phase, lines = state
    if kind is None and lines < _MAX_TRACEBACK_LINES:
        body = _body(text)
        stripped = body.lstrip()
        if stripped.startswith(_TRACEBACK):
            # Chained to the block (the sentence said so), or the traceback of the error's line
            # above; else a block of its own.
            first = phase not in (AFTER_ERROR, AFTER_CHAIN)
            return (TRACEBACK_KIND if first else TRACE_KIND), (IN_TRACEBACK, 1 if first else lines + 1)
        if phase in (IN_TRACEBACK, AFTER_EXCEPTION, AFTER_CHAIN) and stripped.startswith(_CHAIN):
            # (Indented too, as an execution log's record shows its chain.)
            return TRACE_KIND, (AFTER_CHAIN, lines + 1)
        if phase in (IN_TRACEBACK, AFTER_CHAIN):
            if not body.strip() or body[0] in " \t":
                return TRACE_KIND, (phase, lines + 1)
            if _EXCEPTION.fullmatch(body):  # its exception: the block ends here, or a chained one follows
                return TRACE_KIND, (AFTER_EXCEPTION, lines + 1)
        elif phase == AFTER_EXCEPTION and not body.strip():
            return TRACE_KIND, (AFTER_EXCEPTION, lines + 1)
    return kind, (AFTER_ERROR, 0) if kind in _ERROR_LEVELS else NO_TRACEBACK


def in_context(chunks: list, kinds: list, texts: list, width: int, day: str | None = None,
               in_block: tuple = NO_TRACEBACK) -> tuple[list, list, list, str | None, tuple]:
    """What a line's neighbours say about it, for lines written to the pane in order.

    ``chunks`` are ``(line, its rows, row count)``, ``kinds`` each line's own kind (see
    :func:`line_kind`) and ``texts`` its display text (None for the TUI's own notes,
    which take no part); ``day`` and ``in_block`` (a traceback's state, see
    :func:`traceback_kind`) are where the lines before them left off. A traceback's
    lines are one block, and so are the rows of one line's own traceback (a JSON
    record's exception); the first line of each day gets the day's rule (see
    ``format.day_rule``) as its first row, its rows showing the time alone.

    Returns ``(chunks, kinds, row kinds, day, in_block)``. Pure, and a few string
    checks a line, so it runs off the UI thread for a whole pane and on it for a page.
    """
    out: list = []
    line_kinds: list = []
    row_kinds: list = []
    for (line, pieces, count), kind, text in zip(chunks, kinds, texts):
        if text is None:
            out.append((line, pieces, count))
            line_kinds.append(kind)
            row_kinds.extend([kind] * count)
            continue
        kind, in_block = traceback_kind(text, kind, in_block)
        rows = [kind] * count
        if count > 1 and kind not in (TRACEBACK_KIND, TRACE_KIND) and len(pieces) == count:
            at = next((n for n, piece in enumerate(pieces) if _TRACEBACK in piece), 0)
            if at:
                rows[at:] = [TRACE_KIND] * (count - at)
        seen = display_day(text)
        if seen and seen != day:
            day = seen
            pieces, count = [day_rule(seen, width), *pieces], count + 1
            rows.insert(0, DAY_KIND)
        out.append((line, pieces, count))
        line_kinds.append(kind)
        row_kinds.extend(rows)
    return out, line_kinds, row_kinds, day, in_block


def next_line_of(kinds: Sequence[str | None], start: int | None, step: int,
                 wanted: frozenset) -> int | None:
    """The index of the next line (``step`` 1) or the one before (``step`` -1) from
    ``start`` whose kind is ``wanted``, round past either end as a search in a pager
    goes; None when there is none. ``start`` None: from the first line (or the last)."""
    count = len(kinds)
    if not count:
        return None
    if start is None:
        start = -1 if step > 0 else count
    for n in range(1, count + 1):
        index = (start + n * step) % count
        if kinds[index] in wanted:
            return index
    return None


def place_of(kinds: Sequence[str | None], index: int, wanted: frozenset) -> tuple[int, int]:
    """``(n, total)``: line ``index`` is the n-th (from 1) of the ``total`` lines whose kind is ``wanted``."""
    n = total = 0
    for at, kind in enumerate(kinds):
        if kind in wanted:
            total += 1
            if at <= index:
                n = total
    return n, total


def sub_job_color(label: str) -> str:
    """The colour of a sub-job by its label or tag (``training #0``, ``sampling:1``)."""
    kind = re.split(r"[ :#]", label.strip(), maxsplit=1)[0].lower()
    return SUB_JOB_COLORS.get(kind, SUB_JOB_OTHER)


class LogHighlighter(Highlighter):
    """Style one log row; ``query`` marks case-insensitive filter matches."""

    def __init__(self) -> None:
        self.query = ""
        self.row_kind: str | None = UNKNOWN_KIND

    def highlight(self, text: Text) -> None:
        plain = text.plain
        kind = self.row_kind
        if kind == NOTE_KIND or (kind == UNKNOWN_KIND and _NOTE.fullmatch(plain)):
            text.stylize(NOTE)
            return
        if kind == DAY_KIND:
            if day := _DAY.fullmatch(plain):
                text.stylize(FAINT)
                text.stylize(f"bold {MUTED}", day.start(2), day.end(2))
            return
        start = 0
        if head := _HEAD.match(plain):
            text.stylize(TIME, head.start(1), head.end(1))
            level = head.group(2)
            if level:
                style, tint = _LEVELS[level]
                text.stylize(style, head.start(2), head.end(2))
                if tint:  # a warning's or an error's message, in the level's colour
                    text.stylize(tint, head.end(2), len(plain))
                if head.group(3):
                    text.stylize(LOGGER, head.start(3), head.end(3))
            start = head.end()
        elif kind in _LEVELS and _LEVELS[kind][1]:
            text.stylize(_LEVELS[kind][1])  # a warning's or an error's later rows
        if kind in (TRACEBACK_KIND, TRACE_KIND):
            self._traceback_row(text, plain, start)
        else:
            for pair in _PAIR.finditer(plain, start):
                text.stylize(KEY, pair.start(), pair.end())
            for problem in _PROBLEM.finditer(plain, start):
                text.stylize(f"bold {ERR}", problem.start(), problem.end())
        query = self.query.lower()
        lowered = plain.lower()
        if query and len(lowered) == len(plain):  # (a few characters lower-case to two)
            at = lowered.find(query)
            while at >= 0:
                text.stylize(MATCH, at, at + len(query))
                at = lowered.find(query, at + len(query))

    @staticmethod
    def _traceback_row(text: Text, plain: str, start: int) -> None:
        """One row of a traceback: its header and exception in red, each frame's file
        brighter than the rest, the code it ran plain, a chained cause's sentence muted."""
        body = plain[start:]
        stripped = body.strip()
        if frame := _FRAME.match(body):
            text.stylize(MUTED, start, len(plain))
            text.stylize(INFO, start + frame.start(2), start + frame.end(2))
        elif stripped.startswith(_CHAIN):
            text.stylize(f"italic {MUTED}", start, len(plain))
        elif stripped.startswith(_TRACEBACK) or _EXCEPTION.fullmatch(stripped):
            text.stylize(f"bold {ERR}", start + len(body) - len(body.lstrip()), len(plain))
        else:
            text.stylize(TEXT, start, len(plain))  # the code a frame ran
