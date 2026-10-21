# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Save an SVG screenshot of the TUI, read-only, against a real account.

For reviewing how a screen looks without a terminal. The account comes from
``--connection NAME``, else the same configuration ``cortex-training tui``
reads (``--config``, ``CORTEX_TRAINING_CONFIG``, login state):

    python scripts/tui_screenshot.py list.svg
    python scripts/tui_screenshot.py logs.svg --keys "slash,type:gsm8k,enter,enter,wait:1,l,wait:4"
    python scripts/tui_screenshot.py job.svg --size 120x36 --job-id JOB_ID
    python scripts/tui_screenshot.py list.svg --connection NAME

``--keys`` is comma-separated Textual key names, plus ``type:TEXT`` (types each
character) and ``wait:SECONDS``. ``--png`` also writes a PNG beside the SVG
(needs ``rsvg-convert``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _client(connection: str | None):
    """A read-only client, resolved as ``cortex-training tui`` does: ``connection``'s
    Snowflake profile, else whatever the usual configuration gives."""
    import cortex_training._cli as cli
    from cortex_training.tui.__main__ import _build_arg_parser

    from cortex_training.client import TUI_SURFACE

    argv = ["--connection", connection] if connection else []
    args = cli._resolve_args(_build_arg_parser().parse_args(argv))
    args = cli._normalize_connection_args(args)
    client = cli.build_client(args, cli._load_cortex_training_client_class())
    client._telemetry_surface = TUI_SURFACE  # its calls are the TUI's, as in `cortex-training tui`
    client.auto_create_database = False  # a screenshot reads; it never creates a database or schema
    return client


async def _shoot(out: Path, size: tuple[int, int], keys: list[str], job_id: str | None,
                 connection: str | None) -> None:
    from cortex_training.tui.app import CortexTrainingLogTUI

    app = CortexTrainingLogTUI(_client(connection), job_id, poll_interval=1.0)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(1.5)
        for key in keys:
            if key.startswith("wait:"):
                await pilot.pause(float(key[5:]))
            elif key.startswith("type:"):
                await pilot.press(*key[5:])
            else:
                await pilot.press(key)
                await pilot.pause(0.4)
        await pilot.pause(1.5)
        app.save_screenshot(out.name, path=str(out.parent))
        app.workers.cancel_all()
        await pilot.pause(0.2)


def _to_png(svg: Path) -> Path:
    """Render ``svg`` beside itself as a PNG, cell-accurate.

    Textual writes each space as ``&#160;``, which rsvg-convert draws narrower
    than a cell, pulling indented text left. Real spaces, preserved, keep every
    column where the terminal draws it.
    """
    text = svg.read_text(encoding="utf-8").replace("&#160;", " ")
    text = re.sub(r"<svg ", '<svg xml:space="preserve" ', text, count=1)
    fixed = svg.with_suffix(".fixed.svg")
    fixed.write_text(text, encoding="utf-8")
    png = svg.with_suffix(".png")
    subprocess.run(["rsvg-convert", "-w", "1800", str(fixed), "-o", str(png)], check=True)
    fixed.unlink()
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Save an SVG screenshot of the TUI, read-only, against a real account.")
    p.add_argument("output", help="SVG file to write")
    p.add_argument("--size", default="150x42", help="COLUMNSxROWS (default 150x42)")
    p.add_argument("--keys", default="", help="comma-separated keys to press first")
    p.add_argument("--job-id", help="open this job's dashboard directly")
    p.add_argument("--connection", help="use this Snowflake connection profile (default: the usual configuration)")
    p.add_argument("--png", action="store_true", help="also write a PNG beside the SVG (needs rsvg-convert)")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    # A screenshot's caches must never land in the real cache dir.
    os.environ["CORTEX_TRAINING_TUI_CACHE_DIR"] = tempfile.mkdtemp(prefix="cortex-training-shot-")
    columns, rows = (int(n) for n in args.size.lower().split("x"))
    keys = [k for k in args.keys.split(",") if k]
    out = Path(args.output).resolve()
    asyncio.run(_shoot(out, (columns, rows), keys, args.job_id, args.connection))
    if args.png:
        if shutil.which("rsvg-convert") is None:
            raise SystemExit("--png needs rsvg-convert (brew install librsvg)")
        _to_png(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
