# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Save an SVG screenshot of the TUI, read-only, against a real account.

For reviewing how a screen looks without a terminal. Name the account with
``--connection NAME`` (a Snowflake connection profile) or ``--config FILE``: the
script never takes one from ``CORTEX_TRAINING_CONNECTION``,
``CORTEX_TRAINING_CONFIG`` or login state, and says on stderr which account and
database it reads before it shoots. The screenshot shows that account's jobs
(their IDs and names), so check it before you commit or share it.

    python tools/tui_screenshot.py list.svg --connection NAME
    python tools/tui_screenshot.py logs.svg --config config.json --keys "slash,type:gsm8k,enter,enter,wait:1,l,wait:4"
    python tools/tui_screenshot.py job.svg --connection NAME --size 120x36 --job-id JOB_ID

The client is the one ``cortex-training tui`` builds for that account: the same
read limits, the same errors, its calls tagged as the TUI's. ``--keys`` is
comma-separated Textual key names, plus ``type:TEXT`` (types each character)
and ``wait:SECONDS``. ``--png`` also writes a PNG beside the SVG (needs
``rsvg-convert``).
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


def _client(parser: argparse.ArgumentParser, *, connection: str | None, config: str | None):
    """The client ``cortex-training tui --connection NAME`` (or ``--config FILE``)
    builds; ``parser`` reports what keeps it from connecting."""
    from cortex_training.tui.__main__ import _build_arg_parser, _build_read_only_client

    args = _build_arg_parser().parse_args([])
    # The account named here alone, not one the TUI's flags default to from the environment.
    args.connection, args.config = connection, config
    # Its sign-in stays off the app, so the commands a shot shows to copy never name the config file or host.
    client, _sign_in = _build_read_only_client(parser, args)
    return client


async def _shoot(out: Path, size: tuple[int, int], keys: list[str], job_id: str | None, client) -> None:
    from cortex_training.tui.app import CortexTrainingLogTUI

    app = CortexTrainingLogTUI(client, job_id, poll_interval=1.0)
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
    account = p.add_mutually_exclusive_group(required=True)
    account.add_argument("--connection", help="read the account of this Snowflake connection profile")
    account.add_argument("--config", help="read the account this Cortex Training config file gives")
    p.add_argument("--size", default="150x42", help="COLUMNSxROWS (default 150x42)")
    p.add_argument("--keys", default="", help="comma-separated keys to press first")
    p.add_argument("--job-id", help="open this job's dashboard directly")
    p.add_argument("--png", action="store_true", help="also write a PNG beside the SVG (needs rsvg-convert)")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cortex_training.tui.__main__ import _escaped

    columns, rows = (int(n) for n in args.size.lower().split("x"))
    keys = [k for k in args.keys.split(",") if k]
    out = Path(args.output).resolve()
    client = _client(p, connection=args.connection, config=args.config)
    # Said before it shoots: the screenshot shows this account's jobs.
    print(f"{p.prog}: reading {_escaped(client.base_url)}, database {_escaped(client.database)}, "
          f"schema {_escaped(client.schema)}", file=sys.stderr)
    # A screenshot's caches (job logs, GPU telemetry) never land in the real cache dir,
    # and are removed after it. A download thread still writing then can't fail the shot.
    with tempfile.TemporaryDirectory(prefix="cortex-training-shot-", ignore_cleanup_errors=True) as cache:
        os.environ["CORTEX_TRAINING_TUI_CACHE_DIR"] = cache
        asyncio.run(_shoot(out, (columns, rows), keys, args.job_id, client))
    if args.png:
        if shutil.which("rsvg-convert") is None:
            raise SystemExit("--png needs rsvg-convert (brew install librsvg)")
        _to_png(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
