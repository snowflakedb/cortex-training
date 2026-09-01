# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Download an exported Cortex Training checkpoint."""

from __future__ import annotations

import logging
import shlex
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chz
import requests
from recipes.utils import make_client

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    config: str
    job_id: str
    checkpoint_id: str
    output_dir: str = "./cpt-checkpoint"


def checkpoint_download_command(
    config_path: str,
    job_id: str,
    checkpoint_id: str,
    *,
    output_dir: str = "./cpt-checkpoint",
) -> str:
    args = [
        "python",
        "-m",
        "recipes.cpt.download_checkpoint",
        f"config={config_path}",
        f"job_id={job_id}",
        f"checkpoint_id={checkpoint_id}",
        f"output_dir={output_dir}",
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def _target_path(output_dir: Path, filename: str) -> Path:
    relative = Path(filename)
    root = output_dir.resolve()
    target = (root / relative).resolve()
    return target


def _download_file(item: Mapping[str, Any], output_dir: Path) -> None:
    filename = str(item.get("filename") or "")
    url = str(item.get("url") or "")
    if not filename or not url:
        raise ValueError("checkpoint export file needs filename and url")

    target = _target_path(output_dir, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", filename)

    temporary: Path | None = None
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                written = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)

        assert temporary is not None
        expected = item.get("size_bytes")
        if expected is not None and written != int(expected):
            raise IOError(
                f"downloaded {written} bytes for {filename}, expected {expected}"
            )
        temporary.replace(target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(config: Config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = make_client(config.config).export_checkpoint(
        config.job_id,
        config.checkpoint_id,
    )
    files = manifest.get("files")

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in files:
        _download_file(item, output_dir)

    logger.info("Downloaded %d checkpoint files to %s", len(files), output_dir)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
