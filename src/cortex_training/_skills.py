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

"""The agent skill that ships with the package, and installing it where agents load skills.

The skill's files are package data (``cortex_training/skills/``), so every
install has the copy that matches its CLI. ``cortex-training skills install``
copies it into an agent's skills folder and records the version it came from;
``status`` and ``login`` then say when the installed copy no longer matches.
"""

from __future__ import annotations

import re
import shutil
from importlib import resources
from pathlib import Path

SKILL_NAME = "cortex-training-setup"
# Where Claude Code loads a user's skills from; --dir picks another agent's folder.
DEFAULT_ROOT = Path("~/.claude/skills")
# Written next to the installed skill: the cortex-training version it came from.
VERSION_FILE = ".installed-version"


class SkillsError(RuntimeError):
    """The skill can't be installed there; nothing was changed."""


def _packaged() -> dict[str, bytes]:
    """The skill's files as the package ships them, by name."""
    folder = resources.files("cortex_training") / "skills" / SKILL_NAME
    return {entry.name: entry.read_bytes() for entry in folder.iterdir() if entry.is_file()}


def _skill_name(folder: Path) -> str | None:
    """The ``name:`` in ``folder``'s SKILL.md frontmatter, or None if there isn't one."""
    try:
        text = (folder / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    found = re.search(r"^name:\s*['\"]?([^'\"\s]+)", match.group(1), re.MULTILINE) if match else None
    return found.group(1) if found else None


def install(root: Path | None = None) -> tuple[Path, bool]:
    """Copy the skill to ``root/cortex-training-setup``; return the folder and whether it replaced one.

    An existing folder is replaced only if it holds this skill; anything else
    there is left alone and :class:`SkillsError` is raised.
    """
    from cortex_training import __version__

    target = (root or DEFAULT_ROOT).expanduser() / SKILL_NAME
    replaced = target.exists()
    if replaced and _skill_name(target) != SKILL_NAME:
        raise SkillsError(f"{target} exists and isn't the {SKILL_NAME} skill, so it was left alone. Pass --dir to install elsewhere.")
    files = _packaged()
    if replaced:
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name, data in files.items():
        (target / name).write_bytes(data)
    (target / VERSION_FILE).write_text(__version__ + "\n", encoding="utf-8")
    return target, replaced


def out_of_date(root: Path | None = None) -> str | None:
    """The version an installed, out-of-date copy came from; None when it's current or not installed.

    The installed files are compared with the package's; the version file only
    names the old version, so an upgrade that didn't change the skill isn't flagged.
    """
    target = (root or DEFAULT_ROOT).expanduser() / SKILL_NAME
    if _skill_name(target) != SKILL_NAME:
        return None
    try:
        installed = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file() and path.name != VERSION_FILE}
        if installed == _packaged():
            return None
        version = (target / VERSION_FILE).read_text(encoding="utf-8").strip() if (target / VERSION_FILE).is_file() else ""
    except OSError:
        return None
    return version or "an earlier version"


def update_note(root: Path | None = None) -> str | None:
    """The one-line note ``status`` and ``login`` print when the installed skill is out of date."""
    from cortex_training import __version__

    old = out_of_date(root)
    if old is None:
        return None
    came_from = f"cortex-training {old}" if old != "an earlier version" else old
    return (
        f"note: your {SKILL_NAME} agent skill is from {came_from} (you have {__version__}). "
        "Run `cortex-training skills install` to update it."
    )
