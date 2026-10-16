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

"""Client-side store for job display names.

A job's server-side name is its ``comment``: set at creation (``create_job``,
``submit --comment``, the recipes), at most 1,024 bytes, and never editable.
Local aliases name a job on this machine, override the comment in the TUI, and
can name jobs created without one. They live next to the log cache on the
client:

    <cache_root>/aliases.json                          job_id -> display name

Pure stdlib, NO ``textual`` import, so the store is unit-testable like ``format.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from cortex_training.tui.log_cache import _atomic_write_text
from cortex_training.tui.log_cache import cache_root

# A display name keeps at most this many characters (the rename prompt takes no more).
ALIAS_MAX_CHARS = 80
_MAX_ALIAS = ALIAS_MAX_CHARS


class AliasStore:
    """Persistent ``job_id -> display name`` map, shared by every TUI session.

    An alias wins over the job's server ``comment`` as its display name, on
    this machine only; the comment stays visible and searchable beside it.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        base = root if root is not None else cache_root()
        self._path = base / "aliases.json"

    def _read(self) -> dict:
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        names = doc.get("aliases") if isinstance(doc, dict) else None
        return {str(k): str(v) for k, v in names.items() if v} if isinstance(names, dict) else {}

    def all(self) -> dict:
        return self._read()

    def get(self, job_id: str) -> Optional[str]:
        return self._read().get(job_id)

    def set(self, job_id: str, name: Optional[str]) -> None:
        """Set (or, with an empty name, remove) a job's display name."""
        names = self._read()
        clean = " ".join(str(name or "").split())[:_MAX_ALIAS]
        if clean:
            names[job_id] = clean
        else:
            names.pop(job_id, None)
        _atomic_write_text(self._path, json.dumps({"version": 1, "aliases": names}, indent=1, sort_keys=True) + "\n")
