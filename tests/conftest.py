# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_tui_cache(tmp_path, monkeypatch):
    """Keep every test's local caches, the TUI's and the client's training-metrics log, out of ~/.cache."""
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-cache"))
