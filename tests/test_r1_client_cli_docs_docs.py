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

"""The REST reference's checkpoint examples run as written, an untyped checkpoint included."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REST_API = Path(__file__).resolve().parents[1] / "docs" / "reference" / "rest-api.md"

# What GET /{job_id}/checkpoints answers: the newest checkpoint was saved without
# a type, so its record has no checkpoint_type (section 5.7) and it is resumable.
CHECKPOINTS = [
    {"checkpoint_id": "cp_old_resumable", "created_at": "2026-10-01T00:00:00Z", "checkpoint_type": "resumable"},
    {"checkpoint_id": "cp_weights", "created_at": "2026-10-02T00:00:00Z", "checkpoint_type": "weights-only"},
    {"checkpoint_id": "cp_untyped", "created_at": "2026-10-03T00:00:00Z"},
]


class StubClient:
    def __init__(self):
        self.loaded = []
        self.created = []

    def save(self, job_id, checkpoint_type=None):
        return "request-id"

    def poll_request(self, job_id, request_id):
        return {}

    def list_checkpoints(self, job_id):
        return [dict(cp) for cp in CHECKPOINTS]

    def load(self, job_id, checkpoint_id):
        self.loaded.append(checkpoint_id)
        return "request-id"

    def create_job(self, sub_jobs):
        self.created.append(sub_jobs)
        return "sampling-job-id"


def _example(heading: str) -> str:
    text = REST_API.read_text(encoding="utf-8")
    section = text.split(f"### {heading}\n", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match, f"no Python example under {heading}"
    return match.group(1)


def test_the_save_and_load_example_picks_the_untyped_checkpoint_as_resumable():
    client = StubClient()
    exec(_example("13.3 Save and runtime load"), {"client": client, "job_id": "JOB_ID"})
    assert client.loaded == ["cp_untyped"]


def test_the_sampling_example_picks_the_weights_only_checkpoint():
    from cortex_training.client import SubJobConfig

    client = StubClient()
    scope = {"client": client, "training_job_id": "JOB_ID", "SubJobConfig": SubJobConfig}
    exec(_example("13.4 Start sampling from saved weights"), scope)
    assert scope["checkpoint_id"] == "cp_weights"
    (sub_jobs,) = client.created
    assert sub_jobs[0].source_checkpoint_info["checkpoint_id"] == "cp_weights"


@pytest.mark.parametrize("phrase", ["unless the save named one", 'cp["checkpoint_type"]'])
def test_the_reference_no_longer_says(phrase):
    assert phrase not in REST_API.read_text(encoding="utf-8")
