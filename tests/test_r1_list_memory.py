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

"""Listing a large job list again and again (the TUI does every minute) keeps memory flat.

Decoding a multi-megabyte body with non-ASCII text in one call grew the process
by the size of the decoded list on every call on macOS; the list calls decode
large bodies in pieces instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import requests

import cortex_training
from cortex_training import client as client_module


def _jobs(n: int) -> list:
    """Job rows like the service's, with the non-ASCII text real comments carry."""
    return [{"job_id": f"job-{i:05d}", "status": "CANCELLED", "created_at": "2026-10-01T00:00:00Z",
             "comment": f"sft · Qwen3-8B · run {i} — ünïcode 训练 🚀 \\ \"quoted\"  ",
             "sub_jobs": [{"sub_job_id": f"job-{i:05d}:training:0", "job_type": "TRAINING",
                           "training_config": {"optimizer": {"name": "adamw", "lr": 1e-5, "betas": [0.9, 0.999]},
                                               "n_gpus": 8.0, "max_seq_len": 8192.0}}]} for i in range(n)]


def _response(body: bytes) -> requests.Response:
    resp = requests.Response()
    resp.status_code = 200
    resp._content = body
    resp.headers["Content-Type"] = "application/json"
    return resp


def test_a_large_body_decodes_as_resp_json_does():
    body = json.dumps({"jobs": _jobs(4000)}, ensure_ascii=False).encode("utf-8")
    assert len(body) > client_module._PIECEWISE_DECODE_BYTES  # decoded in pieces
    # Pieces end inside multi-byte characters: the incremental decoder carries them over.
    assert client_module._json_body(_response(body)) == _response(body).json()
    small = json.dumps({"jobs": _jobs(3)}, ensure_ascii=False).encode("utf-8")
    assert client_module._json_body(_response(small)) == _response(small).json()


def test_a_large_body_that_is_not_json_raises_as_resp_json_does():
    body = b"<html>" + b"x" * (2 << 20)
    with pytest.raises(requests.exceptions.JSONDecodeError):
        client_module._json_body(_response(body))
    with pytest.raises(requests.exceptions.JSONDecodeError):
        client_module._json_body(_response(b""))


_GROWTH_PROBE = textwrap.dedent(
    """
    import json, resource, sys
    import requests
    from requests.adapters import BaseAdapter
    from cortex_training import CortexTrainingClient

    body = json.dumps({"jobs": json.loads(sys.stdin.read())}, ensure_ascii=False).encode("utf-8")

    class Service(BaseAdapter):
        def send(self, request, **kwargs):
            resp = requests.Response()
            resp.status_code, resp._content = 200, body
            resp.headers["Content-Type"] = "application/json"
            resp.request, resp.url = request, request.url
            return resp

        def close(self):
            pass

    client = CortexTrainingClient(base_url="http://test.local", database="CORTEX_TRAINING_DB", schema="PUBLIC",
                                  max_retries=0)
    client._session.mount("http://", Service())
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB elsewhere
    for _ in range(2):
        assert len(client.list_jobs()) == 6000
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    for _ in range(12):
        assert len(client.list_jobs()) == 6000
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    print(after - before)
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="the probe reads ru_maxrss")
def test_listing_a_large_job_list_again_and_again_does_not_grow_the_process():
    # A fresh interpreter: its peak RSS isn't masked by what earlier tests used.
    src = str(Path(cortex_training.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    rows = json.dumps(_jobs(6000))
    out = subprocess.run([sys.executable, "-c", _GROWTH_PROBE], input=rows, capture_output=True, text=True,
                         env=env, timeout=300, check=True)
    grown_mb = int(out.stdout.split()[-1]) / 1e6
    # The body is about 3 MB and decodes to about 12 MB: twelve more lists grew the
    # peak by well over 100 MB when every decode stayed allocated.
    assert grown_mb < 40, f"the process grew {grown_mb:.0f} MB over twelve lists"
