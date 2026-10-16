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

"""``list_jobs`` takes the same per-call bounds as the account-level pages.

The job TUI's list falls back to it for every account without the
account-level list, and a refresh must not spend the client's ten retries
and ten-minute read timeout before it says it failed.
"""

from __future__ import annotations

import json

import pytest
import requests
from requests.adapters import BaseAdapter

from cortex_training import CortexTrainingClient

PREFIX = "http://test.local/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training"


class _Service(BaseAdapter):
    """Answers every request with ``status``, recording each request's timeout."""

    def __init__(self, status: int, body: dict):
        super().__init__()
        self.status, self.body = status, body
        self.timeouts: list = []
        self.urls: list = []

    def send(self, request, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        self.urls.append(request.url)
        resp = requests.Response()
        resp.status_code = self.status
        resp._content = json.dumps(self.body).encode()
        resp.headers["Content-Type"] = "application/json"
        resp.request, resp.url = request, request.url
        return resp

    def close(self):
        pass


def _client(service: _Service) -> CortexTrainingClient:
    client = CortexTrainingClient(base_url="http://test.local", database="CORTEX_TRAINING_DB", schema="PUBLIC")
    client._session.mount("http://", service)
    return client


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)  # the retries' backoff


def test_list_jobs_is_bounded_when_asked():
    service = _Service(503, {"message": "Service Unavailable"})
    client = _client(service)
    with pytest.raises(requests.exceptions.HTTPError):
        client.list_jobs(max_retries=2, request_timeout=(10, 45))
    assert len(service.urls) == 3  # one attempt and two retries, not eleven
    assert service.timeouts == [(10.0, 45.0)] * 3


def test_list_jobs_keeps_the_clients_own_bounds_by_default():
    service = _Service(200, {"jobs": [{"job_id": "a"}]})
    client = _client(service)
    assert client.list_jobs(status="running") == [{"job_id": "a"}]
    assert service.urls == [f"{PREFIX}?status=running"]
    assert service.timeouts == [client.request_timeout]


@pytest.mark.parametrize("kwargs", [{"max_retries": -1}, {"max_retries": 1.5}, {"request_timeout": 0},
                                    {"request_timeout": (10, -1)}])
def test_list_jobs_checks_its_bounds_before_any_request(kwargs):
    service = _Service(200, {"jobs": []})
    with pytest.raises(ValueError):
        _client(service).list_jobs(**kwargs)
    assert service.urls == []
