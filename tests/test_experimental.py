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

import pytest

from cortex_training import ExperimentalFeatureError
from cortex_training._experimental import EXPERIMENTAL_ENV
from cortex_training._experimental import experimental
from cortex_training._experimental import experimental_enabled
from cortex_training._experimental import require_experimental


class _Client:
    @experimental("shiny")
    def shiny(self, x, *, y=1):
        """Do the shiny thing."""
        return x + y

    @experimental("stream")
    def stream(self):
        yield 1


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)


def test_refused_by_default():
    with pytest.raises(ExperimentalFeatureError, match=f"{EXPERIMENTAL_ENV}=shiny") as exc:
        _Client().shiny(1)
    assert exc.value.feature == "shiny"
    assert "_Client.shiny()" in str(exc.value)


@pytest.mark.parametrize("value", ["shiny", "other, Shiny ", "all", "*"])
def test_enabled_by_env(monkeypatch, value):
    monkeypatch.setenv(EXPERIMENTAL_ENV, value)
    assert _Client().shiny(1, y=2) == 3


def test_other_feature_does_not_enable(monkeypatch):
    monkeypatch.setenv(EXPERIMENTAL_ENV, "other,shinyy")
    with pytest.raises(ExperimentalFeatureError):
        _Client().shiny(1)


def test_generator_refused_at_call_not_iteration():
    with pytest.raises(ExperimentalFeatureError):
        _Client().stream()


def test_wrapper_metadata():
    fn = _Client.shiny
    assert fn.__name__ == "shiny"
    assert fn.__cortex_training_experimental__ == "shiny"
    assert fn.__doc__.startswith("[Experimental: shiny]")
    assert "Do the shiny thing." in fn.__doc__


def test_require_experimental(monkeypatch):
    with pytest.raises(ExperimentalFeatureError, match="the `foo` field"):
        require_experimental("foo", "the `foo` field")
    monkeypatch.setenv(EXPERIMENTAL_ENV, "foo")
    require_experimental("foo", "the `foo` field")


@pytest.mark.parametrize("name", ["", "Shiny", "has-dash", "1x", "all", None])
def test_invalid_names_rejected(name):
    with pytest.raises(ValueError):
        experimental(name)
    with pytest.raises(ValueError):
        experimental_enabled(name)
