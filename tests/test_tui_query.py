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

"""The job list's search: parsing a query, and resolving its other keys against the loaded jobs.

No ``textual`` needed.
"""

from cortex_training.tui.views import job_matches_query
from cortex_training.tui.views import parse_query
from cortex_training.tui.views import resolve_query


def _job(job_id: str, *, seed: int, lr: float, betas: list) -> dict:
    return {"job_id": job_id, "status": "RUNNING", "sub_jobs": [{
        "sub_job_id": f"{job_id}:training:0", "job_type": "training", "seed": seed,
        "training_config": {"n_gpus": 8, "optimizer": {"name": "adamw", "lr": lr, "betas": betas}},
    }]}


JOBS = [_job("j1", seed=42, lr=5e-5, betas=[0.9, 0.95]), _job("j2", seed=7, lr=3e-4, betas=[0.8])]


def _matches(text: str) -> list:
    query = resolve_query(parse_query(text), JOBS)
    return [job["job_id"] for job in JOBS if job_matches_query(job, query)]


def test_quotes_group_words_and_a_wholly_quoted_token_is_a_word():
    query = parse_query("name:\"lora sweep\" model:'qwen3 8b' \"training:0\"")
    assert query.filters == [("name", ":", "lora sweep"), ("model", ":", "qwen3 8b")]
    assert query.terms == ["training:0"] and not query.errors  # quoted: the text, not a filter


def test_an_unbalanced_quote_leaves_every_word_as_typed():
    assert parse_query('lora "sweep 3').terms == ["lora", '"sweep', "3"]
    assert parse_query("bob's run").terms == ["bob's", "run"]  # an apostrophe opens no quote
    assert parse_query('name:"lora sweep').filters == [("name", ":", '"lora')]


def test_a_dotted_key_reads_that_member_in_any_case():
    query = parse_query("Training_Config.Optimizer.LR<1e-4")
    assert query.filters == [("training_config.optimizer.lr", "<", "1e-4")] and not query.errors
    assert _matches("Training_Config.Optimizer.LR<1e-4") == ["j1"]
    assert _matches("training_config.optimizer.betas=0.95") == ["j1"]  # a list matches on any element


def test_an_unknown_key_is_an_error_until_a_loaded_job_has_it():
    query = parse_query("Seed=42 lora")
    assert query.errors == ["unknown field 'Seed'"] and query.unknown == [("seed", "=", "42", "Seed")]
    assert not job_matches_query(JOBS[0], query)  # an unknown key never matches every job
    resolved = resolve_query(query, JOBS)
    assert resolved.filters == [("seed", "=", "42")] and resolved.errors == [] and resolved.terms == ["lora"]
    assert _matches("seed=42") == ["j1"]


def test_a_key_no_job_has_stays_an_error_and_names_the_nested_paths_ending_in_it():
    assert resolve_query(parse_query("betas=0.95"), JOBS).errors == [
        "unknown field 'betas' (did you mean training_config.optimizer.betas?)"]
    assert resolve_query(parse_query("nope=1"), JOBS).errors == ["unknown field 'nope'"]
    assert _matches("betas=0.95") == []
