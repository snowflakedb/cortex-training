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

"""Build TRAJECTORIES and the judge-selected training set from AGENT_TRACE_TABLE.

    python -m recipes.self_training_loop.select connection=<profile> run_tags=seed
    python -m recipes.self_training_loop.select connection=<profile> run_tags=v1-eval,v2-eval report=true

``report=true`` also prints the judge pass rate and accuracy per run tag (sql/judge_results.sql).
"""

from __future__ import annotations

import json
import logging

import chz

from recipes.self_training_loop.store import run_sql_file, snowpark_session

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    # Snowflake connection profile name (connections.toml); the default profile when None.
    connection: str | None = None
    # DATABASE.SCHEMA for TRAJECTORIES and TRAINING_SET_JUDGED.
    schema: str = "CORTEX_TRAINING_DB.SELF_TRAINING_LOOP"
    warehouse: str | None = None
    gateway: str = "SNOWFLAKE"
    # Comma-separated run tags to (re)build.
    run_tags: str = ""
    report: bool = False


def main(config: Config) -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    tags = [t for t in config.run_tags.split(",") if t]
    if not tags:
        raise ValueError("run_tags= is required")
    session = snowpark_session(config.connection, config.schema, config.warehouse)
    summary = {}
    for tag in tags:
        run_sql_file(session, "trajectories.sql", [tag, config.gateway, tag])
        row = session.sql(
            "SELECT COUNT(*), COUNT_IF(finish_reason = 'stop'), COUNT_IF(scores IS NOT NULL), "
            "(SELECT COUNT(*) FROM TRAINING_SET_JUDGED WHERE run_tag = ?), "
            "(SELECT COUNT(DISTINCT task_id) FROM TRAINING_SET_JUDGED WHERE run_tag = ?) "
            "FROM TRAJECTORIES WHERE run_tag = ?",
            params=[tag, tag, tag],
        ).collect()[0]
        summary[tag] = dict(
            zip(("runs", "finished", "judged", "kept", "tasks_kept"), row)
        )
        logger.info("%s: %s", tag, summary[tag])
    if config.report:
        for row in run_sql_file(session, "judge_results.sql", [json.dumps(tags)])[-1]:
            logger.info("%s", row.as_dict())
    return summary


if __name__ == "__main__":
    chz.nested_entrypoint(main)
