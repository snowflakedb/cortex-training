-- Trajectories for one run tag, from AGENT_TRACE_TABLE, and the judge-selected training set.
--
-- Each run has a TruLens record_root span (task metadata), an agt.trajectory span (the
-- conversation as JSON, joined on conversation id) and eval_root spans (judge scores,
-- joined on record id). Statements are separated by semicolons; select.py binds the
-- gateway name and run tag to the ? placeholders in order.

CREATE TABLE IF NOT EXISTS TRAJECTORIES (
  run_tag STRING, record_id STRING, task_id STRING, split STRING, rollout INT, conversation_id STRING,
  n_assistant INT, finish_reason STRING, scores OBJECT, chat_messages VARIANT, snapshotted_at TIMESTAMP_LTZ
);

DELETE FROM TRAJECTORIES WHERE run_tag = ?;

INSERT INTO TRAJECTORIES
WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE(?))
  WHERE start_timestamp >= DATEADD('day', -14, CURRENT_TIMESTAMP())
    AND record_attributes:"ai.observability.app_version"::string = ?
),
roots AS (
  SELECT record_attributes:"ai.observability.record_id"::string AS record_id,
         PARSE_JSON(record_attributes:"ai.observability.call.kwargs.config"::string):metadata AS meta
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'record_root'
    AND record:"name"::string = 'langgraph.graph.state.CompiledStateGraph.invoke'
),
traj0 AS (
  SELECT record_attributes:"agt.conversation_id"::string AS conversation_id,
         TRY_PARSE_JSON(record_attributes:"agt.trajectory"::string) AS msgs
  FROM ev
  WHERE record_attributes:"agt.trajectory" IS NOT NULL
),
traj AS (
  SELECT t.conversation_id, ANY_VALUE(t.msgs) AS msgs, COUNT_IF(f.value:role::string = 'assistant') AS n_assistant
  FROM traj0 t, LATERAL FLATTEN(t.msgs) f
  GROUP BY 1
),
evals AS (
  SELECT record_attributes:"ai.observability.eval.target_record_id"::string AS record_id,
         record_attributes:"ai.observability.eval_root.metric_name"::string AS metric,
         AVG(record_attributes:"ai.observability.eval_root.score"::float) AS score
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'eval_root'
  GROUP BY 1, 2
),
scores AS (
  SELECT record_id, OBJECT_AGG(metric, score::variant) AS scores FROM evals GROUP BY 1
)
SELECT r.meta:run_tag::string, r.record_id, r.meta:task_id::string, r.meta:split::string, r.meta:rollout::int,
       r.meta:conversation_id::string, t.n_assistant,
       IFF(t.msgs[ARRAY_SIZE(t.msgs) - 1]:role::string = 'assistant'
           AND t.msgs[ARRAY_SIZE(t.msgs) - 1]:tool_calls IS NULL, 'stop', 'incomplete'),
       s.scores, t.msgs, CURRENT_TIMESTAMP()
FROM roots r
JOIN traj t ON t.conversation_id = r.meta:conversation_id::string
LEFT JOIN scores s ON s.record_id = r.record_id;

-- Reference-free selection. Judges score 0 to 3, stored as 0, 0.3333, 0.6667 or 1, so the
-- cutoffs are written as 0.66 (2 of 3) and 0.33 (1 of 3). logical_consistency must be 3 of 3:
-- it is the judge that compares the final answer with what the queries returned. Of the runs
-- that pass, the 2 with the highest mean judge score per task are kept.
CREATE OR REPLACE VIEW TRAINING_SET_JUDGED AS
SELECT * FROM (
  SELECT t.*,
         (scores:answer_relevance::float + scores:logical_consistency::float + scores:execution_efficiency::float
          + scores:tool_selection::float + scores:tool_calling::float) / 5 AS judge_mean,
         ROW_NUMBER() OVER (PARTITION BY run_tag, task_id ORDER BY judge_mean DESC, n_assistant) AS rank_in_task
  FROM TRAJECTORIES t
  WHERE split = 'train'
    AND chat_messages IS NOT NULL
    AND finish_reason = 'stop'
    AND scores:answer_relevance::float     >= 0.66
    AND scores:logical_consistency::float  = 1
    AND scores:tool_selection::float       >= 0.66
    AND scores:tool_calling::float         >= 0.66
    AND scores:execution_efficiency::float >= 0.33
)
WHERE rank_in_task <= 2;
