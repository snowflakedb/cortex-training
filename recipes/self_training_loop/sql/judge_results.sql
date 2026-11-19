-- Per run tag: the share of runs that meet every judge cutoff in TRAINING_SET_JUDGED (the
-- reference-free signal a promotion gate uses), and accuracy against the reference answers
-- (correctness, recorded only to check the judges). Baseline, drift and overall.

SELECT t.run_tag,
       COALESCE(IFF(STARTSWITH(t.task_id, 'n'), 'drift', 'baseline'), 'overall') AS tasks,
       COUNT(*) AS runs,
       ROUND(100 * AVG(IFF(t.finish_reason = 'stop'
                           AND t.scores:answer_relevance::float >= 0.66
                           AND t.scores:logical_consistency::float = 1
                           AND t.scores:tool_selection::float >= 0.66
                           AND t.scores:tool_calling::float >= 0.66
                           AND t.scores:execution_efficiency::float >= 0.33, 1, 0)), 1) AS judge_pass_pct,
       ROUND(100 * AVG(t.scores:correctness::float), 1) AS accuracy_pct
FROM TRAJECTORIES t
WHERE t.run_tag IN (SELECT value::string FROM TABLE(FLATTEN(PARSE_JSON(?))))
GROUP BY GROUPING SETS ((t.run_tag, IFF(STARTSWITH(t.task_id, 'n'), 'drift', 'baseline')), (t.run_tag))
ORDER BY 1, 2;
