# On-Policy Distillation

Train a small student model to use tools like a larger teacher, on the
student's own attempts. Each step, the student answers trivia questions by
searching a fixed Wikipedia corpus with three tools over several turns; a frozen
teacher scores every token the student wrote, and the student is updated toward
the tokens the teacher prefers. The default run distills Qwen3.8-27B into
Qwen3.5-4B and raises the student's held-out accuracy from about 69% to about
82% in 40 steps; with Qwen3.5-2B as the student, accuracy rises from about 40%
to 67–74%.

## How it works

The recipe creates three jobs:

- **Student**: a training sub-job and a sampling sub-job of the same model.
  The sampling sub-job plays episodes, and weight sync copies each update from
  training to sampling.
- **Teacher**: a separate sampling-only job. It never trains and never
  receives weight syncs.
- **Judge**: a separate sampling-only job that grades final answers against
  the reference answer. With `judge_job_config=None` the teacher job judges.

An **episode** is one attempt at one question: the student calls tools
(`wiki_search_pages`, `wiki_view_sections`, `wiki_read_section`) until it gives
a final answer, uses `max_turns` turns, writes a `turn_max_tokens`-token turn,
or fills its context. Each training step:

1. The student plays `group_size` episodes for each of `questions_per_step`
   questions. Episodes advance in lockstep: each round is one batched generate
   request for every episode still running, followed by the tool calls it
   produced. Tool results are appended as tokens, so the conversation is never
   re-rendered and the training sequence is exactly what the student was
   sampled on.
2. The judge grades each final answer (logged, and used only with
   `reward_coef > 0`).
3. The teacher scores each episode in one prefill: the same conversation in the
   teacher's chat format (prompt tokens, then everything after them) sent with
   `prompt_logprobs: 0` and `max_tokens: 1` (see
   [Per-token log probabilities](../../docs/reference/rest-api.md#per-token-log-probabilities)).
   The student and teacher share a tokenizer, so the student's tokens are
   scored exactly as written.
4. Each episode is one training sequence. The loss covers only the student's
   own tokens, including the end-of-turn token after each turn; tool results,
   prompts and chat-template markup are context. The teacher scores go in
   `context["teacher_log_probs_shifted"]`, aligned so that position `i` scores
   the token at `i + 1`, and the student trains with the `grpo` loss's teacher
   term (see
   [Teacher term for on-policy distillation](../../docs/reference/rest-api.md#teacher-term-for-on-policy-distillation)).
   Each token's advantage becomes
   `teacher_tau * clamp(log p_teacher - log p_student, -teacher_clip_negative, teacher_clip)`,
   which on average follows the gradient of the reverse KL from student to
   teacher.

This is the formulation from Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
and tinker-cookbook's `distillation/train_on_policy.py`, applied per token over
whole multi-turn episodes, with the KL advantage clipped and the learning rate
decayed.

Every teacher result is checked before training: a missing, mis-sized,
misaligned, or non-finite score raises an error. After each forward-backward
pass, and before the optimizer step, the recipe checks that the server's
teacher term covered exactly the student tokens the teacher scored. Before
creating any job, it checks that the student and teacher share a tokenizer and
that the tool turns it appends match each model's chat template.

The recipe does not send `old_log_probs_shifted`, so the server uses the
current, detached student log probabilities. Each update is fully on-policy
and `eps_clip` has no effect.

## Prerequisites

Install the client and recipe dependencies as described in
[recipes/README.md](../README.md#prerequisites), plus ChromaDB for the title
index:

```bash
uv pip install chromadb
```

Create a connection file from `examples/config/connection.json.template`. The
student, teacher, and judge models are all in the
[model catalog](../../docs/reference/model-compatibility.md).

## Dataset and Models

- Questions: [`willcb/wiki-trivia-questions-v4`](https://huggingface.co/datasets/willcb/wiki-trivia-questions-v4),
  478 trivia questions, each written from one page of the corpus. The dataset
  has a single split, so a fixed sample of `n_eval=100` questions (chosen by
  `seed`) is held out for evaluation and the other 378 are training questions.
- Corpus: [`willcb/rare-wiki-pages`](https://huggingface.co/datasets/willcb/rare-wiki-pages),
  2,590 Wikipedia articles stored as markdown. It is static: no live Wikipedia
  or network search is involved. `wiki_search_pages` searches page **titles**
  with ChromaDB's default local embedding model and returns the 10 closest;
  the index is built on the first run (about a minute, including an 80 MB
  model download) and cached under `index_dir`. Tool results are cut at 8,000
  characters.
- Task, tools, system prompt and judge prompt follow PrimeIntellect's
  `wiki_search` environment.
- Students: `Qwen/Qwen3.5-2B` or `Qwen/Qwen3.5-4B`, full-parameter, thinking off.
- Teachers: `Qwen/Qwen3.8-27B` (92.5% on the same evaluation) or
  `Qwen/Qwen3.8-Flash-Next` (93.5%), thinking off.
- Judge: `Qwen/Qwen3-30B-A3B`, thinking off, temperature 0, answering yes or
  no. It is fixed across teachers so their results are comparable.

## Hardware

Every job runs on H200s. The student uses one training and one sampling GPU
(`configs/qwen35_4b_student.json` or `configs/qwen35_2b_student.json`), the
judge two (`configs/qwen3_30b_a3b_judge.json`), and the teacher two
(`configs/qwen38_27b_teacher.json`) or eight
(`configs/qwen38_flash_next_teacher.json`): six GPUs for the default run and
twelve with Flash-Next. Check capacity before submitting:

```bash
cortex-training capacity --hardware H200
```

Keep the student's training and sampling sub-jobs at the same GPU count. On
H200, a weight sync between a 1-GPU sub-job and a multi-GPU sub-job fails with
`NCCL error: internal error`.

## Run

The default pair, Qwen3.8-27B teaching Qwen3.5-4B:

```bash
python -m recipes.distillation.on_policy.train \
  config=/path/to/config.json
```

The other validated pairs:

```bash
# Qwen3.8-Flash-Next -> Qwen3.5-4B
python -m recipes.distillation.on_policy.train config=/path/to/config.json \
  teacher_job_config=configs/qwen38_flash_next_teacher.json

# Qwen3.8-27B -> Qwen3.5-2B
python -m recipes.distillation.on_policy.train config=/path/to/config.json \
  job_config=configs/qwen35_2b_student.json

# Qwen3.8-Flash-Next -> Qwen3.5-2B
python -m recipes.distillation.on_policy.train config=/path/to/config.json \
  job_config=configs/qwen35_2b_student.json \
  teacher_job_config=configs/qwen38_flash_next_teacher.json
```

`config=` is the Snowflake connection file. A default run takes about 3 hours
on H200s: about 4 minutes per training step plus about 2.5 minutes per
evaluation, with `eval_every=5`. All jobs are cancelled when the run ends,
after the final checkpoint is saved.

## Expected Results

Validated on H200s with the default settings, evaluating after every step
(`eval_every=1`). Accuracy is the share of 200 held-out episodes (100
questions × `eval_samples=2`, temperature 0.7) that the judge accepts; each
evaluation has a standard error of about ±4 points, so the table also gives the
mean of the last ten evaluations.

| Teacher → student | Step 0 | Lowest (step) | Mean of steps 31–40 | Step 40 |
|---|---|---|---|---|
| Qwen3.8-27B → Qwen3.5-4B (default) | 68.5% | 52.5% (5) | 82.2% | 82.5% |
| Qwen3.8-Flash-Next → Qwen3.5-4B | 70.0% | 49.5% (6) | 84.7% | 85.5% |
| Qwen3.8-27B → Qwen3.5-2B | 43.5% | 8.5% (6) | 67.3% | 70.0% |
| Qwen3.8-Flash-Next → Qwen3.5-2B | 38.0% | 0.0% (5) | 73.9% | 73.5% |

Every run dips around steps 3–8, near the peak learning rate: the student
starts calling more tools instead of answering, and the share of episodes that
reach `max_turns` without an answer jumps (to 89–100% for the 2B runs). The
students then recover and end well above where they started: they answer more
often (the 2B's answered share rises from about 50% to 74–78%) and use fewer
turns. Logged metrics include:

| Metric | Meaning |
|---|---|
| `test/correct` | Share of held-out episodes the judge accepts |
| `test/end_answer`, `test/end_max_turns`, `test/end_context`, `test/end_length`, `test/end_empty` | How evaluation episodes ended |
| `test/turns`, `test/tool_calls`, `test/agent_tokens` | Turns, tool calls, and student tokens per evaluation episode |
| `teacher/test/*` | The same, with the teacher as the agent, at step 0 |
| `rollout/*` | The same for the training episodes |
| `train/kl/k1_per_token`, `train/kl/k3_per_token` | Reverse-KL estimates on the training episodes, from the student's sampling log probabilities |
| `teacher_term_token_count`, `teacher_log_ratio_sum`, `teacher_clipped_log_ratio_sum` | The server's teacher-term metrics |
| `teacher/clipped_fraction` | Share of trained tokens whose `log p_teacher - log p_student` falls outside the clip, using the student's sampling log probabilities |

Every episode is also written to `log_path`: `train_episodes.jsonl` and
`eval_episodes.jsonl` record each final answer, its grade, and how the episode
ended, with a few full transcripts per step.

## Configuration Variations

```bash
python -m recipes.distillation.on_policy.train \
  config=/path/to/config.json \
  max_steps=MAX_STEPS \
  questions_per_step=QUESTIONS_PER_STEP \
  group_size=GROUP_SIZE \
  max_turns=MAX_TURNS \
  teacher_clip_negative=TEACHER_CLIP_NEGATIVE \
  reward_coef=REWARD_COEF \
  eval_every=EVAL_EVERY \
  seed=SEED \
  wandb_project=WANDB_PROJECT
```

- `teacher_clip_negative` (default `1`) limits how hard any one token can be
  pushed down; `teacher_clip` (default `3`) limits the push up. With a
  symmetric clip of `3`, the Flash-Next → Qwen3.5-2B run stopped answering at
  step 5 and never recovered; with `1` it recovered by step 10. The other pairs
  ended within a few points either way.
- `reward_coef` (default `0`, pure distillation) adds a group-centred judge
  reward to every token's advantage, alongside the teacher term. `group_size`
  matters only for this reward; the teacher term scores each token on its own.
- `teacher_tau=0` with `reward_coef=1` turns the teacher off and runs the same
  loop as plain GRPO on the judge's reward, for comparison. No teacher job is
  created, so a separate judge is required. This variant is not part of the
  validated configuration.
- `eval_every` (default `5`) sets how often the held-out evaluation runs; it
  does not change training.
- The peak learning rate, GPU counts, and sequence length live in the job
  configs under `configs/`. The teacher's `max_seq_len` must exceed the
  student's, because the teacher scores whole conversations with its own turn
  headers.
- `job_id=`, `teacher_job_id=`, and `judge_job_id=` attach to running jobs
  instead of creating new ones; attached jobs are left running. Several runs
  can share one teacher and one judge this way.

## Checkpoint and Evaluation

The run saves a weights-only checkpoint of the student and logs its
`checkpoint_id`. Serve it with the inference recipe and the sampling-only
configs shipped here:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=recipes/distillation/configs/qwen35_4b_sampling.json \
  source_job_id=STUDENT_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

Use `configs/qwen35_2b_sampling.json` for a Qwen3.5-2B student.

## Troubleshooting

- `student and teacher tokenizers differ`: pick a teacher from the same model
  family; the teacher scores the student's token ids directly.
- `appended tool turns do not match its chat template`: the model's chat
  template formats tool results differently from the Qwen layout this recipe
  appends; pick a Qwen3 or Qwen3.5-family model.
- `generate result has no 'prompt_logprobs'`: the teacher job must be a
  sampling job on a server version that passes `prompt_logprobs` through.
- `the teacher term covered N tokens, expected M`: the server trained on
  different tokens than the teacher scored; check that the student job config
  has not been changed to pack or truncate sequences.
- `NCCL error: internal error` at the first weight sync: the student's
  training and sampling sub-jobs have different GPU counts, one of them 1; give
  both the same count.
- Accuracy falls to 0% and `test/end_max_turns` reaches 100%: the student has
  stopped answering. The teacher scores the first token of an answer much
  lower than the first token of another tool call when the student answers too
  early, so a weak student can learn to never answer, and then no answer
  tokens remain to correct. Keep `teacher_clip_negative=1`, use a teacher closer
  to the student (Qwen3.8-27B pushes about half as hard as Flash-Next), or add
  `reward_coef`. Qwen3.5-0.8B collapsed this way with Flash-Next even with the
  cap, and only partly recovered with Qwen3.8-27B.
- A generate request fails because its response is too large (long episodes
  with per-token log probabilities): lower `max_prompts_per_request`, which
  splits each generate request into chunks. Teacher scoring always uses chunks
  of 8 episodes.

## Provenance

Adapted from tinker-cookbook's on-policy distillation
(`tinker_cookbook/distillation/train_on_policy.py`) and Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
The wiki-search task follows PrimeIntellect's `wiki_search` environment
([prime-rl example](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/wiki-search)).
The `last_validated` date in `recipe.yaml` records the most recent verified run.
