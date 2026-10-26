# On-Policy Distillation

Train a small student model to use tools like a larger teacher, on the
student's own attempts. Each step, the student answers trivia questions by
searching a fixed Wikipedia corpus with three tools over several turns, a
frozen teacher scores every token the student wrote, and the student is updated
toward the tokens the teacher prefers. The default run distills Qwen3.8-27B into
Qwen3.5-4B and raises held-out accuracy from about 69% to about 82% in 40
steps; with Qwen3.5-2B as the student, accuracy rises from about 40% to 67–74%.

## How it works

The example creates three jobs:

- **Student**: a training sub-job and a sampling sub-job of the same model.
  The sampling sub-job plays episodes, and weight sync copies each update from
  training to sampling.
- **Teacher**: a separate sampling-only job. It never trains and never
  receives weight syncs.
- **Judge**: a separate sampling-only job that grades final answers against
  the reference answer. With `judge=None` the teacher job judges.

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
   teacher's chat format, sent as one pre-tokenized prompt with
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

The student and teacher must share a tokenizer, so pick a teacher from the
student's model family. Before it creates any job, the example checks this,
checks that the tool turns it appends match each model's chat template, and
checks that the teacher's window fits the longest conversation it will score.
It also checks every teacher score, and checks that the server's teacher term
covered exactly the scored tokens. Updates are fully on-policy: no
`old_log_probs_shifted` is sent, so `eps_clip` has no effect.

## Run

The default distills Qwen3.8-27B into Qwen3.5-4B, using six H200 GPUs: one
training and one sampling GPU for the student, two for the teacher, and two for
the judge. Check capacity first with `cortex-training capacity`. Install the
dependencies listed in the [examples prerequisites](../README.md#prerequisites).

```bash
python -m examples.distillation.on_policy.train config=/path/to/config.json
```

The other validated pairs:

```bash
# Qwen3.8-Flash-Next -> Qwen3.5-4B (twelve GPUs: the teacher uses eight)
python -m examples.distillation.on_policy.train config=/path/to/config.json \
  teacher=Qwen/Qwen3.8-Flash-Next teacher_gpus=8

# Qwen3.8-27B -> Qwen3.5-2B
python -m examples.distillation.on_policy.train config=/path/to/config.json \
  student=Qwen/Qwen3.5-2B

# Qwen3.8-Flash-Next -> Qwen3.5-2B
python -m examples.distillation.on_policy.train config=/path/to/config.json \
  student=Qwen/Qwen3.5-2B teacher=Qwen/Qwen3.8-Flash-Next teacher_gpus=8
```

A default run takes about 3 hours: about 4 minutes per training step plus about
2.5 minutes per evaluation, with `eval_every=5`. All jobs are cancelled after
the final checkpoint is saved.

- **Questions:** [`willcb/wiki-trivia-questions-v4`](https://huggingface.co/datasets/willcb/wiki-trivia-questions-v4),
  478 trivia questions, each written from one page of the corpus. The dataset
  has a single split, so a fixed sample of `n_eval=100` questions (chosen by
  `seed`) is held out for evaluation and the other 378 are training questions.
  It is loaded at a pinned revision (`QUESTIONS_REVISION` in
  `tasks/wiki_search.py`), so the split matches the results below.
- **Corpus:** [`willcb/rare-wiki-pages`](https://huggingface.co/datasets/willcb/rare-wiki-pages),
  2,590 Wikipedia articles stored as markdown, loaded at a pinned revision
  (`CORPUS_REVISION`). It is static: no live Wikipedia or network search is
  involved. `wiki_search_pages` searches page **titles** with ChromaDB's
  default local embedding model and returns the 10 closest. The title index is
  built on the first run (about a minute, including an 80 MB model download)
  and cached under `index_dir/<corpus revision>`. Tool results are cut at 8,000
  characters. The task, tools, system prompt and judge prompt follow
  PrimeIntellect's `wiki_search` environment.
- **Student:** `Qwen/Qwen3.5-4B` (or `Qwen/Qwen3.5-2B`), full-parameter,
  thinking off.
- **Teacher:** `Qwen/Qwen3.8-27B` (92.5% on the same evaluation) or
  `Qwen/Qwen3.8-Flash-Next` (93.5%), thinking off, sampling only.
- **Judge:** `Qwen/Qwen3-30B-A3B`, thinking off, temperature 0, answering yes or
  no. It is fixed across teachers so their results are comparable.

## How the jobs are built

`make_distillation_config` builds the student and teacher from the
[model catalog](../../model-catalog/README.md), and `make_config` builds the
judge:

- **Student:** the training and inference files for `student=`, `context=`,
  `hardware=` and `lora=` (default `false`), combined like an RL job but without
  router replay. Then `distillation/overrides.json` shrinks it to the validated
  footprint (1 + 1 GPUs, batch 1, peak learning rate 1e-6, micro-batches of up
  to 16,384 tokens), and `student_max_seq_len=` (default 16384) sizes both
  sub-jobs. Keep the training and sampling GPU counts equal: on H200, weight
  sync between a 1-GPU and a multi-GPU sub-job fails.
- **Teacher:** the full-weights inference file for `teacher=` (in
  `teacher_context=`, which defaults to `context=`), sized by
  `teacher_max_seq_len=` (default 20480) and `teacher_gpus=` (default 2, which
  also sets tensor parallelism). The teacher scores whole conversations in its
  own chat format, so its window must exceed the student's by the extra tokens
  its prompt and turn headers add; the example computes that margin before
  creating any job. For the validated pairs the margin is 0.
- **Judge:** the full-weights inference file for `judge=` (in `judge_context=`,
  default `standard`), sized by `judge_max_seq_len=` (default 20480) and
  `judge_gpus=` (default 2).

Each run writes the jobs it builds from the catalog (not `job_config=` files or
attached jobs) to `examples/distillation/configs/` (gitignored). To
customize them, edit copies and pass them back with `job_config=`,
`teacher_job_config=` and `judge_job_config=`. To attach to running jobs
instead, use `job_id=`, `teacher_job_id=` and `judge_job_id=`; attached jobs
are left running, so several runs can share one teacher and one judge. To build
the student and teacher jobs without running the example:

```bash
python -m examples.utils --distillation \
  --student Qwen/Qwen3.5-4B --teacher Qwen/Qwen3.8-27B --context standard \
  --student-max-seq-len 16384 --teacher-max-seq-len 20480 \
  --override examples/distillation/overrides.json -o dist.json
```

This writes `dist.json` (student) and `dist.teacher.json` (teacher). Without
`--override`, the student gets the catalog's full size: 8 training and 8
sampling GPUs. The command does not apply `teacher_gpus=`, so the teacher keeps
the catalog's GPU count.

## Expected Results

Validated on H200s with the default settings except `eval_every=1`. Accuracy is
the share of 200 held-out episodes (100 questions × `eval_samples=2`,
temperature 0.7) that the judge accepts; each evaluation has a standard error of
about ±4 points, so the table also gives the mean of the last ten evaluations.

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
turns. Logged metrics:

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

## Options

| Argument | Default | Meaning |
|---|---|---|
| `student=`, `teacher=`, `judge=`, `context=`, `hardware=`, `lora=` | see above | Which catalog files to use. `judge=None` lets the teacher judge |
| `student_max_seq_len=`, `teacher_max_seq_len=`, `judge_max_seq_len=` | 16384, 20480, 20480 | Context windows. The teacher's must cover the student's plus its header margin |
| `teacher_gpus=`, `judge_gpus=` | 2, 2 | GPUs and tensor parallelism for the teacher and judge. `None` keeps the catalog's |
| `max_steps=`, `questions_per_step=`, `group_size=` | 40, 32, 4 | Loop size. `group_size` matters only for `reward_coef` |
| `max_turns=`, `turn_max_tokens=` | 10, 1024 | Episode limits |
| `lr_schedule=` | `cosine` | Decays to `min_lr_ratio` of the peak after `warmup_steps` (default 2) |
| `teacher_clip=`, `teacher_clip_negative=` | 3, 1 | Limit how hard any one token is pushed up and down. With a symmetric clip of 3, the Flash-Next → Qwen3.5-2B run stopped answering at step 5 and never recovered; with 1 it recovered by step 10 |
| `teacher_tau=` | 1 | Weight of the teacher term |
| `reward_coef=` | 0 | Adds a group-centred judge reward. `teacher_tau=0 reward_coef=1` runs plain GRPO on that reward with no teacher job |
| `eval_every=`, `n_eval=`, `eval_samples=` | 5, 100, 2 | Held-out evaluation. Evaluation does not change training |
| `max_prompts_per_request=`, `teacher_prompts_per_request=` | 32, 8 | Split generate and teacher-scoring requests into chunks |
| `wandb_project=`, `sf_tracking=` | off | [Experiment tracking](../../docs/guides/operations/wandb.md) |

## Checkpoint and Evaluation

The run saves a weights-only checkpoint of the student and logs its
`checkpoint_id`. Serve it with the inference example:

```bash
python -m examples.inference.serve config=/path/to/config.json \
  model=Qwen/Qwen3.5-4B context=standard \
  source_job_id=STUDENT_JOB_ID checkpoint_id=CHECKPOINT_ID
```

The catalog's Qwen3.5-4B inference file uses eight GPUs. To serve on fewer,
write the job with
`python -m examples.utils --model Qwen/Qwen3.5-4B --context standard --inference -o serve.json`,
lower its `n_gpus`, and pass `job_config=serve.json` instead of `model=` and
`context=`. To generate from or evaluate the checkpoint, also pass
`renderer_name=qwen3_5_disable_thinking`, as training uses thinking off.

## Troubleshooting

- `student and teacher tokenizers differ`: pick a teacher from the same model
  family; the teacher scores the student's token ids directly.
- `appended tool turns do not match its chat template`: the model's chat
  template formats tool results differently from the Qwen layout this example
  appends; pick a Qwen3 or Qwen3.5-family model.
- `teacher max_seq_len ... must exceed ...`: raise `teacher_max_seq_len=`.
- `generate result has no 'prompt_logprobs'`: the teacher job must be a
  sampling job on a server version that passes `prompt_logprobs` through.
- `Unknown config keys` naming `teacher_tau`: the training server does not
  support the `grpo` teacher term.
- `the teacher term covered N tokens, expected M`: the server trained on
  different tokens than the teacher scored; check that the student job config
  has not been changed to pack or truncate sequences.
- `NCCL error: internal error` at the first weight sync: the student's
  training and sampling sub-jobs have different GPU counts, one of them 1; give
  both the same count.
- Accuracy falls to 0% and `test/end_max_turns` reaches 100%: the student has
  stopped answering. When the student answers too early, the teacher scores the
  first token of its answer much lower than the first token of another tool
  call, so a weak student can learn to never answer, and then no answer tokens
  remain to correct. Keep `teacher_clip_negative=1`, use a teacher closer to the
  student (Qwen3.8-27B pushes about half as hard as Flash-Next), or add
  `reward_coef`. Qwen3.5-0.8B collapsed this way with Flash-Next even with the
  cap, and only partly recovered with Qwen3.8-27B.
- A generate request fails because its response is too large (long episodes
  with per-token log probabilities): lower `teacher_prompts_per_request=` for
  teacher scoring, or `max_prompts_per_request=` for rollouts and evaluation.

## Provenance

Adapted from tinker-cookbook's on-policy distillation
(`tinker_cookbook/distillation/train_on_policy.py`) and Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
The wiki-search task follows PrimeIntellect's `wiki_search` environment
([prime-rl example](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/wiki-search)).
