# Run GRPO from prime-rl

[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) can run GRPO against
Cortex Training. prime-rl supplies the orchestrator, environments, and configs;
Cortex runs training and sampling as separate sub-jobs. prime-rl has no
in-process plugin hook, so the
[Arctic Platform](https://github.com/Snowflake-AI-Research/Arctic-Platform)
launcher drives an unmodified prime-rl install from the outside — it is a
launcher, not a registered backend. The launcher and its prime-rl pin live in
the Arctic Platform repository.

Two example recipes live in the Arctic Platform repository under
`arctic_platform/integrations/prime_rl/examples/`, rather than under `recipes/`
here (the same split SkyRL uses):

- `qwen3.5-4b/math-i3.toml` — Qwen3.5-4B on the `i3_math` taskset
  (INTELLECT-3-RL math), scored deterministically by math-verify in the rollout
  runtime (no judge model). A difficulty-filtered learnable mid-band and the fixed
  `standard` sampler give a steady climbing reward curve (all-reward ~0.33 → ~0.9).
- `qwen3.5-4b/swe-grade.toml` — Qwen3.5-4B on `r2e-gym` with the `bash` agent
  harness and an agentic grading policy.

## Prerequisites

A Cortex account with capacity for 16 GPUs: 8 for training and 8 for sampling.
Follow [connection setup](../../docs/getting-started/setup.md) to create the connection
file and log in, then confirm capacity:

```bash
cortex-training login ~/cortex-training-config.json
cortex-training capacity
```

The driver machine needs no GPU. SWE additionally needs a sandbox backend; the
recipe uses PrimeIntellect's sandbox service (verifiers' default `prime`
runtime), so no local containers are required.

## Install

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform

# The integration plus the Cortex client stack.
pip install "arctic_platform[prime-rl]"

# prime-rl is user-supplied and version-pinned by the launcher. Install it from
# its repository (its dependency closure does not travel with a plain pin).
PRIME_RL=84e94bf05c29d73e293341b24f6eda2b15bbc900
pip install "prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl@${PRIME_RL}"
```

The two recipe files are in the cloned tree under
`arctic_platform/integrations/prime_rl/examples/`.

## Configure

```bash
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg")
export ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg")
export ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
```

## Run

```bash
cd Arctic-Platform

# Math (Qwen3.5-4B, i3_math — stock subprocess runtime, no sandbox):
python -m arctic_platform.integrations.prime_rl.rl \
  @ arctic_platform/integrations/prime_rl/examples/qwen3.5-4b/math-i3.toml \
  --env-package i3-math

# SWE (Qwen3.5-4B, r2e-gym — PrimeIntellect sandbox):
python -m arctic_platform.integrations.prime_rl.rl \
  @ arctic_platform/integrations/prime_rl/examples/qwen3.5-4b/swe-grade.toml \
  --env-package r2e-gym

# Resolve-only, writes the Cortex job body and exits:
python -m arctic_platform.integrations.prime_rl.rl @ <recipe>.toml --dry-run
```

`--env-package` installs the verifiers environment the recipe names (`i3-math`
for math, `r2e-gym` for SWE). Each recipe runs one 8-GPU training sub-job and
one 8-GPU sampling sub-job; `cortex.training.model_provider = "default"` serves
the dense model from Cortex's default training provider.

## The grading policy (SWE)

`swe-grade.toml` sets `env.agent.grading_policy = true`. The policy terminates a
rollout and forces its reward to 0 whenever the agent does something that should
not earn credit:

- a **format-invalid turn** — a malformed, unknown-tool, or unclosed tool call
  (any non-OK renderer parse status);
- a **per-turn output-budget overrun** — a turn that finishes on `length`
  (`max_completion_tokens`, 40960 here); or
- a **framework turn/token-limit stop** — the run hitting `max_turns` (100) or a
  token cap.

The fraction of disqualified rollouts is logged as
`grading_policy/disqualified`. The policy is applied over the unmodified public
`verifiers` by a startup patch shipped in the Arctic Platform prime-rl
integration (no fork, and nothing to pass at launch); it is off unless a recipe
opts in. Remove the `grading_policy` and `max_turns` lines to run plain SWE.

## Results

Reward below is the mean task reward over all rollouts
(`train/agg/all/agent/reward/mean`), i.e. before any 0-variance filtering.

### SWE — Qwen3.5-4B, r2e-gym, grading policy

![SWE reward curve: mean task reward climbs from ~0.41 to a ~0.85-0.9 plateau while the
grading-policy disqualified fraction falls from ~37% to ~2%](img/prime-rl-swe-reward.png)

After a short warm-up the model stops emitting malformed tool calls — the
disqualified fraction falls from ~37% to ~2% — and reward climbs from ~0.40 to
a ~0.85 plateau and holds there with no collapse.

### Math — Qwen3.5-4B, i3_math

![Math reward curve: mean task reward climbs from ~0.21 to a high ~0.85-0.9 plateau over the
first ~50-60 steps and holds there](img/prime-rl-math-reward.png)

Reward climbs steadily from ~0.21 to a high ~0.85–0.9 plateau over the first
~50–60 steps and holds there, with no collapse. The difficulty filter keeps a
learnable mid-band (so GRPO groups carry reward variance) and the fixed `standard`
sampler lets that improvement show up as a rising curve — the adaptive
`difficulty_pool` sampler instead pins reward flat at the frontier by design.

## More detail

The
[prime-rl integration README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/arctic_platform/integrations/prime_rl/README.md)
in the Arctic Platform repository covers the launcher internals, the exact
prime-rl and verifiers pins, the sandbox options, and the larger
Qwen3.6-35B-A3B recipes.
