# Train an agent on the Harbor tasks you evaluate it on

You evaluate your agent with `harbor run` on a directory of
[Harbor](https://github.com/harbor-framework/harbor) tasks, each with
`instruction.md`, `environment/` and `tests/test.sh`. This recipe trains the
model behind that agent on the same tasks. Every RL rollout is a Harbor trial:
the same task, the same sandbox, scored by the same `tests/test.sh`. Nothing is
rewritten into an RL format, and the score you already track in eval becomes
the reward.

`Qwen/Qwen3.5-4B` is trained with online RL (GSPO loss). SkyRL drives the loop
from a CPU machine, and Cortex Training runs 4 training and 4 sampling H200
GPUs. The code is [`recipes/rl/harbor`](https://github.com/Snowflake-AI-Research/Arctic-Platform/tree/main/recipes/rl/harbor)
in Arctic Platform.

> **Tip — Drive training from OpenHands with `/cortex-training-rl`:** The [`cortex-training-rl`](https://github.com/OpenHands/extensions/tree/main/skills/cortex-training-rl) skill is built into OpenHands' public skill registry (`OpenHands/extensions`). In the OpenHands CLI or GUI, ask OpenHands to post-train your model on your Harbor tasks (or invoke `/cortex-training-rl`) and the agent runs preflight checks, launches `run_qwen35_4b.sh`, monitors rewards, and plots them.

## From eval to training

| | `harbor run` (eval) | This recipe (training) |
| --- | --- | --- |
| Tasks | `-p <task dir>` | `TRAIN_HARBOR_TASKS`, `EVAL_HARBOR_TASKS` |
| Agent | `-a <agent>` | OpenHands (`harbor_rl.agent:OpenHandsHarborAgent`) by default, or your own as `HARBOR_AGENT=module:Class` |
| Skills, tools, prompt | Part of your agent | `HARBOR_SKILL`, `OPENHANDS_TOOLS`, `OPENHANDS_SYSTEM_PROMPT_FILE` |
| Sandbox | `--env`, `--ek key=value` | `HARBOR_ENV`, `HARBOR_ENV_KWARGS` (JSON) |
| Score | `tests/test.sh` writes `/logs/verifier/reward.txt` or `reward.json` | The same file, used as the rollout's reward |
| Model | `-m <model>` | The policy being trained, served by the Cortex sampling job |

`HARBOR_AGENT` must run on the driver. Harbor's installed agents
(`openhands-sdk`, `mini-swe-agent`, `claude-code`, ...) run inside the sandbox,
where task code could read the rollout's API key, so the launcher refuses them.

## How it works

1. **Trial:** Harbor's `Trial` starts the sandbox and runs the agent, with its
   LLM pointed at the rollout's model endpoint.
2. **Tokens:** that endpoint records every chat completion and keeps the exact
   sampled token IDs across turns. Assistant turns are trained; prompts and
   tool outputs are masked.
3. **Verifier:** Harbor's `Verifier` runs `tests/test.sh`, and its reward goes
   back to Cortex for the policy update.

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../../docs/getting-started/setup.md), or run
`cortex-training login` to set up a Snowflake connection profile, then confirm
capacity:

```bash
cortex-training login
cortex-training capacity
```

The driver needs no GPU and no Docker daemon. Rollouts run in Snowflake
sandboxes, which log in through a connection profile in
`~/.snowflake/connections.toml`:
- The `training` profile from [setup](../../docs/getting-started/setup.md)
  works as is. It logs in with a programmatic access token, so it needs no
  browser, which matters because the run logs in again every 20 minutes. The
  commands below pass it as `connection`.
- `role` must be a role that can create sandboxes.
- For tasks that reach the internet, you also need an external access
  integration that allows the hosts they need.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --branch skyrl-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform
```

The run script installs OpenHands and Harbor itself.

## Configure

The Arctic Platform launcher reads its connection only from `ARCTIC_CORTEX_*`
environment variables, so `cortex-training login` alone is not enough. Export
them from your profile:

Start each launcher command below with `python -m cortex_training.env --`. It
runs the command with `ARCTIC_CORTEX_HOST`, `ARCTIC_CORTEX_DATABASE`,
`ARCTIC_CORTEX_SCHEMA` and `ARCTIC_CORTEX_PAT` set from the profile
`cortex-training login` saved (add `connection=NAME` before `--` for another).
The token is only in that command's environment: it's never printed or left in
your shell. The profile must sign in with a programmatic access token (PAT), as
profiles `cortex-training login` creates do. Run it where cortex-training is
installed; it works from any folder.

## 1. Evaluate on the sandbox training uses

Rollouts run in a [Snowflake sandbox](https://github.com/Snowflake-AI-Research/Arctic-Platform/tree/main/arctic_platform/integrations/harbor/snowflake_sandbox)
by default. To get a baseline that training is measured against, evaluate on
that same environment. From the `Arctic-Platform` checkout, `uv run --extra harbor`
provides the pinned `harbor` CLI with that environment importable:

```bash
python -m cortex_training.env -- uv run --extra harbor harbor run -p /path/to/eval_tasks -a <your agent> -m <your model> \
  --env arctic_platform.integrations.harbor.snowflake_sandbox:SnowflakeSandboxEnvironment \
  --ek base_image=sandbox-base --ek connection=training --ek role=MY_ROLE
```

`base_image=sandbox-base` starts each sandbox from the Snowflake catalog image
and replays the task's `environment/Dockerfile` on it, so no image is built or
pushed. `sandbox-base` is Chainguard Linux with Python 3.11, pip and git, and
has `apk` rather than `apt-get`. Replay handles one `FROM` stage with `RUN`,
`COPY`/`ADD` of local files, `ENV`, `ARG`, `WORKDIR`, `USER` and `ENTRYPOINT`,
and refuses anything else.

## 2. Train on the same tasks

From the `Arctic-Platform` checkout. `ARCTIC_CORTEX_JOB_COMMENT` is the note
that identifies your job.

```bash
export TRAIN_HARBOR_TASKS=/path/to/train_tasks
export EVAL_HARBOR_TASKS=/path/to/eval_tasks
export HARBOR_ENV_KWARGS='{"base_image": "sandbox-base", "connection": "training", "role": "MY_ROLE"}'
export HARBOR_SKILL=/path/to/SKILL.md   # optional: ":"-separated SKILL.md paths
export ARCTIC_CORTEX_JOB_COMMENT="your name, harbor rl"

python -m cortex_training.env -- bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=50
```

`HARBOR_ENV_KWARGS` takes the same keys as `--ek`. Add `"egress_eai": "MY_EAI"`
for tasks that reach the internet, such as a `git clone` from GitHub.

Before it provisions GPUs, the launcher runs the first training task once, in
its sandbox, with an agent that does nothing, and grades it. If the sandbox,
the Dockerfile replay or `tests/test.sh` fails, it prints the cause and exits
without creating a Cortex job. `HARBOR_SKIP_TASK_CHECK=1` skips this check.

Snowflake allows 100 live sandboxes per owner, and suspended ones count. A run
keeps at most 90 alive and starts at most 16 at a time. The launcher warns when
sandboxes left from an earlier run would push you over the limit. A run
terminates its sandboxes when it ends, even if it fails. If the driver is killed
outright, clean up as the
[Harbor recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/tree/main/recipes/rl/harbor#1-any-harbor-task-set)
shows.

A task whose `task.toml` sets `[environment] docker_image` to an image already
in a Snowflake image repository (`/DB/SCHEMA/REPO/IMAGE:TAG`) runs on that
image as is and needs no Dockerfile; `prebuilt_image` runs every task on one
such image. To use an image of your own, first
[push it to an image repository](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/working-with-registry-repository)
in your account. This was tested with an image Snowflake already provides;
pushing your own image, and mirroring other images into your repository
(`image_repo`, described in that README), were not tested end to end in this
release.

Other sandboxes: `HARBOR_ENV=host` runs tasks as plain subprocesses on the
driver, with no isolation, so it is only for tasks you trust and needs
`HARBOR_ALLOW_HOST_EXEC=1`. Harbor's other environments (`docker`, `modal`,
`daytona`, `e2b`) are accepted but were not tested with this recipe.

## 3. Read the reward

Each step's reward is the `tests/test.sh` score that `harbor run` reports,
averaged over that step's rollouts. The log is
`~/checkpoints/qwen35-4b-harbor-rl/qwen35-4b-harbor-rl.log`, and checkpoints are
written next to it. Each rollout's conversation and reward are saved under
`trajectories/step_*/`. A rollout that ended without a grade (the sandbox never
started, the agent or the verifier crashed) is saved as `.error` with its
traceback and is left out of the loss. If every rollout of a step errors, the
step is rerun up to 3 times, after waits of 1, 2, and 4 minutes, and then the
run stops.

To also evaluate during training, append
`trainer.eval_before_train=true trainer.eval_interval=10` to the run script:
the model is evaluated on `EVAL_HARBOR_TASKS` before the first step and every
10 steps after it, and each evaluation prints `eval/all/avg_score` in the log.

To compare the trained model with your baseline in `harbor run`,
[deploy](../../docs/guides/inference/deploy-model.md) the `weights-only`
checkpoint the run ends with and evaluate the deployment with the recipe's
agent, as the
[Harbor recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/harbor/README.md#evaluate)
shows.

## Example: SWE-smith code localization as Harbor tasks

`prepare_data.py` exports
[`OpenHands/SWE-smith-py-code-search`](https://huggingface.co/datasets/OpenHands/SWE-smith-py-code-search)
as Harbor task directories. Each task asks for the files, classes and functions
to change for a bug report, and `tests/test.sh` scores file + module + function
F1 (maximum `3.0`):

```bash
uv run recipes/rl/harbor/prepare_data.py \
  --export-swe-smith-to ~/data/swe-smith-harbor-tasks \
  --out ~/data/harbor-rl

export TRAIN_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/train
export EVAL_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/eval
export HARBOR_ENV_KWARGS='{"base_image": "sandbox-base", "connection": "training", "role": "MY_ROLE", "egress_eai": "MY_GITHUB_EAI"}'
export ARCTIC_CORTEX_JOB_COMMENT="your name, harbor localization"

python -m cortex_training.env -- bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=100
```

Each task clones its repository from GitHub when its sandbox starts, so
`egress_eai` must allow `github.com`. In a 25-step run on Snowflake sandboxes,
the mean reward was `0.59` over steps 1–10, `1.81` over steps 11–20 and `2.39`
over steps 21–25.

## Settings

Set these as environment variables before the run script:

| Setting | Variable | Default |
| --- | --- | --- |
| Model | `MODEL` | `Qwen/Qwen3.5-4B` |
| Batch | `BATCH_SIZE` × `N_ROLLOUTS` | 8 prompts × 8 rollouts |
| Turns | `MAX_TURNS` | 10 |
| Context | `MAX_CONTEXT_LENGTH` | 32768 tokens |
| Learning rate | `LR` | 1e-6 |
| Agent tools | `OPENHANDS_TOOLS` | `terminal` plus a finish tool |

Loss settings and everything else pass through as trailing Hydra flags.

## Stop

Ctrl-C, `kill`, or the end of training cancels the Cortex job, and the driver
exits within about 2 minutes, once its rollouts stop. A driver killed
with `kill -9` or by the OOM killer cannot cancel it. The server reclaims that
job after 30 idle minutes; set `ARCTIC_CORTEX_IDLE_TIMEOUT_SECONDS` (at least
300) to shorten that.

The [Harbor recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/harbor/README.md)
in Arctic Platform has every setting and the sandbox details.
