"""Run the examples end to end against a live Cortex Training deployment.

Connects the way a customer does: through the Snowflake connection profile that
`cortex-training login` saves (in $SNOWFLAKE_HOME, else ~/.snowflake). Examples
run in a child process that inherits SNOWFLAKE_HOME, so a throwaway folder
holding one profile works too. Skipped unless you opt in with one of:

- CORTEX_TRAINING_E2E=1: the examples' default connection (the profile saved by
  `cortex-training login`, else CORTEX_TRAINING_HOST and CORTEX_TRAINING_PAT
  when no profile exists, as in CI).
- CORTEX_TRAINING_E2E_CONNECTION=NAME: a named profile, passed to each example
  as connection=NAME.
- CORTEX_TRAINING_E2E_CONFIG=FILE: an old JSON connection file, passed as
  config=FILE. It will be deprecated in a future release. If this and
  CORTEX_TRAINING_E2E_CONNECTION are both set, the profile is used.

Each case runs one example as its README does, cut to a few steps, and passes
when it exits cleanly. The DARE-Bench evaluation first builds the setup its
README describes (a bench checkout, a sandbox venv, and reference predictions),
which needs network access. These submit real jobs on real GPUs:

    CORTEX_TRAINING_E2E=1 python -m pytest tests/test_examples_e2e.py -v -s

If DEBUG_IMAGE_TAG is set, every example also gets debug_image_tag=<tag>.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTION = os.environ.get("CORTEX_TRAINING_E2E_CONNECTION")
CONFIG = os.environ.get("CORTEX_TRAINING_E2E_CONFIG")
USE_DEFAULT = os.environ.get("CORTEX_TRAINING_E2E", "").lower() in ("1", "true", "yes")
DEBUG_IMAGE_TAG = os.environ.get("DEBUG_IMAGE_TAG")
# The environment the run started with, read at import, before any fixture runs.
# conftest points SNOWFLAKE_HOME at an empty folder for every test, so the
# examples get this one instead and find the profiles a customer would.
RUN_ENV = dict(os.environ)
# Per example. CI runs all of them within a fixed total budget.
TIMEOUT_SECONDS = 3600

# The eval tasks, sandbox packages, and reference predictions recipes/sft_quickstart/dare/README.md sets up.
DARE_BENCH_URL = "https://github.com/Snowflake-Labs/dare-bench"
DARE_BENCH_COMMIT = "01447145304c67b861a004ada6d86f29640de61a"
DARE_SANDBOX_PACKAGES = [
    "pandas==2.2.3", "numpy==2.2.6", "scikit-learn==1.6.1", "scipy", "pyarrow", "joblib",
    "xgboost", "lightgbm", "statsmodels",
]
# The evaluator loads every task of the types it runs, so references are built for this type only.
DARE_TASK_TYPE = "regression"

# Module and the settings that keep the run short; {log_path} becomes a fresh directory,
# and {bench_dir} and {sandbox_python} come from the dare_bench fixture.
EXAMPLES = {
    "sft": (
        "recipes.sft_quickstart.conversational.train",
        ["max_steps=2", "max_seq_len=1024", "train_batch_size=4", "log_path={log_path}"],
    ),
    # The only LoRA run on a hybrid-attention model: the others are dense plain-attention
    # or sequence-parallel MoE. 8 data-parallel GPUs, so the server needs batch 8.
    "sft_lora_qwen35": (
        "recipes.sft_quickstart.conversational.train",
        ["model=Qwen/Qwen3.5-9B", "lora=true", "max_steps=2", "max_seq_len=1024",
         "train_batch_size=8", "log_path={log_path}"],
    ),
    "custom_loss": (
        "recipes.sft_quickstart.custom_loss.train",
        ["max_steps=2", "max_seq_len=1024", "train_batch_size=4", "log_path={log_path}"],
    ),
    "rl": (
        "recipes.rl_quickstart.math_grpo.train",
        ["max_steps=2", "problems_per_batch=4", "group_size=4", "max_tokens=512",
         "eval_every=1", "n_test=16", "eval_max_tokens=512", "log_path={log_path}"],
    ),
    "generate": (
        "recipes.inference.generate",
        ["max_tokens=64"],
    ),
    "evaluate": (
        "recipes.inference.evaluate",
        ["task=gsm8k", "max_examples=16", "max_tokens=512", "temperature=0"],
    ),
    "serve": (
        "recipes.inference.serve",
        ["keep_job=false"],
    ),
    "distillation": (
        "recipes.distillation.on_policy.train",
        ["max_steps=2", "questions_per_step=4", "group_size=2", "n_eval=10",
         "eval_samples=1", "eval_every=1", "warmup_steps=0", "log_path={log_path}"],
    ),
    # Two steps, with valid_nll before and after. DARE trains on 8 data-parallel
    # GPUs, and the server rejects a batch smaller than that.
    "dare_train": (
        "recipes.sft_quickstart.dare.train",
        ["max_steps=2", "train_batch_size=8", "eval_every=2", "log_path={log_path}"],
    ),
    # Base model on two v1 (reference-scored) tasks.
    "dare_evaluate": (
        "recipes.sft_quickstart.dare.evaluate",
        ["bench_dir={bench_dir}", "sandbox_python={sandbox_python}", "versions=v1",
         f"task_types={DARE_TASK_TYPE}", "max_examples=2", "log_path={log_path}"],
    ),
    # Sixteen UltraChat conversations answered again by the target, then split.
    "dflash2_data": (
        "recipes.dflash2.generate_data",
        ["out_dir={log_path}", "num_conversations=16", "heldout=4"],
    ),
    # Two steps of draft training on the bundled sample, then the draft served with its
    # target and its acceptance length measured on eight of the same prompts.
    "dflash2": (
        "recipes.dflash2.train",
        ["data_path=recipes/dflash2/data/sample.jsonl", "max_steps=2", "train_batch_size=8",
         "eval_prompts=recipes/dflash2/data/sample.jsonl", "eval_num_prompts=8", "eval_max_tokens=128",
         "log_path={log_path}"],
    ),
    # Heaviest case (MoE, long context, ~16 GPUs), so it runs last.
    "rl_moe": (
        "recipes.rl_quickstart.math_grpo.train",
        ["model=Qwen/Qwen3.6-35B-A3B", "context=long", "lora=true", "max_steps=2",
         "problems_per_batch=4", "group_size=4", "max_tokens=2048", "eval_every=1",
         "n_test=16", "eval_max_tokens=2048", "log_path={log_path}"],
    ),
}


@pytest.fixture
def dare_bench(tmp_path_factory):
    """Build the DARE-Bench evaluation setup from the example's README."""
    root = tmp_path_factory.mktemp("dare")
    checkout = root / "dare-bench"
    # Ignore the user's git config, so a URL rewrite or credential helper cannot redirect the clone.
    # Timeouts are several times the measured setup, so a hang fails this case instead of the whole run.
    git_env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", DARE_BENCH_URL, str(checkout)],
        env=git_env, check=True, timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--quiet", DARE_BENCH_COMMIT], env=git_env, check=True, timeout=600
    )

    sandbox = root / "sandbox"
    subprocess.run([sys.executable, "-m", "venv", str(sandbox)], check=True, timeout=300)
    python = sandbox / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", *DARE_SANDBOX_PACKAGES], check=True, timeout=1200)

    # The script writes each task's verify/simulated_pred_local.csv, which the evaluator reads.
    bench_dir = checkout / "data" / "eval"
    questions = json.loads((bench_dir / "question_list.json").read_text(encoding="utf-8"))
    question_list = root / "question_list.json"
    tasks = [q for q in questions if q["task"] == DARE_TASK_TYPE]
    question_list.write_text(json.dumps(tasks), encoding="utf-8")
    subprocess.run(
        [str(python), str(checkout / "scripts" / "reference_solution.py"),
         "--databases_dir", str(bench_dir / "databases"), "--question_list", str(question_list),
         "--output_dir", str(root / "reference"), "--execute"],
        check=True,
        timeout=1800,
    )
    # The script counts a failed task and still exits 0, so check its outputs here.
    missing = [q["file_path"] for q in tasks
               if not (bench_dir / "databases" / q["file_path"] / "verify" / "simulated_pred_local.csv").is_file()]
    assert not missing, f"reference_solution.py wrote no reference for {missing}"
    return {"bench_dir": bench_dir, "sandbox_python": python}


def connection_args() -> list[str]:
    """The example settings that choose the connection; empty means the examples' default."""
    # A profile wins over the old JSON file, so an environment can provide both for old and new clients.
    if CONNECTION:
        return [f"connection={CONNECTION}"]
    if CONFIG:
        config = Path(CONFIG).expanduser()
        # Setting the variable opts in, so a value that names no file fails rather than skips.
        assert config.is_file(), f"CORTEX_TRAINING_E2E_CONFIG is set to {CONFIG!r}, which is not a file"
        return [f"config={config.resolve()}"]
    return []


@pytest.mark.skipif(
    not (CONNECTION or CONFIG or USE_DEFAULT),
    reason="set CORTEX_TRAINING_E2E=1 (or CORTEX_TRAINING_E2E_CONNECTION=NAME) to run the examples against your profile",
)
@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_runs(name, tmp_path, request):
    connection = connection_args()

    module, settings = EXAMPLES[name]
    values = {"log_path": tmp_path}
    if name == "dare_evaluate":
        # Requested here rather than as an argument, so only this case builds it.
        values.update(request.getfixturevalue("dare_bench"))
    command = [sys.executable, "-m", module, *connection]
    for setting in settings:
        # replace, not str.format, so a setting with a literal brace survives.
        for key, value in values.items():
            setting = setting.replace("{" + key + "}", str(value))
        command.append(setting)
    if DEBUG_IMAGE_TAG:
        command.append(f"debug_image_tag={DEBUG_IMAGE_TAG}")

    try:
        result = subprocess.run(command, cwd=REPO_ROOT, env=RUN_ENV, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # The example has usually submitted a job already, which may still be running.
        pytest.fail(f"{name} did not finish within {TIMEOUT_SECONDS}s: {' '.join(command)}")

    assert result.returncode == 0, f"{name} exited with status {result.returncode}"
