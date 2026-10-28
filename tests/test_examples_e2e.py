"""Run the examples end to end against a live Cortex Training deployment.

Skipped unless CORTEX_TRAINING_E2E_CONFIG names a connection file, the JSON the
examples take as config= (docs/getting-started/connection.json.template). Each
case runs one example as its README does, cut to a few steps, and passes when
it exits cleanly. The DARE-Bench evaluation first builds the setup its README
describes (a bench checkout, a sandbox venv, and reference predictions), which
needs network access. These submit real jobs on real GPUs:

    CORTEX_TRAINING_E2E_CONFIG=~/conn.json python -m pytest tests/test_examples_e2e.py -v -s

If DEBUG_IMAGE_TAG is set, every example also gets debug_image_tag=<tag>.

Each case's output streams live, is written beside its log_path, and is repeated
in the failure if it fails -- see :func:`_run_example` for why all three.
"""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = os.environ.get("CORTEX_TRAINING_E2E_CONFIG")
DEBUG_IMAGE_TAG = os.environ.get("DEBUG_IMAGE_TAG")
# Per example. CI runs all of them within a fixed total budget.
TIMEOUT_SECONDS = 3600
# Repeated from the end of a failing example's output. A server-side traceback arrives
# innermost-last, so the tail is the part worth reading; 400 lines covers a nested one.
FAILURE_TAIL_LINES = 400

# The eval tasks, sandbox packages, and reference predictions examples/sft/dare/README.md sets up.
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
        "examples.sft.conversational.train",
        ["max_steps=2", "max_seq_len=1024", "train_batch_size=4", "log_path={log_path}"],
    ),
    # LoRA on a hybrid-attention model, which no other case covers: the other LoRA
    # runs train a plain-attention dense model or a sequence-parallel MoE one, so a
    # regression in how adapters compose with hybrid attention passes the rest of the
    # suite. Trains on 8 data-parallel GPUs, and the server rejects a smaller batch.
    "sft_lora_qwen35": (
        "examples.sft.conversational.train",
        ["model=Qwen/Qwen3.5-9B", "lora=true", "max_steps=2", "max_seq_len=1024",
         "train_batch_size=8", "log_path={log_path}"],
    ),
    "rl": (
        "examples.rl.math_grpo.train",
        ["max_steps=2", "problems_per_batch=4", "group_size=4", "max_tokens=512",
         "eval_every=1", "n_test=16", "eval_max_tokens=512", "log_path={log_path}"],
    ),
    "generate": (
        "examples.inference.generate",
        ["max_tokens=64"],
    ),
    "evaluate": (
        "examples.inference.evaluate",
        ["task=gsm8k", "max_examples=16", "max_tokens=512", "temperature=0"],
    ),
    "serve": (
        "examples.inference.serve",
        ["keep_job=false"],
    ),
    "distillation": (
        "examples.distillation.on_policy.train",
        ["max_steps=2", "problems_per_batch=4", "group_size=2", "max_tokens=256",
         "eval_max_tokens=256", "eval_every=1", "n_val=16", "n_test=32",
         "n_kl_eval=8", "warmup_steps=0", "log_path={log_path}"],
    ),
    # Two steps, with valid_nll before and after. DARE trains on 8 data-parallel
    # GPUs, and the server rejects a batch smaller than that.
    "dare_train": (
        "examples.sft.dare.train",
        ["max_steps=2", "train_batch_size=8", "eval_every=2", "log_path={log_path}"],
    ),
    # Base model on two v1 (reference-scored) tasks.
    "dare_evaluate": (
        "examples.sft.dare.evaluate",
        ["bench_dir={bench_dir}", "sandbox_python={sandbox_python}", "versions=v1",
         f"task_types={DARE_TASK_TYPE}", "max_examples=2", "log_path={log_path}"],
    ),
    # Heaviest case (MoE, long context, ~16 GPUs), so it runs last.
    "rl_moe": (
        "examples.rl.math_grpo.train",
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
    _run_setup(
        ["git", "clone", "--quiet", "--filter=blob:none", DARE_BENCH_URL, str(checkout)],
        step="clone", root=root, timeout=600, env=git_env,
    )
    _run_setup(
        ["git", "-C", str(checkout), "checkout", "--quiet", DARE_BENCH_COMMIT],
        step="checkout", root=root, timeout=600, env=git_env,
    )

    sandbox = root / "sandbox"
    _run_setup([sys.executable, "-m", "venv", str(sandbox)], step="venv", root=root, timeout=300)
    python = sandbox / "bin" / "python"
    _run_setup(
        [str(python), "-m", "pip", "install", "--quiet", *DARE_SANDBOX_PACKAGES],
        step="pip-install", root=root, timeout=1200,
    )

    # The script writes each task's verify/simulated_pred_local.csv, which the evaluator reads.
    bench_dir = checkout / "data" / "eval"
    questions = json.loads((bench_dir / "question_list.json").read_text(encoding="utf-8"))
    question_list = root / "question_list.json"
    tasks = [q for q in questions if q["task"] == DARE_TASK_TYPE]
    question_list.write_text(json.dumps(tasks), encoding="utf-8")
    _run_setup(
        [str(python), str(checkout / "scripts" / "reference_solution.py"),
         "--databases_dir", str(bench_dir / "databases"), "--question_list", str(question_list),
         "--output_dir", str(root / "reference"), "--execute"],
        step="reference-solution", root=root, timeout=1800,
    )
    # The script counts a failed task and still exits 0, so check its outputs here.
    missing = [q["file_path"] for q in tasks
               if not (bench_dir / "databases" / q["file_path"] / "verify" / "simulated_pred_local.csv").is_file()]
    assert not missing, f"reference_solution.py wrote no reference for {missing}"
    return {"bench_dir": bench_dir, "sandbox_python": python}


def _run_example(
    command: list[str], *, log_file: Path, timeout: float, env: dict[str, str] | None = None
) -> tuple[int | None, str]:
    """Run ``command`` to completion, keeping every byte it writes.

    Returns its exit status and output, or ``(None, output)`` if ``timeout``
    expired and it was killed. stderr is merged into stdout, so the two cannot
    interleave and lose part of a traceback that spans both.

    The output is read here rather than left on the inherited descriptors. An
    example that fails writes its last and largest block as it exits -- a server
    error carries the remote traceback inside the message -- while the test
    runner writes that case's result to the same descriptor. Two writers on one
    pipe clip a block that large, which cost us the innermost frames of a worker
    failure. Holding the bytes here means the copy quoted in the failure is
    written once, by the runner, after this process is reaped. The live echo is
    still best-effort and may be clipped; the quoted copy is the record.
    """
    captured: list[bytes] = []
    timed_out = False
    with subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Unbuffered, because merging the streams only orders them if the example
        # does not hold stdout back: a pipe makes it block-buffered, so it would
        # flush at exit and land after the traceback that stderr already wrote.
        # That ordering decides whether the tail quoted on failure holds the
        # traceback or the last few hundred lines of training progress.
        env={**(env if env is not None else os.environ), "PYTHONUNBUFFERED": "1"},
    ) as process:
        assert process.stdout is not None

        def pump() -> None:
            # read1 returns what has arrived instead of waiting for a full buffer,
            # so an hour-long case still reports progress as it goes.
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            for chunk in iter(lambda: process.stdout.read1(65536), b""):
                captured.append(chunk)
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        # Drain what is left before the context manager closes the pipe.
        reader.join(timeout=60)

    output = b"".join(captured)
    # Written whole, so a case that fails on a developer's machine leaves the
    # untrimmed output behind even though the failure quotes only the tail.
    log_file.write_bytes(output)
    return (None if timed_out else process.returncode), output.decode("utf-8", errors="replace")


def _tail(output: str, log_file: Path) -> str:
    lines = output.splitlines()
    if len(lines) <= FAILURE_TAIL_LINES:
        return f"--- output ({len(lines)} lines), also in {log_file} ---\n{output}"
    head = f"--- last {FAILURE_TAIL_LINES} of {len(lines)} lines; all of them are in {log_file} ---"
    return "\n".join([head, *lines[-FAILURE_TAIL_LINES:]])


def _run_setup(command: list[str], *, step: str, root: Path, timeout: float, env=None) -> None:
    """Run one DARE setup step, failing with its output rather than only a status.

    ``check=True`` would raise with the exit status alone, leaving whatever the
    step printed about why only in the live stream, which is the copy that gets
    clipped. These steps clone, build a venv, and install from the network, so
    that output is usually the whole explanation.
    """
    log_file = root / f"{step}.console.log"
    returncode, output = _run_example(command, log_file=log_file, timeout=timeout, env=env)
    if returncode == 0:
        return
    status = "timed out" if returncode is None else f"exited with status {returncode}"
    pytest.fail(f"dare_bench setup step {step!r} {status}: {' '.join(command)}\n{_tail(output, log_file)}")


@pytest.mark.skipif(CONFIG is None, reason="set CORTEX_TRAINING_E2E_CONFIG to a connection file to run the examples")
@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_runs(name, tmp_path, request):
    config = Path(CONFIG).expanduser()
    # Setting the variable opts in, so a value that names no file fails rather than skips.
    assert config.is_file(), f"CORTEX_TRAINING_E2E_CONFIG is set to {CONFIG!r}, which is not a file"

    module, settings = EXAMPLES[name]
    values = {"log_path": tmp_path}
    if name == "dare_evaluate":
        # Requested here rather than as an argument, so only this case builds it.
        values.update(request.getfixturevalue("dare_bench"))
    command = [sys.executable, "-m", module, f"config={config.resolve()}"]
    for setting in settings:
        # replace, not str.format, so a setting with a literal brace survives.
        for key, value in values.items():
            setting = setting.replace("{" + key + "}", str(value))
        command.append(setting)
    if DEBUG_IMAGE_TAG:
        command.append(f"debug_image_tag={DEBUG_IMAGE_TAG}")

    log_file = tmp_path / f"{name}.console.log"
    returncode, output = _run_example(command, log_file=log_file, timeout=TIMEOUT_SECONDS)
    if returncode is None:
        # The example has usually submitted a job already, which may still be running.
        pytest.fail(
            f"{name} did not finish within {TIMEOUT_SECONDS}s: {' '.join(command)}\n"
            f"{_tail(output, log_file)}"
        )

    assert returncode == 0, (
        f"{name} exited with status {returncode}: {' '.join(command)}\n{_tail(output, log_file)}"
    )
