# Key Concepts

Quick reference for Cortex Training-specific terms. For general ML concepts
(SFT, RL, GRPO, LoRA), see the relevant framework documentation.

## Glossary

### 1. Job

A training run on Cortex Training. A job contains one or more sub-jobs and
selects a single GPU hardware type shared across all sub-jobs. For example, an
RL workflow requires a job with both a training sub-job (for weight updates)
and a sampling sub-job (for generating rollouts).
[Detail →](jobs-and-subjobs.md)

### 2. Sub-job

A unit of work within a job. Types include `training` (forward/backward passes
and optimizer steps), `sampling` (generating text via vLLM), and
`log-probability` (scoring completions). For example, a simple SFT workflow
needs only a training sub-job, while RL needs both training and sampling.
[Detail →](jobs-and-subjobs.md)

### 3. Recipe

A self-contained, runnable training workflow in this repository. Each recipe
includes a Python entry point, configuration files, and a README with expected
results. For example, `recipes/rl/math_grpo` runs GRPO reinforcement learning
on the MATH dataset end to end. See the
[recipe catalog](../../recipes/README.md).

### 4. Endpoint

The Cortex Training API on your Snowflake account. It must be enabled by
Snowflake before you can submit jobs. The endpoint is account-scoped, meaning
it works with any database and schema on your account.

### 5. Capacity

The GPU pool available to your account. Each hardware type (H200, B200, B300)
has its own pool. Run `cortex-training capacity` to check what is available
before submitting a job.

### 6. PAT (Programmatic Access Token)

A token generated in the Snowflake UI that authenticates CLI and SDK requests.
It is tied to your user account and the roles you select when creating it. See
[setup](../getting-started/setup.md) for how to create one.

### 7. Connection Config

A JSON file containing your account hostname, PAT, database, and schema. The
CLI reads this file after you run `cortex-training login`. For example,
`~/cortex-training-config.json`. See [setup](../getting-started/setup.md) for
the template.

### 8. Checkpoint

A saved snapshot of model weights during or after training. Cortex Training
supports resumable training checkpoints (which include optimizer state) and
sampling weights (which are synced to the sampling sub-job for generation).
[Detail →](checkpoints.md)

### 9. Hardware

The GPU type for a job. Set once at job creation and shared across all
sub-jobs. Available types include H200, B200, and B300. Run
`cortex-training capacity --hardware H200` to check a specific type.
[Detail →](hardware.md)

### 10. Backend

In this ecosystem, "backend" can mean two things: the compute backend (Arctic
Platform, which handles model loading, training, and sampling) or the service
backend (the Cortex Training API that manages GPU infrastructure). When in
doubt, context determines which is meant.

### 11. Arctic Platform

The open-source compute engine that powers Cortex Training. When you submit a
job, Arctic Platform is the code that runs on the GPUs — handling model
loading, loss computation, weight synchronization, and inference. See the
[ecosystem overview](../getting-started/ecosystem.md) for how it relates to
this repository.
