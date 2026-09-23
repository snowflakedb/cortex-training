# Ecosystem Overview

## Quick Intro

This page gives you a quick orientation to the projects you'll encounter while
using Cortex Training.

**Cortex Training** (this repository) is the client SDK and CLI for
post-training large language models on Snowflake's managed infrastructure. You
write recipes, submit jobs, and monitor progress from here.

**Arctic Platform** is the open-source compute engine behind the service. When
you submit a training or sampling job, Arctic Platform is the code that runs
it — handling model loading, loss computation, weight synchronization, and
inference.

**Open-source RL frameworks** like SkyRL, VERL, and TRL can drive their own
training loops while delegating the heavy GPU work to Arctic Platform. If you
are coming from one of these frameworks, your recipes run largely
unchanged — the integration swaps the compute backend, not the training logic.

## How It Fits Together

There are two ways to run training with Arctic Platform:

**Self-hosted**: you run Arctic Platform directly on your own GPUs. The
framework drives the loop and Arctic Platform handles the compute locally.
Cortex Training is not involved. See the
[Arctic Platform documentation](https://github.com/Snowflake-AI-Research/Arctic-Platform)
for this path.

**Managed**: you use Snowflake's infrastructure instead of your own GPUs. The
framework still drives the loop, but Arctic Platform routes the GPU work to the
Cortex Training service rather than running it locally. You use this
repository's SDK to set up credentials and capacity, and the service handles
the rest. The remainder of this page covers this path.

## Managed Flow

```
┌──────────────────────────────────────────────────────────────┐
│  Your Machine (CPU only)                                     │
│                                                              │
│  ┌───────────────┐    ┌──────────────────────────────────┐   │
│  │   Framework   │───▶│  Arctic Platform                 │   │
│  │ (SkyRL / VERL │    │  (Cortex dispatch shim)          │   │
│  │  / TRL)       │    │  Translates framework calls into │   │
│  │               │    │  Cortex Training API requests    │   │
│  │  Drives the   │    └──────────────┬───────────────────┘   │
│  │  training     │                   │                       │
│  │  loop         │                   │                       │
│  └───────────────┘                   │                       │
│                                      │ HTTPS                 │
└──────────────────────────────────────┼───────────────────────┘
                                       ▼
                   ┌───────────────────────────────────────────┐
                   │  Cortex Training Service                  │
                   │  (Snowflake-managed GPU infrastructure)   │
                   │                                           │
                   │  Arctic Platform compute engine runs      │
                   │  here: model loading, DeepSpeed training, │
                   │  vLLM sampling, weight sync               │
                   └───────────────────────────────────────────┘
```

1. Your **framework** (SkyRL, VERL, or TRL) runs the training loop on your
   machine — generating rollouts, computing rewards, deciding when to train and
   when to sample. No GPU needed locally.
2. **Arctic Platform's dispatch shim** translates each framework call into an
   HTTP request to the Cortex Training API — reformatting data and managing the
   connection.
3. The **Cortex Training service** allocates GPUs, runs the same Arctic Platform
   compute engine that powers the self-hosted path, and returns results.

This repository covers everything on the client side — SDK, CLI, credentials,
recipes, and job management. To get started, see [Setup](setup.md).

## Framework Integration Status

| Framework | Managed (Cortex) | Guide |
|---|---|---|
| SkyRL | Available | [SkyRL integration](../integrations/skyrl.md) |
| VERL | Not yet available | — |
| TRL | Not yet available | — |
