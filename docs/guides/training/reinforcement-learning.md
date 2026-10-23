# Reinforcement Learning

The [Math GRPO recipe](../../../recipes/rl/README.md) is the current
end-to-end RL example. It demonstrates:

- Colocated training and sampling sub-jobs
- Grouped rollout generation and reward centering
- GRPO loss configuration
- Training-to-sampling weight synchronization
- Held-out MATH-500 evaluation

## Integrated RL frameworks

[SkyRL](../../integrations/skyrl.md) can also run GRPO against Cortex Training,
with its trainer on a CPU driver and training and sampling in Cortex sub-jobs.
Its GSM8K example lives in the Arctic Platform repository and has its own
install.

[prime-rl](../../integrations/prime-rl.md) runs GRPO from its own orchestrator,
driven by the Arctic Platform launcher, with training and sampling in Cortex
sub-jobs. Its math (`i3_math`) and SWE (`r2e-gym`) example recipes live in the
Arctic Platform repository.

To train a model inside an agent you already have, see
[Bring your own agent](../../integrations/bring-your-own-agent.md). The agent
keeps its own loop and sends model calls to a per-rollout endpoint; SkyRL and
Cortex do the training. [OpenHands](../../integrations/openhands.md) is the
worked example.

Additional code, tool-use, and multi-agent RL recipes are planned.
