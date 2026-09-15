# API Request Examples

These JSON files demonstrate individual Cortex Training request bodies:

| File | Purpose |
|---|---|
| `training.json` | Create a training sub-job |
| `sampling.json` | Create a sampling sub-job |
| `rl.json` | Create colocated training and sampling sub-jobs |
| `fwd-bwd.json` | Submit a readable forward/backward batch |
| `generate.json` | Submit prompts for generation |
| `glm-5.3-sampling.json` | Create a single-node GLM-5.3 sampling sub-job on 8 H200 GPUs with an FP8 MLA KV cache and a reasoning parser |

They are wire-format examples, not end-to-end training recipes.
