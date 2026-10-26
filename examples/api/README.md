# API Request Examples

These JSON files demonstrate individual Cortex Training request bodies:

| File | Purpose |
|---|---|
| `training.json` | Create a training sub-job |
| `sampling.json` | Create a sampling sub-job |
| `rl.json` | Create colocated training and sampling sub-jobs |
| `external-weights-training.json` | Create a training sub-job initialized from a Hugging Face model directory uploaded to a stage |
| `fwd-bwd.json` | Submit a readable forward/backward batch |
| `generate.json` | Submit prompts for generation |
| `glm-sampling.json` | Create a 16-GPU GLM-5.2-FP8 sampling sub-job with a 1M-token context and an FP8 MLA KV cache |
| `model-deployment-checkpoint.json` | Create an evaluation model deployment from a weights-only checkpoint (`cortex-training model-deployment create --spec`; not a CreateJob body) |

They are wire-format examples, not end-to-end training recipes.
