# Model Compatibility

The [model catalog](../../model-catalog/README.md) is the source of truth for
which models Cortex Training supports: a model supports a combination of
context, GPU type, training or inference, and LoRA or full weights when the
catalog has a job file for it. Every file today is for H200 GPUs. Treat GPU
counts and optimization values as starting points that might need adjustment
for a specific workload.

| Model | Architecture | Training | Inference | Long-context `max_seq_len` |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | dense | standard | standard | — |
| `Qwen/Qwen3-1.7B` | dense | standard | standard | — |
| `Qwen/Qwen3-4B-Instruct-2507` | dense | standard, long | standard, long | 262,144 |
| `Qwen/Qwen3-8B` | dense | standard | standard | — |
| `Qwen/Qwen3.5-2B` | dense | standard, long | standard, long | 262,144 |
| `Qwen/Qwen3.5-4B` | dense | standard, long | standard, long | 262,144 |
| `Qwen/Qwen3.5-9B` | dense | standard, long | standard, long | 262,144 |
| `Qwen/Qwen3.8-27B` | dense | standard, long | standard, long | 262,144 |
| `Qwen/Qwen3-30B-A3B` | MoE | standard | standard | — |
| `Qwen/Qwen3.6-35B-A3B` | MoE | standard, long | standard, long | 262,144 |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | MoE | standard, long | standard, long | 262,144 |
| `openai/gpt-oss-120b` | MoE | — | long | 131,072 |
| `google/gemma-3-4b-it` | dense | standard, long | standard, long | 131,072 |
| `google/gemma-4-31B-it` | dense | standard, long | standard, long | 262,144 |
| `google/medgemma-1.5-4b-it` | dense | standard, long | standard, long | 131,072 |
| `google/medgemma-4b-it` | dense | standard, long | standard, long | 131,072 |
| `zai-org/GLM-5.3` | MoE | — | standard, long | 720,896 |
| `zai-org/GLM-5.3-Flash` | MoE | standard, long | standard, long | 262,144 |
| `zai-org/GLM-5.3-Flash-BF16` | MoE | standard, long | standard, long | 262,144 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | MoE | standard, long | standard, long | 262,144 |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | MoE | standard, long | standard, long | 262,144 |
| `thinkingmachines/Inkling-2-Small-MXFP8` | MoE | standard, long | standard, long | 262,144 |

`standard` and `long` are each model's smaller and larger `max_seq_len` tiers,
so the boundary differs between models. Most trainable models have LoRA and full
files; some have only one, and inference-only models have full-weight files. The
model's catalog folder lists exactly which files it has. The examples pick a file with
`model=`, `context=`, `hardware=` and `lora=` (see
[Run an example](../../recipes/README.md#run-an-example)).

The examples that tokenize prompts pick a chat renderer from tinker-cookbook
(`recipes.inference.serve` and distillation need none):

- Conversational SFT and the inference `generate`/`evaluate` commands default
  to thinking off, and accept `enable_thinking=` and `renderer_name=`. DARE
  defaults to thinking on.
- Math GRPO takes no renderer arguments and uses tinker-cookbook's top
  recommendation for the model, which is thinking on for hybrid Qwen3 models.
- Distillation takes no renderer arguments: it renders each conversation with
  the model's own chat template from its tokenizer, thinking off, so it also
  runs models that tinker-cookbook has no renderer for.

tinker-cookbook lists no renderer for Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-2B,
the Gemma and MedGemma models, or GLM 5.3 Flash BF16, so
pass `renderer_name=` for those; Math GRPO can't run them. Pick a
renderer from tinker-cookbook's
[renderer list](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/#available-renderers)
whose chat template matches the model's own (its `tokenizer_config.json` on
Hugging Face); a mismatched template trains and samples on the wrong format.
Qwen3-4B-Instruct, GLM 5.3, GLM 5.3 Flash and gpt-oss have only thinking
renderers, so with a thinking-off default pass `enable_thinking=true` or
`renderer_name=`. No tinker-cookbook renderer matches Inkling 2 Small's
tokenizer, so the examples that tokenize prompts don't run it; build its
prompts with its own tokenizer.

## Model notes

- **Inkling 2 Small:** LoRA training uses the `arctic` provider and
  FlexAttention on 32 H200s across four nodes (EP32/SP8, ZeRO-2), with
  rank-16 adapters on the attention `wq_du`/`wv_dv` projections and the routed
  experts (`experts.w1`/`w2`/`w3`). Expert parallelism uses UCCL with a
  2,048-token MoE chunk, and the LM head is chunked at 2,048 tokens with
  `prime_rl.fused_cross_entropy=false`. The `long` tier offloads activations to
  CPU. Sampling serves the MXFP8 checkpoint with TP8 on eight H200s and the
  `inkling` tokenizer, reasoning and tool-call parsers; LoRA sampling runs
  eager with prefix caching off. Router replay is disabled, and there are no
  full-parameter training files.

- **Gemma 4 31B:** both training tiers use SDPA. Full-attention layers have head
  dimension 512, which FlashAttention 3 rejects. Gemma 3 and MedGemma keep
  FlashAttention 3; every one of their layers is 256.
- **DeepSeek V4.1-Flash:** full-parameter and attention LoRA train from the
  official checkpoint. LoRA inference records routes on every layer
  (`swa_bounded_replay` off). The sampler names the attention projections
  `wq_a`, `wkv`, `wq_b`, and `wo_b`. tinker-cookbook renders it as
  `deepseekv4_1`, and `deepseekv4_1_disable_thinking` with thinking off.
- **Qwen3.6-35B-A3B:** LoRA training and checkpoint save/load require the
  `arctic` model provider, and LoRA is limited to `q_proj`, `k_proj`,
  `v_proj`, and `o_proj`. Routed-expert LoRA is not yet supported.
