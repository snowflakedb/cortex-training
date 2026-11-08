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
| `Qwen/Qwen3.8-Flash-Next` | MoE | standard, long | standard, long | 262,144 |
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
| `thinkingmachines/Inkling-Small` | MoE | standard | standard | — |

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
Qwen3.8-Flash-Next, the Gemma and MedGemma models, or GLM 5.3 Flash BF16, so
pass `renderer_name=` for those; Math GRPO can't run them. Pick a
renderer from tinker-cookbook's
[renderer list](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/#available-renderers)
whose chat template matches the model's own (its `tokenizer_config.json` on
Hugging Face); a mismatched template trains and samples on the wrong format.
Qwen3-4B-Instruct, GLM 5.3, GLM 5.3 Flash, gpt-oss and Inkling Small have only thinking
renderers, so with a thinking-off default pass `enable_thinking=true` or
`renderer_name=`.

## Model notes

- **Inkling Small:** text inference uses the official BF16 checkpoint on eight
  H200 GPUs, with a 32,768-token context. Prompts use vLLM's native `inkling`
  tokenizer mode and renderer. The LoRA sampling preset uses 1,024 tokens to
  match training, with served projections `wq_du` and `wv_dv`. It sets
  `INKLING_MULTIMEM_AR=0` so fused decode collectives preserve LoRA. Router replay
  is disabled in both LoRA presets.

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
- **Inkling Small:** text LoRA SFT uses the `arctic` provider, eight H200s,
  expert parallelism of eight, and ZeRO-2. The initial training preset targets
  `q_proj` and `v_proj` with rank four, and supports 1,024 tokens. Eager attention
  preserves Inkling's relative-position bias; sequence parallelism and fused
  LM-head loss are not supported. This preset requires the Inkling training
  loader in the `arctic` provider. Four eight-GPU nodes can use expert parallelism
  of eight with `n_gpus`, `train_batch_size`, and `ds_config.train_batch_size` set
  to 32.
  A local four-node run completed three SFT steps and exported matching updated
  adapters on all 32 ranks. A local gateway run completed three GRPO iterations
  with 24 training GPUs and eight sampling GPUs, including LoRA synchronization
  after each update and changed sampler logprobs. A cold LoRA resume restored
  every saved optimizer field and tensor exactly on all 24 ranks, with the same
  next-step loss, gradient norm and adapters as uninterrupted training.
  Trainer/sampler logprob parity remains a failing check at `atol=1e-3`, `rtol=0`.
  The full-parameter SFT preset uses 24 H200s, EP8/SP1, a global batch of 24,
  1,024 tokens, and ZeRO-2 with CPU optimizer offload. A local run completed
  three short updates and a 1,024-token update; dense weights matched across
  all 24 ranks and expert weights across their three replicas. Allow about
  1.6 TB of anonymous host memory per eight-GPU training node, with additional
  headroom for checkpoint I/O. The full SFT preset's context is smaller than
  the full inference preset's 32,768-token context, so composing them for RL
  requires matching context overrides. Full sync preserved every stored
  sampler tensor exactly against a fresh official checkpoint load on all
  eight sampling ranks.
  These are local validations; hosted validation timestamps remain unset.
  The native vision, audio, and
  multi-token prediction heads are omitted from text SFT.
