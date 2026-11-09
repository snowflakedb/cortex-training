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
  H200 GPUs, configured for 262,144 tokens in full mode and 65,536 in LoRA mode.
  These context lengths have not been live validated. Prompts use vLLM's native
  `inkling` tokenizer mode and renderer. The LoRA sampling preset matches the
  training window, with served projections `wq_du` and `wv_dv`. It sets
  `INKLING_MULTIMEM_AR=0` so fused decode collectives preserve LoRA. Router replay
  is disabled in both training/sampling pairs. Sampling uses chunked prefill
  with an 8,192-token scheduling budget, independent of the context limit.

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
  expert parallelism of eight, and ZeRO-2. The LoRA preset uses rank four for
  `q_proj` and `v_proj` and a configured window of 65,536 tokens. Training uses
  FlexAttention with Triton and query chunks of 2,048 tokens, preserving Inkling's
  learned relative-position bias and its gradients. The native LM-head weights
  are scored in 512-token chunks; dense/shared MLPs use 2,048-token chunks and
  expert dispatch uses 4,096-token chunks. Sequence parallelism remains unsupported.
  This preset requires the updated Inkling training loader and PyTorch 2.13 or
  newer with differentiable FlexAttention score-mod buffers. Four eight-GPU nodes
  can use expert parallelism of eight with `n_gpus`, `train_batch_size`, and
  `ds_config.train_batch_size` set to 32.
  Earlier local validation at 1,024 tokens completed three SFT steps and
  exported matching updated adapters on all 32 ranks. A local gateway run
  completed three GRPO iterations with 24 training GPUs and eight sampling
  GPUs, including LoRA synchronization
  after each update and changed sampler logprobs. A cold LoRA resume restored
  every saved optimizer field and tensor exactly on all 24 ranks, with the same
  next-step loss, gradient norm and adapters as uninterrupted training.
  Trainer/sampler logprob parity remains a failing check at `atol=1e-3`, `rtol=0`.
  The full-parameter SFT preset uses 64 H200s across eight eight-GPU nodes,
  EP16/SP1, a global batch of 64, microbatch one, 262,144 tokens, and ZeRO-2
  without optimizer offload. Full block checkpoints offload activations to CPU,
  requiring substantial host-memory capacity. Adam state and gradient accumulation
  stay in FP32. EP16 places 16 of the 256 experts on each GPU, with four replicas of
  each expert shard. Checkpoint-based memory accounting estimates about
  104 GiB per GPU before activations and communication workspace; this estimate
  does not establish that 262,144-token training fits. This eight-node
  configuration has not been live validated.
  Cross-node DeepEP requires a supported GPU RDMA transport. On EFA, use
  `prime_rl.ep_comm_backend="uccl"` with an EFA-enabled UCCL runtime. A two-node,
  16-rank UCCL dispatch/combine probe matched reference outputs, input gradients,
  and routing-score gradients exactly. A four-node, 32-GPU EP16/UCCL LoRA SFT
  run completed correctly aligned updates at 1,024 and 65,536 tokens with
  activation offload and no optimizer offload. Its first 262,144-token batch
  ran out of GPU memory during expert backward recomputation at 134.4 GiB
  allocated. Chunked expert combines now release their queued communication
  buffers between chunks; the follow-up run exactly reproduced both shorter
  losses and gradient norms. End-to-end 262,144-token validation is pending.
  H200 component validation completed 262,144-token BF16 forward/backward for both
  global and sliding-window attention with the checkpoint's head geometry and
  nonzero learned-bias gradients. Global attention peaked at 16.3 GiB allocated
  (22.2 GiB reserved). The real-width, real-vocabulary chunked LM head also
  completed 262,144-token forward/backward, peaking at 16.2 GiB allocated.
  These are component measurements, not a full-model memory bound. Reduced-model
  full and LoRA steps with activation offload matched eager loss, gradient norm
  and gradients at `atol=1e-6`, `rtol=0`. BF16 left-padding, right-padding, and
  packed-sequence checks matched eager loss, gradient norm, and every trainable
  gradient at `atol=1e-3`, `rtol=0`. Full and LoRA checkpoint/resume reproduced the
  next update with the chunked backend. The eight-node full configuration still
  needs end-to-end validation at its configured context.
  Earlier local full SFT validation used 24 H200s with EP8/SP1 and CPU
  optimizer offload. It completed three short updates and a 1,024-token update;
  dense weights matched across all 24 ranks and expert weights across their
  three replicas. Both full-mode presets now have matching 262,144-token limits
  for RL composition. Initial full sync preserved every stored sampler tensor
  exactly against a fresh official checkpoint load on all eight sampling ranks.
  Cold full resume restored every parameter and optimizer tensor exactly on
  all 24 ranks; the next loss, gradient norm and updated parameters also matched
  uninterrupted training exactly.
  Trained full-weight sync matched sender checksums on all eight sampling
  ranks, with no skipped source tensors.
  These are local validations; hosted validation timestamps remain unset.
  The native vision, audio, and
  multi-token prediction heads are omitted from text SFT.
