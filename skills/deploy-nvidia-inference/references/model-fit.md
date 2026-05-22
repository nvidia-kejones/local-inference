# Model Fit Estimation

Use fit estimates to reject obvious failures and expose assumptions. Do not claim exact runtime memory consumption from this estimator alone.

## Formula Used By V1

For one candidate pair:

```text
weight_bytes = explicit model weight bytes
             or parameters * bytes_per_parameter * storage_factor

live_tokens = max(
  target_context_tokens * expected_concurrent_sequences,
  max_batch_total_tokens,
  expected_live_tokens
)

kv_cache_bytes = kv_cache_bytes_per_token * live_tokens
required_gpu_bytes =
  gpu_resident_weight_bytes
  + kv_cache_bytes
  + batch_workspace_bytes
  + runtime_overhead_bytes

usable_gpu_bytes =
  sum(free_vram_per_selected_gpu - per_gpu_safety_reserve)
```

The safety reserve is the larger of a per-GPU fraction and a per-GPU GiB floor. Runtime overhead is an explicit candidate override or a conservative runtime default plus a fraction of GPU-resident weights and KV cache.

## Candidate Memory Inputs

Prefer explicit data in this order:

1. Actual pinned weight artifact bytes for the deployed precision/quantization.
2. Explicit `weight_bytes` and `kv_cache.bytes_per_token`.
3. Parameter count plus precision/quantization bytes and KV geometry from model config:
   - `num_layers`
   - `hidden_size`
   - `num_attention_heads`
   - `num_key_value_heads` when grouped-query or multi-query attention changes KV size
   - `head_dim` when it is not inferred from hidden size and attention heads
   - KV dtype bytes

`weights.gpu_resident_fraction` exists for offload paths such as llama.cpp. If it is below one, also verify host RAM and latency assumptions.

## Why Parameter Count Is Not Fit

- Quantization and storage metadata change weight bytes.
- KV cache grows with live tokens, context, batching, concurrency, attention geometry, and KV dtype.
- Runtime kernels, graphs, workspaces, allocators, tokenizers, and transient prefill behavior add overhead.
- Tensor parallelism, MIG, topology, model architecture, MoE routing, and partial offload change the memory/performance path.
- Free VRAM is dynamic and allocator fragmentation can matter.

## Interpreting Output

- `fits`: expected to fit after reserve, still benchmark and verify.
- `tight`: close enough that runtime flags, allocator behavior, real prompts, or competing GPU work can flip the result.
- `no_fit`: do not apply without changing model/runtime/precision/context/concurrency/offload assumptions.
- `unknown`: discovery or model memory metadata is too incomplete.

Keep warnings in the recommendation. A candidate with zero KV or zero weight estimate is not a high-confidence fit.
