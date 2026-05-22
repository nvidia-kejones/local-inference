#!/usr/bin/env python3
"""Approximate NVIDIA LLM serving fit estimates for candidate scorecards."""

from __future__ import annotations

import math
from typing import Any

from common_io import as_float, as_int, nested


GIB = 1024**3
PRECISION_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "w8a8": 1.0,
    "int4": 0.5,
    "mxfp4": 0.5,
    "nvfp4": 0.5,
    "awq4": 0.5,
    "gptq4": 0.5,
    "gguf-q4": 0.5,
}
RUNTIME_DEFAULTS = {
    "vllm": {"overhead_gib": 2.0, "overhead_fraction": 0.08},
    "sglang": {"overhead_gib": 2.5, "overhead_fraction": 0.1},
    "tensorrt-llm": {"overhead_gib": 3.0, "overhead_fraction": 0.1},
    "trtllm-serve": {"overhead_gib": 3.0, "overhead_fraction": 0.1},
    "llama.cpp": {"overhead_gib": 1.0, "overhead_fraction": 0.05},
    "ollama": {"overhead_gib": 1.5, "overhead_fraction": 0.08},
}


def estimate_fit(
    host: dict[str, Any], workload: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    warnings: list[str] = []
    runtime = str(candidate.get("runtime", "")).lower()
    target_context = _target_context(workload, candidate)
    sequences = max(
        1,
        as_int(
            nested(workload, "serving", "expected_concurrent_sequences", default=1),
            1,
        ),
    )
    max_batch_tokens = max(
        target_context,
        as_int(
            nested(workload, "serving", "max_batch_total_tokens", default=target_context),
            target_context,
        ),
    )
    live_tokens = max(
        target_context * sequences,
        max_batch_tokens,
        as_int(nested(workload, "serving", "expected_live_tokens", default=0), 0),
    )

    weights_bytes = _weight_bytes(candidate, warnings)
    gpu_weight_fraction = min(
        1.0,
        max(0.0, as_float(nested(candidate, "weights", "gpu_resident_fraction", default=1.0), 1.0)),
    )
    gpu_weights_bytes = int(weights_bytes * gpu_weight_fraction)
    cpu_weights_bytes = max(0, weights_bytes - gpu_weights_bytes)

    kv_bytes_per_token = _kv_bytes_per_token(candidate, warnings)
    kv_cache_bytes = int(kv_bytes_per_token * live_tokens)
    workspace_per_batch_token = as_float(
        nested(candidate, "runtime_estimates", "batch_workspace_bytes_per_token", default=0.0),
        0.0,
    )
    batch_workspace_bytes = int(workspace_per_batch_token * max_batch_tokens)

    defaults = RUNTIME_DEFAULTS.get(runtime, {"overhead_gib": 2.0, "overhead_fraction": 0.1})
    overhead_gib = as_float(
        nested(candidate, "runtime_estimates", "overhead_gib", default=defaults["overhead_gib"]),
        defaults["overhead_gib"],
    )
    overhead_fraction = as_float(
        nested(
            candidate,
            "runtime_estimates",
            "overhead_fraction",
            default=defaults["overhead_fraction"],
        ),
        defaults["overhead_fraction"],
    )
    runtime_overhead_bytes = int(overhead_gib * GIB)
    runtime_overhead_bytes += int((gpu_weights_bytes + kv_cache_bytes) * overhead_fraction)

    required_gpu_bytes = (
        gpu_weights_bytes + kv_cache_bytes + batch_workspace_bytes + runtime_overhead_bytes
    )
    selected_gpus, reserve_bytes, usable_gpu_bytes, budget_source = _gpu_budget(
        host, candidate, warnings
    )
    host_ram_available = as_int(nested(host, "host", "memory", "available_bytes", default=0), 0)
    required_fit_bytes = required_gpu_bytes
    if budget_source == "host_memory_available_uma":
        required_fit_bytes += cpu_weights_bytes

    if not selected_gpus:
        fit_class = "unknown"
        fits = False
        headroom_bytes = None
        warnings.append("No normalized NVIDIA GPU free-memory facts are available.")
    else:
        headroom_bytes = usable_gpu_bytes - required_fit_bytes
        if headroom_bytes < 0:
            fit_class = "no_fit"
            fits = False
        elif headroom_bytes < max(int(required_gpu_bytes * 0.1), GIB):
            fit_class = "tight"
            fits = True
        else:
            fit_class = "fits"
            fits = True

    if cpu_weights_bytes and host_ram_available and cpu_weights_bytes > host_ram_available:
        warnings.append("CPU-resident weight estimate exceeds available host RAM.")
    if gpu_weight_fraction < 1.0:
        warnings.append(
            "GPU fit assumes partial weight residency; verify CPU RAM, offload flags, and latency."
        )

    return {
        "schema_version": "nvidia-model-fit/v1",
        "candidate_id": candidate.get("id") or candidate.get("model_id"),
        "runtime": runtime,
        "workload_basis": {
            "target_context_tokens": target_context,
            "expected_concurrent_sequences": sequences,
            "max_batch_total_tokens": max_batch_tokens,
            "estimated_live_tokens": live_tokens,
        },
        "selected_gpus": selected_gpus,
        "estimates": {
            "weights_bytes": weights_bytes,
            "gpu_resident_weights_bytes": gpu_weights_bytes,
            "cpu_resident_weights_bytes": cpu_weights_bytes,
            "kv_cache_bytes_per_token": kv_bytes_per_token,
            "kv_cache_bytes": kv_cache_bytes,
            "batch_workspace_bytes": batch_workspace_bytes,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "required_gpu_bytes": required_gpu_bytes,
            "required_fit_budget_bytes": required_fit_bytes,
            "fit_budget_source": budget_source,
            "safety_reserve_bytes": reserve_bytes,
            "usable_fit_budget_bytes_after_reserve": usable_gpu_bytes,
            "vram_safety_reserve_bytes": reserve_bytes,
            "usable_gpu_bytes_after_reserve": usable_gpu_bytes,
            "headroom_bytes_after_reserve": headroom_bytes,
        },
        "decision": {
            "fits_after_reserve": fits,
            "fit_class": fit_class,
            "confidence": _confidence(candidate, warnings),
        },
        "warnings": warnings,
    }


def _weight_bytes(candidate: dict[str, Any], warnings: list[str]) -> int:
    explicit = as_int(nested(candidate, "weights", "weight_bytes", default=0), 0)
    if explicit > 0:
        return explicit
    params_billion = as_float(nested(candidate, "weights", "parameters_billion", default=0.0), 0.0)
    if params_billion <= 0:
        warnings.append("Missing weight_bytes and parameters_billion; weight estimate is zero.")
        return 0
    bytes_per_parameter = as_float(
        nested(candidate, "weights", "bytes_per_parameter", default=0.0), 0.0
    )
    if bytes_per_parameter <= 0:
        precision = str(
            nested(candidate, "weights", "quantization", default="")
            or nested(candidate, "weights", "precision", default="")
        ).lower()
        bytes_per_parameter = PRECISION_BYTES.get(precision, 2.0)
        if precision not in PRECISION_BYTES:
            warnings.append(
                "Unknown precision/quantization; defaulted to two bytes per parameter."
            )
    storage_factor = max(
        1.0, as_float(nested(candidate, "weights", "storage_factor", default=1.03), 1.03)
    )
    return int(params_billion * 1_000_000_000 * bytes_per_parameter * storage_factor)


def _kv_bytes_per_token(candidate: dict[str, Any], warnings: list[str]) -> int:
    explicit = as_int(nested(candidate, "kv_cache", "bytes_per_token", default=0), 0)
    if explicit > 0:
        return explicit
    layers = as_int(nested(candidate, "kv_cache", "num_layers", default=0), 0)
    kv_heads = as_int(nested(candidate, "kv_cache", "num_key_value_heads", default=0), 0)
    attention_heads = as_int(nested(candidate, "kv_cache", "num_attention_heads", default=0), 0)
    hidden_size = as_int(nested(candidate, "kv_cache", "hidden_size", default=0), 0)
    head_dim = as_int(nested(candidate, "kv_cache", "head_dim", default=0), 0)
    dtype_bytes = as_float(nested(candidate, "kv_cache", "dtype_bytes", default=2.0), 2.0)
    if not head_dim and hidden_size and attention_heads:
        head_dim = math.ceil(hidden_size / attention_heads)
    if not kv_heads:
        kv_heads = attention_heads
    if layers and kv_heads and head_dim:
        return int(2 * layers * kv_heads * head_dim * dtype_bytes)
    warnings.append("KV-cache metadata is incomplete; KV-cache estimate is zero.")
    return 0


def _gpu_budget(
    host: dict[str, Any], candidate: dict[str, Any], warnings: list[str]
) -> tuple[list[dict[str, Any]], int, int, str]:
    gpu_count = as_int(
        nested(candidate, "deployment", "gpu_count", default=0)
        or nested(candidate, "deployment", "tensor_parallel_size", default=0),
        0,
    )
    if gpu_count <= 0:
        gpu_count = 1

    normalized = []
    for gpu in nested(host, "nvidia", "gpus", default=[]) or []:
        free_bytes = as_int(gpu.get("vram_free_bytes"), 0)
        if free_bytes > 0:
            normalized.append(
                {
                    "index": gpu.get("index"),
                    "uuid": gpu.get("uuid"),
                    "name": gpu.get("name"),
                    "vram_free_bytes": free_bytes,
                }
            )
    normalized.sort(key=lambda gpu: gpu["vram_free_bytes"], reverse=True)
    if not normalized and nested(
        host, "nvidia", "memory_reporting", "system_memory_budget_eligible", default=False
    ):
        return _unified_memory_budget(host, candidate, gpu_count, warnings)

    selected = normalized[:gpu_count]
    if len(selected) < gpu_count:
        warnings.append(f"Requested {gpu_count} GPUs but only {len(selected)} have free VRAM facts.")

    reserve_fraction, reserve_gib_per_gpu = _reserve_policy(candidate)
    reserve_bytes = 0
    usable_bytes = 0
    for gpu in selected:
        per_gpu_reserve = max(
            int(gpu["vram_free_bytes"] * reserve_fraction), int(reserve_gib_per_gpu * GIB)
        )
        reserve_bytes += per_gpu_reserve
        usable_bytes += max(0, gpu["vram_free_bytes"] - per_gpu_reserve)
    if len(selected) > 1:
        warnings.append(
            "Multi-GPU fit is aggregate and approximate; confirm tensor parallel partitioning and topology."
        )
    return selected, reserve_bytes, usable_bytes, "gpu_vram_free"


def _unified_memory_budget(
    host: dict[str, Any],
    candidate: dict[str, Any],
    gpu_count: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], int, int, str]:
    inventory = nested(host, "nvidia", "gpus", default=[]) or []
    if len(inventory) < gpu_count:
        warnings.append(
            f"Requested {gpu_count} GPUs but only {len(inventory)} GPU inventory rows are available."
        )
        return [], 0, 0, "unavailable"

    available_bytes = as_int(nested(host, "host", "memory", "available_bytes", default=0), 0)
    if available_bytes <= 0:
        warnings.append("Unified-memory hint exists but host available-memory facts are missing.")
        return [], 0, 0, "unavailable"

    reserve_fraction, reserve_gib_per_gpu = _reserve_policy(candidate)
    reserve_bytes = max(int(available_bytes * reserve_fraction), int(reserve_gib_per_gpu * GIB))
    selected = [
        {
            "index": gpu.get("index"),
            "uuid": gpu.get("uuid"),
            "name": gpu.get("name"),
            "budget_source": "host.memory.available_bytes",
            "shared_memory_available_bytes": available_bytes,
        }
        for gpu in inventory[:gpu_count]
    ]
    warnings.append(
        "Using host.memory.available_bytes as a conservative current-snapshot unified-memory budget because framebuffer free-memory facts are unavailable."
    )
    warnings.append(
        "Unified-memory fit is heuristic; verify target runtime allocation behavior and host memory pressure before apply."
    )
    if gpu_count > 1:
        warnings.append(
            "Unified-memory fallback uses one shared system-memory budget across selected GPUs; confirm topology and runtime partitioning."
        )
    return (
        selected,
        reserve_bytes,
        max(0, available_bytes - reserve_bytes),
        "host_memory_available_uma",
    )


def _reserve_policy(candidate: dict[str, Any]) -> tuple[float, float]:
    reserve_fraction = as_float(
        nested(candidate, "runtime_estimates", "vram_reserve_fraction", default=0.12), 0.12
    )
    reserve_gib_per_gpu = as_float(
        nested(candidate, "runtime_estimates", "vram_reserve_gib_per_gpu", default=2.0), 2.0
    )
    return reserve_fraction, reserve_gib_per_gpu


def _target_context(workload: dict[str, Any], candidate: dict[str, Any]) -> int:
    workload_context = as_int(
        nested(workload, "serving", "target_context_tokens", default=0)
        or nested(workload, "context", "target_tokens", default=0),
        0,
    )
    candidate_context = as_int(
        candidate.get("max_context_tokens")
        or nested(candidate, "context", "max_model_len", default=0),
        0,
    )
    return max(1, workload_context or candidate_context or 4096)


def _confidence(candidate: dict[str, Any], warnings: list[str]) -> str:
    if any("zero" in warning.lower() for warning in warnings):
        return "low"
    if any("unified-memory" in warning.lower() for warning in warnings):
        return "heuristic"
    if nested(candidate, "weights", "weight_bytes", default=0) and (
        nested(candidate, "kv_cache", "bytes_per_token", default=0)
        or nested(candidate, "kv_cache", "num_layers", default=0)
    ):
        return "medium"
    return "heuristic"
