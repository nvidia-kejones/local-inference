#!/usr/bin/env python3
"""Select serving backend and deployment substrate from host/workload evidence."""

from __future__ import annotations

from typing import Any

from common_io import nested


SUBSTRATE_ALIASES = {
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "kubectl": "kubernetes",
    "docker": "docker",
    "compose": "docker",
    "docker-compose": "docker",
    "docker_compose": "docker",
    "service": "native_service",
    "native": "native_service",
    "native-service": "native_service",
    "native_service": "native_service",
    "systemd": "native_service",
}

BACKEND_SUBSTRATE_MODULES = {
    ("vllm", "docker"): {
        "name": "vllm-compose-v1",
        "status": "implemented",
        "reference": "references/vllm.md",
    },
    ("vllm", "kubernetes"): {
        "name": "vllm-k8s-v1",
        "status": "implemented",
        "reference": "references/vllm.md",
    },
    ("vllm", "native_service"): {
        "name": "vllm-native-service-v1",
        "status": "unsupported_in_v1",
        "reference": "references/vllm.md",
    },
    ("sglang", "docker"): {
        "name": "sglang-compose-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/sglang.md",
    },
    ("sglang", "kubernetes"): {
        "name": "sglang-k8s-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/sglang.md",
    },
    ("sglang", "native_service"): {
        "name": "sglang-native-service-v1",
        "status": "unsupported_in_v1",
        "reference": "references/sglang.md",
    },
    ("tensorrt-llm", "docker"): {
        "name": "trtllm-serve-compose-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/trt-llm.md",
    },
    ("tensorrt-llm", "kubernetes"): {
        "name": "trtllm-serve-k8s-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/trt-llm.md",
    },
    ("tensorrt-llm", "native_service"): {
        "name": "trtllm-serve-native-service-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/trt-llm.md",
    },
    ("trtllm-serve", "docker"): {
        "name": "trtllm-serve-compose-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/trt-llm.md",
    },
    ("trtllm-serve", "kubernetes"): {
        "name": "trtllm-serve-k8s-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/trt-llm.md",
    },
    ("trtllm-serve", "native_service"): {
        "name": "trtllm-serve-native-service-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/trt-llm.md",
    },
    ("llama.cpp", "docker"): {
        "name": "llama-cpp-compose-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/llama-cpp.md",
    },
    ("llama.cpp", "kubernetes"): {
        "name": "llama-cpp-k8s-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/llama-cpp.md",
    },
    ("llama.cpp", "native_service"): {
        "name": "llama-cpp-native-service-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/llama-cpp.md",
    },
    ("ollama", "docker"): {
        "name": "ollama-compose-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/ollama.md",
    },
    ("ollama", "kubernetes"): {
        "name": "ollama-k8s-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/ollama.md",
    },
    ("ollama", "native_service"): {
        "name": "ollama-native-service-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/ollama.md",
    },
}


def build_deployment_selection(
    host: dict[str, Any],
    workload: dict[str, Any],
    candidate: dict[str, Any],
    fit: dict[str, Any],
) -> dict[str, Any]:
    backend = select_backend(workload, candidate, fit)
    substrate = select_substrate(host, workload, candidate, backend["runtime"])
    module = module_for(backend["runtime"], substrate["selected"])
    blockers = []
    blockers.extend(backend["blockers"])
    blockers.extend(substrate["blockers"])
    if module["status"] != "implemented":
        blockers.append(f"{module['name']} is {module['status']}")
    return {
        "schema_version": "nvidia-deployment-selection/v1",
        "selected_backend": backend,
        "selected_substrate": substrate,
        "deployment_module": module,
        "apply_blockers": blockers,
    }


def select_backend(
    workload: dict[str, Any], candidate: dict[str, Any], fit: dict[str, Any]
) -> dict[str, Any]:
    runtime = str(candidate.get("runtime") or "").strip().lower()
    blockers = []
    if not runtime:
        blockers.append("candidate runtime is missing")
    if nested(candidate, "license", "deployment_allowed", default=True) is False:
        blockers.append("candidate license/deployment constraints disallow deployment")
    if nested(candidate, "runtime_support", "blocked", default=False):
        blockers.append("runtime support is marked blocked in candidate metadata")
    fit_class = nested(fit, "decision", "fit_class", default="unknown")
    if fit_class == "no_fit":
        blockers.append("memory fit estimate does not fit after safety reserve")
    return {
        "runtime": runtime or "unknown",
        "model_id": candidate.get("model_id") or candidate.get("id"),
        "fit_class": fit_class,
        "fit_confidence": nested(fit, "decision", "confidence", default="unknown"),
        "rationale": backend_rationale(runtime, workload, candidate),
        "blockers": blockers,
    }


def select_substrate(
    host: dict[str, Any],
    workload: dict[str, Any],
    candidate: dict[str, Any],
    runtime: str,
) -> dict[str, Any]:
    facts = substrate_facts(host)
    requested = requested_substrate(workload, candidate)
    alternatives = {
        "kubernetes": substrate_available(facts, "kubernetes"),
        "docker": substrate_available(facts, "docker"),
        "native_service": substrate_available(facts, "native_service"),
    }
    rationale = []
    blockers = []

    if requested:
        selected = requested
        rationale.append(f"user or candidate requested {requested}")
        if not alternatives.get(requested, False):
            blockers.extend(unavailable_reasons(facts, requested))
            blockers.append("requested substrate is unavailable; not falling back automatically")
    elif alternatives["kubernetes"]:
        selected = "kubernetes"
        rationale.append("Kubernetes is usable and has priority over Docker and native service")
    elif alternatives["docker"]:
        selected = "docker"
        rationale.append("Docker is available and Kubernetes was not selected")
    elif alternatives["native_service"]:
        selected = "native_service"
        rationale.append("Native service manager is usable and container substrates were unavailable")
    else:
        selected = "unavailable"
        blockers.append("no usable deployment substrate was discovered")

    if selected == "docker" and not nested(facts, "docker", "compose_available", default=False):
        blockers.append("Docker is available but Docker Compose is unavailable; current modules require Compose")
    if selected == "native_service" and runtime in {"vllm", "sglang"}:
        blockers.append(f"{runtime} native service deployment is not implemented")

    rejected = []
    for name in ("kubernetes", "docker", "native_service"):
        if name == selected:
            continue
        rejected.append(
            {
                "substrate": name,
                "available": alternatives[name],
                "reasons": unavailable_reasons(facts, name) if not alternatives[name] else [],
            }
        )
    return {
        "selected": selected,
        "requested": requested,
        "priority_order": ["kubernetes", "docker", "native_service"],
        "rationale": rationale,
        "alternatives": alternatives,
        "rejected": rejected,
        "blockers": blockers,
    }


def requested_substrate(workload: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    values = [
        nested(workload, "deployment", "substrate", default=None),
        nested(workload, "deployment", "preferred_substrate", default=None),
        nested(workload, "deployment", "target", default=None),
        nested(candidate, "deployment", "substrate", default=None),
        nested(candidate, "deployment", "preferred_substrate", default=None),
    ]
    for value in values:
        normalized = normalize_substrate(value)
        if normalized:
            return normalized
    return None


def normalize_substrate(value: Any) -> str | None:
    if value is None:
        return None
    return SUBSTRATE_ALIASES.get(str(value).strip().lower())


def substrate_facts(host: dict[str, Any]) -> dict[str, Any]:
    discovered = host.get("deployment_substrates")
    if isinstance(discovered, dict):
        return discovered
    return {
        "kubernetes": {"available": False, "evidence": ["not present in host facts"]},
        "docker": {
            "available": bool(nested(host, "containers", "docker", "available", default=False)),
            "compose_available": False,
            "evidence": ["derived from legacy containers.docker facts"],
        },
        "native_service": {"available": False, "evidence": ["not present in host facts"]},
    }


def substrate_available(facts: dict[str, Any], substrate: str) -> bool:
    return bool(nested(facts, substrate, "available", default=False))


def unavailable_reasons(facts: dict[str, Any], substrate: str) -> list[str]:
    info = facts.get(substrate)
    if not isinstance(info, dict):
        return [f"{substrate} facts were not collected"]
    blockers = info.get("blockers")
    if isinstance(blockers, list) and blockers:
        return [str(item) for item in blockers]
    evidence = info.get("evidence")
    if isinstance(evidence, list) and evidence:
        return [str(item) for item in evidence]
    return [f"{substrate} is not available"]


def module_for(runtime: str, substrate: str) -> dict[str, Any]:
    module = BACKEND_SUBSTRATE_MODULES.get((runtime, substrate))
    if module:
        result = dict(module)
        result.setdefault("expected_outputs", expected_outputs_for(substrate))
        return result
    return {
        "name": f"{runtime or 'unknown'}-{substrate or 'unknown'}-deployment-v1",
        "status": "follow_on_module_not_implemented_in_v1",
        "reference": "references/runtime-selection.md",
        "expected_outputs": expected_outputs_for(substrate),
    }


def expected_outputs_for(substrate: str) -> list[str]:
    if substrate == "kubernetes":
        return ["rendered Kubernetes manifests", "applied state", "rollout and rollback commands"]
    if substrate == "docker":
        return ["rendered Compose/env files", "applied state", "rollback command"]
    if substrate == "native_service":
        return ["service environment file", "service unit", "applied state", "rollback command"]
    return ["reviewed runtime config", "applied state"]


def backend_rationale(
    runtime: str, workload: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    patterns = nested(workload, "serving", "patterns", default={}) or {}
    rationale = [f"selected candidate runtime is {runtime or 'unknown'}"]
    if runtime == "vllm":
        rationale.append("vLLM is the implemented Hugging Face/CUDA baseline")
    elif runtime == "sglang":
        rationale.append("SGLang is preferred for prefix reuse, structured outputs, or agentic flows")
    elif runtime in {"tensorrt-llm", "trtllm-serve"}:
        rationale.append("TensorRT-LLM is preferred when supported optimization work is justified")
    elif runtime == "llama.cpp":
        rationale.append("llama.cpp is preferred for GGUF, quantized portability, or partial offload")
    elif runtime == "ollama":
        rationale.append("Ollama is preferred when model lifecycle simplicity matters most")
    if patterns.get("structured_outputs"):
        rationale.append("workload requests structured outputs")
    if patterns.get("agentic"):
        rationale.append("workload is agentic/tool-adjacent")
    if patterns.get("prefix_reuse") not in {None, "", "low", "false"}:
        rationale.append(f"workload prefix reuse is {patterns.get('prefix_reuse')}")
    if candidate.get("format"):
        rationale.append(f"candidate format is {candidate.get('format')}")
    return rationale
