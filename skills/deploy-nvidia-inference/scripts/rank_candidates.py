#!/usr/bin/env python3
"""Score current model/runtime candidates for a discovered host and workload."""

from __future__ import annotations

import argparse
from typing import Any

from common_io import as_float, as_int, load_structured, nested, write_json
from fitlib import estimate_fit


WEIGHTS = {
    "quality_fit": 0.24,
    "host_fit": 0.25,
    "runtime_support": 0.16,
    "license_deployment": 0.11,
    "quantization": 0.08,
    "context": 0.09,
    "serving_behavior": 0.07,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="normalized host_facts.json")
    parser.add_argument("--workload", required=True, help="workload profile YAML or JSON")
    parser.add_argument("--candidates", required=True, help="current candidate set JSON or YAML")
    parser.add_argument("--out", help="write candidate scorecard JSON here")
    args = parser.parse_args()

    host = load_structured(args.host)
    workload = load_structured(args.workload)
    source = load_structured(args.candidates)
    candidates = source.get("candidates", []) if isinstance(source, dict) else source
    if not isinstance(candidates, list):
        raise SystemExit("candidate set must be a list or an object with a candidates list")

    scored = [score_candidate(host, workload, candidate) for candidate in candidates]
    scored.sort(key=lambda item: (item["blocked"], -item["score"]))
    scorecard = {
        "schema_version": "nvidia-candidate-scorecard/v1",
        "weights": WEIGHTS,
        "workload_profile": args.workload,
        "host_facts": args.host,
        "candidate_source": {
            "path": args.candidates,
            "refresh_notes": source.get("refresh_notes") if isinstance(source, dict) else None,
        },
        "backend_decision": backend_decision(scored, workload),
        "candidates": scored,
    }
    write_json(scorecard, args.out)


def score_candidate(
    host: dict[str, Any], workload: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    fit = estimate_fit(host, workload, candidate)
    component_scores = {
        "quality_fit": quality_score(workload, candidate),
        "host_fit": host_fit_score(fit),
        "runtime_support": runtime_support_score(candidate),
        "license_deployment": license_score(candidate),
        "quantization": quantization_score(candidate),
        "context": context_score(workload, candidate),
        "serving_behavior": behavior_score(workload, candidate),
    }
    blockers = blockers_for(candidate, fit, workload)
    reasons = reasons_for(candidate, fit, component_scores)
    total = sum(component_scores[key] * WEIGHTS[key] for key in WEIGHTS)
    if blockers:
        total *= 0.5
    return {
        "id": candidate.get("id") or candidate.get("model_id"),
        "model_id": candidate.get("model_id") or candidate.get("id"),
        "runtime": candidate.get("runtime"),
        "score": round(total, 4),
        "blocked": bool(blockers),
        "blockers": blockers,
        "component_scores": component_scores,
        "fit": fit,
        "quality_evidence": quality_evidence(workload, candidate),
        "reasons": reasons,
        "pins": {
            "model_revision": candidate.get("model_revision"),
            "container_image": nested(candidate, "deployment", "container_image", default=None),
        },
    }


def normalized_score(value: Any, default: float) -> float:
    score = as_float(value, default)
    if score > 1.0:
        score /= 100.0
    return round(min(1.0, max(0.0, score)), 4)


def quality_score(workload: dict[str, Any], candidate: dict[str, Any]) -> float:
    scores = candidate.get("quality_fit_scores")
    purpose = workload_purpose(workload)
    if isinstance(scores, dict) and purpose in scores:
        return normalized_score(scores[purpose], default=0.5)
    return normalized_score(candidate.get("quality_fit_score"), default=0.5)


def quality_evidence(workload: dict[str, Any], candidate: dict[str, Any]) -> Any:
    evidence = candidate.get("quality_evidence")
    purpose = workload_purpose(workload)
    if isinstance(evidence, dict) and purpose in evidence and isinstance(evidence[purpose], dict):
        return evidence[purpose]
    return evidence


def workload_purpose(workload: dict[str, Any]) -> str:
    return str(
        nested(workload, "intent", "purpose", default="")
        or workload.get("id")
        or workload.get("name")
        or "default"
    )


def host_fit_score(fit: dict[str, Any]) -> float:
    return {"fits": 1.0, "tight": 0.62, "no_fit": 0.0, "unknown": 0.25}.get(
        nested(fit, "decision", "fit_class", default="unknown"), 0.25
    )


def runtime_support_score(candidate: dict[str, Any]) -> float:
    explicit = nested(candidate, "runtime_support", "score", default=None)
    if explicit is not None:
        return normalized_score(explicit, 0.0)
    if nested(candidate, "runtime_support", "validated", default=False):
        return 0.9
    return 0.35


def license_score(candidate: dict[str, Any]) -> float:
    if nested(candidate, "license", "deployment_allowed", default=True) is False:
        return 0.0
    return normalized_score(nested(candidate, "license", "deployment_score", default=1.0), 1.0)


def quantization_score(candidate: dict[str, Any]) -> float:
    quantization = nested(candidate, "weights", "quantization", default=None)
    if not quantization or str(quantization).lower() in {"none", "bf16", "fp16"}:
        return 1.0
    if candidate.get("quantization_available") is True:
        return 1.0
    if candidate.get("quantization_available") is False:
        return 0.0
    return 0.45


def context_score(workload: dict[str, Any], candidate: dict[str, Any]) -> float:
    target = as_int(
        nested(workload, "serving", "target_context_tokens", default=0)
        or nested(workload, "context", "target_tokens", default=0),
        0,
    )
    available = as_int(
        candidate.get("max_context_tokens")
        or nested(candidate, "context", "max_model_len", default=0),
        0,
    )
    if not target:
        return 0.6
    if not available:
        return 0.4
    return round(min(1.0, available / target), 4)


def behavior_score(workload: dict[str, Any], candidate: dict[str, Any]) -> float:
    explicit = candidate.get("serving_behavior_score")
    if explicit is not None:
        return normalized_score(explicit, 0.5)
    runtime = str(candidate.get("runtime", "")).lower()
    patterns = nested(workload, "serving", "patterns", default={}) or {}
    prefix_reuse = str(patterns.get("prefix_reuse", "")).lower() not in {"", "none", "low", "false"}
    structured = bool(patterns.get("structured_outputs"))
    agentic = bool(patterns.get("agentic"))
    peak_performance = as_float(nested(workload, "intent", "throughput_priority", default=0.0), 0.0)
    simplicity = as_float(nested(workload, "intent", "operational_simplicity_priority", default=0.0), 0.0)
    constrained = as_float(nested(workload, "intent", "vram_constraint_priority", default=0.0), 0.0)
    if runtime == "sglang" and (prefix_reuse or structured or agentic):
        return 0.92
    if runtime in {"tensorrt-llm", "trtllm-serve"} and peak_performance >= 0.7:
        return 0.88
    if runtime == "llama.cpp" and (
        str(candidate.get("format", "")).lower() == "gguf" or constrained >= 0.7
    ):
        return 0.86
    if runtime == "ollama" and simplicity >= 0.7:
        return 0.86
    if runtime == "vllm":
        return 0.78
    return 0.55


def blockers_for(
    candidate: dict[str, Any], fit: dict[str, Any], workload: dict[str, Any]
) -> list[str]:
    blockers = []
    if nested(candidate, "license", "deployment_allowed", default=True) is False:
        blockers.append("license/deployment constraint is marked disallowed")
    if nested(fit, "decision", "fit_class", default="unknown") == "no_fit":
        blockers.append("memory fit estimate does not fit after safety reserve")
    target = as_int(nested(workload, "serving", "target_context_tokens", default=0), 0)
    max_context = as_int(candidate.get("max_context_tokens"), 0)
    if target and max_context and max_context < target:
        blockers.append("candidate context limit is below the workload target")
    if nested(candidate, "runtime_support", "blocked", default=False):
        blockers.append("runtime support is marked blocked in candidate metadata")
    return blockers


def reasons_for(
    candidate: dict[str, Any], fit: dict[str, Any], component_scores: dict[str, float]
) -> list[str]:
    reasons = [
        f"fit={nested(fit, 'decision', 'fit_class', default='unknown')} after reserve",
        f"runtime_support={component_scores['runtime_support']}",
        f"context={component_scores['context']}",
    ]
    if not candidate.get("model_revision"):
        reasons.append("model revision is not pinned yet")
    if not nested(candidate, "deployment", "container_image", default=None):
        reasons.append("container image is not pinned yet")
    return reasons


def backend_decision(scored: list[dict[str, Any]], workload: dict[str, Any]) -> dict[str, Any]:
    deployable = [candidate for candidate in scored if not candidate["blocked"]]
    selected = deployable[0] if deployable else (scored[0] if scored else None)
    if selected is None:
        return {
            "status": "no_candidates",
            "selected_candidate_id": None,
            "selected_backend": None,
            "rationale": ["candidate set was empty"],
            "alternatives": [],
            "blockers": ["candidate set was empty"],
        }
    status = "recommended" if deployable else "no_unblocked_candidate"
    alternatives = [
        {
            "id": candidate["id"],
            "backend": candidate["runtime"],
            "score": candidate["score"],
            "blocked": candidate["blocked"],
            "blockers": candidate["blockers"],
        }
        for candidate in scored
        if candidate["id"] != selected["id"]
    ][:3]
    rationale = [
        f"selected highest-ranked unblocked candidate for {workload_purpose(workload)}"
        if deployable
        else "no unblocked candidate exists; selected highest-ranked blocked candidate for review",
        f"backend={selected['runtime']}",
        f"score={selected['score']}",
        f"fit={nested(selected, 'fit', 'decision', 'fit_class', default='unknown')}",
    ]
    return {
        "status": status,
        "selected_candidate_id": selected["id"],
        "selected_model_id": selected["model_id"],
        "selected_backend": selected["runtime"],
        "rationale": rationale,
        "alternatives": alternatives,
        "blockers": selected["blockers"],
        "note": "This is a host/workload-specific backend decision for the refreshed candidate set, not a universal runtime ranking.",
    }


if __name__ == "__main__":
    main()
