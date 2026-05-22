#!/usr/bin/env python3
"""Build host-aware model/runtime recommendations across workload profiles."""

from __future__ import annotations

import argparse
from typing import Any

from common_io import load_structured, write_json
from rank_candidates import WEIGHTS, score_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="normalized host_facts.json")
    parser.add_argument("--profiles", required=True, help="use-case profile JSON or YAML")
    parser.add_argument("--candidates", required=True, help="current candidate set JSON or YAML")
    parser.add_argument("--top", type=int, default=3, help="alternatives to keep per use case")
    parser.add_argument("--out", help="write use_case_recommendations.json here")
    args = parser.parse_args()

    host = load_structured(args.host)
    candidate_source = load_structured(args.candidates)
    profile_source = load_structured(args.profiles)
    candidates = candidate_source.get("candidates", []) if isinstance(candidate_source, dict) else []
    profiles = profile_source.get("profiles", []) if isinstance(profile_source, dict) else []
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("candidate set must provide a non-empty candidates list")
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("profile set must provide a non-empty profiles list")

    recommendations = [
        recommend_for_profile(host, profile, candidates, max(1, args.top)) for profile in profiles
    ]
    report = {
        "schema_version": "nvidia-use-case-recommendations/v1",
        "host_facts": args.host,
        "profile_source": args.profiles,
        "candidate_source": {
            "path": args.candidates,
            "refresh_notes": candidate_source.get("refresh_notes")
            if isinstance(candidate_source, dict)
            else None,
        },
        "weights": WEIGHTS,
        "recommendations": recommendations,
        "presentation_guidance": [
            "Tell the user these are host-aware recommendations for the refreshed candidate set, not a permanent model catalog.",
            "Lead with the recommended model/runtime pair, why it fits this host/use case, fit confidence, and benchmark needed before deployment.",
            "Show blockers and next-best alternatives when no candidate is clearly deployable.",
        ],
    }
    write_json(report, args.out)


def recommend_for_profile(
    host: dict[str, Any],
    profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    top: int,
) -> dict[str, Any]:
    workload = profile.get("workload")
    if not isinstance(workload, dict):
        raise SystemExit(f"profile {profile.get('id') or profile.get('name')!r} needs a workload object")
    workload.setdefault("id", profile.get("id"))
    scored = [score_candidate(host, workload, candidate) for candidate in candidates]
    scored.sort(key=lambda item: (item["blocked"], -item["score"]))
    deployable = [item for item in scored if not item["blocked"]]
    recommendation = deployable[0] if deployable else scored[0]
    alternatives = [item for item in scored if item["id"] != recommendation["id"]][:top]
    return {
        "id": profile.get("id") or profile.get("name"),
        "label": profile.get("label") or profile.get("id") or profile.get("name"),
        "description": profile.get("description"),
        "workload": workload,
        "recommended": recommendation,
        "alternatives": alternatives,
        "status": "recommended" if deployable else "no_unblocked_candidate",
        "notes": recommendation_notes(recommendation, deployable),
    }


def recommendation_notes(recommendation: dict[str, Any], deployable: list[dict[str, Any]]) -> list[str]:
    notes = []
    if not deployable:
        notes.append("Every scored candidate is blocked; use blockers to refresh the candidate set or relax workload assumptions.")
    fit = recommendation.get("fit", {})
    decision = fit.get("decision", {}) if isinstance(fit, dict) else {}
    if decision.get("fit_class") == "tight":
        notes.append("Top candidate is a tight VRAM fit; benchmark real prompts before apply.")
    if decision.get("confidence") in {"low", "heuristic"}:
        notes.append("Top candidate fit confidence is not high; confirm model memory metadata.")
    if not (recommendation.get("pins") or {}).get("model_revision"):
        notes.append("Top candidate still needs a pinned model revision.")
    return notes


if __name__ == "__main__":
    main()
