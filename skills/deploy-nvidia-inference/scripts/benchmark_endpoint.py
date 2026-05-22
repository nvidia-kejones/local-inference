#!/usr/bin/env python3
"""Run a bounded OpenAI-compatible endpoint benchmark profile and emit JSON."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

from common_io import as_int, load_structured, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True, help="benchmark profile JSON or YAML")
    parser.add_argument("--profile-name", default="smoke")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--api-key", help="bearer token; prefer --api-key-env")
    auth.add_argument("--api-key-env", help="environment variable holding a bearer token")
    parser.add_argument("--out", help="write benchmark JSON here")
    args = parser.parse_args()

    selected = load_profile(args.profile, args.profile_name)
    prompts = selected.get("prompts") or ["Say READY."]
    if not isinstance(prompts, list):
        raise SystemExit("benchmark profile prompts must be a list")
    api_key = api_key_from_args(args)
    requests = max(1, as_int(selected.get("requests"), len(prompts)))
    concurrency = max(1, as_int(selected.get("concurrency"), 1))
    started = time.perf_counter()
    tasks = [
        (index, str(prompts[index % len(prompts)]), selected)
        for index in range(requests)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(
                lambda task: complete(args.base_url.rstrip("/"), args.model, task, api_key),
                tasks,
            )
        )
    wall = time.perf_counter() - started
    report = build_report(args, selected, results, wall)
    write_json(report, args.out)
    if report["summary"]["successful_requests"] == 0:
        raise SystemExit(1)


def load_profile(path: str, name: str) -> dict[str, Any]:
    data = load_structured(path)
    profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    selected = profiles.get(name)
    if not isinstance(selected, dict):
        raise SystemExit(f"profile {name!r} not found in {path}")
    return selected


def api_key_from_args(args: argparse.Namespace) -> str | None:
    if args.api_key_env:
        value = os.environ.get(args.api_key_env)
        if not value:
            raise SystemExit(f"environment variable {args.api_key_env!r} is empty or missing")
        return value
    return args.api_key


def complete(
    base_url: str,
    model: str,
    task: tuple[int, str, dict[str, Any]],
    api_key: str | None,
) -> dict[str, Any]:
    index, prompt, profile = task
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": profile.get("temperature", 0),
        "max_tokens": profile.get("max_tokens", 64),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body, headers=headers, method="POST"
    )
    timeout = float(profile.get("timeout_seconds", 120))
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            return {
                "index": index,
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "latency_seconds": round(time.perf_counter() - started, 5),
                "usage": parsed.get("usage") if isinstance(parsed, dict) else None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "index": index,
            "ok": False,
            "status": exc.code,
            "latency_seconds": round(time.perf_counter() - started, 5),
            "error": str(exc),
            "body_excerpt": exc.read().decode("utf-8", errors="replace")[:300],
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "index": index,
            "ok": False,
            "status": None,
            "latency_seconds": round(time.perf_counter() - started, 5),
            "error": str(exc),
        }


def build_report(
    args: argparse.Namespace, profile: dict[str, Any], results: list[dict[str, Any]], wall: float
) -> dict[str, Any]:
    ok = [result for result in results if result.get("ok")]
    latencies = [float(result["latency_seconds"]) for result in ok]
    completion_tokens = sum(
        as_int((result.get("usage") or {}).get("completion_tokens"), 0) for result in ok
    )
    total_tokens = sum(as_int((result.get("usage") or {}).get("total_tokens"), 0) for result in ok)
    summary = {
        "requests": len(results),
        "successful_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "wall_seconds": round(wall, 5),
        "requests_per_second": round(len(ok) / wall, 5) if wall else None,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_second": round(completion_tokens / wall, 5)
        if wall and completion_tokens
        else None,
        "latency_seconds": latency_summary(latencies),
    }
    return {
        "schema_version": "nvidia-endpoint-benchmark/v1",
        "benchmarked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": args.base_url.rstrip("/"),
        "model": args.model,
        "profile": {"path": args.profile, "name": args.profile_name, "values": profile},
        "summary": summary,
        "results": results,
    }


def latency_summary(latencies: list[float]) -> dict[str, float | None]:
    if not latencies:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(latencies)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "min": round(ordered[0], 5),
        "p50": round(statistics.median(ordered), 5),
        "p95": round(ordered[p95_index], 5),
        "max": round(ordered[-1], 5),
        "mean": round(statistics.mean(ordered), 5),
    }


if __name__ == "__main__":
    main()
