#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible inference endpoint and emit JSON evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from common_io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="endpoint root, usually ending before /v1")
    parser.add_argument("--model", help="served model name; inferred from /v1/models when possible")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--api-key", help="optional bearer token; prefer --api-key-env")
    auth.add_argument("--api-key-env", help="environment variable holding a bearer token")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", help="write verification_report.json here")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    api_key = api_key_from_args(args)
    models_check = request_json("GET", f"{base_url}/v1/models", None, api_key, args.timeout)
    model = args.model or first_model_id(models_check.get("json"))
    chat_check = None
    if model:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word READY."}],
            "temperature": 0,
            "max_tokens": 8,
        }
        chat_check = request_json(
            "POST", f"{base_url}/v1/chat/completions", payload, api_key, args.timeout
        )
    report = {
        "schema_version": "nvidia-verification-report/v1",
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": {"base_url": base_url, "contract": "OpenAI-compatible"},
        "command": {
            "script": "smoke_test_endpoint.py",
            "model": args.model,
            "timeout_seconds": args.timeout,
        },
        "checks": {
            "models": summarize(models_check),
            "chat_completions": summarize(chat_check) if chat_check else {
                "ok": False,
                "error": "no model supplied and /v1/models did not provide one",
            },
        },
    }
    report["ok"] = bool(
        nested_ok(report, "checks", "models") and nested_ok(report, "checks", "chat_completions")
    )
    write_json(report, args.out)
    if not report["ok"]:
        raise SystemExit(1)


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "json": json.loads(raw) if raw else None,
                "body_excerpt": raw[:500],
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": str(exc),
            "body_excerpt": raw[:500],
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": None,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": str(exc),
        }


def api_key_from_args(args: argparse.Namespace) -> str | None:
    if args.api_key_env:
        value = os.environ.get(args.api_key_env)
        if not value:
            raise SystemExit(f"environment variable {args.api_key_env!r} is empty or missing")
        return value
    return args.api_key


def first_model_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("id")
    return None


def summarize(check: dict[str, Any] | None) -> dict[str, Any]:
    if not check:
        return {"ok": False, "error": "check not executed"}
    payload = check.get("json")
    summary = {key: value for key, value in check.items() if key != "json"}
    if isinstance(payload, dict):
        summary["response_keys"] = sorted(payload.keys())
        usage = payload.get("usage")
        if isinstance(usage, dict):
            summary["usage"] = usage
    return summary


def nested_ok(data: dict[str, Any], *keys: str) -> bool:
    current: Any = data
    for key in keys:
        current = current.get(key) if isinstance(current, dict) else None
    return bool(isinstance(current, dict) and current.get("ok"))


if __name__ == "__main__":
    main()
