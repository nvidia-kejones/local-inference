#!/usr/bin/env python3
"""Small structured-data helpers with a dependency-free YAML fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_INT_RE = re.compile(r"[-+]?\d+")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")


def load_structured(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if Path(path).suffix.lower() == ".json" or stripped.startswith(("{", "[")):
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return load_simple_yaml(text)


def write_json(data: Any, path: str | Path | None) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if path:
        Path(path).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


def write_yaml(data: Any, path: str | Path | None) -> None:
    payload = dump_yaml(data) + "\n"
    if path:
        Path(path).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the nested mapping subset used by the example workload profile."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise ValueError(f"tabs are not supported in fallback YAML at line {number}")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = _strip_comment(raw_line.strip())
        if content.startswith("- "):
            raise ValueError(
                "fallback YAML loader supports nested mappings only; install PyYAML "
                f"or use JSON for list data (line {number})"
            )
        if ":" not in content:
            raise ValueError(f"expected key:value YAML mapping at line {number}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty YAML key at line {number}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def dump_yaml(data: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(data, dict):
        if not data:
            return f"{prefix}{{}}"
        lines: list[str] = []
        for key, value in data.items():
            if value == {}:
                lines.append(f"{prefix}{key}: {{}}")
            elif value == []:
                lines.append(f"{prefix}{key}: []")
            elif isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(dump_yaml(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return f"{prefix}[]"
        lines = []
        for value in data:
            if isinstance(value, dict):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml(value, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(value)}")
        return "\n".join(lines)
    return f"{prefix}{_yaml_scalar(data)}"


def _strip_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        if char == "#" and quote is None:
            return value[:index].rstrip()
    return value


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith(('"', "'")) and value.endswith(value[0]):
        if value[0] == '"':
            return json.loads(value)
        return value[1:-1].replace("''", "'")
    if value.startswith(("{", "[")):
        return json.loads(value)
    if _INT_RE.fullmatch(value):
        return int(value)
    if _FLOAT_RE.fullmatch(value):
        return float(value)
    return value


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)
