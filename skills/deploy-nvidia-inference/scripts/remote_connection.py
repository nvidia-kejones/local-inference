#!/usr/bin/env python3
"""Normalize remote connection methods for SSH and Brev-managed instances."""

from __future__ import annotations

import argparse
import shlex
from typing import Any

from common_io import load_structured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-file", required=True, help="YAML or JSON connection spec")
    parser.add_argument("--shell-env", action="store_true", help="emit bash assignments")
    parser.add_argument("--json", action="store_true", help="emit normalized JSON")
    args = parser.parse_args()

    connection = normalize_connection(load_structured(args.connection_file))
    if args.json:
        import json

        print(json.dumps(connection, indent=2, sort_keys=True))
        return
    if args.shell_env:
        print(shell_env(connection))
        return
    raise SystemExit("pass --json or --shell-env")


def normalize_connection(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        spec = {"command": spec}
    if not isinstance(spec, dict):
        raise SystemExit("connection spec must be a mapping or a command string")

    raw = str(spec.get("command") or spec.get("connection") or "").strip()
    if raw:
        return parse_command(raw)

    kind = str(spec.get("kind") or spec.get("type") or "").lower()
    if kind == "ssh":
        target = str(spec.get("target") or spec.get("ssh_target") or "").strip()
        if not target:
            raise SystemExit("ssh connection specs need target")
        options = [str(item) for item in spec.get("options", []) or []]
        return {
            "transport": "ssh",
            "raw_command": target,
            "ssh_target": target,
            "ssh_options": options,
        }
    if kind == "brev":
        instance = str(spec.get("instance") or spec.get("brev_instance") or "").strip()
        if not instance:
            raise SystemExit("brev connection specs need instance")
        mode = str(spec.get("mode") or spec.get("brev_mode") or "shell").lower()
        if mode not in {"shell", "ssh"}:
            raise SystemExit("brev mode must be shell or ssh")
        use_host = bool(spec.get("host") or spec.get("brev_host", False))
        return {
            "transport": "brev",
            "raw_command": f"brev {mode} {instance}{' --host' if use_host else ''}".strip(),
            "brev_mode": mode,
            "brev_instance": instance,
            "brev_host": use_host,
        }

    raise SystemExit(
        "connection spec needs a command, or a kind/type of ssh or brev with target/instance"
    )


def parse_command(command: str) -> dict[str, Any]:
    parts = shlex.split(command)
    if not parts:
        raise SystemExit("connection command is empty")
    if parts[0] == "ssh":
        if len(parts) < 2:
            raise SystemExit("ssh connection needs a target")
        target = parts[-1]
        options = parts[1:-1]
        return {
            "transport": "ssh",
            "raw_command": command,
            "ssh_target": target,
            "ssh_options": options,
        }
    if parts[0] == "brev" and len(parts) >= 3 and parts[1] in {"shell", "ssh"}:
        instance = parts[2]
        use_host = "--host" in parts[3:]
        return {
            "transport": "brev",
            "raw_command": command,
            "brev_mode": parts[1],
            "brev_instance": instance,
            "brev_host": use_host,
        }
    if len(parts) == 1:
        return {
            "transport": "ssh",
            "raw_command": command,
            "ssh_target": parts[0],
            "ssh_options": [],
        }
    raise SystemExit(
        "connection command must be 'ssh ...', 'brev shell ...', 'brev ssh ...', or a plain SSH target"
    )


def shell_env(connection: dict[str, Any]) -> str:
    lines = [f"TRANSPORT={shlex.quote(str(connection.get('transport') or 'ssh'))}"]
    lines.append(f"RAW_CONNECTION={shlex.quote(str(connection.get('raw_command') or ''))}")
    lines.append(f"SSH_TARGET={shlex.quote(str(connection.get('ssh_target') or ''))}")
    lines.append(f"SSH_OPTIONS=({ ' '.join(shlex.quote(str(item)) for item in connection.get('ssh_options', []) or []) })")
    lines.append(f"BREV_INSTANCE={shlex.quote(str(connection.get('brev_instance') or ''))}")
    lines.append(f"BREV_MODE={shlex.quote(str(connection.get('brev_mode') or ''))}")
    lines.append(f"BREV_HOST={1 if connection.get('brev_host') else 0}")
    return "\n".join(lines)


def remote_exec_command(connection: dict[str, Any], remote_command: str) -> str:
    if connection.get("transport") == "brev":
        return f"brev exec {shlex.quote(str(connection['brev_instance']))} {shlex.quote(remote_command)}"
    target = shlex.quote(str(connection["ssh_target"]))
    options = " ".join(shlex.quote(str(item)) for item in connection.get("ssh_options", []) or [])
    ssh_prefix = f"ssh {options} " if options else "ssh "
    return f"{ssh_prefix}{target} {shlex.quote(remote_command)}"


def remote_port_forward_command(connection: dict[str, Any], local_port: int, remote_port: int) -> str:
    if connection.get("transport") == "brev":
        return f"brev port-forward {shlex.quote(str(connection['brev_instance']))} -p {local_port}:{remote_port}"
    target = shlex.quote(str(connection["ssh_target"]))
    options = " ".join(shlex.quote(str(item)) for item in connection.get("ssh_options", []) or [])
    if options:
        options = f"{options} "
    return f"ssh {options}-L {local_port}:127.0.0.1:{remote_port} {target}"


def remote_copy_command(connection: dict[str, Any], source: str, destination: str) -> str:
    if connection.get("transport") == "brev":
        return f"brev copy {shlex.quote(source)} {shlex.quote(str(connection['brev_instance']) + ':' + destination)}"
    target = shlex.quote(str(connection["ssh_target"]))
    options = " ".join(shlex.quote(str(item)) for item in connection.get("ssh_options", []) or [])
    if options:
        options = f"{options} "
    return f"scp {options}{shlex.quote(source)} {target}:{shlex.quote(destination)}"


def remote_exists_command(connection: dict[str, Any], remote_path: str) -> str:
    return remote_exec_command(connection, f"test -e {shlex.quote(remote_path)} && printf existing || true")


def remote_mkdir_command(connection: dict[str, Any], remote_dir: str) -> str:
    return remote_exec_command(connection, f"umask 077 && mkdir -p {shlex.quote(remote_dir)} && chmod 700 {shlex.quote(remote_dir)}")


if __name__ == "__main__":
    main()
