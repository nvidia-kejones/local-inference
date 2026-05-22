#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s --ssh-target HOST --compose FILE --env FILE [--remote-dir DIR] [--state-out FILE] --apply --allow-model-downloads [--replace-existing]\n' "$0" >&2
}

ssh_target=
compose_file=
env_file=
remote_dir=.local/share/codex-inference/nvidia-inference
state_out=applied_deployment_state.json
apply=0
allow_downloads=0
replace_existing=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-target) ssh_target=$2; shift 2 ;;
    --compose) compose_file=$2; shift 2 ;;
    --env) env_file=$2; shift 2 ;;
    --remote-dir) remote_dir=$2; shift 2 ;;
    --state-out) state_out=$2; shift 2 ;;
    --apply) apply=1; shift ;;
    --allow-model-downloads) allow_downloads=1; shift ;;
    --replace-existing) replace_existing=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ssh_target" || -z "$compose_file" || -z "$env_file" ]]; then
  usage
  exit 2
fi
if [[ "$ssh_target" == -* ]]; then
  printf 'ssh target must not start with a hyphen\n' >&2
  exit 2
fi
if [[ "$apply" -ne 1 || "$allow_downloads" -ne 1 ]]; then
  printf 'refusing remote write: pass both --apply and --allow-model-downloads after reviewing the plan\n' >&2
  exit 3
fi
if [[ ! -f "$compose_file" || ! -f "$env_file" ]]; then
  printf 'compose and env files must exist locally\n' >&2
  exit 2
fi
if [[ "$compose_file" == -* || "$env_file" == -* ]]; then
  printf 'compose and env file arguments must not start with a hyphen\n' >&2
  exit 2
fi
if [[ ! "$remote_dir" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  printf 'remote dir must be a simple path relative to the SSH login home\n' >&2
  exit 2
fi
if [[ "$remote_dir" == /* || "$remote_dir" == -* || "$remote_dir" =~ (^|/)\.\.(/|$) ]]; then
  printf 'remote dir must stay relative to the SSH login home without traversal components\n' >&2
  exit 2
fi

remote_compose=$remote_dir/docker-compose.yaml
remote_env=$remote_dir/deployment.env
existing=$(ssh "$ssh_target" "test -e '$remote_compose' && printf existing || true")
if [[ "$existing" == "existing" && "$replace_existing" -ne 1 ]]; then
  printf 'remote compose file already exists; capture rollback state and pass --replace-existing only after review\n' >&2
  exit 4
fi

ssh "$ssh_target" "umask 077 && mkdir -p '$remote_dir' && chmod 700 '$remote_dir'"
scp "$compose_file" "$ssh_target:$remote_compose"
scp "$env_file" "$ssh_target:$remote_env"
ssh "$ssh_target" "chmod 600 '$remote_compose' '$remote_env'"
ssh "$ssh_target" "docker compose --env-file '$remote_env' -f '$remote_compose' config >/dev/null"
ssh "$ssh_target" "docker compose --env-file '$remote_env' -f '$remote_compose' up -d"
compose_ps=$(ssh "$ssh_target" "docker compose --env-file '$remote_env' -f '$remote_compose' ps")

APPLY_COMPOSE_PS=$compose_ps python3 - "$state_out" "$ssh_target" "$remote_dir" "$compose_file" "$env_file" "$replace_existing" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import sys

state_out, ssh_target, remote_dir, compose_file, env_file, replace_existing = sys.argv[1:]
quoted_target = shlex.quote(ssh_target)
quoted_remote_env = shlex.quote(f"{remote_dir}/deployment.env")
quoted_remote_compose = shlex.quote(f"{remote_dir}/docker-compose.yaml")
state = {
    "schema_version": "nvidia-applied-deployment-state/v1",
    "applied_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "deployment_module": "vllm-compose-v1",
    "ssh_target": ssh_target,
    "remote_dir": remote_dir,
    "local_rendered_files": {
        "compose": compose_file,
        "environment": env_file,
    },
    "replace_existing": replace_existing == "1",
    "commands": [
        "ssh <target> mkdir -p <remote-dir>",
        "scp <compose> <target>:<remote-dir>/docker-compose.yaml",
        "scp <env> <target>:<remote-dir>/deployment.env",
        "docker compose config",
        "docker compose up -d",
    ],
    "remote_compose_ps": os.environ.get("APPLY_COMPOSE_PS", ""),
    "rollback_command": (
        f"ssh {quoted_target} 'docker compose --env-file {quoted_remote_env} "
        f"-f {quoted_remote_compose} down'"
    ),
}
with open(state_out, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf 'wrote %s\n' "$state_out"
