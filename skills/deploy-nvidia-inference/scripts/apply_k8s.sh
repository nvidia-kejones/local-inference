#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s --manifest FILE --deployment NAME [--connection-file FILE] [--remote-dir DIR] [--namespace NAME] [--context NAME] [--state-out FILE] --apply --allow-model-downloads\n' "$0" >&2
}

manifest=
deployment=
connection_file=
remote_dir=.local/share/codex-inference/kubernetes
namespace=default
context=
state_out=applied_deployment_state.json
apply=0
allow_downloads=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest=$2; shift 2 ;;
    --deployment) deployment=$2; shift 2 ;;
    --connection-file) connection_file=$2; shift 2 ;;
    --remote-dir) remote_dir=$2; shift 2 ;;
    --namespace) namespace=$2; shift 2 ;;
    --context) context=$2; shift 2 ;;
    --state-out) state_out=$2; shift 2 ;;
    --apply) apply=1; shift ;;
    --allow-model-downloads) allow_downloads=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$manifest" || -z "$deployment" ]]; then
  usage
  exit 2
fi
if [[ "$apply" -ne 1 || "$allow_downloads" -ne 1 ]]; then
  printf 'refusing Kubernetes write: pass both --apply and --allow-model-downloads after reviewing the plan\n' >&2
  exit 3
fi
if [[ ! -f "$manifest" ]]; then
  printf 'manifest file must exist locally\n' >&2
  exit 2
fi
if [[ "$manifest" == -* || "$state_out" == -* ]]; then
  printf 'file arguments must not start with a hyphen\n' >&2
  exit 2
fi
if [[ ! "$manifest" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  printf 'manifest path must be a simple path\n' >&2
  exit 2
fi
if [[ ! "$deployment" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  printf 'deployment must be a Kubernetes DNS label\n' >&2
  exit 2
fi
if [[ ! "$namespace" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  printf 'namespace must be a Kubernetes DNS label\n' >&2
  exit 2
fi
if [[ -n "$context" && ( "$context" == -* || ! "$context" =~ ^[A-Za-z0-9._:@/-]+$ ) ]]; then
  printf 'context must be a simple kubectl context name\n' >&2
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

script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "$connection_file" ]]; then
  eval "$(python3 "$script_dir/remote_connection.py" --connection-file "$connection_file" --shell-env)"
else
  command -v kubectl >/dev/null 2>&1 || {
    printf 'kubectl is required for Kubernetes apply\n' >&2
    exit 127
  }
fi

remote_exec() {
  local remote_command=$1
  if [[ -z "$connection_file" ]]; then
    sh -lc "$remote_command"
    return
  fi
  if [[ "$TRANSPORT" == "brev" ]]; then
    command -v brev >/dev/null 2>&1 || {
      printf 'brev CLI is required for Brev connections\n' >&2
      exit 127
    }
    brev exec "$BREV_INSTANCE" "$remote_command"
    return
  fi
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "$remote_command"
}

remote_copy() {
  local src=$1
  local dest=$2
  if [[ -z "$connection_file" ]]; then
    return
  fi
  if [[ "$TRANSPORT" == "brev" ]]; then
    command -v brev >/dev/null 2>&1 || {
      printf 'brev CLI is required for Brev connections\n' >&2
      exit 127
    }
    brev copy "$src" "${BREV_INSTANCE}:${dest}"
    return
  fi
  scp "${SSH_OPTIONS[@]}" "$src" "$SSH_TARGET:$dest"
}

kubectl_base=kubectl
if [[ -n "$context" ]]; then
  kubectl_base="kubectl --context '$context'"
fi
manifest_for_apply=$manifest
if [[ -n "$connection_file" ]]; then
  remote_exec "umask 077 && mkdir -p '$remote_dir' && chmod 700 '$remote_dir'"
  manifest_for_apply="$remote_dir/kubernetes.yaml"
  remote_copy "$manifest" "$manifest_for_apply"
  remote_exec "chmod 600 '$manifest_for_apply'"
fi

can_create_pods=$(remote_exec "$kubectl_base -n '$namespace' auth can-i create pods")
if [[ "$can_create_pods" != "yes" ]]; then
  printf 'kubectl auth can-i create pods returned %s; refusing apply\n' "$can_create_pods" >&2
  exit 4
fi
apply_output=$(remote_exec "$kubectl_base -n '$namespace' apply -f '$manifest_for_apply'")
rollout_output=$(remote_exec "$kubectl_base -n '$namespace' rollout status 'deployment/$deployment' --timeout=600s")
service_output=$(remote_exec "$kubectl_base -n '$namespace' get service '$deployment' -o wide 2>&1 || true")
pods_output=$(remote_exec "$kubectl_base -n '$namespace' get pods -l 'app.kubernetes.io/name=$deployment' -o wide 2>&1 || true")

python3 - "$state_out" "$manifest" "$manifest_for_apply" "$deployment" "$namespace" "$context" "${connection_file:-}" "$can_create_pods" "$apply_output" "$rollout_output" "$service_output" "$pods_output" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import shlex
import sys

(
    state_out,
    manifest,
    manifest_for_apply,
    deployment,
    namespace,
    context,
    connection_file,
    can_create_pods,
    apply_output,
    rollout_output,
    service_output,
    pods_output,
) = sys.argv[1:]
kubectl_prefix = "kubectl"
if context:
    kubectl_prefix += f" --context {shlex.quote(context)}"
state = {
    "schema_version": "nvidia-applied-deployment-state/v1",
    "applied_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "deployment_module": "kubernetes-v1",
    "manifest": manifest,
    "applied_manifest": manifest_for_apply,
    "namespace": namespace,
    "deployment": deployment,
    "context": context or None,
    "remote_connection": connection_file or None,
    "commands": [
        "kubectl auth can-i create pods",
        "kubectl apply -f manifest",
        "kubectl rollout status deployment",
        "kubectl get service",
        "kubectl get pods",
    ],
    "kubectl_can_create_pods": can_create_pods,
    "kubectl_apply": apply_output,
    "kubectl_rollout_status": rollout_output,
    "kubectl_service": service_output,
    "kubectl_pods": pods_output,
    "rollback_command": (
        f"{kubectl_prefix} -n {shlex.quote(namespace)} delete -f {shlex.quote(manifest)}"
    ),
}
with open(state_out, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf 'wrote %s\n' "$state_out"
