#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s [--connection-file FILE | <ssh-target-or-connection>] [ssh-options...]\n' "$0" >&2
}

connection_file=
ssh_target=
remote_exec_prefix=
remote_payload_dir="$(cd "$(dirname "$0")" && pwd)"
remote_payload="$remote_payload_dir/probe_remote_host_payload.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --connection-file)
      connection_file=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n "$connection_file" ]]; then
  eval "$(python3 "$remote_payload_dir/remote_connection.py" --connection-file "$connection_file" --shell-env)"
  if [[ "$TRANSPORT" == "brev" ]]; then
    if ! command -v brev >/dev/null 2>&1; then
      printf 'brev CLI is required for Brev connections\n' >&2
      exit 127
    fi
    brev_args=(brev exec "$BREV_INSTANCE" "@$remote_payload")
    exec "${brev_args[@]}"
  fi
  exec ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" python3 - < "$remote_payload"
fi

if [[ $# -gt 0 ]]; then
  # Backward-compatible direct SSH mode with optional extra ssh options.
  ssh_target=$1
  if [[ "$ssh_target" == -* ]]; then
    printf 'ssh target must not start with a hyphen\n' >&2
    exit 2
  fi
  shift || true
  exec ssh "$@" "$ssh_target" python3 - < "$remote_payload"
fi

usage
exit 2
