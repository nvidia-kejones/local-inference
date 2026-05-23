#!/usr/bin/env python3
"""Render an auditable deployment plan and the v1 vLLM Compose baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from common_io import as_int, load_structured, nested, write_yaml
from deployment_selection import build_deployment_selection
from fitlib import estimate_fit
from remote_connection import (
    normalize_connection,
    remote_exec_command,
    remote_port_forward_command,
)


SKILL_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = SKILL_DIR / "assets" / "templates"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="normalized host_facts.json")
    parser.add_argument("--workload", required=True, help="workload profile YAML or JSON")
    parser.add_argument("--candidate", required=True, help="selected candidate JSON or YAML")
    parser.add_argument("--out", required=True, help="write deployment_plan.yaml here")
    parser.add_argument("--compose-out", help="write rendered vLLM Compose YAML here")
    parser.add_argument("--env-out", help="write rendered vLLM environment file here")
    parser.add_argument("--k8s-out", help="write rendered Kubernetes manifests here")
    parser.add_argument("--service-name", default="nvidia-inference")
    parser.add_argument("--connection-file", help="YAML or JSON remote connection spec")
    parser.add_argument("--remote-dir", help="remote working dir relative to SSH login home")
    parser.add_argument("--ssh-target", default="<ssh-target>")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--substrate",
        choices=["kubernetes", "k8s", "docker", "compose", "native_service", "native", "service"],
        help="override deployment substrate selection for this plan",
    )
    args = parser.parse_args()

    host = load_structured(args.host)
    workload = load_structured(args.workload)
    candidate = load_structured(args.candidate)
    if not isinstance(candidate, dict) or "candidates" in candidate:
        raise SystemExit("--candidate must point to one selected candidate")
    runtime = str(candidate.get("runtime", "")).lower()
    model_id = candidate.get("model_id") or candidate.get("id")
    if not model_id:
        raise SystemExit("selected candidate needs model_id or id")
    connection = load_connection(args)
    if args.substrate:
        workload.setdefault("deployment", {})["preferred_substrate"] = args.substrate

    remote_dir = args.remote_dir or f".local/share/codex-inference/{slug(args.service_name)}"
    host_port = args.port or as_int(nested(candidate, "deployment", "host_port", default=0), 0)
    host_port = host_port or (11434 if runtime == "ollama" else 8000)
    if not loopback_bind(args.bind) and not nested(
        workload, "endpoint", "external_exposure", default=False
    ):
        raise SystemExit(
            "non-loopback --bind requires workload endpoint.external_exposure: true"
        )
    fit = estimate_fit(host, workload, candidate)
    pinning = pinning_state(candidate)
    selection = build_deployment_selection(host, workload, candidate, fit)
    module = selection["deployment_module"]
    namespace = k8s_namespace(workload, candidate)
    apply_blocker_list = unique_blockers(
        apply_blockers(runtime, pinning) + selection["apply_blockers"]
    )
    plan: dict[str, Any] = {
        "schema_version": "nvidia-deployment-plan/v1",
        "mode": "plan",
        "service_name": args.service_name,
        "runtime": runtime,
        "selected_backend": selection["selected_backend"],
        "selected_substrate": selection["selected_substrate"],
        "selection_rationale": {
            "backend": selection["selected_backend"].get("rationale", []),
            "substrate": selection["selected_substrate"].get("rationale", []),
            "rejected_substrates": selection["selected_substrate"].get("rejected", []),
        },
        "selected_candidate": {
            "id": candidate.get("id") or model_id,
            "model_id": model_id,
            "model_revision": candidate.get("model_revision"),
            "container_image": nested(candidate, "deployment", "container_image", default=None),
        },
        "artifact_boundaries": {
            "host_facts": args.host,
            "workload_profile": args.workload,
            "recommendation_input": args.candidate,
            "remote_connection": args.connection_file or args.ssh_target,
            "applied_state": "applied_deployment_state.json",
            "verification_report": "verification_report.json",
        },
        "host_summary": {
            "hostname": nested(host, "host", "hostname", default=None),
            "home_dir": nested(host, "host", "home_dir", default=None),
            "gpu_count": len(nested(host, "nvidia", "gpus", default=[]) or []),
            "driver_version": nested(host, "nvidia", "driver_version", default=None),
            "kubernetes_available": nested(
                host, "deployment_substrates", "kubernetes", "available", default=False
            ),
            "docker_available": nested(host, "containers", "docker", "available", default=False),
            "docker_compose_available": nested(
                host, "deployment_substrates", "docker", "compose_available", default=False
            ),
            "native_service_available": nested(
                host, "deployment_substrates", "native_service", "available", default=False
            ),
            "nvidia_container_toolkit_available": nested(
                host, "containers", "nvidia_container_toolkit", "available", default=False
            ),
        },
        "endpoint": {
            "contract": "OpenAI-compatible where supported",
            "bind_host": args.bind,
            "host_port": host_port,
            "external_exposure_requested": bool(
                nested(workload, "endpoint", "external_exposure", default=False)
            ),
        },
        "deployment_topology": {
            "model_endpoint_count": deployment_endpoint_count(workload),
            "selected_gpu_devices": selected_gpu_devices(fit),
            "shared_hf_cache_path": shared_hf_cache_path(host, workload),
        },
        "fit_estimate": fit,
        "pinning": pinning,
        "apply_blockers": apply_blocker_list,
        "commands": {
            "preflight": preflight_commands(connection, candidate, host_port, selection),
            "verify": verify_commands(
                connection, args.service_name, candidate, model_id, host_port, selection, namespace
            ),
        },
        "rollback": {
            "before_replace": [
                "Record the existing service unit/Compose file, image or binary revision, environment file, port binding, and model cache path.",
                "Capture current endpoint smoke results if an existing endpoint will be replaced.",
            ],
            "guidance": [
                "Stop only the new deployment module first.",
                "Restore the prior config/revision and re-run its documented start command.",
                "Re-run endpoint smoke tests and inspect visible GPU workloads before declaring rollback complete.",
            ],
        },
    }

    if module["status"] == "implemented" and runtime == "vllm" and selection["selected_substrate"]["selected"] == "docker":
        rendered = render_vllm(args, host, workload, candidate, fit, host_port)
        if args.compose_out:
            Path(args.compose_out).write_text(rendered["compose"], encoding="utf-8")
        if args.env_out:
            write_private_text(Path(args.env_out), rendered["env"])
        plan["deployment_module"] = {
            "name": module["name"],
            "status": module["status"],
            "reference": module["reference"],
            "substrate": selection["selected_substrate"]["selected"],
            "rendered_files": {
                "compose": args.compose_out or "<render with --compose-out>",
                "environment": args.env_out or "<render with --env-out>",
            },
            "remote_dir": remote_dir,
        }
        if deployment_endpoint_count(workload) > 1:
            plan["commands"]["preflight"].append(
                remote_exec_command(
                    connection,
                    f"umask 077 && mkdir -p {shlex.quote(shared_hf_cache_path(host, workload))} "
                    f"&& chmod 700 {shlex.quote(shared_hf_cache_path(host, workload))}",
                )
            )
        plan["commands"]["apply"] = [
            "Review deployment_plan.yaml, rendered Compose, and environment files.",
            apply_command(args, remote_dir),
        ]
        plan["rollback"]["new_module_stop_command"] = remote_exec_command(
            connection,
            f"docker compose --env-file {shlex.quote(remote_dir + '/deployment.env')} "
            f"-f {shlex.quote(remote_dir + '/docker-compose.yaml')} down",
        )
    elif module["status"] == "implemented" and runtime == "vllm" and selection["selected_substrate"]["selected"] == "kubernetes":
        manifest = render_vllm_k8s(args, workload, candidate, fit, host_port)
        if args.k8s_out:
            Path(args.k8s_out).write_text(manifest, encoding="utf-8")
        plan["deployment_module"] = {
            "name": module["name"],
            "status": module["status"],
            "reference": module["reference"],
            "substrate": selection["selected_substrate"]["selected"],
            "rendered_files": {
                "kubernetes_manifests": args.k8s_out or "<render with --k8s-out>",
            },
            "namespace": namespace,
        }
        plan["commands"]["apply"] = [
            "Review deployment_plan.yaml and rendered Kubernetes manifests.",
            apply_k8s_command(args, namespace, remote_dir),
        ]
        plan["rollback"]["new_module_stop_command"] = remote_exec_command(
            connection,
            f"kubectl -n {shlex.quote(namespace)} delete -f {shlex.quote(remote_dir + '/kubernetes.yaml')}",
        )
    else:
        plan["deployment_module"] = {
            "name": module["name"],
            "status": module["status"],
            "reference": module["reference"],
            "substrate": selection["selected_substrate"]["selected"],
            "inputs_already_available": [
                "host_facts.json",
                "workload_profile.yaml",
                "selected candidate with pins",
            ],
            "expected_outputs": module["expected_outputs"],
        }
    write_yaml(plan, args.out)


def render_vllm(
    args: argparse.Namespace,
    host: dict[str, Any],
    workload: dict[str, Any],
    candidate: dict[str, Any],
    fit: dict[str, Any],
    host_port: int,
) -> dict[str, str]:
    template = (TEMPLATE_DIR / "docker-compose.vllm.yaml.tmpl").read_text(encoding="utf-8")
    env_template = (TEMPLATE_DIR / "vllm.deployment.env.tmpl").read_text(encoding="utf-8")
    max_model_len = as_int(
        nested(workload, "serving", "target_context_tokens", default=0)
        or candidate.get("max_context_tokens"),
        4096,
    )
    revision = str(candidate.get("model_revision") or "")
    revision_args = (
        f'      - "--revision"\n      - "${{MODEL_REVISION}}"\n' if revision else ""
    )
    extra_args = yaml_command_args(nested(candidate, "deployment", "vllm_args", default=[]))
    tokens = {
        "@@SERVICE_NAME@@": slug(args.service_name),
        "@@REVISION_ARGS@@": revision_args.rstrip(),
        "@@EXTRA_ARGS@@": extra_args.rstrip(),
    }
    for token, value in tokens.items():
        template = template.replace(token, value)
    env_values = {
        "@@SERVICE_NAME@@": slug(args.service_name),
        "@@BIND_HOST@@": args.bind,
        "@@HOST_PORT@@": str(host_port),
        "@@VLLM_IMAGE@@": str(
            nested(candidate, "deployment", "container_image", default="vllm/vllm-openai:latest")
        ),
        "@@MODEL_ID@@": str(candidate.get("model_id") or candidate.get("id")),
        "@@MODEL_REVISION@@": revision,
        "@@SERVED_MODEL_NAME@@": str(
            candidate.get("served_model_name") or candidate.get("model_id") or candidate.get("id")
        ),
        "@@MAX_MODEL_LEN@@": str(max_model_len),
        "@@TENSOR_PARALLEL_SIZE@@": str(
            as_int(nested(candidate, "deployment", "tensor_parallel_size", default=1), 1)
        ),
        "@@NVIDIA_VISIBLE_DEVICES@@": selected_gpu_devices(fit),
        "@@HF_CACHE@@": shared_hf_cache_path(host, workload),
    }
    for token, value in env_values.items():
        env_template = env_template.replace(token, env_escape(value))
    return {"compose": template.rstrip() + "\n", "env": env_template.rstrip() + "\n"}


def render_vllm_k8s(
    args: argparse.Namespace,
    workload: dict[str, Any],
    candidate: dict[str, Any],
    fit: dict[str, Any],
    host_port: int,
) -> str:
    service_name = slug(args.service_name)
    namespace = k8s_namespace(workload, candidate)
    image = str(nested(candidate, "deployment", "container_image", default="vllm/vllm-openai:latest"))
    model_id = str(candidate.get("model_id") or candidate.get("id"))
    served_model_name = str(candidate.get("served_model_name") or model_id)
    revision = str(candidate.get("model_revision") or "")
    max_model_len = as_int(
        nested(workload, "serving", "target_context_tokens", default=0)
        or candidate.get("max_context_tokens"),
        4096,
    )
    tensor_parallel_size = as_int(
        nested(candidate, "deployment", "tensor_parallel_size", default=1), 1
    )
    gpu_count = max(1, len(k8s_gpu_resource_indices(fit)) or tensor_parallel_size)
    command_args = [
        "--model",
        model_id,
        "--served-model-name",
        served_model_name,
    ]
    if revision:
        command_args.extend(["--revision", revision])
    command_args.extend(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-model-len",
            str(max_model_len),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ]
    )
    extra_args = nested(candidate, "deployment", "vllm_args", default=[])
    if isinstance(extra_args, list):
        command_args.extend(str(item) for item in extra_args)
    documents = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": service_name, "namespace": namespace},
            "data": {
                "MODEL_ID": model_id,
                "MODEL_REVISION": revision,
                "SERVED_MODEL_NAME": served_model_name,
                "MAX_MODEL_LEN": str(max_model_len),
                "TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service_name,
                "namespace": namespace,
                "labels": {"app.kubernetes.io/name": service_name},
            },
            "spec": {
                "replicas": as_int(nested(candidate, "deployment", "replicas", default=1), 1),
                "selector": {"matchLabels": {"app.kubernetes.io/name": service_name}},
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": service_name}},
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "image": image,
                                "imagePullPolicy": "IfNotPresent",
                                "args": command_args,
                                "ports": [{"name": "http", "containerPort": 8000}],
                                "env": [
                                    {"name": "HF_HOME", "value": "/root/.cache/huggingface"},
                                    {
                                        "name": "HF_TOKEN",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": f"{service_name}-hf-token",
                                                "key": "HF_TOKEN",
                                                "optional": True,
                                            }
                                        },
                                    },
                                ],
                                "resources": {
                                    "limits": {"nvidia.com/gpu": gpu_count},
                                    "requests": {"nvidia.com/gpu": gpu_count},
                                },
                                "volumeMounts": [
                                    {
                                        "name": "hf-cache",
                                        "mountPath": "/root/.cache/huggingface",
                                    }
                                ],
                                "readinessProbe": {
                                    "httpGet": {"path": "/v1/models", "port": "http"},
                                    "initialDelaySeconds": 20,
                                    "periodSeconds": 10,
                                    "failureThreshold": 30,
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "hf-cache",
                                "emptyDir": {},
                            }
                        ],
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": service_name, "namespace": namespace},
            "spec": {
                "type": "ClusterIP",
                "selector": {"app.kubernetes.io/name": service_name},
                "ports": [
                    {
                        "name": "http",
                        "port": host_port,
                        "targetPort": "http",
                    }
                ],
            },
        },
    ]
    return "\n---\n".join(dump_yaml_document(document) for document in documents) + "\n"


def yaml_command_args(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    lines = []
    for value in values:
        lines.append(f"      - {json.dumps(str(value), ensure_ascii=True)}")
    return "\n".join(lines)


def pinning_state(candidate: dict[str, Any]) -> dict[str, Any]:
    image = str(nested(candidate, "deployment", "container_image", default="") or "")
    return {
        "model_revision_pinned": bool(candidate.get("model_revision")),
        "container_image_digest_pinned": "@sha256:" in image,
        "container_image": image or None,
        "note": "Pin the deployed model revision and image digest in the candidate before apply.",
    }


def apply_blockers(runtime: str, pinning: dict[str, Any]) -> list[str]:
    blockers = []
    if runtime == "vllm":
        if not pinning["model_revision_pinned"]:
            blockers.append("selected model revision is not pinned")
        if not pinning["container_image_digest_pinned"]:
            blockers.append("vLLM container image digest is not pinned")
    if runtime != "vllm":
        blockers.append("runtime apply module is not implemented in v1; use reviewed manual commands")
    return blockers


def apply_k8s_command(args: argparse.Namespace, namespace: str, remote_dir: str) -> str:
    manifest = args.k8s_out or "<k8s-manifest-file>"
    connection_args = (
        f"--connection-file {shlex.quote(args.connection_file)}"
        if args.connection_file
        else ""
    )
    return (
        "scripts/apply_k8s.sh "
        f"--manifest {shlex.quote(manifest)} {connection_args} "
        f"--remote-dir {shlex.quote(remote_dir)} --namespace {shlex.quote(namespace)} "
        f"--deployment {shlex.quote(slug(args.service_name))} "
        "--state-out applied_deployment_state.json --apply --allow-model-downloads"
    )


def preflight_commands(
    connection: dict[str, Any],
    candidate: dict[str, Any],
    host_port: int,
    selection: dict[str, Any],
) -> list[str]:
    selected = selection["selected_substrate"]["selected"]
    commands = [remote_exec_command(connection, "nvidia-smi -L")]
    if selected == "kubernetes":
        commands.extend(
            [
                remote_exec_command(connection, "kubectl version --client=true"),
                remote_exec_command(connection, "kubectl auth can-i create pods"),
                remote_exec_command(connection, "kubectl get nodes -o wide"),
            ]
        )
    elif selected == "docker":
        image = str(nested(candidate, "deployment", "container_image", default="") or "")
        commands.append(remote_exec_command(connection, "docker compose version"))
        if image:
            commands.append(remote_exec_command(connection, f"docker pull {shlex.quote(image)}"))
        commands.append(remote_exec_command(connection, f"ss -H -ltn | grep :{host_port} || true"))
    elif selected == "native_service":
        commands.extend(
            [
                remote_exec_command(connection, "systemctl --version"),
                remote_exec_command(connection, f"ss -H -ltn | grep :{host_port} || true"),
            ]
        )
    else:
        commands.append("No deployment substrate preflight is available because no substrate was selected.")
    return commands


def verify_commands(
    connection: dict[str, Any],
    service_name: str,
    candidate: dict[str, Any],
    model_id: str,
    host_port: int,
    selection: dict[str, Any],
    namespace: str,
) -> list[str]:
    selected = selection["selected_substrate"]["selected"]
    served_model = shlex.quote(str(candidate.get("served_model_name") or model_id))
    smoke = (
        f"python3 scripts/smoke_test_endpoint.py --base-url http://127.0.0.1:{host_port} "
        f"--model {served_model} --out verification_report.json"
    )
    if selected == "kubernetes":
        return [
            remote_exec_command(
                connection,
                f"kubectl -n {shlex.quote(namespace)} port-forward service/{slug(service_name)} {host_port}:{host_port}",
            ),
            remote_port_forward_command(connection, host_port, host_port),
            smoke,
        ]
    return [
        remote_port_forward_command(connection, host_port, host_port),
        smoke,
    ]


def unique_blockers(blockers: list[str]) -> list[str]:
    result = []
    seen = set()
    for blocker in blockers:
        if blocker and blocker not in seen:
            result.append(blocker)
            seen.add(blocker)
    return result


def deployment_endpoint_count(workload: dict[str, Any]) -> int:
    candidates = [
        nested(workload, "deployment", "endpoint_count", default=0),
        nested(workload, "deployment", "model_endpoint_count", default=0),
        nested(workload, "deployment", "service_count", default=0),
    ]
    for value in candidates:
        count = as_int(value, 0)
        if count > 0:
            return count
    for key in ("endpoints", "model_endpoints", "services"):
        entries = nested(workload, "deployment", key, default=None)
        if isinstance(entries, list) and entries:
            return len(entries)
    return 1


def shared_hf_cache_path(host: dict[str, Any], workload: dict[str, Any]) -> str:
    if deployment_endpoint_count(workload) <= 1:
        return "./hf-cache"
    home_dir = str(nested(host, "host", "home_dir", default="") or "").strip()
    if home_dir:
        return f"{home_dir.rstrip('/')}/.cache/huggingface"
    remote_user = str(nested(host, "source", "remote_user", default="") or "").strip()
    if remote_user == "root":
        return "/root/.cache/huggingface"
    if remote_user:
        return f"/home/{remote_user}/.cache/huggingface"
    return "/var/lib/huggingface-cache"


def selected_gpu_devices(fit: dict[str, Any]) -> str:
    selected = nested(fit, "selected_gpus", default=[]) or []
    if not isinstance(selected, list) or not selected:
        return "all"
    devices = [str(gpu.get("index")) for gpu in selected if gpu.get("index") is not None]
    return ",".join(devices) if devices else "all"


def k8s_gpu_resource_indices(fit: dict[str, Any]) -> list[str]:
    selected = nested(fit, "selected_gpus", default=[]) or []
    if not isinstance(selected, list):
        return []
    return [str(gpu.get("index")) for gpu in selected if gpu.get("index") is not None]


def k8s_namespace(workload: dict[str, Any], candidate: dict[str, Any]) -> str:
    namespace = (
        nested(workload, "deployment", "kubernetes_namespace", default=None)
        or nested(workload, "deployment", "namespace", default=None)
        or nested(candidate, "deployment", "kubernetes_namespace", default=None)
        or nested(candidate, "deployment", "namespace", default=None)
        or "default"
    )
    return slug(str(namespace))


def dump_yaml_document(data: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(data, dict):
        if not data:
            return f"{prefix}{{}}"
        lines = []
        for key, value in data.items():
            if value == {}:
                lines.append(f"{prefix}{key}: {{}}")
                continue
            if value == []:
                lines.append(f"{prefix}{key}: []")
                continue
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(dump_yaml_document(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return f"{prefix}[]"
        lines = []
        for value in data:
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml_document(value, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(value)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(data)}"


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "nvidia-inference"


def env_escape(value: str) -> str:
    return value.replace("\n", "").replace("\r", "")


def loopback_bind(value: str) -> bool:
    return value in {"127.0.0.1", "localhost", "::1"}


def load_connection(args: argparse.Namespace) -> dict[str, Any]:
    if args.connection_file:
        return normalize_connection(load_structured(args.connection_file))
    return normalize_connection({"kind": "ssh", "target": args.ssh_target})


def apply_command(args: argparse.Namespace, remote_dir: str) -> str:
    compose_file = args.compose_out or "<compose-file>"
    env_file = args.env_out or "<env-file>"
    connection_args = (
        f"--connection-file {shlex.quote(args.connection_file)}"
        if args.connection_file
        else f"--ssh-target {shlex.quote(args.ssh_target)}"
    )
    return (
        "scripts/apply_vllm_compose.sh "
        f"{connection_args} --remote-dir {shlex.quote(remote_dir)} "
        f"--compose {shlex.quote(compose_file)} --env {shlex.quote(env_file)} "
        "--state-out applied_deployment_state.json --apply --allow-model-downloads"
    )


def ssh_command(target: str, remote_command: str) -> str:
    return f"ssh {shlex.quote(target)} {shlex.quote(remote_command)}"


def write_private_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


if __name__ == "__main__":
    main()
