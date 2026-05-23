#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "skills" / "deploy-nvidia-inference"
SCRIPTS = SKILL / "scripts"
ASSETS = SKILL / "assets"


class DeployNvidiaInferenceTests(unittest.TestCase):
    def run_python(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=True,
        )

    def normalize_example_host(self, tmp: Path) -> Path:
        host_facts = tmp / "host_facts.json"
        self.run_python(
            str(SCRIPTS / "normalize_host_facts.py"),
            str(ASSETS / "host_probe.example.json"),
            "--out",
            str(host_facts),
        )
        return host_facts

    def render_plan(self, tmp: Path, host_facts: Path, *extra: str) -> Path:
        plan = tmp / "deployment_plan.yaml"
        self.run_python(
            str(SCRIPTS / "render_deployment_plan.py"),
            "--host",
            str(host_facts),
            "--workload",
            str(ASSETS / "workload_profile.example.yaml"),
            "--candidate",
            str(ASSETS / "selected_candidate.example.json"),
            "--connection-file",
            str(ASSETS / "remote_connection.example.yaml"),
            "--out",
            str(plan),
            "--compose-out",
            str(tmp / "docker-compose.yaml"),
            "--env-out",
            str(tmp / "deployment.env"),
            "--k8s-out",
            str(tmp / "kubernetes.yaml"),
            *extra,
        )
        return plan

    def rank_candidates(self, tmp: Path, host_facts: Path) -> Path:
        scorecard = tmp / "candidate_scorecard.json"
        self.run_python(
            str(SCRIPTS / "rank_candidates.py"),
            "--host",
            str(host_facts),
            "--workload",
            str(ASSETS / "workload_profile.example.yaml"),
            "--candidates",
            str(ASSETS / "candidates.template.json"),
            "--out",
            str(scorecard),
        )
        return scorecard

    def test_normalize_records_deployment_substrates(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            host_facts = self.normalize_example_host(tmp)
            facts = json.loads(host_facts.read_text(encoding="utf-8"))

        substrates = facts["deployment_substrates"]
        self.assertEqual(substrates["priority_order"], ["kubernetes", "docker", "native_service"])
        self.assertFalse(substrates["kubernetes"]["available"])
        self.assertTrue(substrates["docker"]["available"])
        self.assertFalse(substrates["docker"]["compose_available"])
        self.assertFalse(substrates["native_service"]["available"])

    def test_render_plan_selects_docker_after_kubernetes_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            host_facts = self.normalize_example_host(tmp)
            plan = self.render_plan(tmp, host_facts)
            text = plan.read_text(encoding="utf-8")

        self.assertIn('selected: "docker"', text)
        self.assertIn("Docker is available but Docker Compose is unavailable", text)
        self.assertIn('name: "vllm-compose-v1"', text)

    def test_candidate_scorecard_contains_backend_decision(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            host_facts = self.normalize_example_host(tmp)
            scorecard = self.rank_candidates(tmp, host_facts)
            payload = json.loads(scorecard.read_text(encoding="utf-8"))

        decision = payload["backend_decision"]
        self.assertEqual(decision["status"], "recommended")
        self.assertEqual(decision["selected_backend"], "vllm")
        self.assertEqual(decision["selected_candidate_id"], "replace-org/replace-model-vllm")

    def test_render_plan_prioritizes_usable_kubernetes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            host_facts = self.normalize_example_host(tmp)
            facts = json.loads(host_facts.read_text(encoding="utf-8"))
            facts["deployment_substrates"]["kubernetes"].update(
                {
                    "available": True,
                    "client_available": True,
                    "cluster_reachable": True,
                    "workload_create_allowed": True,
                    "current_context": "test-cluster",
                    "blockers": [],
                }
            )
            host_facts.write_text(json.dumps(facts), encoding="utf-8")
            plan = self.render_plan(tmp, host_facts)
            text = plan.read_text(encoding="utf-8")
            manifest = (tmp / "kubernetes.yaml").read_text(encoding="utf-8")

        self.assertIn('selected: "kubernetes"', text)
        self.assertIn('name: "vllm-k8s-v1"', text)
        self.assertIn('status: "implemented"', text)
        self.assertIn("scripts/apply_k8s.sh", text)
        self.assertIn("--connection-file", text)
        self.assertIn("kubectl -n default delete -f .local/share/codex-inference/nvidia-inference/kubernetes.yaml", text)
        self.assertIn("kind: \"Deployment\"", manifest)
        self.assertIn("kind: \"Service\"", manifest)
        self.assertIn("nvidia.com/gpu", manifest)
        self.assertIn("vllm/vllm-openai@sha256:replace-with-digest", manifest)

    def test_explicit_substrate_request_does_not_silently_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            host_facts = self.normalize_example_host(tmp)
            plan = self.render_plan(tmp, host_facts, "--substrate", "kubernetes")
            text = plan.read_text(encoding="utf-8")

        self.assertIn('selected: "kubernetes"', text)
        self.assertIn("requested substrate is unavailable; not falling back automatically", text)


if __name__ == "__main__":
    unittest.main()
