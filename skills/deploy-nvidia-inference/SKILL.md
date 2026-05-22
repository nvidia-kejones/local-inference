---
name: deploy-nvidia-inference
description: Discover a single remote NVIDIA Linux host over SSH, build refreshable model/runtime recommendations, plan and apply safe inference deployments, and verify OpenAI-compatible endpoints with vLLM, SGLang, TensorRT-LLM/trtllm-serve, llama.cpp, or Ollama. Use when Codex needs to assess NVIDIA GPU fit, choose an inference serving path for a workload, deploy a remote endpoint, benchmark it, or document rollback guidance without treating one runtime or parameter count as universally optimal.
---

# Deploy NVIDIA Inference

Use this skill as a staged remote-deployment workflow. Keep host facts, workload intent, recommendations, applied state, and verification results in separate artifacts so later agents can audit assumptions against what was actually changed.

## Guardrails

- Keep discovery read-only. Use `scripts/probe_remote_host.sh` and `scripts/normalize_host_facts.py` before any install, write, service change, firewall change, or model download.
- Treat every deployment write as an explicit apply action. The v1 scripted apply path is the vLLM Docker Compose baseline and it requires apply/download flags.
- Bind endpoints to `127.0.0.1` by default. Add external exposure only after the user asks for it and the plan records the reverse proxy, authentication, firewall, and rollback implications.
- Prefer an OpenAI-compatible endpoint contract when the selected runtime provides one.
- Do not rank candidates from memory. Refresh model support, runtime support, quantization support, and license/deployment constraints from primary docs and model repos before pinning a candidate set.
- Preserve rollback guidance before replacing an existing inference service. Record commands, image/model revisions, rendered configuration, and verification commands.

## Workflow

1. Capture intent in `workload_profile.yaml`.
   Start from `assets/workload_profile.example.yaml`. Record context length, expected concurrent sequences, batching/live-token expectations, serving patterns, endpoint exposure, quality/latency priorities, and license constraints.
2. Discover the host without changing it.
   Run `scripts/probe_remote_host.sh <ssh-target> > host_probe.raw.json`, then `python3 scripts/normalize_host_facts.py host_probe.raw.json --out host_facts.json`.
3. Refresh the recommendation basis.
   Read [runtime-selection.md](references/runtime-selection.md), [host-discovery.md](references/host-discovery.md), and [model-fit.md](references/model-fit.md). Read only the runtime reference files relevant to the candidate set you are building.
4. Build and score candidate model/runtime pairs.
   Create a candidate file from current primary docs and model metadata rather than a permanent "latest models" list. Use `python3 scripts/rank_candidates.py --host host_facts.json --workload workload_profile.yaml --candidates candidate_set.json --out candidate_scorecard.json` for one workload, or `python3 scripts/recommend_use_cases.py --host host_facts.json --profiles use_case_profiles.json --candidates candidate_set.json --out use_case_recommendations.json` when the user wants host-aware recommendations across use cases.
5. Inspect fit estimates before recommending apply.
   Use `scripts/estimate_model_fit.py` on the winning candidate when fit is tight, KV-cache metadata is uncertain, GPU topology is awkward, MIG is active, or the runtime needs a non-default quantization path.
6. Render a deployment plan.
   Use `scripts/render_deployment_plan.py` to produce `deployment_plan.yaml`. For v1 it can also render the vLLM Compose and environment files. For SGLang, TensorRT-LLM, llama.cpp, and Ollama it emits a bounded follow-on module contract instead of pretending the deployer is implemented.
7. Apply only after the plan is acceptable.
   Use `scripts/apply_vllm_compose.sh --apply --allow-model-downloads ...` for the baseline vLLM Compose module, or execute reviewed plan commands for a documented manual runtime path.
8. Verify and benchmark.
   Use `scripts/smoke_test_endpoint.py` for endpoint verification and `scripts/benchmark_endpoint.py` with an explicit profile before claiming the deployment performs well for the workload. For authenticated endpoints, prefer `--api-key-env` over placing tokens in command arguments.

## Artifact Contract

- `host_facts.json`: normalized read-only host discovery facts and evidence.
- `workload_profile.yaml`: user/workload intent, not discovered host state.
- `candidate_scorecard.json`: scored model/runtime recommendations with fit estimates and blockers.
- `use_case_recommendations.json`: host-aware recommendation matrix across named workload profiles.
- `deployment_plan.yaml`: planned configuration, apply/verify commands, pinning state, and rollback guidance.
- `applied_deployment_state.json`: state recorded by an explicit apply path.
- `verification_report.json`: endpoint smoke-test results; keep benchmark reports beside it or merge them deliberately.

Never overwrite one artifact with another category of state.

## Resources

Scripts:
- `probe_remote_host.sh`: collect read-only SSH probe evidence.
- `normalize_host_facts.py`: convert raw probe evidence into host facts.
- `estimate_model_fit.py`: estimate weights, KV cache, batching/workspace, runtime overhead, and reserve against available VRAM.
- `rank_candidates.py`: score current candidate pairs for workload, support, fit, license, quantization, context, and serving behavior.
- `recommend_use_cases.py`: reuse the scorer across named workload profiles and preserve alternatives/blockers per use case.
- `render_deployment_plan.py`: render plan output and the implemented vLLM Compose baseline.
- `apply_vllm_compose.sh`: explicit remote apply path for the vLLM Compose module.
- `smoke_test_endpoint.py` and `benchmark_endpoint.py`: verify and measure an endpoint with auditable JSON reports.

References:
- Read [runtime-selection.md](references/runtime-selection.md) for the decision rubric.
- Read [host-discovery.md](references/host-discovery.md) before changing probe coverage.
- Read [model-fit.md](references/model-fit.md) before changing memory estimates or candidate schema.
- Read [vllm.md](references/vllm.md), [sglang.md](references/sglang.md), [trt-llm.md](references/trt-llm.md), [llama-cpp.md](references/llama-cpp.md), or [ollama.md](references/ollama.md) only when that runtime is under consideration.

Assets:
- Use the workload, use-case-profile, candidate, host-facts, and benchmark examples as starting shapes, not production recommendations.
- Use templates under `assets/templates/` for Compose, systemd, and reverse-proxy plan material.
