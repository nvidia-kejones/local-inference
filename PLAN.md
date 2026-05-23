# Full Functionality Plan

This project is not complete until the skill can make a defensible deployment decision from the model, workload, host facts, and user constraints, then deploy through the best available substrate.

## Target Outcome

The `deploy-nvidia-inference` skill should:

- Discover a host without changing it.
- Build or validate a current model/runtime candidate set from primary sources and pinned artifacts.
- Decide the best serving backend for the model, workload, and host.
- Decide the best deployment substrate in this order:
  1. Kubernetes, when available on the host or explicitly requested by the user.
  2. Docker Compose or Docker, when available and Kubernetes is not selected.
  3. Native service deployment, when neither Kubernetes nor Docker is available.
- Render an auditable plan before changes.
- Apply only after explicit user approval.
- Verify and benchmark the deployed endpoint before claiming success.
- Preserve rollback guidance and applied state.

## Decision Model

The skill needs two related but separate decisions.

### Serving Backend

Choose among:

- vLLM
- SGLang
- TensorRT-LLM / `trtllm-serve`
- llama.cpp
- Ollama

The backend decision should consider:

- Model architecture, format, revision, license, quantization, and context support.
- Host GPU count, VRAM or UMA memory budget, compute capability, MIG state, topology, CPU RAM, disk/cache capacity, and active workloads.
- Workload requirements: latency, throughput, concurrency, context length, structured output, prefix reuse, agentic/tool usage, operational simplicity, and license constraints.
- Runtime support evidence from current primary docs and model metadata.
- Fit estimate confidence and benchmark requirements.

Acceptance criteria:

- The scorer can rank candidate model/backend pairs across all supported runtimes.
- Blockers are explicit and not hidden by a numeric score.
- Missing deploy-critical evidence prevents apply readiness.
- The recommendation explains why the selected backend is best for this host and workload, not universally best.

### Deployment Substrate

Choose among:

- Kubernetes
- Docker / Docker Compose
- Native service

Selection rules:

- If the user explicitly requests Kubernetes, use Kubernetes when discovery or provided context confirms it is available, and fail the plan with clear blockers when it is not.
- If Kubernetes is available and no substrate is requested, prefer it only when the workload benefits from it or the host appears to be managed for Kubernetes service operation.
- If Kubernetes is not selected, use Docker / Docker Compose when Docker is available and usable.
- If Kubernetes and Docker are unavailable, render a native service deployment plan.
- Never install Kubernetes, Docker, container toolkits, or system packages during discovery.

Acceptance criteria:

- Host discovery records Kubernetes availability, Docker availability, native service manager availability, and permission hints.
- The plan records the selected substrate and why alternatives were rejected.
- Each substrate has separate render/apply/rollback modules.
- Apply paths enforce the selected substrate and do not silently fall back to a different one.

## Work Plan

### Phase 1: Test Harness And Contract Lock

- Add a local test runner for Python helpers and shell guardrails.
- Add golden tests for host normalization, model fit, candidate scoring, backend recommendation, and deployment plan rendering.
- Add tests for malformed candidate sets, missing pins, blocked license, context mismatch, no GPU facts, UMA fallback, Docker unavailable, Kubernetes available, and service fallback.
- Add a fake OpenAI-compatible endpoint test for `smoke_test_endpoint.py` and `benchmark_endpoint.py`.
- Update README and `SKILL.md` language so advertised support matches implemented behavior at every phase.

Exit criteria:

- `make test` or `scripts/test.sh` runs the full local test suite.
- Existing vLLM example flow remains green.

### Phase 2: Discovery Expansion

- Extend `probe_remote_host_payload.py` and normalization to detect:
  - Kubernetes tools and context: `kubectl`, current context, namespace access, node labels, GPU resource visibility, and whether the current user can create workloads.
  - Docker usability: daemon reachability, compose availability, NVIDIA runtime/toolkit hints, and current user permissions.
  - Native service manager: `systemctl` availability, user service support, writable service/config directories, and non-root limitations.
  - Disk/cache suitability for model downloads and container layers.
  - Port conflicts for requested endpoint ports.
- Add a remote no-Python fallback or document and test a bounded manual collector path.

Exit criteria:

- `host_facts.json` contains deployment substrate facts separate from recommendations.
- Discovery remains read-only.

### Phase 3: Candidate Schema And Readiness Gates

- Define a candidate schema with shared fields and runtime-specific fields.
- Add a validator script that checks:
  - Model revision or artifact pin.
  - Runtime/container/binary/package digest or exact version.
  - License/deployment evidence.
  - Runtime support evidence.
  - Memory inputs for weights and KV cache.
  - Context and quantization support.
  - Backend-specific deployment arguments.
- Separate recommendation scoring from apply readiness.

Exit criteria:

- Incomplete candidates can be scored for exploration but cannot be marked apply-ready.
- Plan rendering reports validation blockers consistently.

### Phase 4: Backend Selection Engine

- Refactor scoring into a backend decision report that includes:
  - Ranked candidate/backend pairs.
  - Fit estimate and confidence.
  - Runtime support evidence.
  - License status.
  - Backend-specific strengths and tradeoffs.
  - Required benchmark before production use.
- Preserve workload-specific scoring for use-case matrices.
- Add explicit tie-breaking rules for common cases:
  - vLLM as the general Hugging Face/CUDA baseline.
  - SGLang for prefix reuse, structured outputs, agentic flows, or throughput-sensitive serving.
  - TensorRT-LLM when supported NVIDIA optimization work is justified.
  - llama.cpp for GGUF, constrained VRAM, and partial offload.
  - Ollama for lifecycle simplicity when peak serving performance is not the priority.

Exit criteria:

- The skill can explain the selected serving backend for a concrete host/workload.
- Backend selection is test-covered and not hardcoded to vLLM.

### Phase 5: Deployment Substrate Selection Engine

- Add a substrate selector that consumes host facts, user intent, backend requirements, and workload needs.
- Add substrate-specific blockers and readiness checks:
  - Kubernetes: kube context, namespace, GPU scheduling support, image pull policy, secrets handling, service/port-forward plan, and rollback.
  - Docker: compose availability, NVIDIA runtime, local bind policy, cache mounts, restart policy, and rollback.
  - Native service: binary/runtime availability, model cache path, environment file permissions, systemd user/system mode, port binding, and rollback.
- Render selection evidence into `deployment_plan.yaml`.

Exit criteria:

- Plan output includes `selected_backend`, `selected_substrate`, `selection_rationale`, rejected alternatives, and apply blockers.
- User-requested Kubernetes is honored or blocked explicitly.

### Phase 6: Runtime Deployment Modules

Implement backend modules with the same boundaries: render, preflight, apply, verify, benchmark, rollback.

Priority:

1. Harden vLLM across Docker and Kubernetes.
2. Add SGLang across Docker and Kubernetes.
3. Add Ollama across Docker, Kubernetes, and native service where practical.
4. Add llama.cpp across Docker, Kubernetes, and native service.
5. Add TensorRT-LLM / `trtllm-serve` last because support-matrix and engine/build handling are more complex.

Exit criteria:

- Every advertised backend has at least one implemented deployment path.
- Unsupported backend/substrate combinations fail with clear blockers.

### Phase 7: Kubernetes Support

- Add Kubernetes manifest rendering for selected backends:
  - Deployment or StatefulSet.
  - GPU resource requests/limits.
  - ConfigMap and Secret boundaries.
  - PVC or hostPath/cache strategy.
  - Service and optional port-forward instructions.
  - Readiness/liveness probes when supported.
- Add `apply_k8s.sh` with explicit apply and download/pull acknowledgement flags.
- Record applied state from `kubectl apply`, rollout status, pod logs excerpt, service details, and rollback commands.

Exit criteria:

- Kubernetes plans and apply state are auditable.
- The skill never assumes cluster-admin rights.

### Phase 8: Docker Support

- Generalize the current vLLM Compose module into a Docker substrate module.
- Add backend-specific Compose templates and env templates.
- Enforce pinned image digests and model revisions before apply.
- Add endpoint readiness waits and log capture after startup.
- Keep loopback binding by default unless external exposure is explicitly requested.

Exit criteria:

- Docker apply is implemented for each backend where container deployment is supported.
- Docker apply refuses missing pins and unsafe exposure.

### Phase 9: Native Service Fallback

- Add native service rendering for backends that can run without Docker:
  - llama.cpp server.
  - Ollama where package/binary lifecycle is already present.
  - Other backends only when a reviewed local binary/package path exists.
- Support user systemd services first when root is unavailable.
- Do not install packages automatically.
- Record binary paths, versions, environment files, model cache paths, service files, start/stop commands, logs, and rollback.

Exit criteria:

- Native fallback can render a safe plan when Kubernetes and Docker are unavailable.
- Native apply is blocked unless required binaries already exist or the user explicitly approves a separate install step.

### Phase 10: Verification, Benchmarks, And Rollback

- Extend smoke tests for runtime-specific OpenAI-compatible behavior.
- Add benchmark profiles tied to workload assumptions.
- Add readiness polling after apply.
- Capture structured failure diagnostics from Kubernetes pods, Docker logs, or native service logs.
- Add rollback verification after stop/restore.

Exit criteria:

- The skill only claims deployment success after endpoint smoke verification.
- It only claims workload suitability after running the relevant benchmark profile.

### Phase 11: Real Environment Validation

Validate end to end on:

- SSH host with Docker.
- Brev-managed host with Docker.
- Kubernetes-capable host or cluster context.
- Host with no Docker/Kubernetes but native service path available.
- At least one constrained-memory host.
- At least one multi-GPU host.
- At least one UMA host where framebuffer memory facts are unavailable.

Exit criteria:

- Each validation run keeps artifacts under `outputs/deploy-nvidia-inference/<run-id>/`.
- Known limitations are documented from real failures, not guesses.

## Completion Criteria

The project is complete when:

- The skill chooses the serving backend from host/workload/model evidence.
- The skill chooses Kubernetes, Docker, or native service deployment from host facts and user intent.
- vLLM, SGLang, TensorRT-LLM, llama.cpp, and Ollama have implemented or explicitly blocked backend/substrate combinations.
- Apply readiness requires pinned artifacts and reviewed deployment evidence.
- Discovery, planning, apply, verification, benchmarking, and rollback all produce separate artifacts.
- Local tests and at least one real end-to-end validation per major substrate pass.
