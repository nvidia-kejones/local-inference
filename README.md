# local-inference

Agent Skills for local and remote inference work.

## Skills

### `deploy-nvidia-inference`

Use this Codex Agent Skill to discover a single remote NVIDIA Linux host over SSH, recommend host-aware model/runtime pairs for workloads, render a safe deployment plan, apply the vLLM Docker Compose baseline explicitly, and verify an OpenAI-compatible inference endpoint.

Supported recommendation/deployment targets:

- vLLM
- SGLang
- TensorRT-LLM / `trtllm-serve`
- llama.cpp
- Ollama

The v1 scripted apply path is vLLM Compose. The other runtimes have runtime guidance and deployment module boundaries so they can be added without weakening discovery, planning, fit estimation, or verification.

## Install

Install the skill with the `skills` CLI:

```bash
npx skills add https://github.com/nvidia-kejones/local-inference.git \
  --skill deploy-nvidia-inference \
  --agent codex \
  --global
```

From a local clone, use the repo path instead:

```bash
npx skills add "$PWD" \
  --skill deploy-nvidia-inference \
  --agent codex \
  --global
```

To inspect what the CLI will discover before installing:

```bash
npx skills add "$PWD" --list
```

The skill lives under `skills/deploy-nvidia-inference`, which is a standard `skills` CLI discovery path. The CLI can target Codex globally at `~/.codex/skills/` or install into a project when `--global` is omitted.

Manual Codex install remains a fallback:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/deploy-nvidia-inference "${CODEX_HOME:-$HOME/.codex}/skills/"
```

## Use

Invoke the skill in Codex with a concrete host and goal:

```text
Use $deploy-nvidia-inference to discover user@gpu-host over SSH and recommend the best model/runtime pairs this host can serve for interactive chat, code assistance, long-context RAG, and structured agent workloads.
```

For one deployment workflow:

```text
Use $deploy-nvidia-inference to inspect user@gpu-host, build a workload profile for an 8K-context code assistant with four concurrent sequences, recommend a model/runtime pair, render a safe deployment plan, and stop before apply.
```

The skill keeps recommendation and deployment state separate. During use it aims to produce:

- `host_facts.json`
- `workload_profile.yaml`
- `candidate_scorecard.json`
- `use_case_recommendations.json` when comparing named workload profiles
- `deployment_plan.yaml`
- `applied_deployment_state.json` only after explicit apply
- `verification_report.json`

Discovery is read-only. Installs, model downloads, service changes, deployment writes, firewall changes, and endpoint exposure must be explicit apply decisions. Endpoints bind to `127.0.0.1` by default unless external exposure is requested and reviewed.

## Direct Script Flow

The skill normally drives these helpers, but the core read-only path can also be run from the skill directory:

```bash
cd skills/deploy-nvidia-inference
scripts/probe_remote_host.sh user@gpu-host > host_probe.raw.json
python3 scripts/normalize_host_facts.py host_probe.raw.json --out host_facts.json
python3 scripts/recommend_use_cases.py \
  --host host_facts.json \
  --profiles assets/use_case_profiles.example.json \
  --candidates candidate_set.json \
  --out use_case_recommendations.json
```

Build `candidate_set.json` from current primary runtime/model documentation and model metadata before scoring. The bundled candidate examples are schemas and test shapes, not permanent model recommendations.
