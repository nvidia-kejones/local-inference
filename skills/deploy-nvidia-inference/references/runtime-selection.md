# Runtime Selection Rubric

Read this before building a candidate set. Treat runtime selection as a workload decision constrained by discovered host facts, current runtime support, model format, licensing, and operational appetite.

## V1 Rubric

| Runtime | Prefer when | Check before recommending | V1 deployment status |
| --- | --- | --- | --- |
| vLLM | Hugging Face/CUDA serving needs a production baseline and an OpenAI-compatible API | Model/quantization support, image/driver compatibility, model revision, tensor-parallel fit | Implemented Compose baseline |
| SGLang | Prefix reuse, structured output, agentic flows, or throughput-sensitive serving deserve evaluation | Current server args, model support, memory tuning, OpenAI behavior needed by the client | Follow-on module contract |
| TensorRT-LLM / `trtllm-serve` | NVIDIA hardware, supported model path, and precision/quantization path align for optimization work | Current support matrix, GPU architecture support, precision path, engine/build implications | Follow-on module contract |
| llama.cpp | GGUF, partial GPU offload, constrained VRAM, broad quantized portability, or a simple server matter | GGUF availability, CUDA build/image path, offload flags, endpoint feature expectations | Follow-on module contract |
| Ollama | Model lifecycle simplicity and operational ease outweigh peak serving performance | GPU support, model library/source policy, OpenAI-compatible API coverage, lifecycle commands | Follow-on module contract |

## Refreshable Model Selection

1. Start from the workload profile. Name quality needs, context target, concurrency/live-token assumptions, structured-output/tool needs, latency/throughput emphasis, and license constraints.
2. Use primary sources before candidate scoring:
   - Runtime docs for supported models, server APIs, quantization, container images, and hardware support.
   - Model repository metadata for architecture, context, tokenizer/chat template, file sizes or precision, revision pin, quantization artifacts, and license.
   - NVIDIA docs for driver/container prerequisites when container compatibility is uncertain.
3. Build candidate pairs, not model-only rows. `model A + vLLM BF16` and `model A GGUF + llama.cpp Q4` are different candidates.
4. Record evidence and pins in the candidate set. A deployable candidate needs a model revision and an exact container digest, binary revision, or package revision.
5. Use `rank_candidates.py` to expose tradeoffs. Do not let a single score override fit blockers, license blockers, or missing support evidence.

For broad host advice, run `recommend_use_cases.py` with named workload profiles. Recommend the best known candidate pair for each profile only within the refreshed candidate set, keep the next-best alternatives and blockers, and ask the user which workload should become the deployment plan.

## Quality Signals And Leaderboards

Arena can seed a candidate set when the workload needs a current quality prior. Its text leaderboard has category filters, a `License Type: Open Source` filter, human-preference score/rank, votes, rank spread, and context metadata. Use the relevant category view, record the observed date and filter URL, and prefer the model rows that resolve to an actual deployable open-weight artifact for the target runtime.

Do not use Arena as deployment evidence by itself:

- Arena measures served model behavior under Arena evaluation, not memory fit, GPU support, quantization support, container compatibility, or local throughput.
- A leaderboard label or linked model page still needs artifact, license, model-revision, runtime-support, and quantization verification from primary docs and model repositories.
- Overall chat preference can be the wrong quality signal for code, structured outputs, long-context serving, retrieval/tool behavior, or latency-sensitive agent loops. Use workload-aligned categories or task benchmarks when available.
- Leaderboard ranks move. Keep the observation in candidate evidence instead of hardcoding a permanent "best open models" table in this skill.

Treat a leaderboard score as a quality prior for `quality_fit_score`, then let host fit, runtime support, license review, and real endpoint benchmarks decide the deployable recommendation.

When one candidate has different quality evidence by use case, use purpose-keyed `quality_fit_scores` and `quality_evidence` entries. The scorer falls back to the candidate-wide quality score when a profile has no specific entry.

## Primary Docs To Refresh

- NVIDIA Container Toolkit install/configuration: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`
- NVIDIA CUDA compatibility: `https://docs.nvidia.com/deploy/cuda-compatibility/`
- vLLM Docker and OpenAI-compatible serving: `https://docs.vllm.ai/en/latest/deployment/docker/` and `https://docs.vllm.ai/en/stable/serving/openai_compatible_server/`
- vLLM quantization: `https://docs.vllm.ai/en/stable/features/quantization/`
- SGLang docs and server arguments: `https://docs.sglang.io/` and `https://docs.sglang.io/docs/advanced_features/server_arguments`
- TensorRT-LLM support matrix, precision, and `trtllm-serve`: `https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html`, `https://nvidia.github.io/TensorRT-LLM/reference/precision.html`, and `https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html`
- llama.cpp server and Docker docs: `https://github.com/ggml-org/llama.cpp` and `https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md`
- Ollama OpenAI compatibility, Docker, and GPU docs: `https://docs.ollama.com/api/openai-compatibility`, `https://docs.ollama.com/docker`, and `https://docs.ollama.com/gpu`
- Arena leaderboard quality prior: `https://arena.ai/leaderboard/text?license=open-source`

## Recommendation Shape

Write a recommendation with:

- The selected model/runtime pair and the next-best alternative.
- For a matrix report, the named use case plus why the selected pair is host-optimal for that profile rather than a universal winner.
- The workload reasons that matter.
- Host facts that constrain the decision.
- Fit estimate confidence and uncertain memory inputs.
- Runtime/model/license evidence that was refreshed.
- Pinned revisions to deploy.
- Verification benchmark that will decide whether "optimized" held for this workload.
