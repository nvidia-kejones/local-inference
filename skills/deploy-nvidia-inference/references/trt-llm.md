# TensorRT-LLM Follow-On Module

Use TensorRT-LLM or `trtllm-serve` when NVIDIA hardware, model support, runtime support, and precision/quantization path line up and the optimization effort is justified by the workload.

## Current Checks

- Refresh the TensorRT-LLM support matrix and precision/quantization docs for the selected model architecture and GPU architecture.
- Refresh `trtllm-serve` endpoint and CLI docs. It can expose OpenAI-compatible serving, but supported behavior still needs verification for the chosen release.
- Record whether the path uses a PyTorch backend, TensorRT backend, engine build, pre-quantized model, or other conversion step.
- Pin container/package/model revisions and record any engine artifacts created during apply.

## V1 Module Boundary

`render_deployment_plan.py` records `trtllm-serve-v1` as a follow-on module. Implement it as:

- Inputs: normalized host facts, workload profile, candidate support evidence, precision path, pinned revisions.
- Outputs: support-matrix evidence, build/serve config, explicit build/download/apply commands, applied state, endpoint verification, rollback guidance for generated engines and services.
- Safety: separate planning from engine creation and service changes. Do not convert models, build engines, or install NVIDIA packages during discovery.
