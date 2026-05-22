# llama.cpp Follow-On Module

Use llama.cpp for GGUF-centric serving, quantized portability, constrained VRAM, partial CPU/GPU offload, or a simple server path.

## Current Checks

- Refresh llama.cpp server, Docker, CUDA build/image, and OpenAI-compatible API behavior from primary project docs.
- Select an exact GGUF artifact and quantization. Record model source, revision, file size, context assumptions, and chat template implications.
- Treat partial offload as a workload tradeoff. A fit that spills weights to CPU can still miss latency goals.

## V1 Module Boundary

`render_deployment_plan.py` records `llama-cpp-server-v1` as a follow-on module. Implement it with:

- Inputs: host facts, workload, GGUF artifact metadata, GPU offload fraction or flags, pinned binary/image revision.
- Outputs: server command/container config, explicit artifact download/apply commands, applied state, OpenAI-compatible smoke tests, rollback guidance.
- Safety: loopback bind by default and keep GGUF downloads explicit.
