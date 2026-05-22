# Ollama Follow-On Module

Use Ollama when the simplest operational path and model lifecycle matter more than extracting peak serving performance.

## Current Checks

- Refresh Ollama Docker, GPU support, model library/source policy, and OpenAI compatibility docs.
- Decide whether the model comes from an Ollama library reference or a reviewed local/model-file path.
- Verify the OpenAI-compatible endpoint subset required by the client and benchmark real prompts before recommending it for throughput-sensitive service.

## V1 Module Boundary

`render_deployment_plan.py` records `ollama-container-v1` as a follow-on module. Implement it with:

- Inputs: host facts, workload, model source/lifecycle choice, pinned image/package revision, endpoint binding.
- Outputs: container/service config, explicit model pull/create/apply commands, applied state, endpoint verification, rollback guidance for the model and service lifecycle.
- Safety: keep pulls explicit and do not expose the default service externally without a reviewed plan.
