# SGLang Follow-On Module

Use SGLang as a strong candidate when the workload has prefix reuse, structured-output needs, agentic request shapes, or throughput sensitivity worth testing against the baseline.

## Current Checks

- Refresh SGLang installation/Docker docs, OpenAI-compatible API docs, server arguments, model support, and memory tuning guidance.
- Check shared-memory requirements for container runs and long-prompt prefill tuning.
- Confirm the client features needed from OpenAI compatibility, especially structured outputs, tool behavior, streaming, and model naming.

## V1 Module Boundary

`render_deployment_plan.py` records `sglang-compose-v1` as a follow-on module. Implement it with the same boundaries as the vLLM baseline:

- Inputs: `host_facts.json`, `workload_profile.yaml`, selected pinned candidate, endpoint binding/exposure decision.
- Outputs: rendered container config/env, reviewed apply commands or explicit apply script, `applied_deployment_state.json`, verification commands, rollback command.
- Safety: loopback bind by default; downloads/writes/service changes explicit; no silent Docker/toolkit installs.

Until that module exists, write manual commands in `deployment_plan.yaml` from refreshed primary docs and keep the apply record separate from the recommendation.
