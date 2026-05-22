# Host Discovery

Use discovery to learn what the remote NVIDIA Linux host exposes before deciding how to change it. Discovery mode is read-only.

## Probe Coverage

`scripts/probe_remote_host.sh` opens SSH and runs remote `python3` to capture command, return code, stdout, and stderr for read-only probes:

- GPU list, UUIDs, total/free VRAM, driver version, compute capability when the driver exposes it, and MIG mode when exposed.
- `nvidia-smi` topology/P2P output for NVLink and peer hints.
- Visible compute applications from `nvidia-smi`.
- OS release, kernel, CPU, RAM, and disk paths relevant to model/container caches.
- Docker, containerd, Podman, NVIDIA Container Toolkit, and NVIDIA container CLI visibility.
- Listening sockets and process rows that look like inference services by executable name only.

If remote `python3` is missing, run the same read-only commands manually or add a reviewed fallback collector before treating the probe as complete.

## Normalized Fact Boundary

`normalize_host_facts.py` emits `host_facts.json`. Keep it factual:

- `source.command_status` is evidence of what was run and whether a probe failed.
- `host` is OS/CPU/RAM/disk.
- `nvidia` is driver hint, CUDA compatibility hint from `nvidia-smi`, GPU inventory, MIG state, topology, and visible GPU workloads.
- `containers` records runtime/toolkit visibility. Visibility is not proof the login user can perform apply operations.
- `network` records occupied listening sockets and inference-looking processes found without service changes.

Do not place workload preferences, recommendations, deployment commands, or benchmark conclusions in `host_facts.json`.

## Discovery Review

Before recommending apply:

1. Inspect failed probes. A failed Docker probe may mean Docker is absent, the daemon is down, or the login user lacks permission.
2. Check whether MIG partitions change usable memory and topology assumptions.
3. Check whether existing inference-looking processes or occupied ports conflict with the intended endpoint.
4. Check free VRAM and active GPU processes at the same time. Free VRAM is a snapshot, not a reservation.
5. Treat the CUDA version printed by `nvidia-smi` as a driver compatibility hint, not the host toolkit or a guarantee that a chosen image will run.

The probe avoids process arguments because they often carry tokens or endpoint secrets. If manual diagnosis requires command lines, collect them only after deciding how the resulting raw evidence will be protected.

## Read-Only Rule

Do not install packages, run `nvidia-ctk runtime configure`, restart Docker, change MIG state, pull images, download models, create remote files, edit firewalls, or stop services in discovery mode.
