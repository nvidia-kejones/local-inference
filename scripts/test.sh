#!/usr/bin/env bash
set -euo pipefail

python3 -m py_compile skills/deploy-nvidia-inference/scripts/*.py
bash -n skills/deploy-nvidia-inference/scripts/*.sh
python3 -m unittest discover -s tests -p 'test_*.py'
