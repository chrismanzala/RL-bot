#!/usr/bin/env bash
# One-time setup on the training machine (local or remote GPU server).
# Run from anywhere:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# pip unpacks large wheels (torch is ~2.5 GB with CUDA) in TMPDIR. On servers where
# / is small or full, point it at the project's disk instead.
export TMPDIR="${TMPDIR:-$PWD/.pip-tmp}"
mkdir -p "$TMPDIR"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
rm -rf "$PWD/.pip-tmp"

echo
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
.venv/bin/python train/check_env.py
echo
echo "Ready. Next:  source .venv/bin/activate && python train/train_mappo.py"
