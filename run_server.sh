#!/usr/bin/env bash
# Launch phocinae-server with the sanctioned venv (torch/fastapi/uvicorn).
# Usage: ./run_server.sh [extra env...]
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export PHOC_MODEL_DIR="${PHOC_MODEL_DIR:-/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo}"
export PHOC_HOST="${PHOC_HOST:-127.0.0.1}"
export PHOC_PORT="${PHOC_PORT:-8155}"
exec /home/hermes/decision-model/.venv/bin/python -m phocinae.main "$@"
