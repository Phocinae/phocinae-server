#!/bin/sh
# Launch phocinae-server. Works for a normal `pip install phocinae-server`
# (no repo checkout, no author-specific venv required).
#
# Usage:
#   PHOC_MODEL_DIR=/path/to/Phocinae-Largha-150M-v1 ./run_server.sh
#   PHOC_MODEL_DIR=... PHOC_PORT=8155 PHOC_DEVICE=cpu ./run_server.sh
#
# Env:
#   PHOC_MODEL_DIR  (required) directory of the released model repo
#   PHOC_HOST       default 127.0.0.1
#   PHOC_PORT       default 8155
#   PHOC_DEVICE     auto | cuda | cpu
#   PHOC_PYTHON     interpreter to use (default: python3, else python)
#   PYTHONPATH      the repo root is prepended when run from a checkout
set -eu

# Run from this script's directory (POSIX: no BASH_SOURCE, no pushd).
cd "$(dirname "$0")"

# Use the checkout when there is one, otherwise the installed package.
# (server.py no longer loads a model at import time, so a plain interpreter
# with fastapi/uvicorn/torch/numpy installed is enough.)
if [ -f "phocinae/main.py" ]; then
    PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONPATH
fi

if [ -z "${PHOC_MODEL_DIR:-}" ]; then
    cat >&2 <<'EOF'
run_server.sh: PHOC_MODEL_DIR is not set.

Point it at a local copy of the released model repo (the directory containing
model.safetensors, encoder/config.json, rl_agent_config.json and tokenizer/):

    export PHOC_MODEL_DIR=/path/to/Phocinae-Largha-150M-v1
    ./run_server.sh

Weights: https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1
EOF
    exit 2
fi

if [ ! -d "$PHOC_MODEL_DIR" ]; then
    echo "run_server.sh: PHOC_MODEL_DIR='$PHOC_MODEL_DIR' is not a directory." >&2
    exit 2
fi
if [ ! -f "$PHOC_MODEL_DIR/model.safetensors" ]; then
    echo "run_server.sh: '$PHOC_MODEL_DIR' has no model.safetensors; that is not" >&2
    echo "               the released model repo." >&2
    exit 2
fi

PHOC_HOST="${PHOC_HOST:-127.0.0.1}"
PHOC_PORT="${PHOC_PORT:-8155}"
export PHOC_HOST PHOC_PORT
export PHOC_DEVICE="${PHOC_DEVICE:-auto}"

if [ -n "${PHOC_PYTHON:-}" ]; then
    PY="$PHOC_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "run_server.sh: no python3/python on PATH (set PHOC_PYTHON)." >&2
    exit 127
fi

echo "phocinae-server: http://$PHOC_HOST:$PHOC_PORT  model=$PHOC_MODEL_DIR  device=$PHOC_DEVICE" >&2
exec "$PY" -m phocinae.main "$@"
