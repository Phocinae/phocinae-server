"""Shared helpers for the ad-hoc scripts in tests/ and dev_tools/.

These scripts are run directly (`python tests/<script>.py`), never installed, so
this module is imported by adding the repo root to sys.path. It exists so that
no script hardcodes a machine-specific model path: the model directory always
comes from `PHOC_MODEL_DIR`.

Typical use at the top of a script:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _util import require_model_dir
    MODEL_DIR = require_model_dir()      # exits 0 with a clear SKIP message
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_REPO_URL = "https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1"


def model_dir_or_none():
    """`PHOC_MODEL_DIR` if it is set and looks like the released repo, else None."""
    path = (os.environ.get("PHOC_MODEL_DIR") or "").strip()
    if not path:
        return None
    if not os.path.isdir(path):
        return None
    return path


def require_model_dir(script=None, allow_skip=True):
    """Return the model dir, or exit cleanly when it is not configured.

    A missing `PHOC_MODEL_DIR` is a SKIP (exit 0) because these are optional
    developer scripts. A *set but broken* path is a hard error (exit 2) — that
    is a user typo, not a missing prerequisite.
    """
    path = model_dir_or_none()
    if path is not None:
        return path
    raw = (os.environ.get("PHOC_MODEL_DIR") or "").strip()
    name = script or os.path.basename(__file__)
    if raw:
        print("ERROR %s: PHOC_MODEL_DIR=%r is not an existing directory.\n"
              "     Point it at the released model repo (contains "
              "model.safetensors, encoder/, tokenizer/):\n"
              "       %s"
              % (name, raw, MODEL_REPO_URL))
        raise SystemExit(2)
    if not allow_skip:
        raise SystemExit("PHOC_MODEL_DIR is not set")
    print("SKIP %s: PHOC_MODEL_DIR is not set, so the model cannot be loaded.\n"
          "     Download the released repo (%s)\n"
          "     and point the script at it, e.g.\n"
          "       Linux/macOS:  PHOC_MODEL_DIR=/path/to/hf_repo python %s\n"
          "       PowerShell:   $env:PHOC_MODEL_DIR='D:\\models\\"
          "Phocinae-Largha-150M-v1'; python %s"
          % (name, MODEL_REPO_URL, name, name))
    raise SystemExit(0)


def python_exe():
    """Interpreter used to spawn child server processes: the running one.

    (Historically this was a hardcoded path to the author's virtualenv.)
    """
    import sys
    return sys.executable
