"""uvicorn entrypoint: python -m phocinae.main"""

import logging
import os
import sys

import uvicorn


def _force_utf8_output():
    """Make logging/prints safe on non-UTF-8 consoles (e.g. Windows cp936).

    Model/tokenizer paths and vocabulary diagnostics can contain non-ASCII (CJK)
    text; on a cp936 console the default stdout/stderr encoding makes those
    prints die with UnicodeEncodeError. Reconfiguring the streams here keeps
    `python -m phocinae.main` working without requiring the caller to set
    PYTHONUTF8=1 / PYTHONIOENCODING=utf-8 (if they are set, the streams already
    report utf-8 and are left untouched).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
            if enc != "utf8" and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - never block startup over encoding
            pass


def main():
    _force_utf8_output()
    logging.basicConfig(
        level=os.environ.get("PHOC_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Fail fast with an actionable message (non-zero exit) instead of letting
    # uvicorn die inside the lifespan handler with a bare traceback.
    from phocinae.server import resolve_model_dir

    _model_dir, err = resolve_model_dir()
    if err is not None:
        print(err, file=sys.stderr)
        return 2
    host = os.environ.get("PHOC_HOST", "127.0.0.1")
    port = int(os.environ.get("PHOC_PORT", "8155"))
    uvicorn.run("phocinae.server:app", host=host, port=port, workers=1,
                log_level="warning",
                access_log=os.environ.get("PHOC_ACCESS_LOG", "0") == "1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
