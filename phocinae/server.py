"""FastAPI server exposing the P0 phocinae system-one API.

Endpoints:
  GET  /health                 liveness + model/device info (no auth)
  GET  /v1/models              model catalog
  POST /v1/systemone           {model, state, questions:[...]} -> answers
  POST /v1/systemone/permute   same, with PHOC_PERM_AVG-style option-order
                               averaging forced on for choice questions
  POST /v1/systemone/batch     {"requests": [...]} -> [response, ...]

Extension keys (PHOC_EXTENSIONS=1, default): answer_confidence (calibrated
top probability), action (act_probability), option_scores (calibrated
per-option probability vectors, see /v1/models semantics), routing.

Env config (see README):
  PHOC_MODEL_DIR     path to the released hf_repo directory (required)
  PHOC_HOST/PHOC_PORT
  PHOC_DEVICE        auto | cuda | cpu
  PHOC_BEARER_TOKEN  when set, /v1/* require "Authorization: Bearer <token>"
  PHOC_PERM_AVG      1 -> /v1/systemone also averages choice option orders
  PHOC_EXTENSIONS    0 -> omit answer_confidence/action/routing extension keys
  PHOC_THREADS       torch intra-op threads when running on CPU

Importing this module has no side effects: the engine (model weights, tokenizer,
~30-60 s on CPU) is built in the FastAPI lifespan startup handler, so tooling can
`import phocinae.server` without a model directory. Any /v1/* request that arrives
before the engine is ready gets 503 + Retry-After instead of being mishandled.
"""

import hmac
import json
import logging
import os
import time

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

from .engine import MAX_OPTIONS, MAX_POS, MAX_QUESTIONS, Engine

log = logging.getLogger("phocinae.server")

MODEL_NAME = os.environ.get("PHOC_MODEL_NAME", "Phocinae-Largha-150M-v1")
MAX_BODY = int(os.environ.get("PHOC_MAX_BODY", 2 * 1024 * 1024))
BEARER = os.environ.get("PHOC_BEARER_TOKEN", "") or None
EXTENSIONS = os.environ.get("PHOC_EXTENSIONS", "1") != "0"

START_TIME = time.time()

# Where to download the weights (README / docs/API.md point here as well).
MODEL_REPO_URL = "https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1"

# Set during lifespan startup. Kept at module level so /health can report the
# not-yet-loaded state instead of exploding on a missing attribute.
engine: Optional[Engine] = None


def model_dir_error(detail):
    """Actionable message for a missing/unusable PHOC_MODEL_DIR."""
    return (
        "%s\n"
        "PHOC_MODEL_DIR must point at a downloaded copy of the released model repo\n"
        "(the directory containing model.safetensors, encoder/config.json,\n"
        "rl_agent_config.json and tokenizer/).\n"
        "\n"
        "  1. download the weights: %s  (288.6 MB, fp16 safetensors)\n"
        "  2. point the server at the directory, e.g.\n"
        "       Linux/macOS:  export PHOC_MODEL_DIR=/path/to/Phocinae-Largha-150M-v1\n"
        "                     python -m phocinae.main\n"
        "       PowerShell:   $env:PHOC_MODEL_DIR = 'D:\\models\\Phocinae-Largha-150M-v1'\n"
        "                     python -m phocinae.main\n"
        "       one-shot:     PHOC_MODEL_DIR=/path/to/hf_repo python -m phocinae.main\n"
        "\n"
        "  (download without git-lfs, python:)\n"
        "    from huggingface_hub import snapshot_download\n"
        "    snapshot_download(repo_id='Phocinae/Phocinae-Largha-150M-v1',\n"
        "                      local_dir=r'D:\\models\\Phocinae-Largha-150M-v1')"
        % (detail, MODEL_REPO_URL)
    )


def resolve_model_dir():
    """Resolve PHOC_MODEL_DIR from the environment.

    Returns (model_dir, None) when usable, else (None, actionable_message).
    Shared by the engine loader and by the `phocinae.main` preflight check so a
    fresh user sees one consistent, copy-pasteable error.
    """
    model_dir = (os.environ.get("PHOC_MODEL_DIR") or "").strip()
    if not model_dir:
        return None, model_dir_error("PHOC_MODEL_DIR is not set.")
    if not os.path.isdir(model_dir):
        return None, model_dir_error(
            "PHOC_MODEL_DIR=%r is not an existing directory." % model_dir)
    if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        return None, model_dir_error(
            "PHOC_MODEL_DIR=%r has no model.safetensors." % model_dir)
    return model_dir, None


def load_engine():
    """Build the Engine from the environment. Raises RuntimeError with an
    actionable message when PHOC_MODEL_DIR is unset or unusable."""
    model_dir, err = resolve_model_dir()
    if err is not None:
        raise RuntimeError(err)
    device = os.environ.get("PHOC_DEVICE", "auto")
    perm_avg = os.environ.get("PHOC_PERM_AVG", "0") == "1"
    threads = os.environ.get("PHOC_THREADS") or None
    eng = Engine(model_dir, device=device, model_name=MODEL_NAME,
                 perm_avg=perm_avg, warm=True,
                 threads=int(threads) if threads else None)
    log.info("engine ready: %s on %s (perm_avg=%s)", MODEL_NAME,
             eng.device, perm_avg)
    return eng


# ---------------------------------------------------------------- schemas
class Question(BaseModel):
    id: str
    type: Literal["noul", "choice", "score"]
    options: Optional[List[str]] = None
    threshold: Optional[float] = None


class SystemOneRequest(BaseModel):
    model: str
    state: str
    questions: List[Question]


class BatchRequest(BaseModel):
    requests: List[SystemOneRequest] = Field(default_factory=list)


# ------------------------------------------------------------- body limit
class _BodyTooLarge(Exception):
    pass


def _body_limit_response(max_bytes):
    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=413,
        content={"error": {"code": 413,
                           "message": "request body exceeds the %d-byte limit" % max_bytes}},
    )


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Fast path: declared content-length over the cap -> 413 without reading.
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await _body_limit_response(self.max_bytes)(scope, receive, send)
                        return
                except ValueError:
                    pass
                break
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # The app-level exception handler turns this into a 413.
                    raise _BodyTooLarge()
            return message

        await self.app(scope, limited_receive, send)


# ------------------------------------------------------------------ auth
def auth_guard(request: Request):
    if BEARER is not None:
        header = request.headers.get("authorization", "")
        # Constant-time comparison: a plain != leaks the shared prefix length
        # of the secret through timing. compare_digest needs same-type operands
        # (str or bytes); a non-ASCII configured token raises TypeError, which
        # can only mean "not equal" here, so it fails closed.
        try:
            ok = hmac.compare_digest(header, "Bearer " + BEARER)
        except TypeError:
            ok = False
        if not ok:
            raise HTTPException(status_code=401,
                                detail="missing or invalid bearer token")


# ------------------------------------------------------------- validation
def validate_request(req: SystemOneRequest):
    if req.model != MODEL_NAME:
        raise HTTPException(status_code=422,
                            detail="unknown model %r; available models: [%s]"
                            % (req.model, MODEL_NAME))
    if len(req.questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=422,
                            detail="too many questions (%d > %d)"
                            % (len(req.questions), MAX_QUESTIONS))
    seen = set()
    for q in req.questions:
        if not q.id:
            raise HTTPException(status_code=422, detail="question id must be non-empty")
        if q.id in seen:
            raise HTTPException(status_code=422,
                                detail="duplicate question id %r" % q.id)
        seen.add(q.id)
        if q.type == "choice":
            if q.options is None:
                raise HTTPException(status_code=422,
                                    detail="question %r: choice questions require options" % q.id)
            if not q.options:
                raise HTTPException(status_code=422,
                                    detail="question %r: options must be non-empty" % q.id)
            if len(q.options) > MAX_OPTIONS:
                raise HTTPException(status_code=422,
                                    detail="question %r: too many options (%d > %d)"
                                    % (q.id, len(q.options), MAX_OPTIONS))
        if q.type == "score" and q.options is not None:
            raise HTTPException(status_code=422,
                                detail="question %r: score questions must not carry options" % q.id)
        if q.type == "noul" and q.threshold is not None:
            t = q.threshold
            if not (0.0 <= t <= 1.0):
                raise HTTPException(status_code=422,
                                    detail="question %r: threshold must be within [0, 1]" % q.id)


# -------------------------------------------------------------- responses
def build_response(answers, confs, actions, usage, perm_used, option_scores=None):
    resp = {
        "model": MODEL_NAME,
        "answers": answers,
        "usage": usage,
    }
    if EXTENSIONS:
        resp["answer_confidence"] = confs
        resp["action"] = actions
        resp["option_scores"] = option_scores if option_scores is not None else {}
        resp["routing"] = {
            "model": MODEL_NAME,
            "device": engine.device.type,
            "perm": "avg-k4" if perm_used else "none",
            "backend": "phocinae-pure-torch",
        }
    return resp


# ------------------------------------------------------------------- app
def engine_ready():
    """Dependency for /v1/*: refuse to guess when the model is not loaded yet."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="model not loaded yet (startup still in progress or failed); "
                   "poll /health and retry",
            headers={"Retry-After": "5"})


async def _load_engine_on_startup():
    """FastAPI lifespan entry: the model is loaded here, never at import time."""
    global engine
    engine = load_engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    await _load_engine_on_startup()
    try:
        yield
    finally:
        engine = None


def create_app() -> FastAPI:
    app = FastAPI(title="phocinae-server", version="0.1.5", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY)

    @app.exception_handler(_BodyTooLarge)
    async def _on_body_too_large(request, exc):
        return _body_limit_response(MAX_BODY)

    @app.get("/health")
    def health():
        eng = engine
        if eng is None:
            # Distinct from "ok": the process is up but cannot decide yet.
            return {
                "status": "starting",
                "model": MODEL_NAME,
                "model_loaded": False,
                "device": None,
                "uptime_s": round(time.time() - START_TIME, 1),
            }
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "model_loaded": True,
            "device": eng.device.type,
            "uptime_s": round(time.time() - START_TIME, 1),
        }

    @app.get("/v1/models", dependencies=[Depends(auth_guard),
                                         Depends(engine_ready)])
    def models():
        return {
            "object": "list",
            "data": [{
                "id": MODEL_NAME,
                "object": "model",
                "created": None,
                "owned_by": "phocinae-server",
                "context_length": engine.max_len,
                "head_max_len": engine.head_max_len,
                "max_position_embeddings": MAX_POS,
                "architecture": "mmBERT-small (384 hidden, 22 layers, 6 heads, "
                                 "sliding-window+full attention, RoPE)",
                "task": "typed-decisions system-one",
                "answer_formats": {"noul": "bool", "choice": "option index",
                                   "score": "integer 2-10"},
                "extensions": ["answer_confidence", "action", "routing",
                               "option_scores"],
                "perm_avg": "avg-k4" if engine.perm_avg else "none",
                "option_scores_semantics": {
                    "choice": "calibrated per-option probabilities in request "
                              "option order (averaged over permuted orders when "
                              "perm=avg-k4)",
                    "noul": "[p(false), p(true)]",
                    "score": "level probabilities for answers 2..10",
                },
            }],
        }

    def _serve(req: SystemOneRequest, perm: bool):
        validate_request(req)
        qs = [q.model_dump() for q in req.questions]
        answers, confs, actions, usage, option_scores = engine.run(
            req.state, qs, perm=perm, with_scores=True)
        return build_response(answers, confs, actions, usage,
                              perm_used=perm or engine.perm_avg,
                              option_scores=option_scores)

    @app.post("/v1/systemone", dependencies=[Depends(auth_guard),
                                             Depends(engine_ready)])
    def system_one(req: SystemOneRequest):
        return _serve(req, perm=False)

    @app.post("/v1/systemone/permute", dependencies=[Depends(auth_guard),
                                                     Depends(engine_ready)])
    def system_one_permute(req: SystemOneRequest):
        return _serve(req, perm=True)

    @app.post("/v1/systemone/batch", dependencies=[Depends(auth_guard),
                                                   Depends(engine_ready)])
    def system_one_batch(body: Union[List[SystemOneRequest], BatchRequest]):
        reqs = body.requests if isinstance(body, BatchRequest) else body
        if not reqs:
            raise HTTPException(status_code=422, detail="batch is empty")
        if len(reqs) > 64:
            raise HTTPException(status_code=422,
                                detail="too many batch items (%d > 64)" % len(reqs))
        return [_serve(r, perm=False) for r in reqs]

    @app.get("/")
    def root():
        return {"service": "phocinae-server", "model": MODEL_NAME,
                "docs": "/docs", "health": "/health"}

    return app


# The module-level ASGI app uvicorn resolves for `phocinae.server:app`.
# create_app() itself is side-effect free: the engine is built by the lifespan
# startup handler above, so `import phocinae.server` never loads a model.
app = create_app()
