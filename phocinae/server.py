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
  PHOC_MODEL_DIR     path to the released hf_repo directory
  PHOC_HOST/PHOC_PORT
  PHOC_DEVICE        auto | cuda | cpu
  PHOC_BEARER_TOKEN  when set, /v1/* require "Authorization: Bearer <token>"
  PHOC_PERM_AVG      1 -> /v1/systemone also averages choice option orders
  PHOC_EXTENSIONS    0 -> omit answer_confidence/action/routing extension keys
  PHOC_THREADS       torch intra-op threads when running on CPU
"""

import json
import logging
import os
import time

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

engine: Engine = None  # set by create_app


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
        if header != "Bearer " + BEARER:
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
def create_app() -> FastAPI:
    global engine
    app = FastAPI(title="phocinae-server", version="0.1.0")
    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY)

    @app.exception_handler(_BodyTooLarge)
    async def _on_body_too_large(request, exc):
        return _body_limit_response(MAX_BODY)

    model_dir = os.environ.get(
        "PHOC_MODEL_DIR",
        "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo")
    device = os.environ.get("PHOC_DEVICE", "auto")
    perm_avg = os.environ.get("PHOC_PERM_AVG", "0") == "1"
    threads = os.environ.get("PHOC_THREADS") or None
    engine = Engine(model_dir, device=device, model_name=MODEL_NAME,
                    perm_avg=perm_avg, warm=True,
                    threads=int(threads) if threads else None)
    log.info("engine ready: %s on %s (perm_avg=%s)", MODEL_NAME,
             engine.device, perm_avg)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "model_loaded": True,
            "device": engine.device.type,
            "uptime_s": round(time.time() - START_TIME, 1),
        }

    @app.get("/v1/models", dependencies=[Depends(auth_guard)])
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

    @app.post("/v1/systemone", dependencies=[Depends(auth_guard)])
    def system_one(req: SystemOneRequest):
        return _serve(req, perm=False)

    @app.post("/v1/systemone/permute", dependencies=[Depends(auth_guard)])
    def system_one_permute(req: SystemOneRequest):
        return _serve(req, perm=True)

    @app.post("/v1/systemone/batch", dependencies=[Depends(auth_guard)])
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


app = create_app()
