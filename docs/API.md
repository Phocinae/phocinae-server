# phocinae-server — HTTP API Reference

The real contract of `phocinae/server.py` (v0.1.0). Base URL: `http://127.0.0.1:8155` by default; interactive docs at `/docs`. All bodies are JSON. The API is **deliberately not OpenAI-compatible** (no chat/completions layer — this model is not a chat model).

## Endpoints

| method | path | auth | purpose |
|---|---|---|---|
| GET | `/` | none | service info |
| GET | `/health` | none | liveness + model/device info |
| GET | `/v1/models` | bearer (if set) | model catalog |
| POST | `/v1/systemone` | bearer (if set) | single decision request |
| POST | `/v1/systemone/permute` | bearer (if set) | same, with 4-order permutation averaging on choice questions |
| POST | `/v1/systemone/batch` | bearer (if set) | batch of ≤64 decision requests |

Auth: when `PHOC_BEARER_TOKEN` is set, all `/v1/*` endpoints require `Authorization: Bearer <token>` — otherwise **401**. `/` and `/health` are always unauthenticated.

## GET /

Response: `{"service": "phocinae-server", "model": "<PHOC_MODEL_NAME>", "docs": "/docs", "health": "/health"}`.

## GET /health

| field | type | meaning |
|---|---|---|
| `status` | string | `"ok"` while the service is up |
| `model` | string | served model name (default `Phocinae-Largha-150M-v1`) |
| `model_loaded` | bool | true once the engine finished loading |
| `device` | string | `"cpu"` or `"cuda"` |
| `uptime_s` | float | seconds since process start |

## GET /v1/models

Returns `{"object": "list", "data": [ … ]}` with one entry per served model:

| field | value |
|---|---|
| `id` | model name |
| `object` | `"model"` |
| `created` | `null` |
| `owned_by` | `"phocinae-server"` |
| `context_length` | engine `max_len` (from `rl_agent_config.json`; default 512) |
| `head_max_len` | engine head budget (default 192) |
| `max_position_embeddings` | **8192** (`MAX_POS`) |
| `architecture` | `"mmBERT-small (384 hidden, 22 layers, 6 heads, sliding-window+full attention, RoPE)"` |
| `task` | `"typed-decisions system-one"` |
| `answer_formats` | `{"noul": "bool", "choice": "option index", "score": "integer 2-10"}` |
| `extensions` | `["answer_confidence", "action", "routing", "option_scores"]` |
| `perm_avg` | `"avg-k4"` when `PHOC_PERM_AVG=1`, else `"none"` |
| `option_scores_semantics` | choice: calibrated per-option probabilities in request option order (averaged over permuted orders when perm=avg-k4); noul: `[p(false), p(true)]`; score: level probabilities for answers 2..10 |
## POST /v1/systemone

Request:

```json
{"model": "Phocinae-Largha-150M-v1",
 "state": "The agent restarted nginx after checking the logs and the health endpoint is green.",
 "questions": [
   {"id": "ok",  "type": "noul",   "threshold": 0.65},
   {"id": "act", "type": "choice", "options": ["allow", "ask", "deny"]},
   {"id": "risk","type": "score"}]}
```

| field | constraints |
|---|---|
| `model` | exactly the configured model name (default `Phocinae-Largha-150M-v1`); else **422** |
| `state` | string — the decision context |
| `questions` | list, 1–64 items (`MAX_QUESTIONS`); else **422** |
| `questions[].id` | non-empty string, unique per request; else **422** |
| `questions[].type` | `noul` · `choice` · `score` (unknown → **422**) |
| `questions[].options` | `choice` only: non-empty list of strings, ≤255 (`MAX_OPTIONS`); missing/empty/oversized → **422**; present on `score` → **422** |
| `questions[].threshold` | `noul` only, float in [0, 1]; outside → **422**; omitted → default **0.5** |

Response (200):

```json
{"model": "Phocinae-Largha-150M-v1",
 "answers": {"ok": true, "act": 0, "risk": 2},
 "usage": {"input_tokens": 24, "output_tokens": 0},
 "answer_confidence": {"ok": 0.91, "act": 0.72, "risk": 0.18},
 "action": {"act": {"act_probability": 0.72}},
 "option_scores": {"ok": [0.09, 0.91], "act": [0.72, 0.21, 0.07], "risk": [0.03, 0.18]},
 "routing": {"model": "Phocinae-Largha-150M-v1", "device": "cpu", "perm": "none", "backend": "phocinae-pure-torch"}}
```

| field | shape |
|---|---|
| `answers` | {qid: value} — noul → bool · choice → 0-based option index (int) · score → int 2–10 |
| `usage` | `input_tokens` / `output_tokens` (ints; output is always 0 — no generation) |
| `answer_confidence` | {qid: float} — temperature-calibrated top probability, rounded to 4 dp |
| `action` | {qid: {"act_probability": float}} |
| `option_scores` | {qid: [float…]} — calibrated per-option probabilities (choice: request option order; noul: [p(false), p(true)]; score: levels 2..10) |
| `routing` | `device` (`cpu`/`cuda`) · `perm` (`none`/`avg-k4`) · `backend` (`phocinae-pure-torch`) |

`PHOC_EXTENSIONS=0` omits `answer_confidence`, `action`, `option_scores`, `routing` (then the response is `model` + `answers` + `usage` only).

## POST /v1/systemone/permute

Identical request/response to `/v1/systemone`, but every `choice` question with ≥2 options is evaluated over **4 option orders** (original, reversed, two seeded shuffles — canonical K=4 protocol), per-option probabilities averaged before argmax. `routing.perm` is `"avg-k4"`. Costs ~4× the forward passes; use when order-invariance matters more than latency.

## POST /v1/systemone/batch

Body: either a JSON array of `systemone` requests, or `{"requests": [...]}`. Empty → **422**; more than 64 items → **422**. Returns an ordered list of responses (one per request, same schema as `/v1/systemone`).

## Errors

| code | when |
|---|---|
| **422** | unknown model; unknown question type; >64 questions; empty/duplicate question `id`; choice missing/empty/>255 options; score carrying options; noul threshold outside [0,1]; empty or >64-item batch; malformed body / missing fields (FastAPI validation) |
| **413** | body > 2 MiB (`PHOC_MAX_BODY`); body: `{"error": {"code": 413, "message": "request body exceeds the N-byte limit"}}` |
| **401** | `PHOC_BEARER_TOKEN` set and `Authorization: Bearer <token>` missing or wrong → `{"detail": "missing or invalid bearer token"}` |

## curl examples

```bash
curl -s http://127.0.0.1:8155/health

curl -s http://127.0.0.1:8155/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "Phocinae-Largha-150M-v1",
  "state": "git status",
  "questions": [{"id": "ok", "type": "noul", "threshold": 0.65}]}'

# permutation averaging on a tool choice:
curl -s http://127.0.0.1:8155/v1/systemone/permute -H 'Content-Type: application/json' -d '{
  "model": "Phocinae-Largha-150M-v1", "state": "find the latest news about the launch",
  "questions": [{"id": "tool", "type": "choice",
                 "options": ["web_search", "read_file", "run_command"]}]}'

# batch of two:
curl -s http://127.0.0.1:8155/v1/systemone/batch -H 'Content-Type: application/json' -d '{
  "requests": [
    {"model": "Phocinae-Largha-150M-v1", "state": "a", "questions": [{"id": "q", "type": "noul"}]},
    {"model": "Phocinae-Largha-150M-v1", "state": "b", "questions": [{"id": "q", "type": "score"}]}]}'

# with bearer auth:
curl -s http://127.0.0.1:8155/v1/systemone -H "Authorization: Bearer $PHOC_BEARER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Phocinae-Largha-150M-v1","state":"x","questions":[{"id":"q","type":"score"}]}'
```
## Environment variables

| variable | default | purpose |
|---|---|---|
| `PHOC_MODEL_DIR` | (dev path in source) | directory of the released model repo (`model.safetensors`, `encoder/`, `rl_agent_config.json`, `tokenizer/`) — set it in practice |
| `PHOC_MODEL_NAME` | `Phocinae-Largha-150M-v1` | served model name |
| `PHOC_HOST` / `PHOC_PORT` | `127.0.0.1` / `8155` | listen address |
| `PHOC_DEVICE` | `auto` | `auto` / `cuda` / `cpu` (no GPU → CPU fp32 fallback) |
| `PHOC_PERM_AVG` | `0` | `1` = `/v1/systemone` also averages choice questions over 4 option orders |
| `PHOC_BEARER_TOKEN` | empty | set it → `/v1/*` requires `Authorization: Bearer <token>` |
| `PHOC_THREADS` | — | torch intra-op threads (CPU mode) |
| `PHOC_MAX_BODY` | `2 * 1024 * 1024` (2 MiB) | request-body cap; above it → **413** |
| `PHOC_EXTENSIONS` | `1` | `0` = omit extension keys |
| `PHOC_LOG` / `PHOC_ACCESS_LOG` | `INFO` / `0` | logging level / uvicorn access log |

## Determinism

Same request + same device + same batch shape → bit-identical answers (no sampling anywhere). fp16 vs fp32, or different batch shapes, may differ in the last decimal place.

## Engine-level limits & notes

- ≤64 questions per request, ≤255 options per choice question, 8192-token encoder budget (`MAX_POS`); each option text is truncated to 48 tokens inside the sequence builder.
- Options too long for the token budget raise an engine `ValueError` ("options exceed the N-token model budget") — this surfaces as an HTTP 500, not a 422 (engine limit, not request validation).
- The service listens on 127.0.0.1 by default, has no HA guarantees, and callers should fail closed on any error.
- Client-side helpers (`phocinae/router.py`): escalate gate (τ=0.6, `answer_confidence` below τ → escalate) and tool router (k≤10, permutation-averaged choice) talk to this API over HTTP or in-process — see [README](../README.md).

## Related docs

- Request/response semantics at the model level: the model repo's `docs/protocol.md`.
- Model card & benchmark numbers: the released model repo (`Phocinae-Largha-150M-v1` on Hugging Face).
- Full server README (run instructions, performance, security): [README](../README.md).
