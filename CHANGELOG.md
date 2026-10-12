# Changelog

All notable changes to `phocinae-server` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] — 2026-10-11

### Fixed

- **A clean `pip install phocinae-server` used to return HTTP 500 on every
  inference call.** `phocinae/engine.py` decodes logits with
  `Tensor.numpy()`, but `numpy` was never declared as a dependency and torch
  does not pull it in. The server started fine and `/health` reported `ok`,
  yet every `POST /v1/systemone` died with `RuntimeError: Numpy is not
  available`. `numpy>=1.24` is now declared.
  *If you are on 0.1.5: `pip install numpy` is the immediate workaround; 0.1.6
  fixes it properly.*
- The default `PHOC_MODEL_DIR` pointed at a path that exists only on the
  maintainer's machine, so a fresh user got an error naming a directory they
  had never heard of. It now defaults to empty and produces an actionable
  message (env var, example values for Linux and PowerShell, and the model
  repository URL).
- `run_server.sh` hardcoded the maintainer's virtualenv and model path. It is
  now POSIX `sh`, honours `PHOC_MODEL_DIR` / `PHOC_HOST` / `PHOC_PORT` /
  `PHOC_DEVICE` / `PHOC_PYTHON`, and exits 2 with a copy-pasteable message when
  the model directory is unset or incomplete.
- `tests/test_router.py` required `CUDA_VISIBLE_DEVICES` to be present *and*
  empty, which is unreachable from PowerShell and `cmd` (both delete the
  variable when set to an empty value). An absent variable now counts as empty.
- Author-machine paths (`/home/hermes/...`) were removed from the package,
  `run_server.sh`, `dev_tools/` and all `tests/`.

### Changed

- The engine is now built in a FastAPI **lifespan** instead of at import time.
  `import phocinae.server` no longer loads a model, and requests that arrive
  before the weights are ready get **503 + `Retry-After: 5`** instead of an
  undefined behaviour. `/health` reports `status: "starting"` until then.
- Internal HTTP calls (the router) no longer route loopback/private targets
  through an operating-system HTTP proxy. `requests` ignores the Windows
  `ProxyOverride` bypass list, so on a proxied machine `http://127.0.0.1:...`
  was being sent to the proxy and answered 502. Remote router targets still
  honour the user's proxy configuration.

### Security

- Bearer-token comparison uses `hmac.compare_digest`.
- The router validates the model's choice index (type and range, rejecting
  `bool`) and raises a typed `RouterHTTPError` instead of `IndexError` /
  `TypeError`.

## [0.1.5] — 2026-10-09

Maintenance release: API documentation refresh and version alignment with the
v1.1 model card numbers (τ=0.6 escalation gate).

## [0.1.0] — 2026-10-08

First public release: `POST /v1/systemone` (noul / choice / score),
`/v1/systemone/permute`, `/v1/systemone/batch`, `GET /health`, `GET /v1/models`.
