"""CPU latency smoke: engine.run p50 over representative single requests.
P0 acceptance: CPU fallback p50 <= 150 ms per request.
Run: python tests/cpu_smoke.py"""

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODEL_DIR = os.environ.get(
    "PHOC_MODEL_DIR",
    "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo")

REQ = {
    "state": "The agent was asked to restart the web server. It checked the logs, "
             "removed a stale lock file in /var/run and restarted nginx successfully. "
             "Memory usage returned to normal afterwards.",
    "questions": [
        {"id": "q1", "type": "noul", "threshold": 0.5},
        {"id": "q2", "type": "choice",
         "options": ["Restart the server", "Delete all files", "Do nothing",
                     "Escalate to a human"]},
        {"id": "q3", "type": "score"},
    ],
}


def measure(engine, state, questions, reps=40):
    engine.run(state, questions)  # prime caches
    lat = []
    for _ in range(reps):
        t0 = time.perf_counter()
        engine.run(state, questions)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()
    return {
        "reps": reps,
        "p50_ms": round(statistics.median(lat), 2),
        "p90_ms": round(lat[int(len(lat) * 0.9)], 2),
        "max_ms": round(lat[-1], 2),
    }


def main():
    from phocinae.engine import Engine
    print("loading CPU engine (threads=8)...")
    eng = Engine(MODEL_DIR, device="cpu", warm=True, threads=8)
    res = measure(eng, REQ["state"], REQ["questions"])
    print(json.dumps(res, indent=2))
    ok = res["p50_ms"] <= 150.0
    print("CPU p50 %.2f ms -> %s (acceptance <=150ms)"
          % (res["p50_ms"], "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
