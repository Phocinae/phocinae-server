"""Shape-robustness check for the compiled engine: vary state length across
requests and confirm latency stays low (no per-shape recompile storm)."""
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
MODEL_DIR = "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo"

import torch
from phocinae.engine import Engine


def main():
    eng = Engine(MODEL_DIR, device="cuda", warm=True)
    states = ["The agent restarted nginx."] + [
        "The agent was asked to review a deployment. It checked the logs. " * (n // 60 + 1)
        for n in (60, 200, 500, 1200)]
    questions = [
        {"id": "q1", "type": "noul", "threshold": 0.5},
        {"id": "q2", "type": "choice",
         "options": ["Restart", "Delete", "Nothing", "Escalate"]},
        {"id": "q3", "type": "score"},
    ]
    for st in states:
        lat = []
        for i in range(12):
            t0 = time.perf_counter()
            eng.run(st, questions)
            lat.append((time.perf_counter() - t0) * 1e3)
        lat.sort()
        print("state_tokens~%-4d p50=%.2fms max=%.2fms" %
              (len(eng.tok.encode(st)), statistics.median(lat), lat[-1]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
