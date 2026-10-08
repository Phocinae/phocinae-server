"""Quick experiment: does torch.compile bring the GPU forward under 15ms?"""
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
MODEL_DIR = "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo"

import torch
from phocinae.engine import Engine, QTYPES

REQ = {
    "state": "The agent was asked to restart the web server. It checked the logs, removed a stale lock file in /var/run and restarted nginx successfully. Memory usage returned to normal afterwards.",
    "questions": [
        {"id": "q1", "type": "noul", "threshold": 0.5},
        {"id": "q2", "type": "choice", "options": ["Restart the server", "Delete all files", "Do nothing", "Escalate to a human"]},
        {"id": "q3", "type": "score"},
    ],
}


def bench_forward(eng, b, reps=40):
    with torch.no_grad():
        for _ in range(5):
            eng.model(**{k: v.to(eng.device) for k, v in b.items()})
        lat = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad(), eng._amp_ctx():
                eng.model(b["input_ids"].to(eng.device), b["attention_mask"].to(eng.device),
                          b["marker_pos"].to(eng.device), b["marker_mask"].to(eng.device),
                          b["qtype"].to(eng.device))
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    return statistics.median(lat), lat[0], lat[-1]


def main():
    eng = Engine(MODEL_DIR, device="cuda", warm=True)
    state_ids, items = eng._encode_request(REQ["state"], REQ["questions"])
    rows = []
    for it in items:
        seq, markers = eng._build_sequence_fit(it["spec"], state_ids)
        rows.append({"ids": seq, "markers": markers, "qtype": it["qtype"],
                     "n_opts": it["n_opts"], "spec": it["spec"], "q": it["q"]})
    b = eng._collate(rows)

    p50, mn, mx = bench_forward(eng, b)
    print("eager   forward p50=%.2fms (%.2f..%.2f)" % (p50, mn, mx))

    t0 = time.time()
    cm = torch.compile(eng.model, mode="reduce-overhead", dynamic=True)
    print("compiled in %.1fs" % (time.time() - t0), flush=True)
    eng._compiled = cm
    # warm compiled
    with torch.no_grad(), eng._amp_ctx():
        for _ in range(10):
            cm(b["input_ids"].to(eng.device), b["attention_mask"].to(eng.device),
               b["marker_pos"].to(eng.device), b["marker_mask"].to(eng.device),
               b["qtype"].to(eng.device))
    torch.cuda.synchronize()

    def bench_compiled(reps=40):
        lat = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad(), eng._amp_ctx():
                cm(b["input_ids"].to(eng.device), b["attention_mask"].to(eng.device),
                   b["marker_pos"].to(eng.device), b["marker_mask"].to(eng.device),
                   b["qtype"].to(eng.device))
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1e3)
        lat.sort()
        return statistics.median(lat), lat[0], lat[-1]

    p50, mn, mx = bench_compiled()
    print("compiled forward p50=%.2fms (%.2f..%.2f)" % (p50, mn, mx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
