import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
MODEL_DIR = "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo"
from phocinae.engine import Engine
REQ = {
    "state": "The agent was asked to restart the web server. It checked the logs, removed a stale lock file in /var/run and restarted nginx successfully. Memory usage returned to normal afterwards.",
    "questions": [
        {"id": "q1", "type": "noul", "threshold": 0.5},
        {"id": "q2", "type": "choice", "options": ["Restart the server", "Delete all files", "Do nothing", "Escalate to a human"]},
        {"id": "q3", "type": "score"},
    ],
}
eng = Engine(MODEL_DIR, device="cuda", warm=True)
# warm the run path + caches
eng.run(REQ["state"], REQ["questions"])
import statistics
t_enc = []; t_build = []; t_fwd = []; t_dec = []; t_tot = []
for _ in range(30):
    t0 = time.perf_counter()
    state_ids, items = eng._encode_request(REQ["state"], REQ["questions"])
    t1 = time.perf_counter()
    rows = []
    for it in items:
        seq, markers = eng._build_sequence_fit(it["spec"], state_ids)
        rows.append({"ids": seq, "markers": markers, "qtype": it["qtype"], "n_opts": it["n_opts"], "spec": it["spec"], "q": it["q"]})
    t2 = time.perf_counter()
    b = eng._collate(rows)
    logits, act = eng._forward(b)
    t3 = time.perf_counter()
    dec = eng._decode_rows(logits.numpy(), act.numpy(), rows)
    t4 = time.perf_counter()
    t_enc.append((t1-t0)*1e3); t_build.append((t2-t1)*1e3); t_fwd.append((t3-t2)*1e3); t_dec.append((t4-t3)*1e3); t_tot.append((t4-t0)*1e3)
for name, v in [("encode", t_enc), ("build", t_build), ("forward", t_fwd), ("decode", t_dec), ("total", t_tot)]:
    v.sort()
    print("%-8s p50=%7.2fms  min=%6.2fms  max=%6.2fms" % (name, statistics.median(v), v[0], v[-1]))
