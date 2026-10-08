#!/usr/bin/env python3
"""phocinae-server GPU 鲁棒性（2026-10-08）：跨设备一致 + 压力。

  G1 跨设备一致性：同一批样本 CPU(fp32) vs GPU 前向 argmax 一致率（noul/choice/score）
  G2 GPU 压力：200 次混合请求（含编译预热），无 OOM/崩溃，延迟分位
  G3 GPU 确定性：同输入 5 次 GPU 重放逐位一致
经 run_gpu.sh 串行执行（GPU 独占）。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from phocinae.engine import Engine

MODEL_DIR = "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo"

SAMPLES = [
    ("The system processed 3 invoices without errors.", "choice",
     ["Approve batch", "Retry the batch", "Flag for review"]),
    ("User asked to delete the production database.", "noul",
     ["false: the action is not destructive", "true: the action is destructive"]),
    ("How severe is this security incident?", "score", None),
    ("The customer requested a full refund.", "noul",
     ["false: reject", "true: approve"]),
    ("Pick the next action for the failed deployment.", "choice",
     ["Roll back", "Retry deploy", "Escalate to human", "Ignore"]),
] * 20  # 100 样本


def q_of(sample):
    state, t, opts = sample
    q = {"id": "q1", "type": t}
    if opts:
        q["options"] = opts
    else:
        q["threshold"] = 0.5
    return state, [q]


def main():
    t0 = time.time()
    cpu = Engine(MODEL_DIR, device="cpu", warm=True)
    gpu = Engine(MODEL_DIR, device="cuda", warm=True)
    print("load+compile: %.1fs" % (time.time() - t0))

    # G1 一致性
    mism = 0
    agree = {"noul": 0, "choice": 0, "score": 0}
    tot = {"noul": 0, "choice": 0, "score": 0}
    for s in SAMPLES:
        state, qs = q_of(s)
        a1 = cpu.run(state, qs)[0]
        a2 = gpu.run(state, qs)[0]
        tot[s[1]] += 1
        if a1 == a2:
            agree[s[1]] += 1
        else:
            mism += 1
    print("G1 argmax 一致: noul %d/%d choice %d/%d score %d/%d (mism=%d)"
          % (agree["noul"], tot["noul"], agree["choice"], tot["choice"],
             agree["score"], tot["score"], mism))
    assert mism == 0, "cross-device disagreement: %d" % mism

    # G2 压力 200 次
    lat = []
    for i in range(200):
        state, qs = q_of(SAMPLES[i % len(SAMPLES)])
        t1 = time.time()
        gpu.run(state, qs)
        lat.append((time.time() - t1) * 1000)
    print("G2 200 req: p50=%.2fms p95=%.2fms max=%.2fms"
          % (np.percentile(lat, 50), np.percentile(lat, 95), max(lat)))

    # G3 GPU 确定性
    state, qs = q_of(SAMPLES[0])
    outs = [json.dumps(gpu.run(state, qs)[0], sort_keys=True) for _ in range(5)]
    assert len(set(outs)) == 1, "GPU nondeterminism: %s" % outs
    print("G3 GPU 5 次重放逐位一致")
    print("GPU_ROBUST_OK elapsed=%.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
