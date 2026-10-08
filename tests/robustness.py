#!/usr/bin/env python3
"""phocinae-server 鲁棒性/稳定性实验（2026-10-08，CPU 引擎，进程内 TestClient）。

覆盖：
  R1 畸形输入 fuzz（错类型/缺字段/深嵌套/unicode/空字节/超大载荷 413/超题量 422/未知模型 422）
  R2 并发压测（8 并发 × 共 160 请求，混合题型，无 5xx、无死锁）
  R3 长跑稳定性（1000 请求，RSS 漂移 + 延迟分位稳定性）
  R4 确定性（同输入重放 5 次，逐位一致）
  R5 重启恢复（重建 app 后决策与重启前一致）
  R6 边界尺寸（1 题 / 64 题 / 255 选项 / 超长 state）
用法: PHOC_DEVICE=cpu /home/hermes/decision-model/.venv/bin/python tests/robustness.py
"""
import json
import os
import resource
import time

os.environ.setdefault("PHOC_DEVICE", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from phocinae.server import app  # noqa: E402

client = TestClient(app)
MODEL = "Phocinae-Largha-150M-v1"

PASS = 0
FAIL = 0
RESULTS = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        RESULTS.append("FAIL %s :: %s" % (name, detail))


def req(state, questions):
    return {"model": MODEL, "state": state, "questions": questions}


# ---------- R1 畸形输入 ----------
def r1():
    base = req("The batch has 3 rows.", [
        {"id": "q1", "type": "noul", "threshold": 0.5}])
    cases = [
        ("缺 model", {"state": "x", "questions": []}, 422),
        ("未知 model", {"model": "no-such-model", "state": "x", "questions": []}, 422),
        ("缺 state", {"model": MODEL, "questions": []}, 422),
        ("缺 questions", {"model": MODEL, "state": "x"}, 422),
        ("state 非字符串", {"model": MODEL, "state": {"a": 1}, "questions": []}, 422),
        ("questions 非列表", {"model": MODEL, "state": "x", "questions": "q"}, 422),
        ("未知题型", req("x", [{"id": "q", "type": "banana"}]), 422),
        ("choice 无 options", req("x", [{"id": "q", "type": "choice"}]), 422),
        ("score 不该有 options", req("x", [{"id": "q", "type": "score", "options": ["a"]}]), 422),
        ("noul 缺 threshold=默认0.5", req("x", [{"id": "q", "type": "noul"}]), 200),
        ("重复 id", req("x", [{"id": "q", "type": "noul", "threshold": 0.5},
                             {"id": "q", "type": "score"}]), 422),
        ("65 题超限", req("x", [{"id": "q%d" % i, "type": "score"} for i in range(65)]), 422),
        ("空 state", req("", [{"id": "q", "type": "noul", "threshold": 0.5}]), 200),
        ("state 含空字节", req("a\x00b", [{"id": "q", "type": "score"}]), 200),
        ("state 含控制符", req("a\n\r\t\b\f", [{"id": "q", "type": "score"}]), 200),
        ("state unicode 极端", req("🐬" * 2000 + "汉字混排" * 500,
                                    [{"id": "q", "type": "score"}]), 200),
        ("threshold 越界", req("x", [{"id": "q", "type": "noul", "threshold": 9.9}]), 422),
        ("options 非字符串项", req("x", [{"id": "q", "type": "choice", "options": [1, 2]}]), 422),
        ("JSON 顶层非对象", ["x"], 422),
        ("空 JSON 体", None, 422),
        ("questions 含非对象", req("x", ["str"]), 422),
    ]
    for name, body, want in cases:
        r = client.post("/v1/systemone", json=body if body is not None else "")
        check("R1 " + name, r.status_code == want,
              "got %d want %d" % (r.status_code, want))
    # 大载荷边界：1.5MB < 2MiB 限 → 应 200；3MB → 413
    big_ok = req("x" * 1_500_000, [{"id": "q", "type": "score"}])
    r = client.post("/v1/systemone", json=big_ok)
    check("R1 1.5MB 载荷放行", r.status_code == 200, "got %d" % r.status_code)
    big_no = req("x" * 3_000_000, [{"id": "q", "type": "score"}])
    r = client.post("/v1/systemone", json=big_no)
    check("R1 3MB 载荷 413", r.status_code == 413, "got %d" % r.status_code)
    # 深嵌套 state 当字符串：直接塞 dict 深嵌套 → 422（已测 state 非字符串）
    deep = {"model": MODEL, "state": json.dumps({"a": {"b": [1, [2, [3]]]}}),
            "questions": [{"id": "q", "type": "score"}]}
    r = client.post("/v1/systemone", json=deep)
    check("R1 深嵌套 state 字符串化", r.status_code in (200, 422), "got %d" % r.status_code)


# ---------- R2 并发 ----------
def r2():
    import concurrent.futures as cf

    def one(i):
        t = ["noul", "choice", "score"][i % 3]
        q = {"id": "q1", "type": t}
        if t == "noul":
            q["threshold"] = 0.5
        elif t == "choice":
            q["options"] = ["Approve", "Retry", "Flag"]
        r = client.post("/v1/systemone", json=req("The system processed 3 invoices.", [q]))
        return r.status_code

    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(one, range(160)))
    dt = time.time() - t0
    check("R2 并发 160 请求无 5xx", all(c == 200 for c in codes),
          "bad codes: %s" % sorted(set(codes)))
    check("R2 并发无死锁且吞吐合理", dt < 180, "elapsed %.1fs" % dt)


# ---------- R3 长跑稳定性 ----------
def r3():
    lat = []
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    q = {"id": "q1", "type": "score"}
    for i in range(1000):
        t0 = time.time()
        r = client.post("/v1/systemone", json=req("The batch has %d rows." % (i % 5), [q]))
        lat.append((time.time() - t0) * 1000)
        if r.status_code != 200:
            check("R3 第 %d 次请求 200" % i, False, "got %d" % r.status_code)
            break
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    drift = (rss1 - rss0) / 1024.0
    p50 = np.percentile(lat, 50)
    p95 = np.percentile(lat, 95)
    first = np.percentile(lat[:100], 50)
    last = np.percentile(lat[-100:], 50)
    check("R3 1000 请求全 200", True)
    check("R3 RSS 漂移 < 200MB", drift < 200, "%.1f MB" % drift)
    check("R3 p50 头尾稳定(<2×)", max(last, 1e-6) < 2 * max(first, 1e-6),
          "first %.1fms last %.1fms" % (first, last))
    RESULTS.append("INFO R3 p50=%.1fms p95=%.1fms RSS漂移=%.1fMB" % (p50, p95, drift))


# ---------- R4 确定性 ----------
def r4():
    q = {"id": "q1", "type": "choice", "options": ["A", "B", "C"]}
    outs = []
    for _ in range(5):
        r = client.post("/v1/systemone", json=req("The invoice total is 120 USD.", [q]))
        outs.append(json.dumps(r.json()["answers"]))
    check("R4 同输入 5 次逐位一致", len(set(outs)) == 1, str(outs))


# ---------- R5 重启恢复 ----------
def r5():
    q = {"id": "q1", "type": "choice", "options": ["A", "B", "C"]}
    body = req("The invoice total is 120 USD.", [q])
    a1 = client.post("/v1/systemone", json=body).json()["answers"]
    # 模拟重启：create_app() 重建全新引擎实例（新 app + 新 engine）
    from phocinae.server import create_app
    app2 = create_app()
    c2 = TestClient(app2)
    a2 = c2.post("/v1/systemone", json=body).json()["answers"]
    check("R5 重启后决策一致", a1 == a2, "%s vs %s" % (a1, a2))


# ---------- R6 边界尺寸 ----------
def r6():
    # 1 题
    r = client.post("/v1/systemone", json=req("x", [{"id": "q", "type": "score"}]))
    check("R6 单题", r.status_code == 200, str(r.status_code))
    # 64 题
    qs = [{"id": "q%d" % i, "type": "score"} for i in range(64)]
    r = client.post("/v1/systemone", json=req("x", qs))
    check("R6 64 题", r.status_code == 200, str(r.status_code))
    # 255 选项
    r = client.post("/v1/systemone", json=req("x", [
        {"id": "q", "type": "choice", "options": ["o%d" % i for i in range(255)]}]))
    check("R6 255 选项", r.status_code in (200, 422), str(r.status_code))
    # 超长 state（100K 字符）
    r = client.post("/v1/systemone", json=req("词" * 50_000, [{"id": "q", "type": "score"}]))
    check("R6 100K 字符 state", r.status_code in (200, 413), str(r.status_code))


if __name__ == "__main__":
    t0 = time.time()
    r1(); r2(); r3(); r4(); r5(); r6()
    print("PASS=%d FAIL=%d elapsed=%.1fs" % (PASS, FAIL, time.time() - t0))
    for line in RESULTS:
        print(line)
    raise SystemExit(1 if FAIL else 0)
