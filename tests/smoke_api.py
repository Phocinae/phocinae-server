"""API smoke test: boots the real server on a test port and exercises every
P0 contract branch. Run: python tests/smoke_api.py  (CPU engine; ~40s)."""

import json
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENV = "/home/hermes/decision-model/.venv"
MODEL_DIR = os.environ.get(
    "PHOC_MODEL_DIR",
    "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo")

PORT = 8166
BASE = "http://127.0.0.1:%d" % PORT

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + detail) if detail and not cond else ""))
    if not cond:
        fails.append(name)


def boot(port, env_extra=None):
    env = dict(os.environ)
    env.update({
        "PHOC_MODEL_DIR": MODEL_DIR,
        "PHOC_DEVICE": "cpu",
        "PHOC_THREADS": "8",
        "PYTHONPATH": ROOT,
    })
    if env_extra:
        env.update(env_extra)
    p = subprocess.Popen(
        [VENV + "/bin/python", "-m", "uvicorn", "phocinae.server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p


def wait_ready(base, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited early (rc=%s)" % proc.poll())
        try:
            r = requests.get(base + "/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server not ready after %ds" % timeout)


GOOD_REQ = {
    "model": "Phocinae-Largha-150M-v1",
    "state": "The agent was asked to restart the web server. It checked the logs, "
             "removed a stale lock file and restarted nginx successfully.",
    "questions": [
        {"id": "q1", "type": "noul", "threshold": 0.5},
        {"id": "q2", "type": "choice",
         "options": ["Restart the server", "Delete all files", "Do nothing"]},
        {"id": "q3", "type": "score"},
    ],
}


def main():
    proc = boot(PORT)
    try:
        wait_ready(BASE, proc)

        h = requests.get(BASE + "/health", timeout=10).json()
        check("health ok", h["status"] == "ok" and h["model_loaded"] is True,
              json.dumps(h))

        m = requests.get(BASE + "/v1/models", timeout=10).json()
        check("models catalog", m["data"][0]["id"].startswith("Phocinae-Largha"))
        check("models answers formats",
              m["data"][0]["answer_formats"]["noul"] == "bool"
              and m["data"][0]["answer_formats"]["choice"] == "option index"
              and m["data"][0]["answer_formats"]["score"] == "integer 2-10")

        r = requests.post(BASE + "/v1/systemone", json=GOOD_REQ, timeout=60)
        j = r.json()
        check("systemone 200", r.status_code == 200, str(r.status_code))
        check("answers base keys", set(j) >= {"model", "answers", "usage"})
        a = j["answers"]
        check("noul is bool", isinstance(a["q1"], bool), repr(a))
        check("choice is int index",
              isinstance(a["q2"], int) and 0 <= a["q2"] < 3, repr(a["q2"]))
        check("score is 2..10", isinstance(a["q3"], int) and 2 <= a["q3"] <= 10,
              repr(a["q3"]))
        check("usage present", "input_tokens" in j["usage"])
        check("extension confidence per question",
              set(j.get("answer_confidence", {})) == {"q1", "q2", "q3"})
        check("extension action per question",
              set(j.get("action", {})) == {"q1", "q2", "q3"})
        check("extension routing backend",
              j.get("routing", {}).get("backend") == "phocinae-pure-torch")
        check("extensions do not break base", "answers" in j and "usage" in j)

        # permute endpoint
        rp = requests.post(BASE + "/v1/systemone/permute", json=GOOD_REQ, timeout=60)
        jp = rp.json()
        check("permute 200", rp.status_code == 200)
        check("permute routing flag",
              jp.get("routing", {}).get("perm") == "avg-k4", json.dumps(jp.get("routing")))
        check("permute answers sane", isinstance(jp["answers"]["q1"], bool))

        # batch endpoint
        rb = requests.post(BASE + "/v1/systemone/batch",
                           json={"requests": [GOOD_REQ, GOOD_REQ]}, timeout=120)
        check("batch 200", rb.status_code == 200, str(rb.status_code))
        jb = rb.json()
        check("batch two responses", isinstance(jb, list) and len(jb) == 2)

        # validation branches
        bad = dict(GOOD_REQ); bad["model"] = "nonexistent-model"
        check("unknown model 422",
              requests.post(BASE + "/v1/systemone", json=bad, timeout=10).status_code == 422)
        bad = dict(GOOD_REQ)
        bad["questions"] = [{"id": "c%d" % i, "type": "noul"} for i in range(65)]
        check("too many questions 422",
              requests.post(BASE + "/v1/systemone", json=bad, timeout=10).status_code == 422)
        bad = dict(GOOD_REQ)
        bad["questions"] = [{"id": "q1", "type": "noul"}, {"id": "q1", "type": "score"}]
        check("duplicate ids 422",
              requests.post(BASE + "/v1/systemone", json=bad, timeout=10).status_code == 422)
        bad = dict(GOOD_REQ)
        bad["questions"] = [{"id": "q1", "type": "choice"}]
        check("choice without options 422",
              requests.post(BASE + "/v1/systemone", json=bad, timeout=10).status_code == 422)
        bad = dict(GOOD_REQ)
        bad["questions"] = [{"id": "q1", "type": "choice",
                             "options": ["o%d" % i for i in range(256)]}]
        check("choice 256 options 422",
              requests.post(BASE + "/v1/systemone", json=bad, timeout=10).status_code == 422)
        big = dict(GOOD_REQ)
        big["state"] = "x" * (2 * 1024 * 1024 + 100)
        check("body over 2MiB 413",
              requests.post(BASE + "/v1/systemone", json=big, timeout=10).status_code == 413)

        # default: no auth required
        check("no auth by default",
              requests.post(BASE + "/v1/systemone", json=GOOD_REQ, timeout=60).status_code == 200)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    # bearer branch: second instance with PHOC_BEARER_TOKEN set
    proc2 = boot(PORT + 1, {"PHOC_BEARER_TOKEN": "testtoken"})
    try:
        wait_ready(BASE.replace(str(PORT), str(PORT + 1)), proc2)
        base2 = "http://127.0.0.1:%d" % (PORT + 1)
        check("bearer: 401 without token",
              requests.post(base2 + "/v1/systemone", json=GOOD_REQ,
                            timeout=10).status_code == 401)
        check("bearer: 200 with token",
              requests.post(base2 + "/v1/systemone", json=GOOD_REQ, timeout=60,
                            headers={"Authorization": "Bearer testtoken"}).status_code == 200)
        check("bearer: 401 wrong token",
              requests.post(base2 + "/v1/systemone", json=GOOD_REQ, timeout=10,
                            headers={"Authorization": "Bearer wrong"}).status_code == 401)
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc2.kill()

    print()
    if fails:
        print("smoke_api: FAILED %d checks: %s" % (len(fails), ", ".join(fails)))
    else:
        print("smoke_api: all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
