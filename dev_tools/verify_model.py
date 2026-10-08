"""Numeric parity: phocinae pure-torch engine vs the laya reference stack (CPU fp32).

laya is imported ONLY in this dev tool; the server never touches it. The
harness compares, for P0-shaped requests:
  1. tokenized sequences (ids + marker positions)          -> exact equality
  2. decision logits / action logits over the full batch   -> fp32 tolerance
  3. decoded probabilities / answers / answer_confidence /
     act_probability                                       -> exact / 1e-3

Usage: /home/hermes/decision-model/.venv/bin/python dev_tools/verify_model.py
"""

import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODEL_DIR = "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo"


def render_choice(opts):
    return {"t": "choice", "ins": "Pick the option that best matches the statement.",
            "crit": {str(i): o for i, o in enumerate(opts)}}


def render_noul():
    return {"t": "noul", "ins": "Does the statement hold?",
            "crit": {"false": None, "true": None}}


def render_score(levels):
    return {"t": "score", "ins": "Rate the statement.", "crit": list(levels)}


CASES = [
    {
        "state": "The agent was asked to restart the web server. It checked the logs, "
                 "found a stale lock file in /var/run, removed it, and restarted nginx "
                 "successfully. Memory usage returned to normal.",
        "questions": [
            {"id": "q1", "type": "noul", "threshold": 0.5},
            {"id": "q2", "type": "choice",
             "options": ["Restart the server", "Delete all files", "Do nothing",
                         "Escalate to a human"]},
            {"id": "q3", "type": "score"},
        ],
    },
    {
        "state": "error: segmentation fault in libcrypto\n\n"
                 "Traceback (most recent call last):\n"
                 "  File \"app.py\", line 42, in <module>\n"
                 "    main()\n"
                 "RuntimeError: CUDA out of memory\n",
        "questions": [
            {"id": "a", "type": "choice",
             "options": ["Memory leak", "Disk full", "Network timeout", "OOM",
                         "Permission denied", "Race condition", "Bit flip", "OOM"]},
            {"id": "b", "type": "noul"},
            {"id": "c", "type": "score"},
            {"id": "d", "type": "choice", "options": ["A", "B"]},
        ],
    },
    {
        "state": "Chinese text 中文文本 mixed with English and emoji 🚀 plus "
                 "多行\n换行\tand tab characters, and <h1> markup </h1>.",
        "questions": [
            {"id": "x1", "type": "noul", "threshold": 0.9},
            {"id": "x2", "type": "score"},
            {"id": "x3", "type": "choice",
             "options": ["Continue", "Stop", "Observe", "Human review", "Retry"]},
        ],
    },
    {
        "state": " " * 50 + "The customer asked for a refund because the order "
                 "arrived damaged. The agent apologized and issued a full refund.",
        "questions": [
            {"id": "r1", "type": "choice",
             "options": ["Refund the customer", "Refund the customer", "Refund the customer"]},
            {"id": "r2", "type": "noul"},
        ],
    },
    {
        "state": "Task: summarize the document.\n"
                 "The document describes a series of experiments on model alignment. "
                 "It reports that reinforcement learning from human feedback reduced "
                 "harmful outputs by 42% while keeping helpfulness constant.",
        "questions": [
            {"id": "s1", "type": "score"},
            {"id": "s2", "type": "choice",
             "options": ["Summarize the document", "Translate the document",
                         "Extract keywords", "Answer questions"]},
            {"id": "s3", "type": "noul", "threshold": 0.3},
        ],
    },
]


def build_laya_qdefs(questions):
    """Mirror the P0 mapping into laya qdefs with identical rendered options."""
    ids, internal = [], {}
    for q in questions:
        qid = q["id"]
        if q["type"] == "noul":
            qdef = render_noul()
            opts = ["false: no, the statement does not hold",
                    "true: yes, the statement holds"]
        elif q["type"] == "choice":
            qdef = render_choice(q["options"])
            opts = ["%d: %s" % (i, o) for i, o in enumerate(q["options"])]
        else:
            # score: laya renders "level %d: %s"; to make both pipelines see the
            # same texts the harness feeds crit = the post-colon level labels and
            # the P0 engine uses the rendered strings directly.
            qdef = render_score(["2", "3", "4", "5", "6", "7", "8", "9", "10"])
            opts = ["level %d: %s" % (i, v) for i, v in enumerate(qdef["crit"])]
        ids.append(qid)
        internal[qid] = qdef
        q["_opts"] = opts
    return ids, internal


def main():
    from laya.agent import Agent
    from laya.common import QTYPES, collate_items

    from phocinae.engine import Engine

    print("loading reference (laya.Agent on CPU)...")
    ag = Agent(MODEL_DIR, device="cpu")
    print("loading phocinae engine (CPU fp32)...")
    eng = Engine(MODEL_DIR, device="cpu", warm=False)

    n_fail = 0
    n_logits = 0
    worst = 0.0
    for ci, case in enumerate(CASES):
        state = case["state"]
        questions = [dict(q) for q in case["questions"]]
        ids, internal = build_laya_qdefs(questions)

        # --- reference: laya sequence construction + forward (numpy, like prod)
        items_ref = ag._encode_state(state, ids, internal)
        b_ref = collate_items([items_ref], ag.tok.pad_token_id)
        with torch.no_grad():
            logits_ref, act_ref = ag._forward(b_ref)

        # --- phocinae: same request through the P0 engine internals
        state_ids = eng.tok.encode(state.replace(eng.tok.mask_token, " "))
        rows = []
        for q in questions:
            spec = (q["type"], eng._question_spec(q)[1], q["_opts"])
            seq, markers = eng._build_sequence_fit(spec, state_ids)
            rows.append({"ids": seq, "markers": markers,
                         "qtype": QTYPES[q["type"]], "n_opts": len(q["_opts"]),
                         "spec": spec, "q": q})
        b_mine = eng._collate(rows)
        logits_mine, act_mine = eng._forward(b_mine)
        logits_mine = logits_mine.numpy()
        act_mine = act_mine.numpy()

        # 1) sequences must be identical
        ok = True
        for j in range(len(rows)):
            if rows[j]["ids"] != items_ref[j]["ids"]:
                print("case %d q%d: IDS differ\n  ref : %s\n  mine: %s"
                      % (ci, j, items_ref[j]["ids"][:60], rows[j]["ids"][:60]))
                ok = False
            if rows[j]["markers"] != items_ref[j]["markers"]:
                print("case %d q%d: MARKERS differ ref=%s mine=%s"
                      % (ci, j, items_ref[j]["markers"], rows[j]["markers"]))
                ok = False

        # 2) collated tensors identical
        for key in b_ref:
            if key in ("label", "meta"):
                continue
            if not torch.equal(b_ref[key], b_mine[key]):
                print("case %d: collate %s differs" % (ci, key))
                ok = False

        # 3) logits within fp32 tolerance
        d = float(np.abs(logits_ref - logits_mine).max())
        d_act = float(np.abs(act_ref - act_mine).max())
        worst = max(worst, d, d_act)
        n_logits += 1
        if d > 1e-3 or d_act > 1e-3:
            print("case %d: LOGITS diverge: max|dlogits|=%.6f max|dact|=%.6f"
                  % (ci, d, d_act))
            ok = False

        # 4) decoded answers / confidence / act_probability
        dec_mine = eng._decode_rows(logits_mine, act_mine, rows)
        dec_ref = ag._decode_answers(logits_ref, act_ref, items_ref, ids, internal, 0)
        for j, q in enumerate(questions):
            (ans, conf, actp, p) = dec_mine[j]
            dr = dec_ref[q["id"]]
            if q["type"] == "choice":
                want_ans = int(dr["choice"])
                if ans != want_ans:
                    print("case %d %s: choice answer differs mine=%s ref=%s"
                          % (ci, q["id"], ans, want_ans))
                    ok = False
                want_conf = dr["answer_confidence"]
                if conf != want_conf:
                    print("case %d %s: confidence differs mine=%s ref=%s"
                          % (ci, q["id"], conf, want_conf))
                    ok = False
            elif q["type"] == "score":
                # laya reports expected score; the P0 integer tier must equal
                # the argmax tier of the same probability vector
                want_ans = 2 + int(torch.tensor(
                    [dr["probabilities"][str(i)] for i in range(9)]).argmax())
                if ans != want_ans:
                    print("case %d %s: score differs mine=%s ref=%s"
                          % (ci, q["id"], ans, want_ans))
                    ok = False
            else:
                want_ans = bool(dr["noul"] >= q.get("threshold", 0.5))
                if ans != want_ans:
                    print("case %d %s: noul differs mine=%s ref=%s"
                          % (ci, q["id"], ans, want_ans))
                    ok = False
                want_conf = dr["answer_confidence"]
                if conf != want_conf:
                    print("case %d %s: confidence differs mine=%s ref=%s"
                          % (ci, q["id"], conf, want_conf))
                    ok = False
            if abs(actp - dr["action"]["act_probability"]) > 1e-3:
                print("case %d %s: act_probability differs mine=%s ref=%s"
                      % (ci, q["id"], actp, dr["action"]["act_probability"]))
                ok = False
        if not ok:
            n_fail += 1
    print("cases: %d, failures: %d, worst |d| over logits/act: %.3e"
          % (len(CASES), n_fail, worst))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
