"""Inference engine: sequence construction, batched forward, decoding.

The sequence construction mirrors laya's build_sequence protocol exactly
(head "[qtype] question: <ins>" + [MASK]-prefixed option markers + state +
EOS separators), so the server serves the released checkpoint with the same
input distribution it was trained with. P0 questions are mapped onto that
protocol:

  noul   -> options "false: no, the statement does not hold" /
            "true: yes, the statement holds"; answer = P(true) >= threshold
  choice -> options rendered as "<i>: <option text>" with the P0 list index
            as the label; answer = index of the argmax option
  score  -> 9 ordinal levels "level 2" .. "level 10";
            answer = 2 + argmax (an integer in 2..10)

Permutation averaging (PHOC_PERM_AVG / /v1/systemone/permute) follows the
canonical protocol: K=4 option orders (original, reversed, two seeded
shuffles, seed 0) with choice probabilities averaged before argmax.
"""

import json
import logging
import math
import os
import random
import threading
import time

import torch

from .model import DecisionModel, MMBertEncoder
from .tokenizer import META, GemmaTokenizer
from .safetensors_lite import load_safetensors

log = logging.getLogger("phocinae.engine")

QTYPES = {"choice": 0, "score": 1, "noul": 2}

DEFAULT_INS = {
    "choice": "Pick the option that best matches the statement.",
    "score": "Rate the statement.",
    "noul": "Does the statement hold?",
}
NOUL_OPTIONS = [
    "false: no, the statement does not hold",
    "true: yes, the statement holds",
]
SCORE_LEVELS = 9          # answers 2..10
MAX_OPTIONS = 255
MAX_QUESTIONS = 64
MAX_POS = 8192            # encoder max_position_embeddings
PERM_K = 4                # canonical permutation averaging order count
BUDGET_TIERS = [(192, 512), (512, 1024), (1024, 2048), (2048, 4096),
                (4096, MAX_POS), (MAX_POS, MAX_POS)]


class Engine:
    def __init__(self, model_dir, device="auto", model_name="Phocinae-Largha-150M-v1",
                 perm_avg=False, warm=True, threads=None):
        self.model_name = model_name
        self.perm_avg = perm_avg
        if device in ("auto", None, ""):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.dtype = torch.float16 if self.is_cuda else torch.float32
        if not self.is_cuda and threads:
            torch.set_num_threads(int(threads))
        self._lock = threading.Lock()
        # Constant question-head texts recur on every request: cache once.
        self._head_cache = {}
        # Per-run encode memo for option strings (perm orders repeat them).
        self._opt_cache = {}
        # Lazily torch.compile'd model on CUDA (reduce-overhead, dynamic shapes);
        # falls back to eager if compilation fails.
        self._compiled = None
        self._compile_failed = False

        with open(os.path.join(model_dir, "encoder", "config.json"),
                  encoding="utf-8") as fh:
            ecfg = json.load(fh)
        with open(os.path.join(model_dir, "rl_agent_config.json"),
                  encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        self.max_len = int(self.cfg.get("max_len", 512))
        self.head_max_len = int(self.cfg.get("head_max_len", 192))

        log.info("loading tokenizer from %s", model_dir)
        t0 = time.time()
        self.tok = GemmaTokenizer(os.path.join(model_dir, "tokenizer"))
        self._verify_vocab()
        log.info("tokenizer ready in %.2fs", time.time() - t0)

        log.info("loading weights from %s", os.path.join(model_dir, "model.safetensors"))
        t0 = time.time()
        full_layers = {i for i, t in enumerate(ecfg["layer_types"])
                       if t == "full_attention"}
        encoder = MMBertEncoder(
            vocab=ecfg["vocab_size"],
            d=ecfg["hidden_size"],
            n_layers=ecfg["num_hidden_layers"],
            heads=ecfg["num_attention_heads"],
            inter=ecfg["intermediate_size"],
            eps=float(ecfg.get("layer_norm_eps", 1e-5)),
            full_layers=full_layers,
            theta=float(ecfg["rope_parameters"]["full_attention"]["rope_theta"]),
            local_window=int(ecfg["local_attention"]) // 2,
        )
        head = DecisionModel(
            d=ecfg["hidden_size"],
            head_layers=int(self.cfg.get("head_layers", 2)),
            heads=ecfg["hidden_size"] // 64,
            dropout=float(self.cfg.get("head_dropout", 0.1)),
        )
        head.encoder = encoder
        self.model = head
        weights = load_safetensors(os.path.join(model_dir, "model.safetensors"))
        # The checkpoint's "temperature" tensor is a dummy (all ones); the
        # canonical calibrated temperatures ship in rl_agent_config.json.
        weights.pop("temperature", None)
        self.temperature = [float(t) for t in self.cfg.get("temperature",
                                                           [1.0, 1.0, 1.0])]
        if len(self.temperature) != 3:
            raise RuntimeError("expected temperature vector of size 3")
        weights = {k: v.to(self.dtype) for k, v in weights.items()}
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if missing:
            raise RuntimeError("weights missing for: %s" % ", ".join(sorted(missing)))
        if unexpected:
            raise RuntimeError("unexpected weight keys: %s" % ", ".join(sorted(unexpected)))
        self.model = self.model.to(self.device).eval()
        log.info("model loaded in %.2fs (%s, %d params)",
                 time.time() - t0, self.device, sum(p.numel() for p in self.model.parameters()))
        if warm:
            self.warmup()

    # ------------------------------------------------------------ vocabulary
    def _verify_vocab(self):
        anchors = {0: "<pad>", 1: "<eos>", 2: "<bos>", 3: "<unk>", 4: "<mask>",
                   476: META + "a", 108: "\n", 235248: META}
        for i, want in anchors.items():
            got = self.tok.id2tok.get(i)
            if got != want:
                raise RuntimeError("vocab anchor mismatch: id %d -> %r, expected %r"
                                   % (i, got, want))
        probes = {
            "hello": [25612],
            "hello world": [25612, 2134],
            "the the the": [573, 573, 573],
            "a\nb": [476, 108, 518],
            "a": [476],
            " \t": [235248, 226],
        }
        for text, want in probes.items():
            got = self.tok.encode(text)
            if got != want:
                raise RuntimeError("vocab encode mismatch for %r: got %r want %r"
                                   % (text, got, want))
        rt = "hello world, 中文测试 🚀"
        ids = self.tok.encode(rt)
        back = self.tok.decode(ids)
        if back.replace(" ", "") != rt.replace(" ", ""):
            raise RuntimeError("vocab round-trip failed: %r -> %r" % (rt, back))
        log.info("vocabulary decode-verified (%d tokens, %d merges)",
                 len(self.tok.vocab), len(self.tok.merges))

    # ---------------------------------------------------------- warmup buckets
    def warmup(self):
        buckets = [(1, 64, 2), (1, 128, 4), (2, 128, 4), (4, 128, 4), (4, 256, 9),
                   (8, 256, 9), (16, 256, 9), (16, 512, 16), (32, 512, 16), (64, 512, 32)]
        log.info("warming shape buckets: %s", buckets)
        t0 = time.time()
        with torch.no_grad():
            for B, L, K in buckets:
                b = {
                    "input_ids": torch.randint(5, 2000, (B, L), dtype=torch.long),
                    "attention_mask": torch.ones((B, L), dtype=torch.long),
                    "marker_pos": torch.randint(0, L, (B, K), dtype=torch.long),
                    "marker_mask": torch.ones((B, K), dtype=torch.bool),
                    "qtype": torch.zeros((B,), dtype=torch.long),
                }
                self._forward(b)
                if self.is_cuda:
                    torch.cuda.synchronize()
        log.info("warmup done in %.2fs", time.time() - t0)

    # ------------------------------------------------------------ inference
    def _amp_ctx(self):
        if self.is_cuda:
            return torch.autocast(self.device.type, dtype=torch.float16)
        return torch.autocast("cpu", enabled=False)

    def _forward(self, b):
        with torch.no_grad(), self._amp_ctx():
            m = self.model
            if self.is_cuda and not self._compile_failed:
                if self._compiled is None:
                    try:
                        self._compiled = torch.compile(
                            self.model, mode="reduce-overhead", dynamic=True)
                        log.info("torch.compile active (reduce-overhead, dynamic)")
                    except Exception as exc:  # pragma: no cover
                        self._compile_failed = True
                        log.warning("torch.compile unavailable: %s", exc)
                m = self._compiled if self._compiled is not None else self.model
            logits, act = m(
                b["input_ids"].to(self.device),
                b["attention_mask"].to(self.device),
                b["marker_pos"].to(self.device),
                b["marker_mask"].to(self.device),
                b["qtype"].to(self.device),
            )
        return logits.float().cpu(), torch.softmax(act.float(), -1).cpu()

    # ------------------------------------------------------- sequence building
    def _question_spec(self, q):
        t = q["type"]
        ins = DEFAULT_INS[t]
        if t == "noul":
            return t, ins, list(NOUL_OPTIONS)
        if t == "choice":
            return t, ins, [str(o) for o in q["options"]]
        return t, ins, ["level %d" % (2 + i) for i in range(SCORE_LEVELS)]

    def _build_sequence(self, spec, state_ids, order, head_max_len, max_len):
        t, ins, opts = spec
        head_ids = self._head_cache.get((t, ins))
        if head_ids is None:
            head_ids = self.tok.encode("%s question: %s" % (t, ins))
            self._head_cache[(t, ins)] = head_ids
        opt_ids = []
        for i in order:
            o = opts[i].replace(self.tok.mask_token, " ")
            ot = self._opt_cache.get(o)
            if ot is None:
                ot = self.tok.encode(" " + o)[:48]
                self._opt_cache[o] = ot
                if len(self._opt_cache) > 65536:
                    self._opt_cache.clear()
            opt_ids.append([self.tok.mask_id] + ot)
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[:max(8, opt_budget)]
        ids = [self.tok.cls_token_id] + head_ids + [self.tok.sep_token_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.tok.sep_token_id)
        room = max(0, max_len - len(ids) - 1)
        ids = ids + state_ids[:room] + [self.tok.sep_token_id]
        return ids[:max_len], [m for m in markers if m < max_len]

    def _build_sequence_fit(self, spec, state_ids, order=None):
        k = len(spec[2])
        if order is None:
            order = list(range(k))
        for hml, ml in BUDGET_TIERS:
            seq, markers = self._build_sequence(spec, state_ids, order, hml, ml)
            if len(markers) == k:
                return seq, markers
        raise ValueError("options exceed the %d-token model budget" % MAX_POS)

    def _encode_request(self, state, questions):
        state_ids = self.tok.encode(state.replace(self.tok.mask_token, " "))
        items = []
        for q in questions:
            spec = self._question_spec(q)
            seq, markers = self._build_sequence_fit(spec, state_ids)
            items.append({
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q["type"]],
                "n_opts": len(spec[2]),
                "spec": spec,
                "q": q,
            })
        return state_ids, items

    # ------------------------------------------------------------- collation
    def _collate(self, items):
        n = len(items)
        L = max(len(it["ids"]) for it in items)
        K = max(len(it["markers"]) for it in items)
        ids = [[self.tok.pad_id] * L for _ in range(n)]
        att = [[0] * L for _ in range(n)]
        mpos = [[0] * K for _ in range(n)]
        mmask = [[False] * K for _ in range(n)]
        for i, it in enumerate(items):
            ids[i][:len(it["ids"])] = it["ids"]
            att[i][:len(it["ids"])] = [1] * len(it["ids"])
            for j, m in enumerate(it["markers"]):
                mpos[i][j] = m
                mmask[i][j] = True
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
            "marker_pos": torch.tensor(mpos, dtype=torch.long),
            "marker_mask": torch.tensor(mmask, dtype=torch.bool),
            "qtype": torch.tensor([it["qtype"] for it in items], dtype=torch.long),
        }

    # --------------------------------------------------------------- decoding
    @staticmethod
    def _probs(logits_row, k, t_scale):
        z = [float(v) / t_scale for v in logits_row[:k]]
        m = max(z)
        p = [math.exp(v - m) for v in z]
        s = sum(p)
        return [v / s for v in p]

    def _decode_rows(self, logits, act, items):
        """Returns per-item (answer_value, confidence, act_probability).

        confidence follows laya's canonical answer_confidence:
        float(clip(max(p[:k]), 0, 1)) — the calibrated top probability.
        """
        out = []
        for j, it in enumerate(items):
            k = it["n_opts"]
            t = it["q"]["type"]
            t_scale = self.temperature[QTYPES[t]]
            p = self._probs(logits[j], k, t_scale)
            conf = round(float(min(1.0, max(p[:k]))), 4)
            act_prob = round(float(act[j, 0]), 4)
            if t == "choice":
                ans = max(range(k), key=lambda i: p[i])
            elif t == "score":
                ans = 2 + max(range(k), key=lambda i: p[i])
            else:
                thr = it["q"].get("threshold")
                thr = 0.5 if thr is None else float(thr)
                ans = bool(p[1] >= thr)
            out.append((ans, conf, act_prob, p))
        return out

    # ------------------------------------------------------------- permutation
    def _perm_orders(self, item, rng):
        k = item["n_opts"]
        if item["q"]["type"] != "choice" or k < 2:
            return [list(range(k))]
        o1 = list(reversed(range(k)))
        o2 = list(range(k))
        rng.shuffle(o2)
        o3 = list(range(k))
        rng.shuffle(o3)
        return [list(range(k)), o1, o2, o3]

    # ----------------------------------------------------------------- run
    def run(self, state, questions, perm=None, usage=True, with_scores=False):
        """Run one system-one request. Returns (answers, confidence, action, usage).

        perm: None -> engine default; False -> single pass; True -> permute-avg.
        with_scores: True -> also return the per-option calibrated probability
        vectors as a 5th element: {qid: [p0, p1, ...]} in canonical option
        order (choice: request option order; noul: [false, true];
        score: levels 2..10). Choice vectors are averaged over permuted
        orders when perm=True.
        """
        if perm is None:
            perm = self.perm_avg
        with self._lock:
            state_ids, items = self._encode_request(state, questions)
            rng = random.Random(0)
            rows = []
            row_meta = []  # (question_idx, order)
            for qi, it in enumerate(items):
                orders = self._perm_orders(it, rng) if perm \
                    else [list(range(it["n_opts"]))]
                for order in orders:
                    seq, markers = self._build_sequence_fit(it["spec"], state_ids, order)
                    rows.append({"ids": seq, "markers": markers,
                                 "qtype": it["qtype"], "n_opts": it["n_opts"],
                                 "spec": it["spec"], "q": it["q"]})
                    row_meta.append((qi, order))
            b = self._collate(rows)
            logits, act = self._forward(b)
            decoded = self._decode_rows(logits.numpy(), act.numpy(), rows)

            # regroup per question; average choice probabilities over orders
            n_q = len(questions)
            per_q = [[] for _ in range(n_q)]
            for (qi, order), d in zip(row_meta, decoded):
                per_q[qi].append((order,) + d)
            answers, confs, actions = {}, {}, {}
            scores = {}
            for qi, rowsq in enumerate(per_q):
                q = questions[qi]
                order0 = rowsq[0][0]
                n_opts = len(order0)
                if len(rowsq) == 1:
                    _, ans, conf, actp, p = rowsq[0]
                else:
                    pal = [0.0] * n_opts
                    for order, _ans, _conf, _actp, p in rowsq:
                        for pos, opt_idx in enumerate(order):
                            pal[opt_idx] += p[pos]
                    m = len(rowsq)
                    pal = [v / m for v in pal]
                    p = pal
                    conf = round(float(min(1.0, max(pal[:n_opts]))), 4)
                    actp = rowsq[0][3]
                    ans = max(range(n_opts), key=lambda i: pal[i])
                if q["type"] == "choice":
                    pass  # ans is already the option index
                elif q["type"] == "score":
                    ans = 2 + max(range(len(p)), key=lambda i: p[i])
                else:
                    thr = q.get("threshold")
                    thr = 0.5 if thr is None else float(thr)
                    ans = bool(p[1] >= thr)
                answers[q["id"]] = ans
                confs[q["id"]] = conf
                actions[q["id"]] = {"act_probability": actp}
                scores[q["id"]] = [round(float(v), 4) for v in p]
            total_tokens = int(b["attention_mask"].sum().item())
            usage = {"input_tokens": total_tokens, "output_tokens": 0}
            if with_scores:
                return answers, confs, actions, usage, scores
            return answers, confs, actions, usage
