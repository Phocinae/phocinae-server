"""phocinae-router: escalate 升级门（E1，τ=0.6 冻结）+ 工具路由（k≤10）。

两个组件共用同一推理后端，支持两种接入模式：

  - 进程内 Engine 模式：直接持有 phocinae.engine.Engine（零网络、最低延迟）
  - HTTP 模式：指向本机 phocinae-server（http://127.0.0.1:8155），经
    POST /v1/systemone（escalate 门，单遍判定）与
    POST /v1/systemone/permute（工具路由，4 排列平均）

口径（2026-10-07 冻结，证据见 09_常态探索 发布材料）：

  - escalate 门（E1）：τ=0.6。answer_confidence（温度标定 top 概率，已与
    laya canonical 对齐）低于 τ 的决策才升级外部模型。E1 typed en 400 例
    实测：本地 acc 0.797（独立复现 0.7825，如实并排）→ 保留集 acc 0.886（+0.089 kept-subset），大模型
    调用占比 100%→45.7%，省费 54.4%（τ=0.5 档 82.8%）。τ 为默认值，允许按调用覆盖（新域须重扫 τ）。
  - 工具路由：k≤10 菜单（JevBench tool_selection k≤10 12/12=1.0 域）。
    choice 题 options=工具名（附一行描述时渲染为 "name: description"，
    banking77 渲染先例）；PHOC_PERM_AVG 口径 4 种选项排列平均压翻转
    （/v1/systemone/permute 或 engine.run(perm=True)）。

依赖：engine/HTTP 两种模式共用 phocinae.server 的 option_scores 扩展键
（每选项标定概率向量）取 per_option_scores；HTTP 模式额外需要 requests
（惰性导入，纯 engine 模式不依赖）。
"""

import logging
import os

log = logging.getLogger("phocinae.router")

MODEL_NAME = os.environ.get("PHOC_MODEL_NAME", "Phocinae-Largha-150M-v1")

# E1 升级门冻结口径（2026-10-07）：保留集 acc 0.886、省费 54%。
TAU_DEFAULT = 0.6
MAX_TOOLS = 10          # 工具路由菜单上限（JevBench tool_selection 域界）
PERM_K = 4              # 排列平均口径（canonical，与 engine.PERM_K 一致）


class RouterError(RuntimeError):
    """Router 组件错误基类。"""


class RouterHTTPError(RouterError):
    """HTTP 模式：phocinae-server 不可达 / 非 200 / 响应缺扩展键。"""


class Router:
    """escalate 升级门 + 工具路由。

    Router(engine=Engine(...))              -> 进程内模式
    Router(base_url="http://127.0.0.1:8155") -> HTTP 模式
    """

    def __init__(self, engine=None, base_url=None, model=None,
                 tau=TAU_DEFAULT, timeout=60.0, bearer_token=None):
        if engine is not None and base_url is not None:
            raise ValueError("router: give either engine or base_url, not both")
        self.tau = float(tau)
        self.model = model or MODEL_NAME
        self.timeout = float(timeout)
        self.bearer_token = (bearer_token
                             or os.environ.get("PHOC_BEARER_TOKEN") or None)
        if engine is not None:
            self.mode = "engine"
            self._engine = engine
        elif base_url:
            self.mode = "http"
            self.base_url = base_url.rstrip("/")
        else:
            raise ValueError("router: need engine (in-process) or base_url (HTTP)")

    # ------------------------------------------------------------- transport
    def _systemone(self, state, questions, perm):
        """Run one system-one request; return the normalized response dict
        (answers / answer_confidence / action / option_scores / usage /
        routing), identical shape in both modes."""
        if self.mode == "engine":
            answers, confs, actions, usage, scores = self._engine.run(
                state, questions, perm=perm, with_scores=True)
            return {
                "model": self.model,
                "answers": answers,
                "usage": usage,
                "answer_confidence": confs,
                "action": actions,
                "option_scores": scores,
                "routing": {
                    "model": self.model,
                    "device": self._engine.device.type,
                    "perm": "avg-k4" if perm else "none",
                    "backend": "phocinae-engine-inprocess",
                },
            }
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RouterError(
                "HTTP mode requires requests: pip install requests") from exc
        url = self.base_url + ("/v1/systemone/permute" if perm
                               else "/v1/systemone")
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = "Bearer " + self.bearer_token
        payload = {"model": self.model, "state": state,
                   "questions": questions}
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=self.timeout)
        except requests.RequestException as exc:
            raise RouterHTTPError("phocinae-server unreachable at %s: %s"
                                  % (self.base_url, exc)) from exc
        if resp.status_code != 200:
            raise RouterHTTPError("phocinae-server %s -> %d: %s"
                                  % (url, resp.status_code,
                                     resp.text[:400].replace("\n", " ")))
        try:
            body = resp.json()
        except ValueError as exc:
            raise RouterHTTPError("non-JSON response from %s" % url) from exc
        return body

    def _confidence_map(self, body, questions):
        confs = body.get("answer_confidence") or {}
        missing = [q["id"] for q in questions if q["id"] not in confs]
        if missing:
            raise RouterHTTPError(
                "answer_confidence missing for %s — is PHOC_EXTENSIONS=0 or "
                "not a phocinae-server? (escalate gate needs the calibrated "
                "confidence extension)" % missing)
        return confs

    # ------------------------------------------------------------- escalate
    def escalate(self, state, questions, tau=None):
        """E1 升级门：本地判定一次，answer_confidence < τ 的决策 → escalate。

        questions 与 /v1/systemone 同规格（至少 1 题，noul/choice/score 均可）。
        tau 缺省为 E1 冻结口径 0.6；覆盖仅用于新域重扫实验。

        单题返回（扁平）：{decision, confidence, escalate, tau, gate,
                          question_id, usage}
        多题返回（聚合）：{decision: {id: 值}, confidence: {id: float},
                          escalate, escalated_ids, tau, gate,
                          per_question: [...], usage}
        """
        tau = float(TAU_DEFAULT if tau is None else tau)
        if not questions:
            raise ValueError("escalate: questions must be non-empty")
        body = self._systemone(state, questions, perm=False)
        confs = self._confidence_map(body, questions)
        per = []
        for q in questions:
            c = float(confs[q["id"]])
            per.append({
                "id": q["id"],
                "decision": body["answers"][q["id"]],
                "confidence": c,
                "escalate": bool(c < tau),
            })
        escalated_ids = [e["id"] for e in per if e["escalate"]]
        gate_meta = {
            "tau": tau,
            "gate": "E1-tau0.6",
            "usage": body.get("usage"),
        }
        if len(questions) == 1:
            out = dict(per[0])
            out.pop("id")
            out["question_id"] = questions[0]["id"]
            out.update(gate_meta)
            return out
        return {
            "decision": dict(body["answers"]),
            "confidence": dict(confs),
            "escalate": bool(escalated_ids),
            "escalated_ids": escalated_ids,
            "per_question": per,
            **gate_meta,
        }

    # ----------------------------------------------------------- tool route
    @staticmethod
    def _normalize_tools(tools):
        """Normalize tools into (names, descriptions, render_fn).

        Accepts: list[str] | list[{name, description|desc}] | {name: desc}.
        """
        names, descs = [], []
        if isinstance(tools, dict):
            items = list(tools.items())
        elif isinstance(tools, (list, tuple)):
            items = []
            for t in tools:
                if isinstance(t, str):
                    items.append((t, None))
                elif isinstance(t, dict) and "name" in t:
                    items.append((t["name"],
                                  t.get("description") or t.get("desc")))
                else:
                    raise ValueError("route_tool: bad tool spec %r" % (t,))
        else:
            raise ValueError("route_tool: tools must be a list or a dict")
        seen = set()
        for name, desc in items:
            name = str(name).strip()
            if not name:
                raise ValueError("route_tool: tool names must be non-empty")
            if name in seen:
                raise ValueError("route_tool: duplicate tool name %r" % name)
            seen.add(name)
            names.append(name)
            descs.append(str(desc).strip() if desc else None)

        def render(name, desc):
            # 05 件 §4.1 D2 口径：options=工具名（+一行描述，banking77 渲染先例）
            if not desc:
                return name
            return "%s: %s" % (name, desc[:160])

        return names, descs, render

    def route_tool(self, state, tools, tau=None, perm=True):
        """工具路由：state + ≤10 个工具描述 → choice 题 + 4 排列平均。

        tools 三种形态：
          ["web_search", "read_file", ...]
          [{"name": "web_search", "description": "..."}, ...]
          {"web_search": "search the public web", ...}

        返回：{tool, tool_index, confidence, per_option_scores,
              per_question: [{id, tool, confidence, escalate}],
              escalate, tau, gate, usage, routing}
        per_option_scores 键=工具名，值=4 排列平均后的标定概率；
        escalate = confidence < τ（低于 τ 的工具选择建议升级大模型）。
        """
        names, descs, render = self._normalize_tools(tools)
        k = len(names)
        if k < 1:
            raise ValueError("route_tool: tools must be non-empty")
        if k > MAX_TOOLS:
            raise ValueError("route_tool: %d tools > %d menu cap "
                             "(JevBench tool_selection domain bound)"
                             % (k, MAX_TOOLS))
        options = [render(n, d) for n, d in zip(names, descs)]
        questions = [{"id": "tool", "type": "choice", "options": options}]
        body = self._systemone(state, questions, perm=perm)
        confs = self._confidence_map(body, questions)
        idx = body["answers"]["tool"]
        conf = float(confs["tool"])
        scores_raw = body.get("option_scores") or {}
        raw = scores_raw.get("tool")
        if not raw or len(raw) != k:
            raise RouterHTTPError(
                "option_scores missing for tool routing — needs a "
                "phocinae-server with the option_scores extension "
                "(this repo's server.py)")
        per_option_scores = {n: float(raw[i]) for i, n in enumerate(names)}
        tau = float(TAU_DEFAULT if tau is None else tau)
        return {
            "tool": names[idx],
            "tool_index": idx,
            "confidence": conf,
            "per_option_scores": per_option_scores,
            "per_question": [{
                "id": "tool",
                "tool": names[idx],
                "confidence": conf,
                "escalate": bool(conf < tau),
            }],
            "escalate": bool(conf < tau),
            "tau": tau,
            "gate": "E1-tau0.6",
            "usage": body.get("usage"),
            "routing": body.get("routing"),
        }
