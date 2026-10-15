"""phocinae-router: escalate 升级门（E1，τ=0.6 冻结）+ 工具路由（k≤10）。

两个组件共用同一推理后端，支持两种接入模式：

  - 进程内 Engine 模式：直接持有 phocinae.engine.Engine（零网络、最低延迟）
  - HTTP 模式：指向本机 phocinae-server（http://127.0.0.1:8155），经
    POST /v1/systemone（escalate 门，单遍判定）与
    POST /v1/systemone/permute（工具路由，4 排列平均）

口径（2026-10-07 冻结，证据见 09_常态探索 发布材料）：

  - escalate 门（E1）：τ=0.6。answer_confidence（温度标定 top 概率，已与
    laya canonical 对齐）低于 τ 的决策才升级外部模型。E1 typed en 400 例
    实测：本地 acc 0.791（基准策展清洗分；未策展读数 0.906，独立复现 0.9055，如实并排）→ 保留集 acc 0.9936（+0.0876 kept-subset，相对未策展基线 0.906），大模型
    调用占比 100%→45.0%，省费 55.0%（τ=0.5 档 79.6%）。τ 为默认值，允许按调用覆盖（新域须重扫 τ）。
  - 工具路由：k≤10 菜单（JevBench tool_selection k≤10 12/12=1.0 域）。
    choice 题 options=工具名（附一行描述时渲染为 "name: description"，
    banking77 渲染先例）；PHOC_PERM_AVG 口径 4 种选项排列平均压翻转
    （/v1/systemone/permute 或 engine.run(perm=True)）。

依赖：engine/HTTP 两种模式共用 phocinae.server 的 option_scores 扩展键
（每选项标定概率向量）取 per_option_scores；HTTP 模式额外需要 requests
（惰性导入，纯 engine 模式不依赖）。
"""

import ipaddress
import logging
import os
import urllib.parse

log = logging.getLogger("phocinae.router")

MODEL_NAME = os.environ.get("PHOC_MODEL_NAME", "Phocinae-Largha-150M-v1")

# E1 升级门冻结口径（2026-10-09 v1.1）：保留集 acc 0.9936、省费 55.0%。
TAU_DEFAULT = 0.6
MAX_TOOLS = 10          # 工具路由菜单上限（JevBench tool_selection 域界）
PERM_K = 4              # 排列平均口径（canonical，与 engine.PERM_K 一致）


class RouterError(RuntimeError):
    """Router 组件错误基类。"""


class RouterHTTPError(RouterError):
    """HTTP 模式：phocinae-server 不可达 / 非 200 / 响应缺扩展键或字段非法。"""


def _is_loopback_target(url):
    """True when `url` points at this machine / a private LAN address.

    Why it matters: `requests` reads the operating-system HTTP proxy settings
    (on Windows that is the WinINET registry key) and, unlike urllib, it does
    NOT honour the OS `ProxyOverride` bypass list. On a proxied machine — very
    common in China — a plain `requests.post("http://127.0.0.1:8155/...")` is
    therefore sent to the proxy and fails (502 / connection reset), which takes
    the whole local router path down. Loopback and private targets are never
    legitimately meant to go through a forward proxy, so we bypass it for them
    and leave the user's proxy configuration untouched for every other host.
    """
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        return False
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


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
    @staticmethod
    def _requests():
        """Lazy `requests` import (engine mode does not need it).

        Kept on the class so tests/instrumentation can swap it. A module-level
        import would also break `import phocinae.router` in environments that
        only use the in-process engine.
        """
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RouterError(
                "HTTP mode requires requests: pip install requests") from exc
        return requests

    def _post(self, url, payload, headers):
        """POST with the proxy bypass described in `_is_loopback_target`.

        A fresh Session per call with `trust_env` kept as-is for remote hosts
        (their proxy config may be intentional); for loopback/private targets
        proxies are explicitly disabled for this one request.
        """
        requests = self._requests()
        session = requests.Session()
        kwargs = {}
        if _is_loopback_target(url):
            kwargs["proxies"] = {"http": None, "https": None}
        return session.request("POST", url, json=payload, headers=headers,
                               timeout=self.timeout, **kwargs)

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
        requests = self._requests()
        url = self.base_url + ("/v1/systemone/permute" if perm
                               else "/v1/systemone")
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = "Bearer " + self.bearer_token
        payload = {"model": self.model, "state": state,
                   "questions": questions}
        try:
            resp = self._post(url, payload, headers)
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

    @staticmethod
    def _confidence_value(confs, qid, where):
        """Calibrated confidence must be a real number in [0, 1].

        A missing key, a string or None must surface as RouterHTTPError, not as
        a KeyError/TypeError from deep inside the gate arithmetic.
        """
        if qid not in confs:
            raise RouterHTTPError(
                "%s: no answer_confidence for question %r (got keys %s); "
                "PHOC_EXTENSIONS must not be 0"
                % (where, qid, sorted(confs)[:10]))
        raw = confs[qid]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RouterHTTPError(
                "%s: answer_confidence for %r is %r, expected a number in [0, 1]"
                % (where, qid, raw))
        c = float(raw)
        if not 0.0 <= c <= 1.0:
            raise RouterHTTPError(
                "%s: answer_confidence for %r is %r, outside [0, 1]"
                % (where, qid, raw))
        return c

    @staticmethod
    def _answers_map(body, where):
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise RouterHTTPError(
                "%s: response has no 'answers' object (got %s) — is the base "
                "URL really a phocinae-server?" % (where, type(answers).__name__))
        return answers

    @staticmethod
    def _tool_index(answer, k, where):
        """Validate a choice answer as a usable option index.

        The server may return anything (or a proxy may return a mangled body):
        a bool (subclass of int), a float, a string or an out-of-range int must
        become RouterHTTPError, never IndexError/TypeError.
        """
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise RouterHTTPError(
                "%s: choice answer %r is not an integer index" % (where, answer))
        if not 0 <= answer < k:
            raise RouterHTTPError(
                "%s: choice answer index %d out of range (0..%d)"
                % (where, answer, k - 1))
        return answer

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
        answers = self._answers_map(body, "escalate")
        per = []
        for q in questions:
            qid = q["id"]
            if qid not in answers:
                raise RouterHTTPError(
                    "escalate: no answer for question %r (got %s)"
                    % (qid, sorted(answers)[:10]))
            c = self._confidence_value(confs, qid, "escalate")
            per.append({
                "id": qid,
                "decision": answers[qid],
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
        answers = self._answers_map(body, "route_tool")
        if "tool" not in answers:
            raise RouterHTTPError(
                "route_tool: response has no answer for question 'tool' "
                "(got %s)" % sorted(answers)[:10])
        # Validate before use: `names[idx]` with an unchecked index was an
        # IndexError/TypeError instead of a RouterHTTPError (S6).
        idx = self._tool_index(answers["tool"], k, "route_tool")
        conf = self._confidence_value(confs, "tool", "route_tool")
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
