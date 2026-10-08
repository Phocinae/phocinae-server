# phocinae-server

斑海豹（Phocinae-Largha-150M）本机决策服务：把 1.5 亿参数的决策模型打包成
一行 `pip install` 级别依赖的 FastAPI 服务，把 `POST /v1/systemone` 契约带给
任何本机工具。**不提供对外服务**——它监听 127.0.0.1，只服务你自己机器上的
agent 与工作流。

## 定位

- 决策模型不是聊天模型：本服务**刻意不实现** chat / Ollama / OpenAI-compat。
- 推理内核为**纯 torch 自写前向**（运行时零 laya/transformers 依赖），权重直读
  发布的 hf_repo safetensors；与官方推理栈做过数值对齐（见 `dev_tools/verify_model.py`，
  5 组跨题型用例，logits 最大偏差 0）。
- 支持 noul（是否）/ choice（多选一）/ score（2–10 打分）三种题型，一次请求最多
  64 题、choice 最多 255 选项、上下文预算 8192 token。

## Install

```bash
git clone https://github.com/Phocinae/phocinae-server.git
cd phocinae-server
python -m venv .venv && . .venv/bin/activate
pip install fastapi uvicorn torch
```

模型权重：设置 `PHOC_MODEL_DIR` 指向斑海豹 hf_repo 目录（含
`model.safetensors`、`encoder/config.json`、`rl_agent_config.json`、`tokenizer/`）。

## 运行

```bash
PHOC_MODEL_DIR=/path/to/hf_repo python -m phocinae.main
# 默认 http://127.0.0.1:8155 ，/docs 有交互式文档
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PHOC_MODEL_DIR` | (必设) | hf_repo 目录 |
| `PHOC_HOST` / `PHOC_PORT` | 127.0.0.1 / 8155 | 监听地址 |
| `PHOC_DEVICE` | auto | auto / cuda / cpu |
| `PHOC_PERM_AVG` | 0 | 1 = 对 choice 题做 4 种选项排列平均（压翻转） |
| `PHOC_BEARER_TOKEN` | 空 | 设置后 /v1/* 要求 `Authorization: Bearer <token>` |
| `PHOC_THREADS` | — | CPU 模式 torch 线程数 |
| `PHOC_MAX_BODY` | 2MiB | 请求体上限（超限 413） |
| `PHOC_EXTENSIONS` | 1 | 0 = 不返回 answer_confidence/action/routing 扩展键 |

## API

```
GET  /health                存活 + 模型/设备信息（无需鉴权）
GET  /v1/models             模型目录
POST /v1/systemone          单请求判定
POST /v1/systemone/permute  同左，choice 题选项排列平均
POST /v1/systemone/batch    批量（≤64 个请求）
```

请求：

```json
{"model": "Phocinae-Largha-150M-v1",
 "state": "agent 刚刚执行了 rm -rf /var/log/app",
 "questions": [
   {"id": "ok", "type": "noul", "threshold": 0.65},
   {"id": "act", "type": "choice",
    "options": ["allow", "ask", "deny"]},
   {"id": "risk", "type": "score"}]}
```

响应：`answers` 里 noul=布尔、choice=选项下标、score=2–10 整数；`usage` 给 token
计数；`answer_confidence`（温度标定的 top 概率）、`action.act_probability`、
`routing`（设备/后端/排列平均开关）为扩展键，可关闭。

## 性能（实测，本机 RTX 5090 / 20 核 CPU）

| 路径 | p50 |
|---|---|
| GPU fp16 + torch.compile（整请求） | **1.75 ms** |
| CPU fp32 8 线程（整请求） | 35.97 ms |

GPU 前向经 `torch.compile(reduce-overhead, dynamic)` 融合（eager 14.1ms → 编译后
~2ms）；加载时预热常用 batch 形状，首问无形状抖动。

## 安全声明

- 默认**仅监听 127.0.0.1**，默认无鉴权——不要把它暴露到局域网/公网；需要时用
  `PHOC_BEARER_TOKEN` 加 bearer。
- 判定结果来自 150M 决策模型，**不是安全审计的完备防护**；用于审批门时必须配合
  确定性规则层（见 phocinae-guard 的 L0 表），并把灰区交给人工。
- 服务崩溃时应由调用方 fail-closed（本服务不保证任何高可用）。

## 兼容性

- typesafe-sdk-python：把 base_url 指向本服务即可接入（字段按官方形状对齐）。
- 降级链：无 GPU 自动回退 CPU（fp32）；GPU 显存不足时 torch 侧回退由上层处理。

## Router 组件（escalate 升级门 + 工具路由）

`phocinae/router.py` 提供两个 harness 决策组件，共用本服务推理后端，支持两种
接入模式：**进程内 Engine**（零网络、最低延迟）与 **HTTP**（指向
`http://127.0.0.1:8155`）。

```python
from phocinae.engine import Engine
from phocinae.router import Router

router = Router(engine=Engine(MODEL_DIR, device="cpu"))     # 进程内
router = Router(base_url="http://127.0.0.1:8155")           # HTTP（另需 requests）
```

### escalate 升级门（E1，τ=0.6 冻结口径）

本地单遍判定一次，`answer_confidence`（温度标定 top 概率，已与 laya canonical
对齐）低于 τ 的决策升级外部模型/人工：

```python
res = router.escalate(
    "The agent was asked to restart the web server. It checked the logs and restarted nginx.",
    [{"id": "ok", "type": "noul"}])
# 单题 → {"decision": bool/int, "confidence": 0.786, "escalate": False,
#         "tau": 0.6, "gate": "E1-tau0.6", "question_id": "ok", "usage": {...}}
# 多题 → {"decision": {id:值}, "confidence": {id:float}, "escalate": bool,
#         "escalated_ids": [...], "per_question": [...], "tau", "gate", "usage"}
```

**口径（2026-10-07 冻结，实测）**：τ=0.6；E1 typed en 400 例本地 acc 0.789 →
组合 acc 0.7948（+0.6pp，噪声内无劣化）；大模型调用占比 100%→18%，**省费 82%**；
门控决策 perm 翻转 8.87% ≤ 13% 基线。证据：
`09_常态探索/release_prep_20261007/08_发布材料_20261007/04_省费评估_20261007.md`。
τ 缺省为冻结值 0.6，允许按调用覆盖（`tau=` 参数）——**新域须重扫 τ**，勿直接
沿用。HTTP 模式要求服务端扩展键开启（`PHOC_EXTENSIONS=1`，默认开启）。

### 工具路由（k≤10）

state + ≤10 个工具描述 → choice 题（options=工具名，附一行描述时渲染为
`name: description`，banking77 渲染先例）→ 4 排列平均（
`/v1/systemone/permute` 口径，压选项顺序翻转）：

```python
res = router.route_tool(
    "The user wants to find the latest news about the product launch.",
    [{"name": "web_search", "description": "search the public web"},
     {"name": "read_file", "description": "read a local file"},
     {"name": "run_command", "description": "run a shell command"},
     {"name": "list_files", "description": "list directory files"},
     {"name": "fetch_url", "description": "fetch a URL"},
     {"name": "ask_user", "description": "ask the human user"}])
# → {"tool": "web_search", "tool_index": 0, "confidence": 0.5561,
#    "per_option_scores": {"web_search": 0.5561, ...},  # 排列平均后的标定概率
#    "per_question": [{"id":"tool","tool":...,"confidence":...,"escalate":...}],
#    "escalate": bool, "tau": 0.6, "gate": "E1-tau0.6", "usage", "routing"}
```

`tools` 支持三种形态：`["name", ...]`、`[{"name", "description"}]`、
`{"name": "description"}`。工具名须唯一且非空；k>10 直接报错（JevBench
tool_selection 域界）。口径：k≤10 菜单 JevBench 12/12=1.0（本地原生零升级）；
大菜单（k>10）不接终选，走层级路由或外部分诊。

### 服务端配合

工具路由的 `per_option_scores` 需要本服务新增的 `option_scores` 扩展键（逐选项
标定概率向量：choice=请求选项序、noul=[p(false), p(true)]、score=2–10 各档；
perm 时已按 4 排列平均）。引擎模式与 HTTP 模式输出同构。

### 测试

```bash
CUDA_VISIBLE_DEVICES='' PHOC_DEVICE=cpu \
  /home/hermes/decision-model/.venv/bin/python tests/test_router.py
# 可选 HTTP 对拍（额外拉起一个本机 CPU server 实例）：加 PHOC_ROUTER_HTTP_TEST=1
```

覆盖：escalate 门低/高置信样本各若干条（本机 CPU 实测标定 conf）、多题聚合、
τ 覆盖、6 工具 ×4 场景路由、确定性（两次运行逐位一致）、参数校验（k>10/重名/
空表）、HTTP 与引擎模式对拍（决策一致、conf 偏差 ≤0.005）。

## License

Apache-2.0（见 LICENSE）。模型权重另见其发布的模型卡。
