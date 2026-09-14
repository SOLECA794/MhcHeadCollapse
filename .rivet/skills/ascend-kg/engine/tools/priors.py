"""Priors：optix 先验提取（P1-1b，从 stages/recommend.py 迁出的证据产出）。

定位：recommend 的「选参逻辑」已随 run.py 退役，但其**证据产出**（optix 推荐 +
profiling 信号映射）对 LLM 决策有先验价值——本模块把它做成纯产出函数，注入
agent_loop 的 decide context（priors 字段）。

边界（plan §3.2）：
- 先验只影响 LLM，不影响校验：回传仍走 _parse_decision 全量域校验；
  先验值本身过 within_domain，不合格的直接不展示；
- optix 不可用 → 降级 EVIDENCE_MAP 启发式 → 再降级空列表；
- 不滤 tried（历史已试参数在 history 可见，context 中明示「参考先验非指令」，
  保留 LLM 的探索自由度）；no-op（先验值 == 当前值）跳过。
"""
import json
import subprocess
from pathlib import Path

from engine.state import within_domain

# optix 推荐名 → 本 orchestrator 参数 key
OPTIX_PARAM_MAP = {
    "MAX_MODEL_LEN": "max-model-len",
    "MAX_NUM_SEQS": "max-num-seqs",
    "MAX_NUM_BATCHED_TOKENS": "max-num-batched-tokens",
    "GPU_MEMORY_UTILIZATION": "gpu-memory-utilization",
    "BLOCK_SIZE": "block-size",
    "ENABLE_PREFIX_CACHING": "enable-prefix-caching",
    "ENABLE_CHUNKED_PREFILL": "enable-chunked-prefill",
}

# 压测/并行度参数，不落 vllm serve 启动参数
BENCH_ONLY = {"CONCURRENCY", "REQUESTRATE", "TENSOR_PARALLEL_SIZE",
              "PIPELINE_PARALLEL_SIZE", "DATA_PARALLEL_SIZE"}

# 证据映射表：(信号条件, 候选参数, 理由, 优先级[数字小=优先])
EVIDENCE_MAP = [
    ("wait_top",   "max-num-batched-tokens", "下发/等待瓶颈: 对齐 batched tokens", 1),
    ("decode_dom", "gpu-memory-utilization", "decode 小 matmul 主导: 提高显存利用率", 2),
    ("prefill_dom", "max-num-partial-prefills", "prefill 主导: partial prefill 调优", 3),
    ("prefill_dom", "long-prefill-token-threshold", "prefill 主导: 长输入阈值", 3),
]

# 本地启发式表（optix 不可用时兜底）
HEURISTIC = [
    {"param": "max-num-batched-tokens", "value": 32768, "reason": "本地启发式: 对齐上限收窄调度"},
    {"param": "gpu-memory-utilization", "value": 0.92, "reason": "本地启发式: 提高显存利用"},
]

# optix 子进程结果缓存：context 只依赖 flow/model/target/workload（与轮次无关），
# 同一 run 内多轮 decide 复用一次子进程调用（120s timeout 的重操作不逐轮重跑）。
_OPTIX_CACHE: dict = {}


def _build_optix_context(cfg) -> dict:
    return {
        "engine": "vllm",
        "hardware": {
            "single_card_mem_gb": 64, "world_size": 1,
            "num_per_nodes": 1, "num_nodes": 1,
            "platform": "ascend",
            "max_num_batched_tokens_ceiling": 32768,
        },
        "model": {"config_path": f"{cfg.model}/config.json", "torch_dtype": "bfloat16"},
        "workload": {
            "input_len_avg": cfg.workload["input_len_avg"],
            "input_len_max": cfg.workload["input_len_max"],
            "output_len_avg": cfg.workload["output_len_avg"],
            "output_len_max": cfg.workload["output_len_max"],
        },
        "target": cfg.target,
    }


def _run_optix(cfg, optix_script) -> dict | None:
    """跑一次 optix 推荐子进程（按 cfg 关键字段缓存）。失败/超时返回 None。"""
    cache_key = json.dumps([cfg.flow_id, cfg.model, cfg.target, cfg.workload],
                           ensure_ascii=False, sort_keys=True)
    if cache_key in _OPTIX_CACHE:
        return _OPTIX_CACHE[cache_key]
    ctx_path = Path("/tmp/optix_ctx.json")
    ctx_path.write_text(json.dumps(_build_optix_context(cfg)), encoding="utf-8")
    out = None
    try:
        r = subprocess.run(
            ["python", str(optix_script), "--context", str(ctx_path)],
            capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            out = json.loads(r.stdout)
    except Exception:
        out = None
    _OPTIX_CACHE[cache_key] = out
    return out


def _evidence_score(signals: dict) -> dict:
    """信号 → 参数字典优先级分数（数字小优先；无信号给默认分 9）。"""
    score = {}
    if not signals:
        return score
    prefill_ms = signals.get("prefill_matmul_ms", 0)
    decode_ms = signals.get("decode_matmul_ms", 0)
    for cond, param, reason, prio in EVIDENCE_MAP:
        hit = False
        if cond == "wait_top" and signals.get("wait_top_total_ms", 0) > 100:
            hit = True
        elif cond == "decode_dom" and decode_ms > prefill_ms:
            hit = True
        elif cond == "prefill_dom" and prefill_ms > decode_ms:
            hit = True
        if hit:
            score[param] = min(score.get(param, 9), prio)
    return score


def build_priors(flow, cfg, state, signals) -> tuple:
    """产出 decide context 的 priors 字段。返回 (priors, notes)。

    priors = [{"param", "value", "reason", "src": "optix|heuristic", "prio"}]，
    全部经过 domain 可用性（非 blocked/locked、有 domain 条目）+ 值域校验 +
    no-op 过滤；notes = 降级链说明（optix 失败/信号摘要），供日志审计。
    """
    domain = flow.params.domain or {}
    current = state.get("current_params") or dict(getattr(flow.params, "defaults", None) or {})
    notes = []

    optix = _run_optix(cfg, getattr(flow.paths, "optix_script", None)) \
        if getattr(flow.paths, "optix_script", None) else None
    optix_map = {}
    if optix and optix.get("status") == "ok":
        for rec in optix.get("recommendations", []):
            key = OPTIX_PARAM_MAP.get(rec.get("name"))
            if key:
                optix_map[key] = rec
    else:
        notes.append("optix need_more_info/失败 → 本地启发式表")

    score = _evidence_score(signals)
    if signals:
        notes.append(f"profiling 信号: MFU={signals.get('mfu_pct')}%, "
                     f"wait_top={signals.get('wait_top_total_ms')}ms, "
                     f"prefill={signals.get('prefill_matmul_ms')}ms/decode={signals.get('decode_matmul_ms')}ms")

    priors = []
    for key in list(optix_map) + [h["param"] for h in HEURISTIC]:
        if key in (flow.params.blocked or []):
            continue
        # 域可用性：无 domain 条目 / locked 的不展示（BENCH_ONLY 类 optix 名
        # 不会被 OPTIX_PARAM_MAP 映射，天然到不了这里）
        d = domain.get(key)
        if not d or d.get("locked"):
            continue
        if key in optix_map:
            val, reason, src = optix_map[key].get("value"), \
                optix_map[key].get("reason", ""), "optix"
        else:
            h = next((h for h in HEURISTIC if h["param"] == key), None)
            val, reason, src = (h["value"], h["reason"], "heuristic") if h \
                else (None, "", "heuristic")
        if val is None or val == "":
            continue
        # 先验值本身过域校验：不合格的不展示（LLM 不会被诱导回传非法值）
        if not within_domain(key, val, domain):
            notes.append(f"先验 {key}={val} 越域，不展示")
            continue
        cur = current.get(key)
        try:
            if cur is not None and float(cur) == float(val):
                continue  # no-op：先验值 == 当前运行值
        except (TypeError, ValueError):
            pass
        priors.append({"param": key, "value": val, "reason": reason,
                       "src": src, "prio": score.get(key, 9)})

    priors.sort(key=lambda x: (x["prio"], x["param"]))
    return priors, notes
