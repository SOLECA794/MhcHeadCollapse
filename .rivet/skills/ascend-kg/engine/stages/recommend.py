"""③推荐（run.py 链路遗留 stage，已随 P1-1 退役归档——证据资产迁 engine/tools/priors.py）：
optix context → recommend_params.py → 证据映射 → 选本轮单参数变更。"""
import json
import subprocess
from pathlib import Path

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
    ctx_path = Path("/tmp/optix_ctx.json")
    ctx_path.write_text(json.dumps(_build_optix_context(cfg)), encoding="utf-8")
    try:
        r = subprocess.run(
            ["python", str(optix_script), "--context", str(ctx_path)],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except Exception:
        return None


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


def run(flow, ctx):
    """推荐本轮单参数候选，产出写 ctx["cand"]。
    candidate=None 表示无候选（后续 apply 跳过，仅保留现状）。"""
    cfg = ctx["cfg"]
    state = ctx["state"]
    signals = ctx.get("signals")
    current = state.get("current_params", dict(flow.params.defaults))
    tried = set()
    for r in state.get("rounds", []):
        tried.update(r.get("changed", []))
    notes = []

    # 1) optix 推荐
    optix = _run_optix(cfg, flow.paths.optix_script)
    optix_map = {}
    if optix and optix.get("status") == "ok":
        for rec in optix.get("recommendations", []):
            key = OPTIX_PARAM_MAP.get(rec.get("name"))
            if key:
                optix_map[key] = rec
    else:
        notes.append("optix need_more_info/失败 → 本地启发式表")

    # 2) 证据映射打分
    score = _evidence_score(signals)
    if signals:
        notes.append(f"profiling 信号: MFU={signals.get('mfu_pct')}%, "
                     f"wait_top={signals.get('wait_top_total_ms')}ms, "
                     f"prefill={signals.get('prefill_matmul_ms')}ms/decode={signals.get('decode_matmul_ms')}ms")

    # 3) 候选池：optix 推荐 ∩ 参数域 ∩ 未试过 ∩ 未屏蔽 ∩ 非 no-op
    pool = []
    for key in list(optix_map) + [h["param"] for h in HEURISTIC]:
        if key in tried or key in flow.params.blocked:
            continue
        domain = flow.params.domain.get(key)
        if not domain or domain.get("locked"):
            continue
        # 值：optix 优先，否则启发式
        val = None
        reason = ""
        src = "optix"
        if key in optix_map:
            val = optix_map[key].get("value")
            reason = optix_map[key].get("reason", "")
        else:
            val = next((h["value"] for h in HEURISTIC if h["param"] == key), None)
            reason = next((h["reason"] for h in HEURISTIC if h["param"] == key), "")
            src = "heuristic"
        if val is None or val == "":
            continue
        # no-op 守卫：候选值 == 当前值 → 无实际变更，跳过（避免无意义重启）
        cur = current.get(key)
        try:
            if cur is not None and float(cur) == float(val):
                continue
        except (TypeError, ValueError):
            pass
        pool.append({"param": key, "value": val, "reason": reason,
                     "src": src, "prio": score.get(key, 9)})

    if not pool:
        ctx["cand"] = {"candidate": None, "new_params": current,
                       "rationale": "候选已穷尽或全被屏蔽/已试", "optix_notes": notes,
                       "kg_notes": [], "source": "none"}
        return

    # 4) 按优先级排序，选未达护栏的单参数
    pool.sort(key=lambda x: x["prio"])
    chosen = pool[0]
    new_params = dict(current)
    new_params[chosen["param"]] = chosen["value"]

    # 5) KG 纠偏已摘除（P1-1c/B5）：block-size 约束统一走 domain locked（case5 flow
    #    已是 {"locked": true, "fixed": 128}），kg_correct 的 KG 在线确认只是装饰性
    #    note。kg_notes 机制保留给 domain 节点（r0-domain 的 dropped 记录）。
    kg_notes = []

    rationale = (f"{chosen['param']}={chosen['value']} "
                 f"[{chosen['src']}] prio={chosen['prio']}: {chosen['reason']}")
    ctx["cand"] = {"candidate": chosen["param"], "new_params": new_params,
                   "rationale": rationale, "optix_notes": notes, "kg_notes": kg_notes,
                   "source": chosen["src"], "prio": chosen["prio"]}
