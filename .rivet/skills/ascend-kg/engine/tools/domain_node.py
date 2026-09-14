"""domain_node —— KG 构建动作空间的循环内必走节点（plan §7.1 的 in-loop 化）。

定位（2026-08-27 case5 复盘后改造）：KG 检索 → 暂停点（LLM 提取）→ 与手写默认
domain 合并。此前 setup_domain.py 是引擎外的可选手动前置步骤，落地后从未被任何
run 走过（被「flow JSON 已有手写 domain」遮蔽）；本模块把检索逻辑迁进来，由
agent_loop 在 round 0 前作为必走节点调用。

合并规则（用户拍板「完全以 KG 为准 + 结合默认动作空间」的字段级语义）：
  merged = {**手写, **{p: {**手写.get(p,{}), **kg[p]} for p in kg}}
  —— KG 显式给出的字段覆盖同名参数；KG 未给的字段从手写继承（执行元数据
     kind/compose/path 兜底，防 LLM 提取漏写导致 apply 渲染坏）；
     KG 新增参数原样采用（必须过 validate_kg_domain 结构校验）。

KG 故障（不可用/无黄金命中）不硬失败也不静默降级：照常走暂停点，把 kg_status
回传主会话 agent 决策（use_default 降级 / terminate / 其他）。

纯 stdlib。零 LLM 依赖——「提取」由主会话 agent 在暂停点完成。
"""
import json
from pathlib import Path

from engine import kg

ENGINE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ENGINE_DIR / "domain_cache"

# flow.backend → 框架关键词（用于构造 KG 检索 query）。
# 注：sglang flow 的 backend 是 "sglang"（历史 key "sglang-npu" 曾不匹配导致报错）。
FRAMEWORK = {
    "vllm-ascend": "vLLM-Ascend",
    "mindie": "MindIE",
    "sglang": "SGLang",
    "sglang-npu": "SGLang",
}

QUERIES = [
    "{fw} 参数调优指南",
    "{fw} 服务化自动寻优 config.toml",
]

SCORE_FLOOR = 0.83

# 暂停点回传 schema（agent_result.json 的 result 字段）
DOMAIN_GOAL = (
    "构建本轮调优的动作空间：从 context.sources 的 KG 参数调优资料中提取结构化 "
    "domain（参数名/kind/risk/取值域），与 context.default_domain（手写默认动作空间）"
    "合并——同名参数以你的提取为准（未提到的字段从默认继承），默认没有的参数可新增。"
    "KG 不可用（kg_status=unavailable）或材料不足时，可回传 use_default 降级用默认"
    "动作空间，或 terminate 终止本次调优。只把最终决策写进 agent_result.json 的 "
    "result 字段。"
)

DOMAIN_SCHEMA = {
    "action": "ok | use_default | terminate",
    "domain": ("{<param>: {kind/risk/min/max/step/values/flag/locked/compose/path}} "
               "—— action=ok 时必填；与 flow.params.domain 同构"),
    "interaction_priors": ("list，可选（P0-1）：[{params: [参数名...], evidence: 'source 摘录'}] "
                           "—— 指南中明确表述的参数联动关系（如「A 与 B 同开收益叠加」）；"
                           "仅提取有据可依的组，无明确表述就不提取"),
    "reason": "string: 一句话理由（含来源依据，如哪些 source 支撑了新增/覆盖）",
}

# 字段词表 + 引擎消费语义（进暂停任务 context.rules，供提取时遵循）
DOMAIN_RULES = [
    "domain 条目字段词表：kind（cli/env/sub/json，缺省=cli）、risk（low/mid/high）、"
    "min/max/step（连续参数）、values（枚举列表）、flag（布尔开关）、locked（锁定）、"
    "compose（派生参数的 {json_key: sub_param} 映射）、path（json 参数的 config 键路径列表）",
    "引擎消费语义（决定参数怎么生效，提取时务必给对）：kind=env → 渲染成 export 行；"
    "kind=cli → 渲染成 --flag；kind=sub → 不独立渲染，只作 compose 父参数的输入；"
    "kind=json → patch 到 config.json 的 path 位置（path 必填，如 "
    "['BackendConfig','ScheduleConfig','xx']）",
    "新参数 kind 判定：指南写 `--xx` → cli；全大写 ENV_VAR → env；写成配置键/嵌套 "
    "toml/json 键 → json（必须给 path）",
    "取值域优先用指南的推荐值/取值范围；只给单点推荐值时收窄 min/max 或用 values 枚举；"
    "无据可依时不要自造宽域",
    "risk 标注保守：改 dtype/图模式等可能破坏精度的标 high；未知新参数标 mid",
    "同名参数只写你要覆盖的字段即可（其余从默认继承）；整条不改的参数不必回传",
    "engine 校验会丢弃结构不合格条目（kind=json 缺 path、数值参数缺 min/max 等）并记录原因",
    "interaction_priors（可选）：只提取指南中明确表述的参数联动（≥2 个参数一组），"
    "每组必须带 evidence（source 摘录）；组内参数必须是你提取/默认 domain 里存在的名字；"
    "无明确联动表述就不回传此字段——宁缺勿滥（漏掉的由引擎的 gap 检测兜底）",
]


def search_materials(flow) -> dict:
    """确定性 KG 检索：按 flow.backend 映射框架词，两条 query 取黄金命中全文。

    返回 {"framework", "kg_status", "hits", "sources"}：
      kg_status = "ok" | "unavailable"（无 Key/health 非 200，hits/sources 为空，
      照常交暂停点让 agent 决策）
    """
    fw = FRAMEWORK.get(getattr(flow, "backend", ""))
    if not fw:
        return {"framework": None, "kg_status": "unmapped-backend",
                "hits": [], "sources": []}

    if not kg.kg_available():
        return {"framework": fw, "kg_status": "unavailable",
                "hits": [], "sources": []}

    hits, sources, seen = [], [], set()
    for q in QUERIES:
        query = q.format(fw=fw)
        for x in kg.kg_search(query, top_k=5):
            score = float(x.get("score", 0) or 0)
            nid = x.get("id") or x.get("node_id")
            if not nid or nid in seen:
                continue
            seen.add(nid)
            hits.append({"id": nid, "score": score,
                         "source_file": x.get("source_file", ""), "query": query})
            if score > SCORE_FLOOR:
                text = kg.kg_source(nid, max_length=5000)
                if text:
                    sources.append({"id": nid, "score": score, "text": text})

    return {"framework": fw,
            "kg_status": "no-golden-hits" if not sources else "ok",
            "hits": hits, "sources": sources}


def dump_raw(flow_id: str, backend: str, materials: dict) -> Path:
    """落盘原始检索材料 domain_cache/{flow_id}_raw.json（记录 + 暂停点恢复依据）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = CACHE_DIR / f"{flow_id}_raw.json"
    raw = {"flow_id": flow_id, "backend": backend, "framework": materials.get("framework"),
           "kg_status": materials.get("kg_status"),
           "hits": materials.get("hits", []), "sources": materials.get("sources", [])}
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw_path


def validate_kg_domain(kg_domain, handwritten: dict) -> tuple:
    """结构校验 LLM 提取的 domain，防炸下游消费方（渲染/within_domain/sampler）。

    返回 (valid, dropped)：valid = 通过校验的 {param: entry}；
    dropped = {param: 不合格原因}（条目级丢弃，不整轮作废）。
    """
    if not isinstance(kg_domain, dict) or not kg_domain:
        return {}, {"_all": "domain 非法（非 dict 或为空）"}
    valid, dropped = {}, {}
    for name, entry in kg_domain.items():
        if not isinstance(name, str) or not name or isinstance(name, bool):
            dropped[repr(name)] = "参数名非法"
            continue
        if not isinstance(entry, dict):
            dropped[name] = "条目非 dict"
            continue
        d = entry
        kind = d.get("kind")
        if kind not in (None, "cli", "env", "sub", "json"):
            dropped[name] = f"kind={kind!r} 非法（cli/env/sub/json）"
            continue
        if kind == "json":
            path = d.get("path")
            # 手写同名参数有 path 时可继承，此处只校验 KG 显式给的 path
            hw_path = (handwritten.get(name) or {}).get("path")
            if path is None and hw_path is None:
                dropped[name] = "kind=json 缺 path（config 键路径必填）"
                continue
            if path is not None and not (isinstance(path, list) and path
                                         and all(isinstance(k, str) and k for k in path)):
                dropped[name] = "path 非法（须非空 str 列表）"
                continue
        if d.get("values") is not None and not (isinstance(d["values"], list) and d["values"]):
            dropped[name] = "values 非法（非空列表）"
            continue
        if d.get("risk") not in (None, "low", "mid", "high"):
            dropped[name] = f"risk={d['risk']!r} 非法（low/mid/high）"
            continue
        numeric_needed = not (d.get("flag") or d.get("values")
                              or d.get("locked") or d.get("compose")
                              or kind in ("sub", "json"))
        if numeric_needed and "min" not in d and "max" not in d:
            hw = handwritten.get(name) or {}
            if "min" not in hw and "max" not in hw and not (hw.get("flag") or hw.get("values")):
                dropped[name] = "数值参数缺 min/max（within_domain 直接下标访问，缺了会炸）"
                continue
        if "min" in d and "max" in d:
            try:
                if float(d["min"]) > float(d["max"]):
                    dropped[name] = "min > max"
                    continue
            except (TypeError, ValueError):
                dropped[name] = "min/max 非数值"
                continue
        valid[name] = d
    return valid, dropped


def merge_domain(handwritten: dict, valid_kg: dict) -> dict:
    """字段级合并：手写为底，KG 条目覆盖同名字段、缺失字段从手写继承。"""
    merged = dict(handwritten or {})
    for name, entry in (valid_kg or {}).items():
        merged[name] = {**(merged.get(name) or {}), **entry}
    return merged


def validate_interactions(kg_inter, merged_domain: dict) -> tuple:
    """结构校验 LLM 提取的交互先验组（P0-1 层 2 数据源）。

    输入形态：[{params: [参数名...], evidence: str}, ...]（也容忍 [p1, p2] 裸列表，
    无 evidence 的裸列表直接丢弃——先验必须可溯源，防 LLM 编造联动）。
    校验：params 非空列表、≥2 个、全部在 merged_domain 中（合并后的动作空间）、
    组内去重、evidence 非空 str。
    返回 (valid_groups, dropped)：valid = [{"params": [...]}, ...]（归一形态）。
    """
    valid, dropped = [], {}
    if not kg_inter:
        return valid, dropped
    if not isinstance(kg_inter, list):
        return valid, {"_all": "interaction_priors 非法（非 list）"}
    for i, g in enumerate(kg_inter):
        tag = f"[{i}]"
        if isinstance(g, list):
            dropped[tag] = "裸列表形态缺 evidence，丢弃（先验必须可溯源）"
            continue
        if not isinstance(g, dict):
            dropped[tag] = "条目非 dict"
            continue
        params, evidence = g.get("params"), g.get("evidence")
        if not isinstance(params, list) or len(params) < 2:
            dropped[tag] = f"params 非法（须 ≥2 个参数的列表）: {params!r}"
            continue
        params = [p for p in params if isinstance(p, str)]
        if len(set(params)) != len(params) or len(params) < 2:
            dropped[tag] = "组内参数重复/清洗后不足 2 个"
            continue
        unknown = [p for p in params if p not in (merged_domain or {})]
        if unknown:
            dropped[tag] = f"组内参数不在合并后 domain: {unknown}"
            continue
        if not isinstance(evidence, str) or not evidence.strip():
            dropped[tag] = "缺 evidence（source 摘录必填）"
            continue
        valid.append({"params": params, "evidence": evidence.strip()})
    return valid, dropped


def domain_task_context(flow, materials: dict, rules=None) -> dict:
    """构造暂停点任务 context（sources 截断防任务文件膨胀）。"""
    sources = [{"id": s["id"], "score": s["score"], "text": s["text"][:4000]}
               for s in materials.get("sources", [])]
    return {
        "flow_id": flow.flow_id,
        "backend": getattr(flow, "backend", None),
        "framework": materials.get("framework"),
        "kg_status": materials.get("kg_status"),
        "hits": materials.get("hits", []),
        "sources": sources,
        "default_domain": flow.params.domain or {},
        "rules": rules if rules is not None else list(DOMAIN_RULES),
    }


def record_domain(flow_id: str, domain: dict, meta: dict) -> Path:
    """落盘 domain_cache/{flow_id}.json —— 本次 run 最终动作空间的记录（非覆盖源）。

    语义变更（2026-08-27）：registry.load_flow 不再读此文件覆盖手写 domain；
    必走节点每次 fresh run 重新检索合并，此文件仅记录「上次 run 用了什么」。
    resume 回装（agent_loop）是唯一消费方：跨暂停续跑时把 KG 覆盖字段
    重新叠加回新加载的手写 domain（见 load_cached_domain）。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{flow_id}.json"
    path.write_text(json.dumps({"flow_id": flow_id, "domain": domain, "meta": meta},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_cached_domain(flow_id: str) -> dict | None:
    """读 domain_cache/{flow_id}.json 的 domain 字段；缺失/损坏/非 dict 返回 None。"""
    path = CACHE_DIR / f"{flow_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    domain = data.get("domain") if isinstance(data, dict) else None
    return domain if isinstance(domain, dict) and domain else None
