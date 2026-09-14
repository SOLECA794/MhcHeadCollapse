"""Sampler：值空间探索（阶段 B 核心，确定性纯函数）。

职责：把 LLM 的「选维度」扩展成「选维度 + 值策略」，产出值序列供逐轮执行；
同时提供值级化 tried 集合与 backbone 准入的「补扫邻域」判定。

分工（plan §1）：LLM 只负责「选哪个参数 + 给中心值」，值序列展开 / 补扫判定
在这里确定性完成——「LLM 选维度（智能）+ 值级粗网格/线搜索（确定性）」。
本模块不依赖 LLM、零副作用，供 agent_loop 决策上下文与 grouping 组合候选复用。
"""


def tried_values(state: dict) -> set:
    """值级化 tried 集合：{(param, value), ...}（plan §2.2，替代参数级 tried）。

    试过 max-num-seqs=128 不再排除试 160——值级去重是值空间探索的前提。
    action v2（P0-1）：组合轮 {"params": {...}} 额外记 frozenset 组合键——
    同一组合（参数集+取值全同）不重复试；组内单个 (param, value) 也照记，
    保持值级视图完整。
    """
    tried = set()
    for r in state.get("rounds") or []:
        a = r.get("action") or {}
        if isinstance(a.get("params"), dict) and a["params"]:
            tried.add(frozenset(a["params"].items()))
            for p, v in a["params"].items():
                if v is not None:
                    tried.add((p, v))
            continue
        p, v = a.get("param"), a.get("value")
        if p is not None and v is not None:
            tried.add((p, v))
    return tried


def value_sequence(param: str, domain: dict, center) -> dict:
    """LLM 给中心值 → 值序列（值策略）。返回 {"values": [...], "strategy": ...}。

    - 布尔（flag）：single，单点（2 态天然全覆盖）
    - 枚举（values）：二元枚举 → ab 对照；多值枚举 → single（LLM 逐个给值）
    - 连续（min/max/step）：line_search，{中心-1step, 中心, 中心+1step}（clip 到域）
    """
    d = domain.get(param, {})
    if d.get("flag"):
        return {"values": [center], "strategy": "single"}
    if d.get("values"):
        vals = list(d["values"])
        if len(vals) == 2 and center in vals:
            other = vals[1] if vals[0] == center else vals[0]
            return {"values": [center, other], "strategy": "ab"}
        return {"values": [center], "strategy": "single"}
    step = d.get("step")
    if not step:
        return {"values": [center], "strategy": "single"}
    seq = []
    for v in (center - step, center, center + step):
        if (d.get("min") is None or v >= d["min"]) and (d.get("max") is None or v <= d["max"]):
            if v not in seq:
                seq.append(v)
    return {"values": seq, "strategy": "line_search"}


def pending_neighborhood(state: dict, flow, target=None) -> list:
    """找出「已 keep 但未确认局部最优」的参数及待补扫值（backbone 准入条件，plan §2.2）。

    判定：参数最新 keep 值 center 的邻域 {center±step} 未被试过 → 需补扫。
    只对有收益（keep）参数补扫，不对全部 21 个。布尔/枚举天然全覆盖、锁定参数跳过。
    返回 [{"param": p, "values": [待补扫值...], "reason": ...}, ...]。
    """
    domain = flow.params.domain or {}
    best_kept = {}
    for r in state.get("rounds") or []:
        a = r.get("action") or {}
        if isinstance(a.get("params"), dict) and a["params"]:
            if r.get("kept"):  # action v2：组合轮 keep 按本次生效值记账
                best_kept.update(a["params"])
            continue
        p = a.get("param")
        if p and r.get("kept"):
            best_kept[p] = a.get("value")
    tried = tried_values(state)
    out = []
    for p, center in best_kept.items():
        d = domain.get(p, {})
        if d.get("flag") or d.get("values") or d.get("locked") or d.get("compose"):
            continue  # 布尔/枚举天然全覆盖；锁定/派生不可补扫
        step = d.get("step")
        if not step:
            continue
        neighbors = []
        for cand in (center - step, center + step):
            if (d.get("min") is None or cand >= d["min"]) and (d.get("max") is None or cand <= d["max"]):
                if (p, cand) not in tried:
                    neighbors.append(cand)
        if neighbors:
            out.append({"param": p, "values": neighbors,
                        "reason": "keep 未确认局部最优，补扫邻域（backbone 准入）"})
    return out
