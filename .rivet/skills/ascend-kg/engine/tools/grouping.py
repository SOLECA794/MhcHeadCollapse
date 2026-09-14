"""分组降维 + 组合（阶段 B 组合层，unmerged-notes 核心方案）。

partially separable decomposition + Cooperative Coevolution：
把 21 维组合搜索压到 2-3 维交互子集。交互检测四层（①先验→②收窄→③gap探针→④leave-one-out），
产出分组：backbone（无交互，固定单参数最优值）+ 交互子集（圈出，重新组合搜索）。

确定性纯函数，消费 state["rounds"] 历史产出「组合计划」，供 agent_loop 值级探索收敛后
调用；真机执行组合搜索（多参数 apply）复用现有 apply/verify 原语，不在本模块。
"""
import itertools

from engine.tools.sampler import value_sequence, pending_neighborhood


def kept_params(state: dict, flow, target=None) -> dict:
    """② 收窄：从 rounds 提取「已 keep 且有收益」的参数 → {param: 最新 keep 值}。

    交互只可能发生在「跑出收益的参数」里——没收益的参数不值得怀疑交互。
    这是交互检测成本从 C(21,2) 压到 k+1 的根本（notes §4 层②）。
    action v2（P0-1）：组合轮 keep 时把组内每个参数按本次生效值记账。
    注意：组合上下文的 keep ≠ 单参数各自最优——backbone 取值若来自组合轮，
    final_sweep 的「组合 vs 独立最优」对拍负责兜底揭示该差异。
    """
    out = {}
    for r in state.get("rounds") or []:
        a = r.get("action") or {}
        if isinstance(a.get("params"), dict) and a["params"]:
            if r.get("kept"):
                out.update(a["params"])
            continue
        p = a.get("param")
        if p and r.get("kept"):
            out[p] = a.get("value")
    return out


def group(flow, kept: dict, prior_interactions=None) -> dict:
    """分组：backbone + 交互子集。

    ① 先验（prior_interactions：KG 场景联动组，如 prefill 类/decode 类/调度类）∩ ② 收窄（kept）：
    - 交互子集 = kept ∩ 先验交互组（组内 ≥2 个 kept 参数才圈出）
    - backbone = kept − 交互子集（无交互，固定单参数最优值）

    无先验交互信息时保守：全部 kept 进 backbone，gap 探针（③）发现交互后再拆。
    """
    interactive = set()
    for grp in (prior_interactions or []):
        hit = [p for p in grp if p in kept]
        if len(hit) >= 2:
            interactive.update(hit)
    interactive = interactive & set(kept)
    backbone = {p: v for p, v in kept.items() if p not in interactive}
    return {"backbone": backbone, "interactive": sorted(interactive)}


def combination_candidates(flow, interactive: list, kept: dict) -> list:
    """交互子集 → 组合候选（笛卡尔积，含各自可取值域）。

    交互子集 ≤3 参数时全组合可穷举（27/9/4），比 PSO 可靠（notes §3 降维表）。
    每个候选 = {param: value}，叠加在 backbone 上。连续参数在 kept 最优值附近取 3 档。
    """
    if not interactive:
        return []
    domain = flow.params.domain or {}
    dims = []
    for p in interactive:
        d = domain.get(p, {})
        if d.get("values"):
            vals = list(d["values"])
        elif d.get("flag"):
            vals = [True, False]
        else:
            center = kept.get(p)
            if center is None:
                center = (d.get("min", 0) + d.get("max", 0)) / 2
            vals = value_sequence(p, domain, center)["values"]
        dims.append([(p, v) for v in vals])
    return [dict(combo) for combo in itertools.product(*dims)]


def final_sweep(backbone: dict, interactive_best: dict) -> dict:
    """终局 SWEEP：backbone + 交互最优组合 → 一次组合验证候选（notes §3 兜底）。

    直接回答核心问题「局部最优拼起来是否真最优」——组合 vs 独立最优的 gap。
    """
    combo = dict(backbone)
    combo.update(interactive_best)
    return combo


def plan_composition(state: dict, flow, target=None, prior_interactions=None) -> dict:
    """分组降维主编排：值级探索收敛后调用，产出分组 + 组合计划。

    返回 {
      "backbone":          {param: 固定最优值},
      "interactive":       [交互子集参数],
      "combinations":      [交互子集组合候选...]（叠加在 backbone 上真机测）,
      "final_sweep":       {backbone + 交互最优 的终局验证候选},
      "pending_neighborhood": [待补扫邻域的参数]（backbone 准入，前提 1 未满足项）,
    }
    """
    kept = kept_params(state, flow, target)
    grp = group(flow, kept, prior_interactions)
    interactive = grp["interactive"]
    combos = combination_candidates(flow, interactive, kept)
    # 交互最优：无真机组合结果时退化为 kept 值（组合搜索执行后再回填 argmin）
    interactive_best = {p: kept[p] for p in interactive}
    return {
        "backbone": grp["backbone"],
        "interactive": interactive,
        "combinations": combos,
        "final_sweep": final_sweep(grp["backbone"], interactive_best),
        "pending_neighborhood": pending_neighborhood(state, flow, target),
    }
