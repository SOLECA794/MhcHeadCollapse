"""目标函数标量化（Objective，阶段 A 核心）：单轮 metrics + 精度对拍 → 标量 value + feasible。

把 keep/rollback 的二元布尔升级为标量，供 Sampler/Pruner/best 消费（阶段 B/C 依赖）。
value 统一最小化；feasible=False 表示该点不可接受（护栏破坏 / 精度 B 类语义破坏 / 主指标缺失）。

精度 A/B 分类（问题④解法）：A 类数值漂移 feasible=True 但 value 加小惩罚；B 类语义破坏
feasible=False 硬否决。分类在 tools/accuracy.py 的 check_accuracy 产出（detail.cls）。

依赖方向：本模块 import tools/targeting.primary_key（单向），decide 反向 import 本模块
（compute_objective + noise_floor）——本模块永远不 import decide，否则成环。
"""
from engine.tools.targeting import primary_key


def noise_floor(flow, metrics: dict | None, baseline: dict | None, key: str) -> float:
    """噪声地板 = noise_k × min(基线相对噪声, 本轮相对噪声)（P0-A，2026-08-28 case7 后改 min）。

    相对噪声来自 run_benchmark 的 stats（MAD/median）。池化口径会混入 rep 间系统性
    漂移（基线会话冷启动 rep 等）：case7 基线 TTFT 池化 17.9% vs 轮内 2-6%，原 max()
    口径把地板永久锁在最差一次测量上——所有 ≤5% 真实收益被判噪声、影子记账零触发、
    r11 吞吐 +8.0% 被噪声大于阈值的 TTFT guard 随机拦截。min() 取测量更稳的一侧作为
    噪声可信尺度；跨时段系统性漂移由中途锚点轮（P1-B 重锚）+ 终局 drift_check 兜底，
    单轮幸运采样由 borderline 重测（含 unstable 轮确认）兜底。
    任一侧 stats 缺失（旧 state resume / 非 bench 产物）返回 0 —— hyst_eff 退回
    hyst，行为零回归。flow.perf.noise_k 缺省 1.5（单侧 ~87% 置信）；
    disable_noise_floor=true 一键关闭。

    定义在 objective（非 decide）：compute_objective 的 guard 展宽也消费它，
    而 decide 反向 import objective——放 decide 会成环。
    """
    if getattr(flow.perf, "disable_noise_floor", False):
        return 0.0
    k = float(getattr(flow.perf, "noise_k", 1.5) or 0)
    if k <= 0:
        return 0.0
    # 指标各自的相对噪声键（#6 修：ttot 此前落到 ttft 噪声——TTOT 守门
    # 的地板一直在用 TTFT 的离散度）；其余键（p95/p99/tpot/tps 目前无独立噪声键）
    # 沿用 ttft 噪声作代理。
    noise_key = {"thr_tok_s": "thr_rel_noise",
                 "ttot_ms": "ttot_rel_noise"}.get(key, "ttft_rel_noise")
    noises = []
    for src in (metrics, baseline):
        v = ((src or {}).get("stats") or {}).get(noise_key)
        if isinstance(v, (int, float)) and v > 0:
            noises.append(float(v))
    return k * min(noises) if noises else 0.0


def compute_objective(flow, metrics, baseline, target=None,
                      accuracy_match=None, accuracy_detail=None) -> dict:
    """合成单轮标量 objective。

    返回：
      {
        "value": float | None,   # 越小越优；None = 无主指标可评估（护栏破坏/精度 B/缺指标）
        "feasible": bool,        # False = 该点不可接受，调用方强制回滚
        "kind": "ok" | "guard_violated" | "accuracy_B" | "accuracy_A" | "no_metric",
        "components": {...}      # 主指标/改善百分比/护栏/精度 分项，供决策上下文
      }
    """
    key = primary_key(flow, target)
    up = key == "thr_tok_s"

    base = (baseline or {}).get(key)
    cur = (metrics or {}).get(key)
    if base is None or cur is None or base <= 0:
        return {"value": None, "feasible": False, "kind": "no_metric",
                "components": {"reason": f"{key} 缺失/非法（cur={cur}, base={base}）"}}

    improved_pct = (cur - base) / base * 100 if up else (base - cur) / base * 100

    # 护栏（退化容限，恒对比 baseline；与 keep_verdict 同守，这里产出 feasible）。
    # 噪声展宽（P0-A，case7 r11 教训）：守门指标自身的噪声地板超过配置容限时，
    # 容限按地板展宽（max(room, 1+floor)）——阈值小于噪声的守门是随机拦截
    # （r11 吞吐 +8.0% 被 TTFT +11.4% 拦，而同配置两次测量 TTFT 摆动 10pt+）。
    # 展宽只放宽退化上限：catastrophic 退化（超 max(room, 1+floor)）仍被拦；
    # 展宽发生时 components 记 guard_widened 供审计。
    # 业务画像放宽（--profile，装配期 cli._apply_guard_policy 已把放宽后的 room
    # 烘进 flow.perf）：batch=灾难档/balanced=显式预算。两层放宽正交——统计展宽
    # 管「退化是否超出噪声可解释」，业务放宽管「代价是否商业可接受」；
    # perf.guard_policy 存在时 components 记 guard_policy，与 guard_widened 分开审计。
    if up:
        room = getattr(flow.perf, "ttft_room", 1.10)
        gkey = "ttft_ms"
        guard = "TTFT 退化超容限"
    else:
        room = getattr(flow.perf, "ttot_room", 1.10)
        gkey = "ttot_ms"
        guard = "TTOT 退化超容限"
    gfloor = noise_floor(flow, metrics, baseline, gkey)
    limit_room = max(room, 1.0 + gfloor)
    guard_ok = (metrics.get(gkey) or 0) <= (baseline.get(gkey) or 0) * limit_room
    if not guard_ok:
        comp = {"improved_pct": improved_pct, "guard": guard}
        if gfloor > 0 and (1.0 + gfloor) > room:
            comp["guard_widened"] = True  # 已按噪声展宽仍拦 = 超出噪声可解释的退化
            comp["guard_limit_room"] = round(limit_room, 4)
        policy = getattr(flow.perf, "guard_policy", None)
        if policy:
            comp["guard_policy"] = policy  # 画像放宽后仍拦 = 超灾难档/预算的退化
        return {"value": None, "feasible": False, "kind": "guard_violated",
                "components": comp}

    # p95 尾延迟（ttft 目标可选，同 keep_verdict L1 容限）
    p95_room = getattr(flow.perf, "p95_room", None)
    if (not up and p95_room and metrics.get("ttft_p95") and baseline.get("ttft_p95")
            and metrics["ttft_p95"] > baseline["ttft_p95"] * p95_room):
        return {"value": None, "feasible": False, "kind": "guard_violated",
                "components": {"improved_pct": improved_pct, "guard": "p95 超容限"}}

    # 精度分类（A/B）：B 类硬否决；A 类放行但加惩罚
    kind = "ok"
    acc_comp = None
    if accuracy_match is False:
        cls = (accuracy_detail or {}).get("cls")
        if cls == "B":
            return {"value": None, "feasible": False, "kind": "accuracy_B",
                    "components": {"improved_pct": improved_pct,
                                   "accuracy": accuracy_detail}}
        kind = "accuracy_A"
        acc_comp = accuracy_detail

    # value 构造（统一最小化）：ttft → ttft_ms；throughput → -thr_tok_s
    value = -cur if up else cur
    # A 类数值漂移惩罚：value 向变差方向移动 eps（ttft 增大 / throughput 更接近 0）。
    # 用 value + |value|*eps 保证统一最小化语义下两方向都「变差」。
    if kind == "accuracy_A":
        eps = float(getattr(flow.perf, "accuracy_penalty", 0.005))
        value = value + abs(value) * eps

    comp = {"improved_pct": improved_pct, "primary": {key: cur}, "accuracy": acc_comp}
    # 放行但依赖展宽（退化超配置容限、仍在噪声可解释范围）→ 记审计标记：
    # case7 r11 的镜像场景——keep 成立依赖了展宽，复盘时必须可分辨。
    if (gfloor > 0 and (1.0 + gfloor) > room
            and (metrics.get(gkey) or 0) > (baseline.get(gkey) or 0) * room):
        comp["guard_widened"] = True
        comp["guard_limit_room"] = round(limit_room, 4)
    # 画像放宽生效时无条件透传：keep/拦在哪个业务契约下成立，复盘必须可分辨
    policy = getattr(flow.perf, "guard_policy", None)
    if policy:
        comp["guard_policy"] = policy
    return {"value": value, "feasible": True, "kind": kind, "components": comp}
