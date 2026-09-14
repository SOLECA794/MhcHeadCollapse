"""确定性判定：单轮滞回 keep（本模块）+ 跨轮终止（should_terminate）。

职责划分（DESIGN §4.4 层1，确定性、不依赖 LLM、不可绕过）：
  - keep_verdict / is_kept  —— 单轮结果是否保留（滞回门槛 + 护栏）。
  - should_terminate        —— 跨轮是否终止（穷尽 / 平台期 / 不健康）。
  agent_loop 与 run.py 两条入口共用本模块，保证判定一致。

目标感知（flow.target，DESIGN 目标化改造）：主指标按优化目标选择——
  ttft       → ttft_ms（↓优），护栏 = TTOT 退化 ≤ ttot_room（可选 p95_room）
  throughput → thr_tok_s（↑优），护栏 = TTFT 退化 ≤ ttft_room
三个共享助手（primary_key/keep_verdict/best_candidate）同时供 agent_loop、
stages/decide.py、stages/orchestrate.py、run.py 使用，保证两条入口判定一致。
"""

import itertools
from dataclasses import dataclass
from typing import Optional

from engine.tools.targeting import _eff_target, primary_key
from engine.tools.objective import compute_objective, noise_floor


def decision_block(flow, target=None):
    """按生效目标取决策块：throughput 取 decision_throughput（吞吐 goal/rules），
    否则取 decision（低时延取向）。供 should_terminate 与 agent_loop 共用，
    避免两处各写一份「按 target 选块」逻辑导致分叉。"""
    if _eff_target(flow, target) == "throughput":
        dt = getattr(flow, "decision_throughput", None)
        if dt is not None and getattr(dt, "goal", None):
            return dt
    return flow.decision


def keep_verdict(flow, metrics: dict | None, baseline: dict | None, target=None,
                 best: dict | None = None) -> tuple:
    """目标感知的单轮滞回判定。返回 (kept, improved_pct, rationale)。"""
    kept, _pct, why, _det = _keep_verdict_impl(flow, metrics, baseline, target, best)
    return kept, _pct, why


def keep_verdict_detail(flow, metrics: dict | None, baseline: dict | None, target=None,
                        best: dict | None = None) -> dict:
    """keep_verdict 的结构化版本（P0-2）：额外透出 hyst_eff / borderline。

    borderline = 改善过了配置门槛（hyst）但被噪声地板（hyst_eff）拦下的临界轮
    ——信号方向对但量级不足，调用方（try_action）可重测一次再终判。
    """
    kept, pct, why, det = _keep_verdict_impl(flow, metrics, baseline, target, best)
    det.update({"kept": kept, "improved_pct": pct, "rationale": why})
    return det


# 已搬迁至 objective.noise_floor（P0-A：compute_objective 的 guard 展宽也要消费，
# 放本模块会 objective→decide 成环）。别名保留给历史 import 路径（含本模块内部调用）。
_noise_floor = noise_floor


def _keep_verdict_impl(flow, metrics: dict | None, baseline: dict | None, target=None,
                       best: dict | None = None) -> tuple:
    """判定实现（keep_verdict / keep_verdict_detail 共用，返回多一个 detail dict）。

    改善基准（复利水位线，7.3）：best（rolling best）优先，缺失回退 baseline。
    护栏判定：复用 compute_objective 的 feasible（恒对比 baseline，退化容限 + p95 容限
    + 噪声展宽），避免护栏逻辑在此处重复实现（阶段 A 偏差 1 修复）。精度不在此层
    处理——accuracy 的 A/B 在 agent_loop.try_action 单独判定，故这里 accuracy=None。
    改善门槛：flow.perf.watermark_eps 显式配置覆盖 hyst；噪声地板（P0-2，P0-A 改 min
    口径——取测量更稳一侧，防基线会话漂移永久锁死地板）在其上取 max。
    unstable 轮确认（P0-A 配套）：本轮噪声 > 2× 基线且 > 2% 时，过线改善也标
    borderline——min 口径放松假阴性防护后，幸运轮假 keep 由一次重测兜底。
    """
    if not metrics or not baseline:
        return False, 0.0, "metrics/baseline 缺失，判不 keep", {"borderline": False}
    key = primary_key(flow, target)
    up = key == "thr_tok_s"
    # 改善参考点：rolling best 优先（复利水位线），缺 key 回退 baseline
    ref = best if (best and best.get(key)) else baseline
    base = ref.get(key)
    cur = metrics.get(key)
    if base is None or cur is None or base <= 0:
        return False, 0.0, f"{key} 数据缺失/非法（cur={cur}, base={base}），判不 keep", \
            {"borderline": False}
    improved_pct = (cur - base) / base * 100 if up else (base - cur) / base * 100
    # 吞吐口径噪声大于首 token，独立门槛 thr_hyst（缺省回退 hyst，保持零配置可跑）
    hyst = getattr(flow.perf, "thr_hyst", None) if up else getattr(flow.perf, "hyst", 0.003)
    if hyst is None:
        hyst = getattr(flow.perf, "hyst", 0.003)
    # 复利水位线 ε（7.3）：显式配置覆盖 hyst；缺省 None → 保持原 hyst 零回归
    eps = getattr(flow.perf, "watermark_eps", None)
    if eps is not None:
        hyst = eps
    # 噪声地板（P0-2/P0-A）：有效门槛 = max(配置 hyst, k × min(基线相对噪声, 本轮相对噪声))。
    # 依据（case5 教训）：hyst 0.3% << 实测 ±2% 波动，无地板时大量轮次被噪声
    # 驱动假 keep/假 rollback。rationale 必须透出地板来源，保证判定可审计。
    floor = _noise_floor(flow, metrics, baseline, key)
    hyst_eff = max(hyst, floor)
    # unstable 轮（P0-A 配套）：本轮噪声显著高于基线（>2× 且 >2%）→ 测量不可信，
    # 过线改善也标 borderline 触发重测（case7 r12 轮内噪声 11% vs 基线 3.6% 的教训）。
    noise_key = {"thr_tok_s": "thr_rel_noise",
                      "ttot_ms": "ttot_rel_noise"}.get(key, "ttft_rel_noise")
    r_rel = ((metrics.get("stats") or {}).get(noise_key))
    b_rel = ((baseline.get("stats") or {}).get(noise_key))
    unstable = (isinstance(r_rel, (int, float)) and isinstance(b_rel, (int, float))
                and r_rel > 2 * b_rel and r_rel > 0.02)
    # 护栏 + 退化容限：复用 objective 的 feasible（恒对比 baseline），消除重复实现
    obj = compute_objective(flow, metrics, baseline, target,
                            accuracy_match=None, accuracy_detail=None)
    if not obj["feasible"]:
        guard = obj["components"].get("guard", "护栏")
        note = "（guard 已按噪声展宽仍拦）" if obj["components"].get("guard_widened") else ""
        return False, improved_pct, (f"{key} {base:.1f}->{cur:.1f} "
                                     f"({improved_pct:+.1f}%), {guard}{note}"), {"borderline": False}
    floor_note = (f"，噪声地板 {floor*100:.2f}%（>{hyst*100:g}% 配置门槛，按地板判）"
                  if floor > hyst else "")
    unstable_note = "，本轮噪声异常 → 重测确认" if unstable else ""
    detail = {"borderline": ((hyst * 100 <= improved_pct < hyst_eff * 100)
                             or (unstable and improved_pct >= hyst_eff * 100)),
              "hyst": hyst, "hyst_eff": hyst_eff, "noise_floor": floor,
              "unstable_round": unstable}
    if improved_pct >= hyst_eff * 100:
        return True, improved_pct, (f"{key} {base:.1f}->{cur:.1f} ({improved_pct:+.1f}%), "
                                    f"改善达 {hyst_eff*100:.2f}% 门槛{floor_note}{unstable_note}"), detail
    return False, improved_pct, (f"{key} {base:.1f}->{cur:.1f} ({improved_pct:+.1f}%), "
                                 f"未达 {hyst_eff*100:.2f}% 改善门槛{floor_note}"), detail


def is_kept(metrics: dict | None, baseline: dict | None, flow, target=None,
            best: dict | None = None) -> tuple:
    """单轮结果是否保留。返回 (kept, rationale)。metrics/baseline 缺失 → 不 keep。"""
    kept, _improved_pct, rationale = keep_verdict(flow, metrics, baseline, target, best)
    return kept, rationale


def pareto_verdict(flow, metrics: dict | None, baseline: dict | None,
                   main_ref, sec_ref, main_target=None) -> dict:
    """双轴 Pareto 判定（副目标通道，case8 档C教训）。

    场景：吞吐 run 里 TTFT 赢家（或反之）在副目标影子账记了账，但主目标持平
    → 永远过不了 keep_verdict（它只看主指标）→ 从未复测、从未进组合。本判定
    给这类参数一条独立通道：叠在当前主目标最优配置上复测，Pareto 严格占优
    （同主指标 + 更好副指标）才 keep。

    参照点必须是对称配置（「加/不加该参数」的两个配置），不是 r0 基线——
    main_ref/sec_ref 由调用方取最近 keep 轮实测（无 keep 即基线）：
      - 主指标 vs main_ref：退化 ≤ 主噪声地板（含噪声容差，floor=0 时要求完全不退）；
      - 副指标 vs sec_ref：改善 ≥ max(副目标 hyst, 副噪声地板)——收益超噪声才叫真改善。
    主目标水位线 state["best"] 不消费本判定（防副目标轮污染主目标记账）。
    """
    if not metrics or not baseline or not main_ref or not sec_ref:
        return {"ok": False, "main_pct": None, "sec_pct": None,
                "rationale": "metrics/baseline/参照值缺失，判不 keep"}
    key_main = primary_key(flow, main_target)
    other = "ttft" if key_main == "thr_tok_s" else "throughput"
    key_sec = primary_key(flow, other)
    cur_m, cur_s = metrics.get(key_main), metrics.get(key_sec)
    if cur_m is None or cur_s is None:
        return {"ok": False, "main_pct": None, "sec_pct": None,
                "rationale": f"主/副指标缺失（{key_main}={cur_m}, {key_sec}={cur_s}）"}
    up_m, up_s = key_main == "thr_tok_s", key_sec == "thr_tok_s"
    main_pct = (cur_m - main_ref) / main_ref * 100 if up_m else (main_ref - cur_m) / main_ref * 100
    sec_pct = (cur_s - sec_ref) / sec_ref * 100 if up_s else (sec_ref - cur_s) / sec_ref * 100
    floor_m = _noise_floor(flow, metrics, baseline, key_main)
    floor_s = _noise_floor(flow, metrics, baseline, key_sec)
    hyst_s = getattr(flow.perf, "thr_hyst", None) if up_s else getattr(flow.perf, "hyst", 0.003)
    eff_s = max(hyst_s if hyst_s is not None else 0.003, floor_s)
    ok = main_pct >= -floor_m * 100 and sec_pct >= eff_s * 100
    why = (f"主 {key_main} {main_ref:.1f}->{cur_m:.1f}（{main_pct:+.1f}%，容差 -{floor_m*100:.2f}%），"
           f"副 {key_sec} {sec_ref:.1f}->{cur_s:.1f}（{sec_pct:+.1f}%，门槛 {eff_s*100:.2f}%）")
    return {"ok": ok, "main_pct": main_pct, "sec_pct": sec_pct,
            "main_floor": floor_m, "sec_eff": eff_s,
            "rationale": f"Pareto 占优成立，{why}" if ok else f"Pareto 占优不成立，{why}"}


def best_candidate(state, flow, round_key: str, metrics: dict | None, target=None,
                   value=None) -> dict | None:
    """按 objective value 判更优，返回新的 best 记录 dict 或 None（未更优/指标缺失）。

    value 提供时（agent_loop）：best = argmin(value)（统一最小化，阶段 A 偏差 2 修复），
    让 A 类精度惩罚参与 best 排序；
    value 缺失（run.py 过渡期）：回退主指标 argmin（ttft ↓ / thr ↑），行为不变。

    best 记录 = {"round": rk, <key>: cur, "value": value}（value=None 时缺省 value 键）。"""
    key = primary_key(flow, target)
    cur = (metrics or {}).get(key)
    if cur is None:
        return None
    if value is not None:
        bv = (state.get("best") or {}).get("value")
        if bv is None or value < bv:
            return {"round": round_key, key: cur, "value": value}
        return None
    # 回退主指标 argmin（run.py 过渡期，无 objective value）
    bv = (state.get("best") or {}).get(key)
    up = key == "thr_tok_s"
    if bv is None or (cur > bv if up else cur < bv):
        return {"round": round_key, key: cur}
    return None


# ---------------- 跨轮终止判定（Pruner，确定性、不依赖 LLM）----------------


@dataclass
class TerminationReason:
    """should_terminate 的返回值。kind 见 should_terminate 文档。"""
    kind: str        # exhausted | plateau
    detail: str      # 人类可读原因


def _consec_no_improve(rounds: list) -> int:
    """连续无 keep 轮数（从最近一轮往回数，遇 keep 即断；精修轮透明跳过）。

    修复 orchestrate.py 旧实现的 double-count：调用方传入的 rounds 已含当前轮记录，
    故这里只做一次 takewhile 回溯，不再对「当前 decision」额外 +1。
    旧实现对同一轮先在回溯里数到、又在 `if decision != "keep": consec += 1` 再加，
    导致首轮 rollback 即 consec=2、keep 后紧跟 1 个 rollback 即停——均过早停机。

    refine 轮（②a 邻域补扫）不计入：补扫 rollback 是「确认原值已局部最优」的预期
    结果，不是探索失败——若计入会误触发平台期终止。粗筛/终局 sweep 轮（sweep 标记）
    同理不计入：粗筛恒回滚是 OFAT 设计（只测量不占领），不是探索失败。
    无测量轮（metrics=None，P1-A）不计入：apply 失败/高风险被拒的轮没有产出任何
    改善证据，「无测量 ≠ 无改善证据」——计入会让环境故障烧掉探索名额（case7 r6/r8
    apply 失败占掉 13 个 plateau 名额中的 2 个）。
    """
    consec = 0
    for r in reversed(rounds):
        if r.get("refine") or r.get("sweep") or r.get("metrics") is None:
            continue
        if not r.get("kept"):
            consec += 1
        else:
            break
    return consec


def consec_rejected(rounds: list) -> int:
    """连续「提议被确定性拒绝」轮数（#10 修，2026-08-29）。

    被拒轮（高风险无信号 / 组合幕队列外，decision=rejected/rejected_offqueue）
    不 apply 不压测——与 apply 失败（环境故障，P1-A 刻意不计 plateau）不同，
    这是 LLM 决策面在空转：没有测量产出地烧探索预算，且熔断不触发（apply
    没跑）。连拒 ≥3 视为 exhausted——与熔断阈值同级的环境/决策面止损。
    """
    n = 0
    for r in reversed(rounds):
        if r.get("sweep") or r.get("refine"):
            continue  # 锚点/粗筛/精修轮不打破也不贡献连拒计数
        if r.get("decision") in ("rejected", "rejected_offqueue"):
            n += 1
        else:
            break
    return n


def _n_tunable_params(flow) -> int:
    """可调参数量（动作空间去 locked/compose）——plateau 阈值的缩放基准。

    与 agent_loop._screen_params 的粗筛入选口径一致（覆盖保证的计数口径）；
    取不到 domain（测试桩/旧 flow）返回 0，阈值退回 max(flow 配置, 3)。
    """
    params = getattr(flow, "params", None)
    dom = getattr(params, "domain", None) or {}
    return sum(1 for d in dom.values()
               if isinstance(d, dict) and not d.get("locked") and not d.get("compose"))


def should_terminate(state: dict, flow, target=None, max_rounds=None) -> Optional[TerminationReason]:
    """纯函数：基于已落盘的历史轮次判定是否该终止。无副作用，不记账。

    返回 None = 继续；返回 TerminationReason = 终止。

    LLM 的语义停止洞察通过另一条路径保留：agent_loop 里 LLM 回传「无推荐」(value=null)
    → 本轮记 no_candidate → 下一次 should_terminate 走 exhausted 终止。即 LLM 提供
    「软终止」(optimization ceiling)，本函数提供「硬终止」(reliability floor)。

    两类准则（优先级从高到低）：
      exhausted  —— 最近一轮无候选（LLM 主动放弃推荐 / 动作空间穷尽）或连拒 ≥3（#10）
      plateau    —— 连续 consec_stop 轮无 keep（未达 hyst 改善门槛）

    （原 unhealthy 准则已删：agent_loop 从不产出 unhealthy/rollback_failed
    decision——服务不健康在活跃路径走 apply 失败熔断 fail{N}-fuse 与收尾兜底，
    不经终止判定；生产者只存在于已退役的 stages/decide.py。）

    consec_stop 下限（②b→2026-08-28 收紧）：max(flow 配置, 3, 可调参数量//2+1)。
    连续劣化**超过参数量一半**才允许平台期终止——覆盖保证：参数没试到一半之前，
    探索不被平台期打断（案例6 教训：24 参数只实测 3 个即停）。可调参数量按动作空间
    去 locked/compose 计（与粗筛幕入选口径一致，_n_tunable_params）。flow 显式配置
    只能抬阈值不能压低（max 语义）；max_rounds 不再参与缩放（保留形参兼容调用方）。

    调用契约：state["rounds"] 已含当前轮记录（agent_loop append 后再调用）。
    """
    rounds = state.get("rounds", [])
    if not rounds:
        return None
    last = rounds[-1]
    last_decision = last.get("decision", "")
    # 准则 1：终态性异常——穷尽/连拒，看最近一轮即可，不依赖历史长度
    if last_decision == "no_candidate":
        return TerminationReason("exhausted", "无更多可落地候选（已穷尽或 LLM 放弃推荐）")
    # #10：连拒 ≥3 = 决策面失效（高风险无信号/队列外反复提）——无测量烧预算，
    # 熔断也够不着（apply 没跑）。按 exhausted 止损，与 P1-A（apply 失败不计）正交。
    if consec_rejected(rounds) >= 3:
        return TerminationReason("exhausted", "连续 3 轮提议被确定性拒绝（无测量空转烧预算）")
    # 准则 3：平台期——连续无改善计数，按生效目标取 consec_stop 阈值
    consec_stop = getattr(decision_block(flow, target), "consec_no_improve_stop", 2)
    consec_stop = max(consec_stop, 3, _n_tunable_params(flow) // 2 + 1)
    consec = _consec_no_improve(rounds)
    if consec >= consec_stop:
        return TerminationReason("plateau", f"连续 {consec} 轮无改善（未达 hyst 门槛）")
    return None
