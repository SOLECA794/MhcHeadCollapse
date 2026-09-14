"""P0-2 统计功效：噪声地板 hyst_eff + borderline 临界重测 + 终止判定回归。

核心意图（为什么这些行为重要）：hyst 0.3% 远小于真机 ±2% 采样波动时，
无地板的 keep 判定会被噪声驱动假 keep/假 rollback（case5 教训）——
这些用例锁住「噪声大 → 门槛抬高」与「临界轮可被重测救回」两条语义。
"""
from types import SimpleNamespace

import pytest

from engine.tools.decide import keep_verdict_detail, should_terminate
from engine.stages.verify import _mad


def mk_flow(**perf):
    """最小 flow 桩：只填 keep 判定消费的 perf 字段（ttft 目标）。"""
    base = {"hyst": 0.003, "noise_k": 1.5}
    base.update(perf)
    return SimpleNamespace(perf=SimpleNamespace(**base), target="ttft")


def mk_metrics(ttft, rel_noise=None, ttot=100.0):
    m = {"ttft_ms": ttft, "ttot_ms": ttot}
    if rel_noise is not None:
        m["stats"] = {"ttft_rel_noise": rel_noise}
    return m


BASE = {"ttft_ms": 100.0, "ttot_ms": 100.0}


# ---------------- 噪声地板（hyst_eff） ----------------


def test_no_stats_zero_regression():
    """旧 state（无 stats）resume：地板 0，hyst_eff==hyst，行为与改造前一致。"""
    det = keep_verdict_detail(mk_flow(), mk_metrics(99.0), BASE)
    assert det["noise_floor"] == 0.0
    assert det["hyst_eff"] == det["hyst"] == 0.003
    assert det["kept"] is True  # 改善 1% > 0.3%，老口径本就该 keep


def test_noise_floor_blocks_marginal_keep():
    """本轮噪声 2% → 地板 3%（min 口径取更稳一侧=基线 1%）：改善 1% 的轮不再假 keep，
    且标记 borderline（可重测）。"""
    det = keep_verdict_detail(mk_flow(),
                              mk_metrics(99.0, rel_noise=0.02),
                              {**BASE, "stats": {"ttft_rel_noise": 0.01}})
    assert det["noise_floor"] == 0.015  # 1.5 × min(0.02, 0.01)
    assert det["hyst_eff"] == 0.015
    assert det["kept"] is False
    assert det["borderline"] is True  # 1% 过了配置门槛 0.3% 但被地板拦下
    assert "噪声地板" in det["rationale"]


def test_floor_takes_tighter_side():
    """P0-A min 口径：基线池化噪声被 rep 间漂移抬高（case7 TTFT 17.9%）时，
    地板不再被最差一次测量锁死——轮内噪声 1% → 地板 1.5%，13% 真改善可入账
    （case7 影子记账零触发的直接解）。"""
    det = keep_verdict_detail(mk_flow(),
                              mk_metrics(87.0, rel_noise=0.01),
                              {**BASE, "stats": {"ttft_rel_noise": 0.1786}})
    assert det["noise_floor"] == 0.015  # 1.5 × min(0.1786, 0.01)
    assert det["kept"] is True  # 改善 13% > 1.5%
    assert det["borderline"] is False


def test_unstable_round_flags_borderline_for_remeasure():
    """P0-A 配套：本轮噪声显著高于基线（>2× 且 >2%，case7 r12 轮内 11% vs 基线
    3.6%）→ 即使改善过了地板（6% > 1.5×3.6%=5.4%）也标 borderline 触发重测——
    min 口径放松假阴性防护后，幸运轮假 keep 由一次重测兜底。"""
    det = keep_verdict_detail(mk_flow(),
                              mk_metrics(94.0, rel_noise=0.11),
                              {**BASE, "stats": {"ttft_rel_noise": 0.036}})
    assert det["kept"] is True          # 改善 6% ≥ 地板 5.4% → 常规口径本就 keep
    assert det["borderline"] is True    # 但本轮噪声异常 → 仍标临界触发重测
    assert det["unstable_round"] is True


def test_stable_round_not_flagged():
    """正常轮（噪声与基线同量级）不触发 unstable 重测——重测预算只花在可疑轮上。"""
    det = keep_verdict_detail(mk_flow(),
                              mk_metrics(95.0, rel_noise=0.02),
                              {**BASE, "stats": {"ttft_rel_noise": 0.036}})
    assert det["unstable_round"] is False
    assert det["borderline"] is False or det["kept"] is True


def test_clear_improve_passes_floor():
    """改善 5% > 地板 3%：真改善不受地板误伤。"""
    det = keep_verdict_detail(mk_flow(),
                              mk_metrics(95.0, rel_noise=0.01),
                              {**BASE, "stats": {"ttft_rel_noise": 0.02}})
    assert det["kept"] is True
    assert det["borderline"] is False


def test_below_configured_hyst_not_borderline():
    """改善 0.1% 连配置门槛都没过 → 不是临界轮（重测也救不回，不该触发）。"""
    det = keep_verdict_detail(mk_flow(),
                              mk_metrics(99.9, rel_noise=0.01),
                              {**BASE, "stats": {"ttft_rel_noise": 0.02}})
    assert det["kept"] is False
    assert det["borderline"] is False


def test_disable_noise_floor():
    """disable_noise_floor=true 一键回到纯 hyst 口径（逃生开关）。"""
    det = keep_verdict_detail(mk_flow(disable_noise_floor=True),
                              mk_metrics(99.0, rel_noise=0.01),
                              {**BASE, "stats": {"ttft_rel_noise": 0.02}})
    assert det["noise_floor"] == 0.0
    assert det["kept"] is True


def test_remeasure_flip():
    """临界重测语义：原始 + 重测的 median 取均值后过地板 → 终判翻转为 keep。

    这是 try_action 里 remeasure 块的行为契约——单次采样抖动不该误杀方向正确的改善。
    基线噪声 2% → 地板 3%；原始 99ms（改善 1%，临界）+ 重测 94ms → 均值 96.5（3.5% > 3%）。
    """
    flow = mk_flow()
    base = {**BASE, "stats": {"ttft_rel_noise": 0.02}}
    m1 = mk_metrics(99.0, rel_noise=0.02)
    det1 = keep_verdict_detail(flow, m1, base)
    assert det1["borderline"] and not det1["kept"]
    merged = dict(m1, ttft_ms=(99.0 + 94.0) / 2)  # try_action 的均值合并口径
    det2 = keep_verdict_detail(flow, merged, base)
    assert det2["kept"] is True


def test_mad_sanity():
    """MAD 基本语义：常数列 0；对称离群列 = 半距。"""
    assert _mad([5.0, 5.0, 5.0]) == 0.0
    assert _mad([4.0, 5.0, 6.0]) == 1.0
    assert _mad([1.0, 2.0, 3.0, 4.0]) == 1.0  # median 2.5 → |1.5,0.5,0.5,1.5| median 1.0


# ---------------- 终止判定回归（含 action v2 组合轮） ----------------


def _mk_state_flow():
    flow = SimpleNamespace(
        perf=SimpleNamespace(hyst=0.003),
        target="ttft",
        decision=SimpleNamespace(consec_no_improve_stop=2),
        decision_throughput=None)
    return flow


def test_terminate_exhausted_on_no_candidate():
    state = {"rounds": [{"round": "r1", "decision": "no_candidate", "kept": False}]}
    term = should_terminate(state, _mk_state_flow())
    assert term and term.kind == "exhausted"


def test_terminate_plateau_consec():
    """②b：plateau 阈值最低 3——flow 配 2 也被抬底，2 个 rollback 不再停（防探索一次失败即死）。"""
    rb = {"decision": "rollback", "kept": False, "metrics": {"ttft_ms": 101.0}}
    state = {"rounds": [{"round": "r1", **rb}, {"round": "r2", **rb}]}
    assert should_terminate(state, _mk_state_flow()) is None
    state["rounds"].append({"round": "r3", **rb})
    term = should_terminate(state, _mk_state_flow())
    assert term and term.kind == "plateau"


def test_apply_fail_rounds_not_counted_toward_plateau():
    """P1-A：metrics=None（apply 失败/高风险被拒）的轮不计 plateau——无测量 ≠ 无改善
    证据，环境故障不该烧探索名额（case7 r6/r8 各占掉 plateau 名额）。"""
    fb = {"decision": "rollback", "kept": False, "metrics": None}  # apply 失败轮
    rb = {"decision": "rollback", "kept": False, "metrics": {"ttft_ms": 101.0}}
    state = {"rounds": [{"round": "r1", **rb}, {"round": "r2", **fb},
                        {"round": "r3", **fb}, {"round": "r4", **rb}]}
    # 2 个失败轮被跳过 → 有效连续无改善 = 2 < 3，不停
    assert should_terminate(state, _mk_state_flow()) is None
    state["rounds"].append({"round": "r5", **rb})
    assert should_terminate(state, _mk_state_flow()).kind == "plateau"


def test_terminate_plateau_scaling_and_user_override():
    """②b：随动作空间缩放（连续劣化超过可调参数量一半才停）；用户显式配高值保留。"""
    rb = {"decision": "rollback", "kept": False, "metrics": {"ttft_ms": 101.0}}

    def dom_flow(n, stop=2):
        dom = {f"p{i}": {"min": 1, "max": 9, "step": 1} for i in range(n)}
        return SimpleNamespace(
            perf=SimpleNamespace(hyst=0.003), target="ttft",
            params=SimpleNamespace(domain=dom, defaults={}),
            decision=SimpleNamespace(consec_no_improve_stop=stop),
            decision_throughput=None)

    # 10 可调参数 → 阈值 10//2+1=6：5 个 rollback 不停、6 个停（max_rounds 不再参与）
    flow10 = dom_flow(10)
    state5 = {"rounds": [{"round": f"r{i}", **rb} for i in range(1, 6)]}
    state6 = {"rounds": [{"round": f"r{i}", **rb} for i in range(1, 7)]}
    assert should_terminate(state5, flow10, max_rounds=8) is None
    assert should_terminate(state6, flow10, max_rounds=8).kind == "plateau"
    # 21 可调参数（vllm flow 口径）→ 阈值 11
    flow21 = dom_flow(21)
    state10 = {"rounds": [{"round": f"r{i}", **rb} for i in range(1, 11)]}
    assert should_terminate(state10, flow21) is None
    # locked/compose 不计入缩放基准
    flow_l = dom_flow(10)
    flow_l.params.domain["locked-p"] = {"min": 1, "max": 1, "locked": True}
    flow_l.params.domain["compose-p"] = {"compose": {"k": "sub"}}
    state5b = {"rounds": [{"round": f"r{i}", **rb} for i in range(1, 6)]}
    assert should_terminate(state5b, flow_l) is None  # 仍是 10 参数口径的阈值 6
    # 用户显式配 12 > 6：6 个 rollback 不停（显式配置只能抬不能压）
    flow12 = dom_flow(10, stop=12)
    assert should_terminate(state6, flow12) is None


def test_refine_rounds_transparent_to_plateau():
    """②a：精修轮 rollback 不计探索 plateau（确认局部最优是预期结果非探索失败）。"""
    state = {"rounds": [{"round": "r1", "decision": "keep", "kept": True,
                         "action": {"param": "a", "value": 5}},
                        {"round": "f1", "decision": "rollback", "kept": False,
                         "refine": True},
                        {"round": "f2", "decision": "rollback", "kept": False,
                         "refine": True}]}
    # 2 个 refine rollback 紧跟 keep → consec=0，不终止
    assert should_terminate(state, _mk_state_flow()) is None


def test_kept_combo_breaks_plateau_streak():
    """组合轮 keep 打断连续无改善计数——组合幕产物不误触发值级平台期终止。"""
    state = {"rounds": [{"round": "r1", "decision": "rollback", "kept": False,
                         "metrics": {"ttft_ms": 101.0}},
                        {"round": "c1", "decision": "keep", "kept": True,
                         "action": {"params": {"a": 1, "b": 2}}},
                        {"round": "c2", "decision": "rollback", "kept": False,
                         "metrics": {"ttft_ms": 101.0}}]}
    term = should_terminate(state, _mk_state_flow())
    assert term is None  # c2 后只连续 1 轮无改善


# ---------------- #10：连拒终止（2026-08-29 修） ----------------


def _rej(round_key, decision="rejected"):
    return {"round": round_key, "action": {"param": "x", "value": 1}, "kept": False,
            "decision": decision, "metrics": None, "action_changed": []}


def test_consec_rejected_counts_and_terminates():
    """连拒 ≥3 → exhausted：被拒轮不 apply 不压测（熔断也够不着），无测量空转
    烧探索预算——LLM 反复提高风险无信号/队列外动作时必须止损，不能烧满 max_rounds。"""
    from engine.tools.decide import consec_rejected
    rounds = [_rej("r1"), _rej("r2"), _rej("r3")]
    assert consec_rejected(rounds) == 3
    keep = {"round": "r0", "action": {}, "kept": True,
            "decision": "keep", "metrics": {"ttft_ms": 10.0}}
    state = {"rounds": [keep] + rounds}  # keep 在前：连拒链从尾部回溯不被打断
    term = should_terminate(state, _mk_state_flow())
    assert term is not None and term.kind == "exhausted"
    assert "被确定性拒绝" in term.detail


def test_consec_rejected_apply_failure_breaks_streak():
    """apply 失败轮（环境故障，P1-A 刻意不烧探索名额）打破连拒链——两类「无测量」
    成因不同：一个是 LLM 决策面空转，一个是环境坏了交熔断，不能互相顶替。"""
    from engine.tools.decide import consec_rejected
    rounds = [_rej("r1"), _rej("r2"),
              {"round": "r3", "action": {"param": "x", "value": 1}, "kept": False,
               "decision": "rollback", "metrics": None, "fail_reason": "ERROR: OOM"},
              _rej("r4")]
    assert consec_rejected(rounds) == 1  # 从尾部只数到 r4；r3 打断


def test_consec_rejected_sweep_refine_transparent():
    """锚点/粗筛/精修轮不打破也不贡献连拒计数（与 _consec_no_improve 同口径）。"""
    from engine.tools.decide import consec_rejected
    rounds = [_rej("r1"), {"round": "a1", "action": {}, "kept": False,
                           "decision": "anchor", "sweep": True, "metrics": {}},
              _rej("r2"), _rej("r3")]
    assert consec_rejected(rounds) == 3  # a1 透明跳过但链条延续：r1+r2+r3 连拒


def test_noise_floor_uses_own_rel_noise_per_metric():
    """#6 回归：key="ttot_ms" 的地板必须用 ttot_rel_noise——TTOT 守门此前一直
    拿 TTFT 的离散度当地板（TTFT 噪声小 → TTOT 真收益被误判噪声、真劣化被放过）。"""
    from engine.tools.objective import noise_floor
    flow = mk_flow(noise_k=1.0)
    m = {"stats": {"ttft_rel_noise": 0.01, "ttot_rel_noise": 0.06}}
    b = {"stats": {"ttft_rel_noise": 0.02, "ttot_rel_noise": 0.04}}
    assert noise_floor(flow, m, b, "ttot_ms") == pytest.approx(0.04)  # min(6%,4%)，非 ttft 的 1%
    assert noise_floor(flow, m, b, "ttft_ms") == pytest.approx(0.01)
    assert noise_floor(flow, m, b, "thr_tok_s") == 0.0  # 无 thr_rel_noise → 零回归回退
