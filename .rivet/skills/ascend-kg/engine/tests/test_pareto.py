"""副目标 Pareto 通道（case8 档C教训）：pareto_verdict 双轴判定 + _pareto_recheck 节点。

核心意图（为什么这些行为重要）：吞吐 run 里 TTFT -16% 的参数因主目标持平
永远过不了 keep_verdict（只看主指标）→ 从未复测、从未进组合幕（组合准入 =
kept-only）——但「同主指标 + 更好副指标」对最终配置是严格 Pareto 占优。
通道把它升格为可信证据：当前最优配置上 reps=3 复测 → 双轴判定 → keep 进
交付配置（backbone/final_sweep），主目标水位线 best 不被污染。
"""
from types import SimpleNamespace

import pytest

import engine.agent_loop as al
from engine.tools.decide import pareto_verdict
from engine.tools import grouping


def mk_flow(target="throughput"):
    return SimpleNamespace(
        flow_id="t",
        params=SimpleNamespace(domain={}, defaults={}),
        perf=SimpleNamespace(hyst=0.003, ttot_room=1.10, ttft_room=1.10, noise_k=1.5),
        target=target)


def mk_cfg(target="throughput", pareto_max=1, reps=3):
    return SimpleNamespace(flow_id="t", model="/m", target=target, port=8000, device=0,
                           max_rounds=8, reps=reps, pareto_max=pareto_max)


# case8 口径：吞吐主目标，骨干=FULL_DECODE_ONLY（r1 实测）
BASE = {"ttft_ms": 15.0, "thr_tok_s": 188.6,
        "stats": {"ttft_rel_noise": 0.03, "thr_rel_noise": 0.0044}}
BACKBONE = {"ttft_ms": 16.451, "thr_tok_s": 258.9}  # r1 keep 轮实测 = 当前配置参照


def m_metrics(thr, ttft, thr_rel=0.004, ttft_rel=0.03):
    return {"thr_tok_s": thr, "ttft_ms": ttft,
            "stats": {"thr_rel_noise": thr_rel, "ttft_rel_noise": ttft_rel}}


# ---------------- pareto_verdict（纯函数） ----------------


def test_pareto_verdict_ok_when_flat_main_and_real_secondary():
    """主持平（-0.5% ≤ 容差 0.6%）+ 副真改善（+17.9% ≥ 门槛 4.5%）→ 占优成立。"""
    det = pareto_verdict(mk_flow(), m_metrics(257.5, 13.5), BASE,
                         main_ref=BACKBONE["thr_tok_s"], sec_ref=BACKBONE["ttft_ms"],
                         main_target="throughput")
    assert det["ok"] is True
    assert det["main_pct"] == pytest.approx(-0.54, abs=0.01)
    assert det["sec_pct"] == pytest.approx(17.94, abs=0.05)


def test_pareto_verdict_rejects_main_degradation_beyond_floor():
    """主退化 -3.4% 超容差 0.6% → 不成立（通道不是副目标无条件放行）。"""
    det = pareto_verdict(mk_flow(), m_metrics(250.0, 13.5), BASE,
                         main_ref=BACKBONE["thr_tok_s"], sec_ref=BACKBONE["ttft_ms"],
                         main_target="throughput")
    assert det["ok"] is False


def test_pareto_verdict_rejects_marginal_secondary():
    """副改善 +2.7% 低于地板 4.5% → 不成立（reps=1 粗筛单点必须过噪声地板才叫真）。"""
    det = pareto_verdict(mk_flow(), m_metrics(257.5, 16.0), BASE,
                         main_ref=BACKBONE["thr_tok_s"], sec_ref=BACKBONE["ttft_ms"],
                         main_target="throughput")
    assert det["ok"] is False


def test_pareto_verdict_missing_refs():
    """参照值/指标缺失 → 显式判不 keep（不裸抛）。"""
    det = pareto_verdict(mk_flow(), m_metrics(257.5, 13.5), BASE,
                         main_ref=None, sec_ref=16.4, main_target="throughput")
    assert det["ok"] is False
    det2 = pareto_verdict(mk_flow(), {"thr_tok_s": 257.5}, BASE,
                          main_ref=258.9, sec_ref=16.4, main_target="throughput")
    assert det2["ok"] is False


# ---------------- _pareto_recheck（节点） ----------------


def _mk_state():
    """case8 形态最小 state：r1 骨干 keep + scr28 影子赢家来源轮 + 影子账。"""
    return {
        "baseline": dict(BASE, params={}),
        "current_params": {"cudagraph-mode": "FULL_DECODE_ONLY"},
        "best": {"round": "r1", "thr_tok_s": 258.9, "value": -258.9},
        "best_by_target": {"ttft": {"round": "scr28", "value": 12.581,
                                    "params": {"cudagraph-mode": "FULL_DECODE_ONLY",
                                               "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": True}}},
        "rounds": [
            {"round": "r1", "action": {"param": "cudagraph-mode", "value": "FULL_DECODE_ONLY"},
             "kept": True, "decision": "keep", "metrics": dict(BACKBONE)},
            {"round": "scr28", "action": {"param": "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK",
                                          "value": True},
             "kept": False, "decision": "screen", "sweep": True,
             "metrics": {"ttft_ms": 12.581, "thr_tok_s": 186.2}},
        ],
        "apply_fail_streak": 0,
    }


P_TRANSPOSE = "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK"


def _patch_ok(monkeypatch, metrics):
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": True, "log": [], "rolled_back": False})
    monkeypatch.setattr(al, "verify",
                        lambda f, c, s, rk: {"metrics": metrics, "smoke_ok": True,
                                             "accuracy_match": None, "graph_ok": None})
    rb = []
    monkeypatch.setattr(al, "rollback_to_last",
                        lambda f, c, s, rk: rb.append(rk) or None)
    return rb


def test_pareto_recheck_keeps_into_delivery_config(monkeypatch, tmp_path):
    """占优成立 → keep 进 rounds（pareto 标记）→ kept_params 收编（进 backbone/
    final_sweep，交付配置含它）；主目标水位线 best 不更新；不触发回滚。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    rb = _patch_ok(monkeypatch, m_metrics(257.5, 13.5))
    state = _mk_state()
    al._pareto_recheck(mk_flow(), mk_cfg(), state)
    rec = state["rounds"][-1]
    assert rec["round"] == "p1" and rec["pareto"] is True and rec["kept"] is True
    assert rec["decision"] == "keep"
    assert grouping.kept_params(state, mk_flow()).get(P_TRANSPOSE) is True
    assert state["best"] == {"round": "r1", "thr_tok_s": 258.9, "value": -258.9}  # 未污染
    assert rb == []


def test_pareto_recheck_rolls_back_when_not_dominant(monkeypatch, tmp_path):
    """副改善不够（+2.7% < 地板）→ 回滚 + rollback_to_last 物理还原。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    rb = _patch_ok(monkeypatch, m_metrics(257.5, 16.0))
    state = _mk_state()
    al._pareto_recheck(mk_flow(), mk_cfg(), state)
    rec = state["rounds"][-1]
    assert rec["round"] == "p1" and rec["kept"] is False and rec["decision"] == "rollback"
    assert rec["pareto"] is True
    assert P_TRANSPOSE not in grouping.kept_params(state, mk_flow())
    assert rb == ["p1"]


def test_pareto_recheck_gates(monkeypatch, tmp_path):
    """pareto_max=0 关闭；非单参数来源跳过；参数已 keep 同值跳过；幂等不重跑。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    calls = []
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: calls.append(rk) or {"ok": True, "log": []})
    # 关闭
    al._pareto_recheck(mk_flow(), mk_cfg(pareto_max=0), _mk_state())
    assert calls == []
    # 非单参数来源（组合轮形态）
    st = _mk_state()
    st["rounds"][1] = {"round": "c1", "action": {"params": {P_TRANSPOSE: True}},
                       "kept": True, "decision": "keep", "metrics": dict(BACKBONE)}
    st["best_by_target"]["ttft"]["round"] = "c1"
    al._pareto_recheck(mk_flow(), mk_cfg(), st)
    assert calls == []
    # 参数已 keep 同值 → 通道无事可做
    st = _mk_state()
    st["rounds"].insert(0, {"round": "r0x", "action": {"param": P_TRANSPOSE, "value": True},
                            "kept": True, "decision": "keep", "metrics": dict(BACKBONE)})
    al._pareto_recheck(mk_flow(), mk_cfg(), st)
    assert calls == []
    # 幂等：p1 已存在（同参数）→ 不再跑
    st = _mk_state()
    st["rounds"].append({"round": "p1", "pareto": True, "kept": False,
                         "action": {"param": P_TRANSPOSE, "value": True},
                         "metrics": m_metrics(257.5, 16.0)})
    al._pareto_recheck(mk_flow(), mk_cfg(), st)
    assert calls == []
    # 基线种子（round="r0"，副目标从未改善）→ 静默返回，不是 loud skip
    st = _mk_state()
    st["best_by_target"]["ttft"]["round"] = "r0"
    al._pareto_recheck(mk_flow(), mk_cfg(), st)
    assert calls == []


def test_pareto_recheck_apply_failure_counts_streak(monkeypatch, tmp_path):
    """apply 失败：P0-B/P1-A 同口径（streak+1、fail_reason 摘录、metrics=None）。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": False, "log": "boot\nERROR: NPU OOM",
                                                 "rolled_back": True})
    state = _mk_state()
    al._pareto_recheck(mk_flow(), mk_cfg(), state)
    rec = state["rounds"][-1]
    assert rec["round"] == "p1" and rec["kept"] is False and rec["metrics"] is None
    assert "NPU OOM" in rec["fail_reason"]
    assert state["apply_fail_streak"] == 1


def test_pareto_recheck_hard_guard_rolls_back(monkeypatch, tmp_path):
    """smoke 失败/精度B/图捕获未生效 → 强制回滚（与 try_action 同口径，硬护栏不因通道放宽）。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": True, "log": [], "rolled_back": False})
    monkeypatch.setattr(al, "verify",
                        lambda f, c, s, rk: {"metrics": m_metrics(257.5, 13.5),
                                             "smoke_ok": False,
                                             "accuracy_match": None, "graph_ok": None})
    rb = []
    monkeypatch.setattr(al, "rollback_to_last", lambda f, c, s, rk: rb.append(rk) or None)
    state = _mk_state()
    al._pareto_recheck(mk_flow(), mk_cfg(), state)
    rec = state["rounds"][-1]
    assert rec["kept"] is False and "smoke" in rec["rationale"]
    assert rb == ["p1"]


def test_pareto_recheck_ref_falls_back_to_baseline(monkeypatch, tmp_path):
    """无任何 keep 轮（全回滚 run）→ 参照回退 baseline：对照 defaults 复测而非 last-keep。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    refs = {}
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": True, "log": [], "rolled_back": False})

    def fake_verify(f, c, s, rk):
        # 主持平（189.5 vs 基线 188.6）+ 副 -20%（12.0 vs 15.0）→ 参照是基线时必 keep
        return {"metrics": m_metrics(189.5, 12.0), "smoke_ok": True,
                "accuracy_match": None, "graph_ok": None}

    monkeypatch.setattr(al, "verify", fake_verify)
    real_pv = al.pareto_verdict
    monkeypatch.setattr(al, "pareto_verdict",
                        lambda f, m, b, mr, sr, t: refs.update(main=mr, sec=sr)
                        or real_pv(f, m, b, mr, sr, t))
    monkeypatch.setattr(al, "rollback_to_last", lambda f, c, s, rk: None)
    st = _mk_state()
    st["rounds"] = [st["rounds"][1]]  # 只留 scr 来源轮（kept 轮删掉 → 无 keep）
    st["best"] = None
    al._pareto_recheck(mk_flow(), mk_cfg(), st)
    assert refs["main"] == pytest.approx(BASE["thr_tok_s"])  # 基线 thr，非 r1 的 258.9
    assert refs["sec"] == pytest.approx(BASE["ttft_ms"])
    assert st["rounds"][-1]["round"] == "p1"  # 主持平+副 -20% → keep 成立
