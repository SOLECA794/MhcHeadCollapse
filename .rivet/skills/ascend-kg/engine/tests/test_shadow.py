"""双目标影子记账（方案 1）：一轮测量服务两个目标。

核心意图：主目标回滚但副目标大赢的 trade-off（案例6 FULL_DECODE_ONLY：TTFT
-6.6% 回滚 / 吞吐 +58%）不被单目标判定丢弃——配置与实测值记入
best_by_target[副目标]，第二遍换目标的 pass 无需重测。
"""
from types import SimpleNamespace

import engine.agent_loop as al


def mk_flow(hyst=0.003, ttot_room=1.10, ttft_room=1.10):
    return SimpleNamespace(
        perf=SimpleNamespace(hyst=hyst, ttot_room=ttot_room, ttft_room=ttft_room),
        target="ttft")


def mk_cfg(target="ttft"):
    return SimpleNamespace(flow_id="t", model="/m", target=target, port=8000, device=0,
                           max_rounds=8, max_refine_rounds=4)


# 案例6 实测数据回放：baseline 13.1ms/139.5t/s，FULL_DECODE_ONLY 14.0ms/220.1t/s
BASE = {"ttft_ms": 13.1, "ttot_ms": 100.0, "thr_tok_s": 139.5, "params": {}}
CAND = {"ttft_ms": 14.0, "ttot_ms": 100.0, "thr_tok_s": 220.1}


def test_tradeoff_shadow_kept():
    flow = mk_flow()
    state = {"baseline": dict(BASE), "current_params": {}}
    # 主目标 ttft：14.0 劣化 → try_action 会回滚；副目标 throughput：+58%
    new = al._shadow_update(flow, state, dict(CAND),
                            {"cudagraph-mode": "FULL_DECODE_ONLY"}, "r1", "ttft")
    assert new is not None and new["target"] == "throughput"
    ob = state["best_by_target"]["throughput"]
    assert ob["round"] == "r1"
    assert ob["params"] == {"cudagraph-mode": "FULL_DECODE_ONLY"}
    assert ob["metrics"]["thr_tok_s"] == 220.1
    assert ob["improved_pct"] > 50  # +58% 量级
    # 主目标记账不碰（防双写漂移）
    assert "best" not in state


def test_shadow_guard_violated_no_update():
    """吞吐目标 guard：TTFT 退化超 ttft_room → 影子判定拒绝（复用副目标自己的护栏）。"""
    flow = mk_flow(ttft_room=1.10)
    state = {"baseline": dict(BASE)}
    bad_ttft = dict(CAND, ttft_ms=20.0)  # 13.1×1.10=14.4 → 20 超容限
    assert al._shadow_update(flow, state, bad_ttft, {"x": 1}, "r1", "ttft") is None
    assert "best_by_target" not in state or "throughput" not in state["best_by_target"]


def test_shadow_below_threshold_no_update():
    """副目标改善 < hyst → 不更新（保持 baseline 起点）。"""
    flow = mk_flow(hyst=0.05)
    state = {"baseline": dict(BASE)}
    weak = dict(CAND, thr_tok_s=140.0)  # +0.36% < 5%
    assert al._shadow_update(flow, state, weak, {"x": 1}, "r1", "ttft") is None


def test_shadow_monotonic_against_cur_best():
    """改善 vs baseline 但劣于已记副目标 best → 不回退。"""
    flow = mk_flow()
    state = {"baseline": dict(BASE),
             "best_by_target": {"throughput": {
                 "target": "throughput", "value": -220.1, "round": "r1",
                 "params": {"a": 1}, "improved_pct": 58.0,
                 "metrics": {"thr_tok_s": 220.1}}}}
    mid = dict(CAND, thr_tok_s=180.0)  # vs baseline +29% 但 < 220.1
    assert al._shadow_update(flow, state, mid, {"b": 2}, "r2", "ttft") is None
    assert state["best_by_target"]["throughput"]["metrics"]["thr_tok_s"] == 220.1


def test_baseline_best_dual_targets():
    flow = mk_flow()
    bb = al._baseline_best(flow, BASE, "throughput")
    assert bb["value"] == -139.5 and bb["round"] == "r0" and bb["params"] == {}
    # 无 thr 指标的后端（vLLM 模板不报 token 数）→ throughput 条目不成立
    no_thr = {k: v for k, v in BASE.items() if k != "thr_tok_s"}
    assert al._baseline_best(flow, no_thr, "throughput") is None


def test_shadow_no_metric_backend_disables():
    """基线无副目标指标 → 影子记账整体不生效（返回 None，不建记账键）。"""
    flow = mk_flow()
    no_thr = {k: v for k, v in BASE.items() if k != "thr_tok_s"}
    state = {"baseline": no_thr}
    assert al._shadow_update(flow, state, {"ttft_ms": 12.0, "ttot_ms": 100.0},
                             {}, "r1", "ttft") is None


def test_dual_target_summary_shape():
    flow = mk_flow()
    state = {"baseline": dict(BASE), "best": {"round": "r2", "ttft_ms": 12.5, "value": 12.5},
             "best_by_target": {"throughput": {
                 "target": "throughput", "value": -220.1, "round": "r1",
                 "params": {"cudagraph-mode": "FULL_DECODE_ONLY"},
                 "improved_pct": 58.0, "metrics": {"thr_tok_s": 220.1}}}}
    s = al._dual_target_summary(flow, state, "ttft")
    assert s["main_target"] == "ttft" and s["main_best"]["round"] == "r2"
    assert s["other_best"]["params"]["cudagraph-mode"] == "FULL_DECODE_ONLY"
    assert "trade-off" in s["note"]


def test_try_action_wiring_captures_config_before_rollback(monkeypatch):
    """接线点验证：影子记账发生在回滚前——记录的是候选配置而非回滚后的 defaults。"""
    flow = mk_flow()
    captured = {}

    def fake_shadow(f, st_, m, params, rk, tgt, am=None, ad=None):
        captured["params"] = dict(params or {})
        captured["metrics"] = dict(m)
        return None

    monkeypatch.setattr(al, "_shadow_update", fake_shadow)

    def fake_apply(f, cfg, st_, action, rk):
        st_["current_params"] = {"cudagraph-mode": "FULL_DECODE_ONLY"}  # apply 后候选配置
        return {"ok": True, "log": []}

    def fake_verify(f, cfg, st_, rk):
        return {"smoke_ok": True, "metrics": dict(CAND)}

    def fake_rollback(f, cfg, st_, rk):
        st_["current_params"] = {}  # 回滚恢复 defaults
        return {"ok": True}

    monkeypatch.setattr(al, "apply_action", fake_apply)
    monkeypatch.setattr(al, "verify", fake_verify)
    monkeypatch.setattr(al, "rollback_to_last", fake_rollback)
    monkeypatch.setattr(al, "run_benchmark", lambda *a, **k: dict(CAND))

    state = {"baseline": dict(BASE), "current_params": {}}
    rec = al.try_action(flow, mk_cfg(), state, {"param": "cudagraph-mode",
                                                "value": "FULL_DECODE_ONLY"}, "r1")
    assert rec["decision"] == "rollback"  # 主目标 ttft 劣化 → 回滚
    assert captured["params"] == {"cudagraph-mode": "FULL_DECODE_ONLY"}  # 回滚前的候选配置
    assert captured["metrics"]["thr_tok_s"] == 220.1
    assert state["current_params"] == {}  # 回滚已恢复
