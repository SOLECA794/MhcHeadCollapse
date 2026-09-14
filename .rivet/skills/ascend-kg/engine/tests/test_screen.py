"""粗筛幕（③）：全参数最大对比度单点快筛，值级幕前必经。

核心意图：LLM 是 exploit 型选择器（案例6：24 参数只实测 3 个），粗筛幕用
确定性单点快筛保证每个可设参数至少 1 次真机登场，榜单把 LLM 的选参依据
从信念排序变成实测排序。
"""
from types import SimpleNamespace

import engine.agent_loop as al
from engine.tools.decide import _consec_no_improve


def mk_flow(domain=None, defaults=None, target="ttft"):
    return SimpleNamespace(
        flow_id="t",
        params=SimpleNamespace(domain=domain or {}, defaults=defaults or {}),
        perf=SimpleNamespace(hyst=0.003, ttot_room=1.10, ttft_room=1.10),
        target=target)


def mk_cfg(target="ttft", screen=True, reps=3):
    return SimpleNamespace(flow_id="t", model="/m", target=target, port=8000, device=0,
                           max_rounds=8, max_refine_rounds=4, reps=reps, screen=screen)


# 案例6 口径的基线/候选（BASE/CAND 同 test_shadow）
BASE = {"ttft_ms": 13.1, "ttot_ms": 100.0, "thr_tok_s": 139.5}


# ---------- 哨兵值选取 ----------

def test_sentinel_selection():
    domain = {
        "max-num-seqs": {"min": 32, "max": 512, "step": 32},              # 无 default → max
        "gpu-memory-utilization": {"min": 0.8, "max": 0.92, "step": 0.02},  # 无 default → max
        "enforce-eager": {"min": 0, "max": 1, "step": 1, "flag": True},   # flag 无 default → True
        "enable-chunked-prefill": {"min": 0, "max": 1, "step": 1, "flag": True},  # default True → False
        "block-size": {"min": 128, "max": 128, "step": 0, "locked": True},     # locked → 排除
        "compilation-config": {"kind": "cli", "compose": {"cudagraph_mode": "cudagraph-mode"}},  # compose → 排除
        "broken-param": {"min": 1, "step": 1},                              # 域不完整 → 排除
    }
    defaults = {"gpu-memory-utilization": 0.9, "enable-chunked-prefill": True}
    sent = dict(al._screen_params(mk_flow(domain, defaults)))
    assert sent == {
        "max-num-seqs": 512,
        "gpu-memory-utilization": 0.8,    # default 0.9 离 min 0.10 > 离 max 0.02 → 取 min
        "enforce-eager": True,
        "enable-chunked-prefill": False,  # default True → 翻转
    }


def test_sentinel_enum_full_expand():
    """枚举参数全值展开（case7：单哨兵 vals[0] 错过 FULL_DECODE_ONLY）。"""
    domain = {
        "cudagraph-mode": {"kind": "sub", "risk": "mid",
                           "values": ["FULL_AND_PIECEWISE", "FULL_DECODE_ONLY", "PIECEWISE", "NONE"]},
        "kv-cache-dtype": {"kind": "cli", "risk": "high", "values": ["fp8", "bf16", "auto"]},
    }
    pairs = al._screen_params(mk_flow(domain, {}))
    assert pairs == [("cudagraph-mode", "FULL_AND_PIECEWISE"),
                     ("cudagraph-mode", "FULL_DECODE_ONLY"),
                     ("cudagraph-mode", "PIECEWISE"),
                     ("cudagraph-mode", "NONE"),
                     ("kv-cache-dtype", "fp8"), ("kv-cache-dtype", "bf16"),
                     ("kv-cache-dtype", "auto")]
    # default 已知的值 = 基线状态本身，跳过
    pairs_d = al._screen_params(mk_flow(domain, {"kv-cache-dtype": "auto"}))
    assert ("kv-cache-dtype", "auto") not in pairs_d
    assert ("kv-cache-dtype", "fp8") in pairs_d and ("kv-cache-dtype", "bf16") in pairs_d


def test_sentinel_risk_order():
    """风险升序：易炸参数（high）殿后，失败/熔断时低风险画像已到手（case7 scr 连锁教训）。"""
    domain = {  # 声明序故意让 high 在最前
        "kv-cache-dtype": {"kind": "cli", "risk": "high", "values": ["fp8"]},
        "static-kernel": {"kind": "sub", "risk": "mid", "flag": True},
        "async-scheduling": {"kind": "cli", "flag": True},                # 未标注 → low
        "max-num-seqs": {"min": 32, "max": 512, "step": 32, "risk": "low"},
    }
    order = [p for p, _ in al._screen_params(mk_flow(domain, {}))]
    assert order.index("async-scheduling") < order.index("static-kernel") < order.index("kv-cache-dtype")
    assert order.index("max-num-seqs") < order.index("static-kernel")


def test_sentinel_numeric_default_closer_to_max_picks_min():
    domain = {"max-num-seqs": {"min": 32, "max": 512, "step": 32}}
    sent = dict(al._screen_params(mk_flow(domain, {"max-num-seqs": 448})))
    assert sent["max-num-seqs"] == 32  # default 靠上端 → 取下端才是最大对比度


# ---------- 粗筛单轮 ----------

def test_screen_round_measures_and_restores(monkeypatch):
    """成功轮：记 metrics + 双轴 pct + 影子记账 + 还原 current_params（无物理回滚）。"""
    flow = mk_flow()
    cfg = mk_cfg()
    state = {"baseline": dict(BASE, params={}), "current_params": {}}
    calls = []

    monkeypatch.setattr(al, "apply_action", lambda f, c, s, a, rk:
                        (s.__setitem__("current_params", {a["param"]: a["value"]}),
                         calls.append("apply"),
                         {"ok": True, "log": [], "rolled_back": False})[2])
    monkeypatch.setattr(al, "verify", lambda f, c, s, rk:
                        (calls.append(("verify", c.reps)),
                         {"metrics": {"ttft_ms": 14.0, "ttot_ms": 100.0, "thr_tok_s": 220.1},
                          "smoke_ok": True, "accuracy_match": None, "graph_ok": None})[1])
    monkeypatch.setattr(al, "rollback_to_last", lambda *a, **k: calls.append("rollback") or {"ok": True})
    monkeypatch.setattr(al, "_shadow_update", lambda *a, **k: calls.append("shadow") or None)

    rec, entry = al._screen_round(flow, cfg, state, "cudagraph-mode", "FULL_DECODE_ONLY", "scr1")
    assert entry["applied"] and entry["ok"]
    assert entry["ttft_pct"] < 0 and entry["thr_pct"] > 50          # -6.6% / +58%
    assert entry["main_pct"] == entry["ttft_pct"]                    # target=ttft
    assert rec["decision"] == "screen" and rec["sweep"] and rec["rolled_back"]
    assert rec["metrics"]["thr_tok_s"] == 220.1
    assert state["current_params"] == {}                             # state 还原基线
    assert calls == ["apply", ("verify", 1), "shadow"]               # reps=1、无逐轮物理回滚


def test_screen_round_apply_fail_continues(monkeypatch):
    flow = mk_flow()
    state = {"baseline": dict(BASE, params={}), "current_params": {}}
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": False, "log": ["编译超时"], "rolled_back": True})
    monkeypatch.setattr(al, "verify", lambda *a: (_ for _ in ()).throw(AssertionError("apply 失败不应 verify")))
    rec, entry = al._screen_round(flow, mk_cfg(), state, "enable-static-kernel", True, "scr1")
    assert not entry["ok"] and not entry["applied"]
    assert "apply 失败" in entry["note"] and rec["metrics"] is None


def test_screen_round_accuracy_B_blocks(monkeypatch):
    """B 类语义破坏：不进榜单/不影子记账，但 metrics 照记（数据是数据）。"""
    flow = mk_flow()
    state = {"baseline": dict(BASE, params={}), "current_params": {}}
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": True, "log": [], "rolled_back": False})
    monkeypatch.setattr(al, "verify", lambda f, c, s, rk: {
        "metrics": {"ttft_ms": 5.0, "ttot_ms": 100.0}, "smoke_ok": True,
        "accuracy_match": False, "accuracy_detail": {"cls": "B"}, "graph_ok": None})
    monkeypatch.setattr(al, "_shadow_update", lambda *a, **k: (_ for _ in ()).throw(AssertionError("B 类不应影子记账")))
    rec, entry = al._screen_round(flow, mk_cfg(), state, "kv-cache-dtype", "fp8", "scr1")
    assert not entry["ok"] and "B 类" in entry["note"]
    assert rec["metrics"]["ttft_ms"] == 5.0 and state["current_params"] == {}


# ---------- 幕编排 ----------

def test_run_screening_idempotent_and_flag_off(monkeypatch):
    flow = mk_flow({"a": {"min": 1, "max": 9, "step": 1}, "b": {"min": 1, "max": 9, "step": 1}})
    state = {"rounds": [], "baseline": dict(BASE, params={}), "current_params": {}, "phase": "value",
             "screening": [{"param": "a", "value": 9, "round": "scr1", "ok": True}]}
    ran = []
    monkeypatch.setattr(al, "_screen_round", lambda f, c, s, p, v, rk:
                        ran.append((p, rk)) or
                        ({"round": rk, "action": {"param": p, "value": v}, "kept": False,
                          "decision": "screen", "metrics": dict(BASE), "sweep": True,
                          "rolled_back": True, "rationale": ""},
                         {"param": p, "value": v, "round": rk, "ok": True, "applied": False}))
    monkeypatch.setattr(al, "save_state", lambda *a: None)
    monkeypatch.setattr(al, "rollback_to_last", lambda *a, **k: {"ok": True})

    al._run_screening(flow, mk_cfg(), state)
    assert ran == [("b", "scr2")]                 # a 已完成跳过；编号续接
    assert len(state["screening"]) == 2
    al._run_screening(flow, mk_cfg(), state)      # 再跑：全部完成 → 无动作
    assert len(ran) == 1

    state2 = {"rounds": [], "baseline": dict(BASE, params={}), "current_params": {}, "phase": "value"}
    al._run_screening(flow, mk_cfg(screen=False), state2)   # --screen off → 整幕跳过
    assert "screening" not in state2


def test_run_screening_final_physical_rollback_only_when_dirty(monkeypatch):
    """有成功 apply → 幕末一次物理回滚；全部 apply 失败（apply_action 内已自愈）→ 不再回滚。"""
    flow = mk_flow({"a": {"min": 1, "max": 9, "step": 1}})
    rb = []
    monkeypatch.setattr(al, "save_state", lambda *a: None)
    monkeypatch.setattr(al, "rollback_to_last", lambda f, c, s, rk: rb.append(rk) or {"ok": True})
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": True, "log": [], "rolled_back": False})
    monkeypatch.setattr(al, "verify", lambda f, c, s, rk: {
        "metrics": dict(BASE), "smoke_ok": True, "accuracy_match": None, "graph_ok": None})

    st_ok = {"rounds": [], "baseline": dict(BASE, params={}), "current_params": {}, "phase": "value"}
    al._run_screening(flow, mk_cfg(), st_ok)
    assert rb == ["scr1"]

    rb.clear()
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": False, "log": ["挂"], "rolled_back": True})
    st_fail = {"rounds": [], "baseline": dict(BASE, params={}), "current_params": {}, "phase": "value"}
    al._run_screening(flow, mk_cfg(), st_fail)
    assert rb == []


def test_screen_digest_shape():
    state = {"screening": [
        {"param": "cudagraph-mode", "value": "FULL_DECODE_ONLY", "ok": True,
         "ttft_pct": -6.6, "thr_pct": 58.0, "main_pct": -6.6},
        {"param": "async-scheduling", "value": True, "ok": True,
         "ttft_pct": -1.5, "thr_pct": 2.0, "main_pct": -1.5},
        {"param": "static", "value": True, "ok": False, "note": "apply 失败: 编译超时"},
    ]}
    d = al._screen_digest(state, "ttft")
    assert d.index("cudagraph-mode") < d.index("async-scheduling")  # |效应| 降序
    assert "58.0%" in d and "✗" in d and "编译超时" in d
    assert al._screen_digest({"screening": None}, "ttft") == ""


def test_screen_digest_recheck_feedback():
    """置信反馈（case7：4 正向复测 3 噪声）：证伪标记 + 保真标记 + 头部汇总警示。"""
    flow = mk_flow(target="throughput")  # primary_key → thr_tok_s
    state = {
        "baseline": {"thr_tok_s": 417.1, "params": {}},
        "screening": [
            {"param": "CPU_AFFINITY_CONF", "value": True, "ok": True,
             "thr_pct": 5.1, "main_pct": 5.1},
            {"param": "cudagraph-mode", "value": "FULL_AND_PIECEWISE", "ok": True,
             "thr_pct": 3.3, "main_pct": 3.3},
            {"param": "enforce-eager", "value": True, "ok": True,
             "thr_pct": -55.3, "main_pct": -55.3},
            {"param": "static", "value": True, "ok": False, "note": "apply 失败"},
        ],
        "rounds": [
            # 值级复测 CPU_AFFINITY：-0.3% 与粗筛 +5.1% 异号 → 证伪
            {"round": "r1", "action": {"param": "CPU_AFFINITY_CONF", "value": True},
             "metrics": {"thr_tok_s": 415.8}, "sweep": False},
            # 值级复测 cudagraph：+3.1% 同号且 ≥1% → 保真
            {"round": "r4", "action": {"param": "cudagraph-mode",
                                       "value": "FULL_AND_PIECEWISE"},
             "metrics": {"thr_tok_s": 430.0}, "sweep": False},
            # 组合轮（params 形态）不参与匹配
            {"round": "r7", "action": {"params": {"async-scheduling": True}},
             "metrics": {"thr_tok_s": 427.5}, "sweep": False},
            # 粗筛轮自身（sweep）不参与
            {"round": "scr1", "action": {"param": "enforce-eager", "value": True},
             "metrics": {"thr_tok_s": 186.2}, "sweep": True},
        ],
    }
    d = al._screen_digest(state, "throughput", flow)
    assert "正向已复测 2 项，1 项证伪不可兑现" in d
    assert "勿当预验证结论" in d                       # 1×2 ≥ 2 → 多数证伪警示触发
    line_caf = next(l for l in d.splitlines() if l.startswith("  CPU_AFFINITY_CONF"))
    assert "复测已证伪→不可兑现" in line_caf
    line_cg = next(l for l in d.splitlines() if l.startswith("  cudagraph-mode"))
    assert "复测保真" in line_cg
    # 负向条目（enforce-eager）未被值级复测 → 无标记；static 失败条目不参与
    line_eag = next(l for l in d.splitlines() if l.startswith("  enforce-eager"))
    assert "复测" not in line_eag


def test_screen_recheck_falsify_on_nearzero():
    """复测近零（|pct|<1%）也算证伪——粗筛 +5.1% → 复测 +0.0% 是噪声不是弱保真。"""
    flow = mk_flow(target="throughput")
    state = {
        "baseline": {"thr_tok_s": 417.1, "params": {}},
        "screening": [{"param": "mnbt", "value": 40960, "ok": True,
                       "thr_pct": 2.6, "main_pct": 2.6}],
        "rounds": [{"round": "r2", "action": {"param": "mnbt", "value": 40960},
                    "metrics": {"thr_tok_s": 417.3}, "sweep": False}],  # +0.05%
    }
    _, falsified = al._screen_recheck(state, "throughput", flow)
    assert ("mnbt", 40960) in falsified


def test_screen_recheck_incremental_vs_watermark():
    """复测口径 = 该轮相对当时水位线的增量（case8 教训）：r2 叠 FULL_DECODE_ONLY
    骨干，vs 基线 +35.2%（旧口径同号 → 误标「复测保真」），实为 -1.5% 增量
    （rollback）——误差棒失真直接喂错了 r4 停机决策。"""
    flow = mk_flow(target="throughput")
    state = {
        "baseline": {"thr_tok_s": 188.6},
        "screening": [
            {"param": "cudagraph-mode", "value": "FULL_DECODE_ONLY", "ok": True,
             "thr_pct": 37.6, "main_pct": 37.6},
            {"param": "mnbt", "value": 32768, "ok": True, "thr_pct": 7.3, "main_pct": 7.3},
        ],
        "rounds": [
            # r1 keep 骨干：当时水位线=基线 → +37.2% 同号 → 保真，且推进水位线
            {"round": "r1", "action": {"param": "cudagraph-mode", "value": "FULL_DECODE_ONLY"},
             "kept": True, "metrics": {"thr_tok_s": 258.9}, "sweep": False},
            # r2 叠骨干：vs 水位线 258.9 → -1.5% 异号 → 证伪（vs 基线则是 +35.2%）
            {"round": "r2", "action": {"param": "mnbt", "value": 32768},
             "kept": False, "metrics": {"thr_tok_s": 255.1}, "sweep": False},
        ],
    }
    rechecked, falsified = al._screen_recheck(state, "throughput", flow)
    assert ("cudagraph-mode", "FULL_DECODE_ONLY") in rechecked
    assert ("cudagraph-mode", "FULL_DECODE_ONLY") not in falsified
    assert ("mnbt", 32768) in falsified  # 旧口径（vs 基线）会把它误标保真


def test_plateau_and_history_transparent_to_screen_rounds():
    rounds = [
        {"round": "scr1", "action": {}, "kept": False, "decision": "screen", "sweep": True},
        {"round": "scr2", "action": {}, "kept": False, "decision": "screen", "sweep": True},
        {"round": "r1", "action": {}, "kept": False, "decision": "rollback",
         "metrics": {"ttft_ms": 101.0}},
    ]
    assert _consec_no_improve(rounds) == 1  # 只数 r1（有测量），粗筛轮透明
    flow = mk_flow()
    hist = al._fmt_history({"rounds": rounds + [
        {"round": "r2", "action": {"param": "x", "value": 1}, "kept": True,
         "decision": "keep", "metrics": {"ttft_ms": 12.0}}]}, flow)
    assert "scr1" not in hist and "r2" in hist  # 历史摘要同样跳过粗筛轮


# ---------------- #1：幕末回滚目标与 resume 兜底（2026-08-29 修） ----------------


def test_screen_finalize_rolls_back_to_baseline_script(monkeypatch, tmp_path):
    """幕末回滚目标 = scr1_prev（施加第一个哨兵前的脚本=真基线），不是末轮 scrN_prev
    ——_apply 备份的是「施加前正在跑的脚本」，末轮 prev 是 baseline+倒数第二个哨兵，
    用它恢复 = 服务侧脏、state 侧干净的分叉（r1 profiling 跑在脏服务上）。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    rb_keys = []
    monkeypatch.setattr(al, "rollback_to_last",
                        lambda f, c, s, rk: rb_keys.append(rk) or {"ok": True})
    monkeypatch.setattr(al, "_screen_round",
                        lambda f, c, s, p, v, rk: (
                            {"round": rk, "action": {"param": p, "value": v}, "kept": False,
                             "decision": "screen", "sweep": True, "metrics": {}},
                            {"param": p, "value": v, "ok": True, "applied": True}))
    state = {"phase": "value", "screening": [
        {"param": "a", "value": 1, "ok": True, "applied": False},   # 上 run 已还原
        {"param": "b", "value": 2, "ok": True, "applied": False}],
        "rounds": [], "apply_fail_streak": 0, "current_params": {}}
    flow = mk_flow(domain={"a": {"flag": True}, "b": {"flag": True},
                           "c": {"flag": True}})
    cfg = mk_cfg()
    monkeypatch.setattr(al, "_fuse_check", lambda f, c, s: None)
    al._run_screening(flow, cfg, state)
    assert rb_keys == ["scr1"]            # 不是旧口径的 scrN（此处 N=3）
    assert all(not e["applied"] for e in state["screening"])  # 还原成功 → 清标志


def test_screen_finalize_resume_with_empty_todo(monkeypatch, tmp_path):
    """中断恰好落在「循环结束~幕末回滚」之间：resume 时 todo 空，回滚仍须执行
    （旧代码 `if not todo: return None` 在回滚块之前，会整个跳过）。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    rb_keys = []
    monkeypatch.setattr(al, "rollback_to_last",
                        lambda f, c, s, rk: rb_keys.append(rk) or {"ok": True})
    state = {"phase": "value", "screening": [
        {"param": "a", "value": 1, "ok": True, "applied": True}],  # dirty 残留
        "rounds": [], "apply_fail_streak": 0, "current_params": {}}
    al._run_screening(mk_flow(), mk_cfg(), state)  # domain 空 → todo 空 → 兜底回滚
    assert rb_keys == ["scr1"]
    assert state["screening"][0]["applied"] is False


def test_screen_finalize_failure_keeps_applied_flag(monkeypatch, tmp_path):
    """回滚未恢复健康（健康等待超时）：保留 applied 标志，下次 resume 重试。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    monkeypatch.setattr(al, "rollback_to_last",
                        lambda f, c, s, rk: {"ok": False})
    state = {"phase": "value", "screening": [
        {"param": "a", "value": 1, "ok": True, "applied": True}],
        "rounds": [], "apply_fail_streak": 0, "current_params": {}}
    al._run_screening(mk_flow(), mk_cfg(), state)
    assert state["screening"][0]["applied"] is True  # 未还原，标志保留
