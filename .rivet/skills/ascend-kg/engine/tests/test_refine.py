"""②a 探索/精修预算分离：_drain_refines 的行为契约。

核心意图：keep 参数的邻域补扫是「精修」——不占 max_rounds 探索预算、由独立
max_refine_rounds 封顶、队列按 tried_values 幂等重算（resume 不重扫）、keep 后
center 移动链式前进。全部 monkeypatch try_action，不碰真机。
"""
from types import SimpleNamespace

import engine.agent_loop as al


DOMAIN = {
    "a": {"kind": "cli", "min": 1, "max": 9, "step": 2},
    "b": {"kind": "cli", "min": 0, "max": 10, "step": 5},
    "locked-p": {"kind": "cli", "min": 1, "max": 4, "step": 1, "locked": True, "fixed": 1},
}


def mk_flow(domain=None):
    return SimpleNamespace(flow_id="t", target="ttft",
                           params=SimpleNamespace(domain=domain or DOMAIN, blocked=[]))


def mk_cfg(refine=4):
    return SimpleNamespace(flow_id="t", model="/m", target="ttft", port=8000, device=0,
                           max_rounds=8, max_refine_rounds=refine)


def kept_round(p, v):
    return {"round": "r1", "action": {"param": p, "value": v}, "kept": True,
            "decision": "keep"}


def fake_try(log, decision="rollback"):
    def _f(flow, cfg, state, action, round_key):
        log.append((round_key, action["param"], action["value"]))
        rec = {"round": round_key, "action": action, "changed": [action["param"]],
               "kept": decision == "keep", "decision": decision, "metrics": {}}
        if decision == "keep":  # keep 记账由 pending_neighborhood 读 best_kept
            pass
        return rec
    return _f


def test_drain_with_monkeypatch(monkeypatch):
    log = []
    monkeypatch.setattr(al, "try_action", fake_try(log))
    state = {"rounds": [kept_round("a", 5)]}
    al._drain_refines(mk_flow(), mk_cfg(), state)
    # 邻域 3/7 均未试 → 两轮精修
    assert [(rk, p, v) for rk, p, v in log] == [("f1", "a", 3), ("f2", "a", 7)]
    assert all(r.get("refine") for r in state["rounds"][1:])
    assert [r["round"] for r in state["rounds"]] == ["r1", "f1", "f2"]


def test_drain_budget_cap(monkeypatch, capsys):
    """max_refine_rounds=1 → 只跑 1 轮，剩余邻域大声告警。"""
    log = []
    monkeypatch.setattr(al, "try_action", fake_try(log))
    state = {"rounds": [kept_round("a", 5)]}
    al._drain_refines(mk_flow(), mk_cfg(refine=1), state)
    assert len(log) == 1
    out = capsys.readouterr().out
    assert "精修预算耗尽" in out and "a" in out


def test_drain_keep_chains_and_idempotent(monkeypatch):
    """refine keep → center 移动链式前进；再次 drain 幂等（tried 去重不重扫）。"""
    log = []
    calls = {"n": 0}

    def _f(flow, cfg, state, action, round_key):
        log.append((round_key, action["param"], action["value"]))
        # 第 1 次精修（a=3）keep：center 5→3
        keep = calls["n"] == 0
        calls["n"] += 1
        return {"round": round_key, "action": action, "changed": ["a"],
                "kept": keep, "decision": "keep" if keep else "rollback"}

    monkeypatch.setattr(al, "try_action", _f)
    state = {"rounds": [kept_round("a", 5)]}
    al._drain_refines(mk_flow(), mk_cfg(refine=4), state)
    # f1: a=3 keep；f2: 新 center=3 的邻域 {1,5}，5 已试 → a=1
    assert log == [("f1", "a", 3), ("f2", "a", 1)]
    # 幂等：全部邻域已进 tried（含 keep 的 3、rollback 的 1/7? 7 未试！）
    # f2 rollback 后 pending 仍有 a=7（原 center=5 的 +step 未试，best_kept[a]=3 → 邻域 {1,5}）
    # best_kept 是最新 keep 值 3 → 邻域 {1,5} 都试过 → 空，drain 结束（7 留给下一轮 decide）
    before = len(state["rounds"])
    al._drain_refines(mk_flow(), mk_cfg(refine=4), state)
    assert len(state["rounds"]) == before  # 无新轮：幂等


def test_drain_skips_locked_and_multi_param(monkeypatch):
    """locked 参数不补扫；只有 keep 过的参数进队列（未 keep 的 b 不扫）。"""
    log = []
    monkeypatch.setattr(al, "try_action", fake_try(log))
    state = {"rounds": [kept_round("a", 5)]}  # b 从未 keep
    al._drain_refines(mk_flow(), mk_cfg(), state)
    assert all(p == "a" for _, p, _ in log)


def test_drain_no_pending_noop(monkeypatch):
    """无 keep 轮 → 队列空 → 零调用（fresh run r0 后首轮前不误触发）。"""
    log = []
    monkeypatch.setattr(al, "try_action", fake_try(log))
    al._drain_refines(mk_flow(), mk_cfg(), {"rounds": []})
    assert log == []
