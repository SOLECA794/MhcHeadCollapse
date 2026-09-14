"""P0-1 两幕编排：action v2 决策解析 + 组合幕编排纯函数（真实 case5 flow）。

核心意图：
- 组合动作逐参数全量校验、任一越界整单拒绝（组合不放宽护栏）；
- 有效决策（单/组合）绝不被误判成软终止（旧 bug 回归锚点）；
- 组合队列成员校验/预算截断/交互最优提取 —— 组合幕的行为契约。
"""
from types import SimpleNamespace

import engine.agent_loop as al


# ---------------- action v2：_parse_decision 双形态 ----------------


def test_parse_single_param(case5_flow):
    out = al._parse_decision({"action": {"param": "max-num-seqs", "value": 160}},
                             case5_flow)
    assert out == {"param": "max-num-seqs", "value": 160}


def test_parse_combo_params(case5_flow):
    out = al._parse_decision(
        {"action": {"params": {"VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE": True,
                               "VLLM_ASCEND_ENABLE_PREFETCH_MLP": True}}},
        case5_flow)
    assert out == {"params": {"VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE": True,
                              "VLLM_ASCEND_ENABLE_PREFETCH_MLP": True}}


def test_parse_combo_type_coercion(case5_flow):
    """组合内数值字符串收敛后过校验（MindIE config 拒收 str 的教训）。"""
    out = al._parse_decision(
        {"action": {"params": {"max-num-seqs": "160"}}}, case5_flow)
    assert out == {"params": {"max-num-seqs": 160}}


def test_parse_combo_out_of_domain_rejects_whole(case5_flow):
    """组合内任一参数越界 → 整单拒绝（None），不是部分通过。"""
    assert al._parse_decision(
        {"action": {"params": {"max-num-seqs": 160, "ghost-p": 1}}}, case5_flow) is None


def test_parse_combo_over_three_rejects(case5_flow):
    assert al._parse_decision(
        {"action": {"params": {f"p{i}": i for i in range(4)}}}, case5_flow) is None


def test_parse_soft_terminate_marker(case5_flow):
    """无推荐 → {"candidate": None}；有效组合决策不得再被误判为软终止（旧 bug）。"""
    out = al._parse_decision({"action": {}, "reason": "穷尽"}, case5_flow)
    assert out == {"candidate": None}
    # 软终止判据 = param/params 双键都不在，而不是旧版的 `.get("candidate") is None`
    assert "param" not in out and "params" not in out


def test_action_helpers():
    assert al.action_label({"param": "p", "value": 1}) == "p=1"
    assert al.action_label({"params": {"a": 1, "b": True}}) in ("a=1+b=True", "b=True+a=1")
    assert al.action_changed({"param": "p", "value": 1}) == ["p"]
    assert set(al.action_changed({"params": {"a": 1, "b": 2}})) == {"a", "b"}
    assert al.action_changed({"candidate": None}) == []


# ---------------- 组合幕：队列匹配 / 先验并集 / 交互最优 ----------------

P1 = "VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE"
P2 = "VLLM_ASCEND_ENABLE_PREFETCH_MLP"


def test_match_queue_exact(case5_flow):
    q = [{P1: True, P2: False}, {P1: False, P2: True}]
    assert al._match_queue_combo({P1: True, P2: False}, q) == {P1: True, P2: False}


def test_match_queue_backbone_tolerance(case5_flow):
    """LLM 把 backbone 参数一起带回：任一队列组合是子集 → 收窄到该组合。"""
    q = [{P1: True, P2: False}]
    out = al._match_queue_combo({P1: True, P2: False, "max-num-seqs": 160}, q)
    assert out == {P1: True, P2: False}


def test_match_queue_offqueue_rejected(case5_flow):
    q = [{P1: True, P2: False}]
    assert al._match_queue_combo({P1: True, P2: True}, q) is None


def test_match_queue_type_lenient():
    """True/1 同哈希：LLM 布尔回传形态宽松匹配。"""
    q = [{"a": 1, "b": 2}]
    assert al._match_queue_combo({"a": True, "b": 2}, q) == {"a": 1, "b": 2}


def test_eff_priors_union_dedup(case5_flow):
    """手写 ∪ KG（domain 节点同源提取），frozenset 去重。"""
    state = {"domain": {"kg_interactions": [
        {"params": [P2, "max-num-batched-tokens"], "evidence": "x"},
        {"params": [P1, P2], "evidence": "y"},  # 与手写第一组同集合 → 去重
    ]}}
    priors = al._eff_priors(case5_flow, state)
    flat = [tuple(sorted(g)) for g in priors]
    assert flat.count(tuple(sorted([P1, P2]))) == 1
    assert tuple(sorted([P2, "max-num-batched-tokens"])) in flat


def test_enter_composition_and_interactive_best(case5_flow):
    """先验组两端 keep → 组合幕入场：交互子集圈出、队列=flag 笛卡尔积、组合最优提取。"""
    state = {"rounds": [
        {"round": "r1", "action": {"param": P1, "value": True}, "kept": True,
         "objective": {"value": 13.0}, "metrics": {"ttft_ms": 13.0}},
        {"round": "r2", "action": {"param": P2, "value": True}, "kept": True,
         "objective": {"value": 12.8}, "metrics": {"ttft_ms": 12.8}},
    ], "best": None, "current_params": dict(case5_flow.params.defaults),
        "baseline": {"ttft_ms": 14.3}, "phase": "value"}
    cfg = SimpleNamespace(max_combo_rounds=6, target="ttft")
    al._enter_composition(case5_flow, cfg, state)
    assert state["phase"] == "composition"
    assert state["composition_plan"]["interactive"] == sorted([P1, P2])
    assert len(state["composition_queue"]) == 4  # 2 flag × 2 flag

    # 无 kept 组合轮 → 交互最优退化为单参数 keep 值
    ib = al._interactive_best(state, [P1, P2])
    assert ib == {P1: True, P2: True}

    # kept 组合轮 → objective value 最小的组合胜出（统一最小化语义）
    state["rounds"].append(
        {"round": "c1", "action": {"params": {P1: True, P2: False}}, "kept": True,
         "objective": {"value": 12.6}, "metrics": {"ttft_ms": 12.6}})
    state["rounds"].append(
        {"round": "c2", "action": {"params": {P1: False, P2: True}}, "kept": True,
         "objective": {"value": 12.9}, "metrics": {"ttft_ms": 12.9}})
    assert al._interactive_best(state, [P1, P2]) == {P1: True, P2: False}


def test_enter_composition_budget_truncation(case5_flow):
    """组合候选超预算 → 截断 + 大声声明（截断非静默丢弃）。"""
    state = {"rounds": [
        {"round": "r1", "action": {"param": P1, "value": True}, "kept": True,
         "objective": {"value": 13.0}, "metrics": {"ttft_ms": 13.0}},
        {"round": "r2", "action": {"param": P2, "value": True}, "kept": True,
         "objective": {"value": 12.8}, "metrics": {"ttft_ms": 12.8}},
    ], "best": None, "current_params": dict(case5_flow.params.defaults),
        "baseline": {"ttft_ms": 14.3}, "phase": "value"}
    al._enter_composition(case5_flow, SimpleNamespace(max_combo_rounds=2, target="ttft"),
                          state)
    assert len(state["composition_queue"]) == 2


def test_enter_composition_no_interaction_skips(case5_flow):
    """先验组未命中（kept 只含组外参数）→ 无交互子集 → phase=done 跳过组合幕。"""
    state = {"rounds": [
        {"round": "r1", "action": {"param": "max-num-seqs", "value": 160},
         "kept": True, "objective": {"value": 12.5}, "metrics": {"ttft_ms": 12.5}},
    ], "best": None, "current_params": dict(case5_flow.params.defaults),
        "baseline": {"ttft_ms": 14.3}, "phase": "value"}
    al._enter_composition(case5_flow, SimpleNamespace(max_combo_rounds=6, target="ttft"),
                          state)
    assert state["phase"] == "done"
    assert not state.get("composition_queue")


def test_combo_rounds_filter():
    state = {"rounds": [
        {"round": "r1"}, {"round": "c1"}, {"round": "c2"}, {"round": "sweep"}]}
    assert [r["round"] for r in al._combo_rounds(state)] == ["c1", "c2"]


def test_run_composition_phase_orchestration(case5_flow, monkeypatch):
    """组合幕主循环编排：LLM 选队列组合 → 收窄匹配 → try_action → 队列弹出 → 软终止出口。

    try_action/agent_gate 打桩（真机 apply/verify 已由 E2E 覆盖），此处锁编排契约。
    """
    q = [{P1: True, P2: True}, {P1: False, P2: True}]
    state = {"rounds": [], "phase": "composition", "composition_queue": [dict(c) for c in q],
             "composition_backbone": {}, "composition_plan": {"interactive": [P1, P2]},
             "best": None, "current_params": {}, "baseline": {"ttft_ms": 14.3}}
    calls = {"decisions": [{"action": {"params": {P1: True, P2: True, "max-num-seqs": 160}}},
                           {"action": {}, "reason": "剩余组合必劣化"}], "i": 0}

    def fake_consume(ctx):
        d = calls["decisions"][min(calls["i"], len(calls["decisions"]) - 1)]
        calls["i"] += 1
        return d

    def fake_try(flow, cfg, st, action, rk):
        return {"round": rk, "action": action, "changed": list(action.get("params", {})),
                "kept": True, "decision": "keep", "metrics": {"ttft_ms": 13.0},
                "objective": {"value": 13.0}, "rationale": "stub"}

    monkeypatch.setattr(al.agent_gate, "consume_result", fake_consume)
    monkeypatch.setattr(al, "try_action", fake_try)
    cfg = SimpleNamespace(max_combo_rounds=6, target="ttft")
    ret = al._run_composition_phase(case5_flow, cfg, state)
    assert ret is None  # 正常走完（软终止出口，非暂停）
    c_recs = [r for r in state["rounds"] if r["round"].startswith("c")]
    # c1：带回 backbone 的回传被收窄到队列组合本身（max-num-seqs 剥离）
    assert c_recs[0]["action"] == {"params": {P1: True, P2: True}}
    assert c_recs[0]["kept"] is True
    # c2：软终止 no_candidate 记轮
    assert c_recs[1]["decision"] == "no_candidate"
    assert state["phase"] == "done"
    assert state["composition_queue"] == [{P1: False, P2: True}]  # 已试组合弹出


def test_run_composition_phase_offqueue_rejected(case5_flow, monkeypatch):
    """队列外自造组合 → rejected_offqueue 记轮不执行，下一轮继续（不炸不逃逸）。"""
    state = {"rounds": [], "phase": "composition",
             "composition_queue": [{P1: True, P2: False}],
             "composition_backbone": {}, "composition_plan": {"interactive": [P1, P2]},
             "best": None, "current_params": {}, "baseline": {"ttft_ms": 14.3}}
    decisions = [{"action": {"params": {P1: True, P2: True}}},  # 不在队列
                 {"action": {}, "reason": "放弃"}]
    it = {"i": 0}

    def fake_consume(ctx):
        d = decisions[min(it["i"], 1)]
        it["i"] += 1
        return d

    monkeypatch.setattr(al.agent_gate, "consume_result", fake_consume)
    monkeypatch.setattr(al, "try_action",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应执行队列外组合")))
    ret = al._run_composition_phase(case5_flow, SimpleNamespace(max_combo_rounds=6, target="ttft"),
                                    state)
    assert ret is None
    assert state["rounds"][0]["decision"] == "rejected_offqueue"
    assert state["phase"] == "done"
