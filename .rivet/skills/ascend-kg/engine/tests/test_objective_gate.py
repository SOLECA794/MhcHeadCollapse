"""P1-2b 补齐：objective 四分支 + agent_gate 协议 + state schema 迁移。

核心意图：objective 的 feasible=False 路径必须「无 value」（下游 best 排序不
消费不可接受点）；A 类精度惩罚在两个目标方向上都「变差」（统一最小化语义）；
agent_gate 的 task_id 校验防陈旧结果串轮（resume 幂等基石）。
"""
import json
from types import SimpleNamespace

import pytest

from engine.tools.objective import compute_objective
from engine.cli import _apply_guard_policy, parse_args
from engine import agent_gate
from engine.state import validate_and_migrate, snapshot_state, SCHEMA_VERSION


def mk_flow(**perf):
    target = perf.pop("target", "ttft")  # target 是 flow 属性，不进 perf
    base = {"hyst": 0.003}
    base.update(perf)
    return SimpleNamespace(perf=SimpleNamespace(**base), target=target)


BASE = {"ttft_ms": 100.0, "ttot_ms": 100.0}


# ---------------- compute_objective 四分支 ----------------


def test_objective_ok_value_minimize():
    obj = compute_objective(mk_flow(), {"ttft_ms": 95.0, "ttot_ms": 100.0}, BASE)
    assert obj["feasible"] and obj["kind"] == "ok"
    assert obj["value"] == 95.0  # ttft → value = 原值（最小化）


def test_objective_guard_violated_no_value():
    """TTOT 退化超容限 → feasible=False 且 value=None（不可接受点不进排序）。"""
    obj = compute_objective(mk_flow(), {"ttft_ms": 95.0, "ttot_ms": 130.0}, BASE)
    assert not obj["feasible"] and obj["value"] is None
    assert obj["kind"] == "guard_violated"


def test_objective_guard_widened_by_noise():
    """P0-A guard 噪声展宽：吞吐目标 + 主指标大改善，TTFT 退化 17.6% 超配置容限
    10%，但 TTFT 自身噪声 15% → 容限展宽到 22.5% → 放行并记 guard_widened。
    case7 r11 教训：吞吐 +8.0% 被 TTFT +11.4% 拦，而同配置 TTFT 摆动 10pt+——
    阈值小于噪声的守门是随机拦截。"""
    flow = mk_flow(target="throughput", ttft_room=1.10, noise_k=1.5)
    base = {"ttft_ms": 100.0, "ttot_ms": 100.0, "thr_tok_s": 500.0,
            "stats": {"ttft_rel_noise": 0.15}}
    cur = {"ttft_ms": 117.6, "ttot_ms": 100.0, "thr_tok_s": 540.0,
           "stats": {"ttft_rel_noise": 0.15}}
    obj = compute_objective(flow, cur, base)
    assert obj["feasible"] and obj["kind"] == "ok"
    assert obj["components"].get("guard_widened") is True
    assert obj["components"].get("guard_limit_room") == pytest.approx(1.225, rel=1e-3)


def test_objective_guard_widening_still_blocks_beyond_floor():
    """展宽只放宽退化上限，不取消护栏：退化 30% > max(10%, 22.5%) 仍被拦。"""
    flow = mk_flow(target="throughput", ttft_room=1.10, noise_k=1.5)
    base = {"ttft_ms": 100.0, "ttot_ms": 100.0, "thr_tok_s": 500.0,
            "stats": {"ttft_rel_noise": 0.15}}
    cur = {"ttft_ms": 130.0, "ttot_ms": 100.0, "thr_tok_s": 540.0,
           "stats": {"ttft_rel_noise": 0.15}}
    obj = compute_objective(flow, cur, base)
    assert not obj["feasible"] and obj["kind"] == "guard_violated"


def test_objective_guard_not_widened_without_noise():
    """无 stats（旧 state resume）→ 地板 0，容限原样 10%，17.6% 退化照拦——零回归。"""
    flow = mk_flow(target="throughput", ttft_room=1.10)
    base = {"ttft_ms": 100.0, "ttot_ms": 100.0, "thr_tok_s": 500.0}
    cur = {"ttft_ms": 117.6, "ttot_ms": 100.0, "thr_tok_s": 540.0}
    obj = compute_objective(flow, cur, base)
    assert not obj["feasible"] and obj["kind"] == "guard_violated"
    assert "guard_widened" not in obj["components"]


def test_objective_no_metric():
    obj = compute_objective(mk_flow(), {"ttot_ms": 100.0}, BASE)
    assert not obj["feasible"] and obj["value"] is None and obj["kind"] == "no_metric"


def test_objective_accuracy_B_hard_veto():
    obj = compute_objective(mk_flow(), {"ttft_ms": 95.0, "ttot_ms": 100.0}, BASE,
                            accuracy_match=False,
                            accuracy_detail={"cls": "B"})
    assert not obj["feasible"] and obj["value"] is None
    assert obj["kind"] == "accuracy_B"


def test_objective_accuracy_A_penalty_direction():
    """A 类数值漂移：惩罚必须让 value 变差（增大）而非变好——防惩罚反向。"""
    obj = compute_objective(mk_flow(), {"ttft_ms": 95.0, "ttot_ms": 100.0}, BASE,
                            accuracy_match=False, accuracy_detail={"cls": "A"})
    assert obj["feasible"] and obj["kind"] == "accuracy_A"
    assert obj["value"] > 95.0  # ttft 口径：惩罚向上


def test_objective_accuracy_A_penalty_throughput_direction():
    """吞吐口径（value=-thr，越大越优）：A 类惩罚同样向变差方向（value 增大 → 更接近 0）。"""
    flow = SimpleNamespace(perf=SimpleNamespace(hyst=0.003, accuracy_penalty=0.01),
                           target="throughput")
    base = {"thr_tok_s": 200.0, "ttft_ms": 100.0}
    obj = compute_objective(flow, {"thr_tok_s": 220.0, "ttft_ms": 100.0}, base,
                            accuracy_match=False, accuracy_detail={"cls": "A"})
    assert obj["feasible"]
    # value=-220；惩罚 = value + |value|*eps = -220+2.2 = -217.8（向 0 移动 = 变差方向）
    assert obj["value"] == pytest.approx(-220.0 + 220.0 * 0.01)


# ---------------- 业务画像 guard 放宽（--profile，case7 r11 教训） ----------------

def test_guard_policy_batch_passes_r11_mirror():
    """case7 r11 镜像：吞吐 +8.0%（417.1→450.5）、TTFT +11.4%（33.98→37.84）。
    online：37.84 > 33.98×1.10=37.38 被拦（复现案例事实）；batch：守门退化为
    灾难档 1.5 → 放行，components 记 guard_policy 与 guard_widened 分账。"""
    base = {"thr_tok_s": 417.1, "ttft_ms": 33.98}
    cur = {"thr_tok_s": 450.5, "ttft_ms": 37.84}

    flow_online = mk_flow(target="throughput", ttft_room=1.10)
    assert not compute_objective(flow_online, cur, base)["feasible"]  # 案例原结局

    flow_batch = mk_flow(target="throughput", ttft_room=1.10)
    _apply_guard_policy(flow_batch.perf, "batch", None)
    obj = compute_objective(flow_batch, cur, base)
    assert obj["feasible"] and obj["kind"] == "ok"
    assert obj["components"]["guard_policy"]["profile"] == "batch"
    assert "guard_widened" not in obj["components"]  # 放行靠业务契约，非统计展宽


def test_guard_policy_batch_catastrophic_still_blocks():
    """batch 只放宽到灾难档，不取消护栏：TTFT +80% 仍拦，guard_policy 透传。"""
    flow = mk_flow(target="throughput", ttft_room=1.10)
    _apply_guard_policy(flow.perf, "batch", None)
    base = {"thr_tok_s": 417.1, "ttft_ms": 33.98}
    cur = {"thr_tok_s": 450.5, "ttft_ms": 61.16}  # +80% > 1.5 灾难线
    obj = compute_objective(flow, cur, base)
    assert not obj["feasible"] and obj["kind"] == "guard_violated"
    assert obj["components"]["guard_policy"]["profile"] == "batch"


def test_guard_policy_balanced_budget_boundary():
    """balanced = 配置容限上再放宽 budget：0.15 → 上限 1.25 放行 r11 镜像；
    0.01 → 上限 1.11（37.72 < 37.84）仍拦——预算边界两侧行为可预期。"""
    base = {"thr_tok_s": 417.1, "ttft_ms": 33.98}
    cur = {"thr_tok_s": 450.5, "ttft_ms": 37.84}

    flow_b15 = mk_flow(target="throughput", ttft_room=1.10)
    _apply_guard_policy(flow_b15.perf, "balanced", 0.15)
    obj = compute_objective(flow_b15, cur, base)
    assert obj["feasible"]
    assert obj["components"]["guard_policy"] == {
        "profile": "balanced", "budget": 0.15,
        "orig": {"ttft_room": 1.10, "ttot_room": 1.10}}

    flow_b01 = mk_flow(target="throughput", ttft_room=1.10)
    _apply_guard_policy(flow_b01.perf, "balanced", 0.01)
    obj = compute_objective(flow_b01, cur, base)
    assert not obj["feasible"] and obj["kind"] == "guard_violated"


def test_apply_guard_policy_variants():
    """装配单测：online 不动无元数据（零回归）；batch 尊重 flow 的
    catastrophic_room 覆盖；balanced 双 room 对称放宽且 orig 留痕。"""
    online = SimpleNamespace(ttft_room=1.10, ttot_room=1.10)
    _apply_guard_policy(online, "online", None)
    assert online.ttft_room == 1.10 and not hasattr(online, "guard_policy")

    batch = SimpleNamespace(ttft_room=1.10, ttot_room=1.08, catastrophic_room=2.0)
    _apply_guard_policy(batch, "batch", None)
    assert batch.ttft_room == 2.0 and batch.ttot_room == 2.0
    assert batch.guard_policy["orig"]["ttot_room"] == 1.08  # 覆写前生效值可审计

    bal = SimpleNamespace(ttft_room=1.10, ttot_room=1.08)
    _apply_guard_policy(bal, "balanced", 0.05)
    assert bal.ttft_room == pytest.approx(1.15) and bal.ttot_room == pytest.approx(1.13)


def test_profile_balanced_requires_budget():
    """balanced 无预算无意义 → argparse 层拒绝（防静默按 budget=0 跑）。"""
    with pytest.raises(SystemExit):
        parse_args(["--flow", "x", "--profile", "balanced"])
    args = parse_args(["--flow", "x", "--profile", "balanced", "--ttft-budget", "0.15"])
    assert args.ttft_budget == 0.15





@pytest.fixture
def gate_files(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_gate, "TASK_FILE", tmp_path / "agent_task.json")
    monkeypatch.setattr(agent_gate, "RESULT_FILE", tmp_path / "agent_result.json")
    return tmp_path


def test_gate_emit_consume_roundtrip(gate_files):
    ctx = {"round_key": "r3", "stage": "decide"}
    agent_gate.emit_task(ctx, goal="g", skill_refs=[], schema={}, context={})
    assert ctx["pending"] and ctx["pending_task_id"] == "r3-decide"
    # 未写 result → None；写后同 task_id → 命中
    assert agent_gate.consume_result(ctx) is None
    agent_gate.RESULT_FILE.write_text(json.dumps(
        {"task_id": "r3-decide", "status": "done", "result": {"x": 1}}))
    assert agent_gate.consume_result(ctx) == {"x": 1}


def test_gate_stale_task_id_rejected(gate_files):
    """跨 task_id 的残留 result 必须拒收（防上轮结果被误消费成新决策——历史 bug 源）。"""
    agent_gate.RESULT_FILE.write_text(json.dumps(
        {"task_id": "r1-decide", "status": "done", "result": {"old": True}}))
    assert agent_gate.consume_result({"round_key": "r2", "stage": "decide"}) is None


def test_gate_corrupt_json_and_undone(gate_files):
    agent_gate.RESULT_FILE.write_text("{broken json")
    assert agent_gate.consume_result({"round_key": "r1", "stage": "decide"}) is None
    agent_gate.RESULT_FILE.write_text(json.dumps(
        {"task_id": "r1-decide", "status": "error", "result": {}}))
    assert agent_gate.consume_result({"round_key": "r1", "stage": "decide"}) is None


def test_gate_emit_clears_stale_result(gate_files):
    """emit_task 清旧 result（防 stale 被下一暂停点误读），task 幂等重写安全。"""
    agent_gate.RESULT_FILE.write_text(json.dumps({"task_id": "x", "status": "done"}))
    ctx = {"round_key": "r1", "stage": "domain"}
    agent_gate.emit_task(ctx, goal="g", skill_refs=[], schema={}, context={})
    assert not agent_gate.RESULT_FILE.exists()


# ---------------- state schema：validate_and_migrate ----------------


def test_migrate_v1_int_round_and_defaults():
    old = {"rounds": [{"round": 2, "action": {"param": "p", "value": 1},
                       "kept": True, "decision": "keep"}],
           "best": None, "status": "complete", "kg_notes": []}
    m = validate_and_migrate(old)
    assert m["rounds"][0]["round"] == "r2"  # int → str 归一写回
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["phase"] == "value" and m["composition_queue"] is None


def test_migrate_loud_errors():
    with pytest.raises(ValueError, match=r"rounds\[0\]\.round"):
        validate_and_migrate({"rounds": [{"round": ["x"], "action": {},
                                          "kept": False, "decision": "k"}]})
    # round 键缺失 = 无身份可发明，仍拒载（#4 只降级 action/kept/decision）
    with pytest.raises(ValueError, match="round 非法"):
        validate_and_migrate({"rounds": [{"action": {}, "kept": False, "decision": "k"}]})
    with pytest.raises(ValueError, match="高于本引擎"):
        validate_and_migrate({"schema_version": 99, "rounds": []})


def test_migrate_mutable_defaults_isolated():
    a = validate_and_migrate({"rounds": []})
    b = validate_and_migrate({"rounds": []})
    a["current_params"]["leak"] = 1
    assert "leak" not in b["current_params"]  # 共享默认值串污防回归


def test_snapshot_state_real_legacy(tmp_path):
    """旧 state 文件（无 schema_version）经 snapshot_state 读入不报错、行为不变。"""
    p = tmp_path / "run_state.json"
    p.write_text(json.dumps({"rounds": [{"round": 1, "action": {"param": "p", "value": 2},
                                         "kept": False, "decision": "rollback",
                                         "unknown_key_x": 1}],
                             "best": None, "status": "pending", "kg_notes": ["n"]}))
    s = snapshot_state(p)
    assert s["rounds"][0]["round"] == "r1"
    assert s["status"] == "pending"  # 语义字段原样保留


# ---------------- #3/#4：v1 state 兼容迁移（2026-08-29 修） ----------------


def _v1_state():
    """实测 v1 形态（state/_vllm_stale_* 等旧 run）：无 schema_version、指标键
    ttfb_*、rounds 无 decision、best_by_target 为空。"""
    return {
        "rounds": [
            {"round": 1, "action": {"param": "x", "value": 8}, "kept": True,
             "metrics": {"ttfb_ms": 40.6, "ttfb_p95": 55.0},
             "rationale": "ttfb_ms 44.1->40.6 (+7.9%)"},
            {"round": 2, "action": {"param": "y", "value": 4}, "kept": False,
             "metrics": {"ttfb_ms": 41.2, "ttfb_p95": 56.0}}],
        "baseline": {"ttfb_ms": 44.1, "ttot_ms": 900.0, "params": {}},
        "best": {"round": 1, "ttfb_ms": 40.6},
        "best_by_target": {"ttfb": {"round": 1, "value": 40.6}},
        "status": "complete"}


def test_migrate_ttfb_keys_recursive_and_loud(capsys):
    """#3 回归：ttfb_ms/ttfb_p95/best_by_target 目标键全容器改名 ttft*，rationale
    历史文案值不动（机器只读键，改写文案=伪造历史），命中大声打印。"""
    m = validate_and_migrate(_v1_state())
    assert m["baseline"]["ttft_ms"] == 44.1 and "ttfb_ms" not in m["baseline"]
    assert m["best"]["ttft_ms"] == 40.6
    assert m["rounds"][0]["metrics"] == {"ttft_ms": 40.6, "ttft_p95": 55.0}
    assert m["rounds"][1]["metrics"]["ttft_ms"] == 41.2
    assert m["best_by_target"]["ttft"]["value"] == 40.6  # 目标键 ttfb→ttft
    assert "ttfb_ms 44.1->40.6" in m["rounds"][0]["rationale"]  # 值不迁
    out = capsys.readouterr().out
    assert "ttfb→ttft" in out and "baseline.ttfb_ms" in out


def test_migrate_v1_missing_decision_downgraded(capsys):
    """#4 回归主场景：v1 rounds 缺 decision（实测两台旧 run 卡死点）→ 从 kept
    推导补默认 + 大声告警，不再 raise——resume 起得来，plateau 连拒等判定
    消费的 decision 语义还原正确。"""
    s = _v1_state()
    for r in s["rounds"]:
        r.pop("decision", None)
    m = validate_and_migrate(s)
    assert m["rounds"][0]["decision"] == "keep"       # kept=True 推导
    assert m["rounds"][1]["decision"] == "rollback"   # kept=False 推导
    assert "缺必填键" in capsys.readouterr().out


def test_migrate_missing_action_kept_defaults(capsys):
    """action/kept 缺 → {} / False 保守默认 + 告警（decision 仍按补后的 kept 推导）。"""
    m = validate_and_migrate({"rounds": [{"round": "r1", "decision": "rollback"}]})
    assert m["rounds"][0]["action"] == {} and m["rounds"][0]["kept"] is False
    assert "补默认" in capsys.readouterr().out


def test_snapshot_loads_real_v1_ttfb_state(tmp_path, capsys):
    """端到端：磁盘上的 v1 ttfb state 经 snapshot_state 读入 → 键已迁移、decision
    已补、可正常继续（不 raise、不裸 KeyError）。"""
    p = tmp_path / "run_state.json"
    s = _v1_state()
    s["rounds"][0].pop("decision", None)
    s["rounds"][1].pop("decision", None)
    p.write_text(json.dumps(s))
    m = snapshot_state(p)
    assert m["rounds"][0]["metrics"]["ttft_ms"] == 40.6
    assert m["rounds"][0]["decision"] == "keep"
    assert m["schema_version"] == SCHEMA_VERSION
