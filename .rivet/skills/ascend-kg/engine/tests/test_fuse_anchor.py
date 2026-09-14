"""P0-B apply 连续失败熔断 + P1-A 失败根因摘录 + P1-B 中途漂移锚点。

核心意图（case7 教训——为什么这些行为重要）：
- 熔断：scr14-24 连锁 11 轮 apply 失败空转 1h27m（每轮烧 ~10min 健康等待），
  引擎毫无察觉直到外层会话人工干预——环境故障必须停下交人（continue/terminate），
  不该伪装成参数探索继续烧时间；
- 根因摘录：apply 失败原因进 rounds，复盘不再翻 /tmp 大日志；
- 锚点：漂移只在终局 drift_check 测（-46.5% 为时已晚），run 后半段的候选对比的
  是数小时前的基线，host 劣化被记成参数劣化——每 N 轮零重启复测当前配置，
  |漂移|>5% 即重锚基线/水位线。
"""
from types import SimpleNamespace

import pytest

import engine.agent_loop as al


def mk_flow(target="ttft"):
    return SimpleNamespace(
        flow_id="t",
        params=SimpleNamespace(domain={}, defaults={}),
        perf=SimpleNamespace(hyst=0.003, ttot_room=1.10, ttft_room=1.10),
        target=target)


def mk_cfg(target="ttft", anchor_every=5, reps=3):
    return SimpleNamespace(flow_id="t", model="/m", target=target, port=8000, device=0,
                           max_rounds=8, reps=reps, anchor_every=anchor_every)


# ---------------- P1-A：失败根因摘录 ----------------


def test_fail_reason_text_hit_and_next_line():
    """文本日志：命中失败关键词的最后一处取其行+次行（OOM/崩溃的上下文）。"""
    log = "INFO server starting\nINFO loading model\nERROR: device core dumped\nINFO exit"
    assert "ERROR: device core dumped" in al._fail_reason(log)
    assert "INFO exit" in al._fail_reason(log)  # 行 + 次行拼接


def test_fail_reason_file_path(tmp_path):
    """日志文件路径：读末 4000 字符后同样摘录（_apply 失败返回的是日志路径）。"""
    p = tmp_path / "start_r3.log"
    p.write_text("x" * 5000 + "\nERROR: NPU OOM when allocating 56GB\nbye", encoding="utf-8")
    assert "NPU OOM" in al._fail_reason(str(p))


def test_fail_reason_no_hit_falls_back_to_last_line():
    assert al._fail_reason("all fine\nnothing wrong") == "nothing wrong"


def test_fail_reason_empty():
    assert al._fail_reason(None) == ""
    assert al._fail_reason("") == ""
    assert al._fail_reason("   \n  ") == ""


# ---------------- P0-B：熔断 ----------------


def test_fuse_check_below_threshold(monkeypatch, tmp_path):
    """streak < 3：不熔断、不碰 agent_gate。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    called = []
    monkeypatch.setattr(al.agent_gate, "consume_result", lambda ctx: called.append(1))
    state = {"apply_fail_streak": 2, "rounds": []}
    assert al._fuse_check(mk_flow(), mk_cfg(), state) is None
    assert called == []  # 阈值未到，不该有任何暂停点交互


def test_fuse_check_pause_then_continue(monkeypatch, tmp_path):
    """streak=3 → 暂停（exit 2，context 带 recent_failures）；回传 continue → 清零续跑。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    results = [None]  # 第一次 consume 无结果 → emit 暂停
    emitted = {}

    def fake_consume(ctx):
        return results.pop(0) if results else None

    def fake_emit(ctx, **kw):
        ctx["pending_task_id"] = f"{ctx['round_key']}-{ctx['stage']}"
        emitted.update(kw)

    monkeypatch.setattr(al.agent_gate, "consume_result", fake_consume)
    monkeypatch.setattr(al.agent_gate, "emit_task", fake_emit)
    state = {"apply_fail_streak": 3, "rounds": [
        {"round": "scr12", "action": {"param": "a", "value": 1}, "decision": "screen",
         "kept": False, "metrics": None, "fail_reason": "ERROR: OOM"}],
        "current_params": {"a": 1}}
    assert al._fuse_check(mk_flow(), mk_cfg(), state) == 2
    assert state["status"] == "pending"
    assert emitted["context"]["fail_streak"] == 3
    assert emitted["context"]["recent_failures"][0]["fail_reason"] == "ERROR: OOM"
    assert "kill_by_port" in emitted["context"]["note"]  # 排查提示含孤儿清理脚本
    # 回传 continue：失败计数清零（再给 3 次机会），续跑
    results.append({"action": "continue", "reason": "清理了残留 EngineCore"})
    assert al._fuse_check(mk_flow(), mk_cfg(), state) is None
    assert state["apply_fail_streak"] == 0
    assert state["fuse"]["resumed"] is True and state["fuse"]["streak"] == 3


def test_fuse_check_terminate(monkeypatch, tmp_path):
    """回传 terminate → status=complete、exit 0 收尾（不写成 crash）。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    monkeypatch.setattr(al.agent_gate, "consume_result",
                        lambda ctx: {"action": "terminate", "reason": "卡坏"})
    monkeypatch.setattr(al.agent_gate, "emit_task",
                        lambda ctx, **kw: ctx.update(pending_task_id="x"))
    state = {"apply_fail_streak": 4, "rounds": []}
    assert al._fuse_check(mk_flow(), mk_cfg(), state) == 0
    assert state["status"] == "complete"
    assert state["fuse"]["terminated"] is True and state["fuse"]["streak"] == 4


def test_screen_round_failure_counts_streak(monkeypatch):
    """粗筛单轮 apply 失败：streak +1、根因摘录进 rounds、entry 记失败。"""
    flow, cfg = mk_flow(), mk_cfg()
    state = {"baseline": {"ttft_ms": 13.1, "params": {}}, "current_params": {},
             "apply_fail_streak": 0}
    monkeypatch.setattr(al, "apply_action",
                        lambda f, c, s, a, rk: {"ok": False, "log":
                                                "boot\nERROR: NPU OOM 56GB", "rolled_back": True})
    rec, entry = al._screen_round(flow, cfg, state, "max-num-seqs", 512, "scr1")
    assert state["apply_fail_streak"] == 1
    assert rec["decision"] == "screen" and rec["metrics"] is None
    assert "NPU OOM" in rec["fail_reason"]
    assert not entry["ok"] and not entry["applied"]


# ---------------- P1-B：中途漂移锚点 ----------------


def _anchor_state(ttft_prev_keep):
    """可重锚的最小 state：r1 keep 过（current_params=r1 配置）+ 基线 + 水位线。"""
    return {
        "anchor_since": 5, "anchors": None,
        "baseline": {"ttft_ms": 460.0, "ttot_ms": 1000.0, "thr_tok_s": 200.0,
                     "stats": {"ttft_rel_noise": 0.03}},
        "best": {"round": "r1", "ttft_ms": 420.0, "value": 420.0},
        "rounds": [{"round": "r1", "kept": True,
                    "metrics": {"ttft_ms": ttft_prev_keep, "ttot_ms": 980.0}}],
        "current_params": {"max-num-seqs": 256}}


def test_anchor_no_drift_records_and_returns(monkeypatch, tmp_path):
    """|漂移|≤5%：只记录锚点轮（decision=anchor + sweep 透明），不动基线/水位线。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    monkeypatch.setattr(al, "run_benchmark",
                        lambda cfg, s, rk: {"ttft_ms": 447.0, "ttot_ms": 1000.0,
                                            "stats": {"ttft_rel_noise": 0.02}})
    state = _anchor_state(440.0)  # 参考 = 最近 keep 轮实测 440 → 漂移 +1.6%
    al._maybe_anchor(mk_flow(), mk_cfg(anchor_every=5), state)
    a = state["anchors"][0]
    assert a["ok"] and a["prev"] == 440.0 and a["cur"] == 447.0
    assert a["drift_pct"] == pytest.approx(1.59, abs=0.05)
    rec = state["rounds"][-1]
    assert rec["decision"] == "anchor" and rec["sweep"] is True
    assert state["baseline"]["ttft_ms"] == 460.0   # 未重锚
    assert state["anchor_since"] == 0               # 计数已清


def test_anchor_reanchor_scales_baseline_and_best(monkeypatch, tmp_path):
    """|漂移|>5%：基线全指标 + rolling best 按漂移比缩放，r0 归档 baseline_r0。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    monkeypatch.setattr(al, "run_benchmark",
                        lambda cfg, s, rk: {"ttft_ms": 300.0, "ttot_ms": 700.0, "thr_tok_s": 280.0,
                                            "stats": {"ttft_rel_noise": 0.02}})
    state = _anchor_state(440.0)  # cur 300 / prev 440 → -31.8%，ratio=0.6818
    al._maybe_anchor(mk_flow(), mk_cfg(anchor_every=5), state)
    ratio = 300.0 / 440.0
    assert state["baseline"]["ttft_ms"] == pytest.approx(460.0 * ratio)
    assert state["baseline"]["ttot_ms"] == pytest.approx(1000.0 * ratio)
    assert state["baseline"]["thr_tok_s"] == pytest.approx(200.0 * ratio)
    assert state["baseline"]["stats"] == {"ttft_rel_noise": 0.02}  # 换锚点新鲜噪声
    assert state["best"]["ttft_ms"] == pytest.approx(420.0 * ratio)
    assert state["best"]["value"] == pytest.approx(420.0 * ratio)
    assert state["baseline_r0"]["ttft_ms"] == 460.0  # 重锚前的 r0 归档
    assert state["anchors"][-1]["drift_pct"] == pytest.approx(-31.82, abs=0.05)


def test_anchor_chain_compares_to_previous_anchor(monkeypatch, tmp_path):
    """连续锚点接力（仅当配置未变）：第二锚点与上一锚点实测比，而非 keep 轮/基线。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    monkeypatch.setattr(al, "run_benchmark",
                        lambda cfg, s, rk: {"ttft_ms": 445.0, "stats": {}})
    state = _anchor_state(440.0)
    # a1 记录的 config 与当前一致（锚点后无 keep）→ 锚点链接力成立
    state["anchors"] = [{"round": "a1", "ok": True, "cur": 443.0,
                         "config": dict(state["current_params"])}]
    al._maybe_anchor(mk_flow(), mk_cfg(anchor_every=5), state)
    a2 = state["anchors"][-1]
    assert a2["prev"] == 443.0  # 上一锚点 cur，非 keep 轮的 440
    assert a2["drift_pct"] == pytest.approx(0.45, abs=0.05)


def test_anchor_keep_between_anchors_not_drift(monkeypatch, tmp_path):
    """#2 回归：两锚之间发生 keep（配置已变）→ 不得接力上一锚点 cur——keep 改善
    被当漂移会把基线/水位线按收益比例假重锚（水位线推到不可达，之后全 rollback）。
    参照必须回退「当前配置的上次测量」= 最近 keep 轮实测。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    # a1 跑在 keep 前的差配置上（ttft 600）；keep 后当前配置实测 445 ≈ keep 轮 440。
    # 旧口径 prev=600 → -25.8% 假重锚；修后 prev=440（同配置）→ +1.1% 不重锚。
    monkeypatch.setattr(al, "run_benchmark",
                        lambda cfg, s, rk: {"ttft_ms": 445.0, "stats": {}})
    state = _anchor_state(440.0)
    state["anchors"] = [{"round": "a1", "ok": True, "cur": 600.0, "config": {}}]
    al._maybe_anchor(mk_flow(), mk_cfg(anchor_every=5), state)
    assert state["anchors"][-1]["prev"] == 440.0  # 回退最近 keep 轮实测，非 a1 的 600
    assert state["anchors"][-1]["drift_pct"] == pytest.approx(1.14, abs=0.05)
    assert state["baseline"]["ttft_ms"] == 460.0  # 未假重锚（旧口径此处已被 ×0.742）
    assert state["best"]["value"] == 420.0


def test_anchor_gates_and_bench_failure(monkeypatch, tmp_path):
    """anchor_every=0 关闭；计数未到不测；锚点压测失败记 ok=False 不重锚。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    calls = []
    monkeypatch.setattr(al, "run_benchmark",
                        lambda cfg, s, rk: (calls.append(rk),
                                            {"ttft_ms": 999.0, "stats": {}})[1])
    # 关闭
    state = _anchor_state(440.0)
    al._maybe_anchor(mk_flow(), mk_cfg(anchor_every=0), state)
    assert calls == [] and state.get("anchors") is None
    # 计数未到
    state = _anchor_state(440.0)
    state["anchor_since"] = 4
    al._maybe_anchor(mk_flow(), mk_cfg(anchor_every=5), state)
    assert calls == []
    # 压测失败（服务不可用）→ 记失败锚点，不产出 rounds 锚点轮
    monkeypatch.setattr(al, "run_benchmark", lambda cfg, s, rk: None)
    state = _anchor_state(440.0)
    al._maybe_anchor(mk_flow(), mk_cfg(anchor_every=5), state)
    assert state["anchors"][0]["ok"] is False
    assert state["rounds"][-1]["round"] == "r1"  # 无锚点轮 append


# ---------------- #7：熔断 task_id episode 序号（2026-08-29 修） ----------------


def _mk_gate(tmp_path, monkeypatch, task_id, result):
    """真实 consume_result + 临时结果文件（含指定 task_id 的陈旧/新鲜结果）。"""
    import json
    import engine.agent_loop as al
    rf = tmp_path / "agent_result.json"
    rf.write_text(json.dumps({"status": "done", "task_id": task_id, "result": result}),
                  encoding="utf-8")
    monkeypatch.setattr(al.agent_gate, "RESULT_FILE", rf)
    monkeypatch.setattr(al.agent_gate, "emit_task",
                        lambda ctx, **kw: ctx.update(pending_task_id="emit"))


def test_fuse_stale_result_not_consumed_across_episodes(monkeypatch, tmp_path):
    """#7 回归：同 run 第二次 3 连败（episode=1）时，残留的上一 episode continue
    （task_id fail3-e0-fuse）不得被自动消费——agent 必须被重新问过。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    _mk_gate(tmp_path, monkeypatch, "fail3-e0-fuse",
             {"action": "continue", "reason": "上次清理过"})
    state = {"apply_fail_streak": 3, "fuse_episode": 1, "rounds": []}
    assert al._fuse_check(mk_flow(), mk_cfg(), state) == 2  # 重新暂停，非自动续跑
    assert state["apply_fail_streak"] == 3                  # 未被陈旧 continue 清零
    assert state["pending"]["round_key"] == "fail3-e1"      # 新 episode 的 task


def test_fuse_episode_advances_after_consume(monkeypatch, tmp_path):
    """同 episode 崩溃幂等保留：消费成功 → episode 推进持久化 + streak 清零。"""
    monkeypatch.setattr(al.st, "RUN_STATE", tmp_path / "rs.json")
    _mk_gate(tmp_path, monkeypatch, "fail3-e0-fuse",
             {"action": "continue", "reason": "清理了 EngineCore"})
    state = {"apply_fail_streak": 3, "fuse_episode": 0, "rounds": []}
    assert al._fuse_check(mk_flow(), mk_cfg(), state) is None  # 同 id 消费 → continue
    assert state["apply_fail_streak"] == 0
    assert state["fuse_episode"] == 1
