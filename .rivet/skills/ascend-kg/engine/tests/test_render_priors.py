"""P1-2b/P1-1b 补齐：渲染矩阵（kind×flag×fixed×compose）+ priors 产出 + 诊断节流。

渲染矩阵是 apply 落地的契约：布尔 False 省略 vs sub-param False 显式 emit、
compose 分组形态、json path patch——历史上多次踩坑点全部锁死。
"""
from types import SimpleNamespace

import engine.agent_loop as al
from engine.state import params_to_args, params_to_env, params_to_json_config
from engine.tools import priors


# ---------------- params_to_args / env / json_config 渲染矩阵 ----------------

DOMAIN = {
    "seqs": {"kind": "cli", "min": 1, "max": 512, "step": 32},
    "flag-on": {"kind": "cli", "flag": True},
    "env-flag": {"kind": "env", "flag": True},
    "env-num": {"kind": "env", "min": 0, "max": 100},
    "json-p": {"kind": "json", "path": ["ScheduleConfig", "maxBatchSize"]},
    "compilation-config": {"compose": {"cudagraph_mode": "cudagraph-mode"}},
    "additional-config": {"compose": {"ascend_compilation_config": {
        "pt2_cd": "pt2-cd-flag"}}},
    "cudagraph-mode": {"kind": "sub"},
    "pt2-cd-flag": {"kind": "sub", "flag": True},
}


def test_args_bool_false_omitted_true_rendered():
    s = params_to_args({"flag-on": True}, DOMAIN, 8000)
    assert "--flag-on" in s
    s2 = params_to_args({"flag-on": False}, DOMAIN, 8000)
    assert "--flag-on" not in s2  # 布尔 flag False 省略（vLLM 正向启用语义）


def test_args_env_sub_json_skipped():
    s = params_to_args({"env-flag": 1, "env-num": 5, "json-p": 8,
                        "cudagraph-mode": "FULL"}, DOMAIN, 8000)
    assert "--env-flag" not in s and "--env-num" not in s
    assert "--json-p" not in s and "--cudagraph-mode" not in s


def test_args_string_value_quoted():
    s = params_to_args({"seqs": 128, "flag-on": True}, DOMAIN, 8000, base_args="X")
    assert "--seqs 128" in s


def test_args_compose_flat_and_grouped():
    """compose 扁平（compilation-config）与分组（additional-config）两种形态。"""
    params = {"cudagraph-mode": "FULL_DECODE_ONLY", "pt2-cd-flag": False}
    s = params_to_args(params, DOMAIN, 8000)
    assert "'{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'" in s
    # 分组形态：sub flag 的 False 显式 emit 成 false（区别于独立 flag 的省略！）
    assert "\"pt2_cd\": false" in s


def test_args_compose_unset_omitted():
    s = params_to_args({}, DOMAIN, 8000)
    assert "compilation-config" not in s  # sub-param 未设置 → 整个 compose 省略


def test_env_flag_and_number():
    e = params_to_env({"env-flag": True, "env-num": 5}, DOMAIN)
    assert "export env-flag=1" in e and "export env-num=5" in e
    assert "export env-flag=0" in params_to_env({"env-flag": False}, DOMAIN)


def test_json_config_path_patch():
    base = {"ScheduleConfig": {"existing": 1}, "Other": True}
    out = params_to_json_config(base, {"json-p": 16}, DOMAIN)
    assert out["ScheduleConfig"]["maxBatchSize"] == 16
    assert out["ScheduleConfig"]["existing"] == 1 and out["Other"] is True
    assert base["ScheduleConfig"].get("maxBatchSize") is None  # 深拷贝不改 base


def test_json_config_missing_midlevel_created():
    out = params_to_json_config({}, {"json-p": 4}, DOMAIN)
    assert out["ScheduleConfig"]["maxBatchSize"] == 4


# ---------------- priors.build_priors（P1-1b：降级链 + 域过滤 + no-op） ----------------


def mk_cfg(**kw):
    d = dict(flow_id="f", model="/m", target="ttft", port=8000, device=0,
             workload={"input_len_avg": 184, "input_len_max": 264,
                       "output_len_avg": 116, "output_len_max": 128})
    d.update(kw)
    return SimpleNamespace(**d)


def mk_flow(domain=None, blocked=None, optix=None):
    return SimpleNamespace(
        params=SimpleNamespace(domain=domain or {}, blocked=blocked or []),
        paths=SimpleNamespace(optix_script=optix or "/x.py"))


PD = {
    "max-num-batched-tokens": {"kind": "cli", "min": 4096, "max": 32768, "step": 4096},
    "gpu-memory-utilization": {"kind": "cli", "min": 0.8, "max": 0.95, "step": 0.01},
    "locked-p": {"kind": "cli", "locked": True, "fixed": 1},
}


def test_priors_optix_hit(monkeypatch):
    monkeypatch.setattr(priors, "_run_optix", lambda cfg, script: {
        "status": "ok",
        "recommendations": [
            {"name": "MAX_NUM_BATCHED_TOKENS", "value": 16384, "reason": "r1"},
            {"name": "BLOCK_SIZE", "value": 999, "reason": "locked 会被滤"},
            {"name": "NOT_MAPPED", "value": 1, "reason": "映射外"},
        ]})
    flow = mk_flow(PD)
    pri, notes = priors.build_priors(flow, mk_cfg(), {"current_params": {}}, None)
    params = [p["param"] for p in pri]
    assert "max-num-batched-tokens" in params
    assert "locked-p" not in params            # locked 不展示
    assert all(p["value"] != 999 for p in pri)  # block-size 无 domain 条目被滤


def test_priors_out_of_domain_dropped(monkeypatch):
    """先验值越域 → 不展示（LLM 不被诱导回传非法值）+ notes 记录。"""
    monkeypatch.setattr(priors, "_run_optix", lambda cfg, script: {
        "status": "ok",
        "recommendations": [{"name": "GPU_MEMORY_UTILIZATION", "value": 1.5,
                             "reason": "超 max"}]})
    pri, notes = priors.build_priors(mk_flow(PD), mk_cfg(), {"current_params": {}}, None)
    params = {p["param"] for p in pri}
    assert "gpu-memory-utilization" not in params  # 越域 optix 先验被滤
    assert any("越域" in n for n in notes)
    # optix 提过的参数以 optix 为准（值被滤即不出现，启发式兜底不接管该参数）；
    # 另一参数由启发式正常补位
    assert "max-num-batched-tokens" in params


def test_priors_heuristic_fallback_and_noop(monkeypatch):
    """optix 失败 → 启发式表兜底；先验值 == 当前值 → no-op 跳过。"""
    monkeypatch.setattr(priors, "_run_optix", lambda cfg, script: None)
    state = {"current_params": {"gpu-memory-utilization": 0.92}}  # 启发式值同当前
    pri, notes = priors.build_priors(mk_flow(PD), mk_cfg(), state, None)
    params = {p["param"]: p for p in pri}
    assert "gpu-memory-utilization" not in params       # no-op 被滤
    assert params["max-num-batched-tokens"]["src"] == "heuristic"
    assert any("启发式" in n for n in notes)


def test_priors_evidence_prio(monkeypatch):
    """wait_top 信号 → max-num-batched-tokens prio=1 排最前。"""
    monkeypatch.setattr(priors, "_run_optix", lambda cfg, script: None)
    signals = {"wait_top_total_ms": 230, "prefill_matmul_ms": 1, "decode_matmul_ms": 2}
    pri, _ = priors.build_priors(mk_flow(PD), mk_cfg(), {"current_params": {}}, signals)
    assert pri[0]["param"] == "max-num-batched-tokens" and pri[0]["prio"] == 1


def test_priors_optix_cached_once():
    """optix 子进程按 cfg 缓存：同 run 多轮 decide 只跑一次（120s 重操作不逐轮重跑）。"""
    import json as _json
    from pathlib import Path

    priors._OPTIX_CACHE.clear()
    script = Path("/tmp/fake_optix_test.py")
    marker = Path("/tmp/fake_optix_test.marker")
    marker.unlink(missing_ok=True)
    script.write_text(
        "import json\n"
        f"open({str(marker)!r}, 'a').write('x')\n"
        "print(json.dumps({'status': 'ok', 'recommendations': []}))")
    try:
        cfg = mk_cfg()
        out1 = priors._run_optix(cfg, script)
        out2 = priors._run_optix(cfg, script)  # 第二次应命中缓存
        assert out1 == out2 == {"status": "ok", "recommendations": []}
        assert marker.read_text() == "x"  # 子进程只跑了一次
        # cfg 变化（换 target）→ 新 key → 重新跑子进程
        priors._run_optix(mk_cfg(target="throughput"), script)
        assert marker.read_text() == "xx"
    finally:
        script.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)


# ---------------- _maybe_diagnose 节流（P1-3） ----------------


def mk_flow_msot():
    return SimpleNamespace(
        flow_id="f", target="ttft",
        params=SimpleNamespace(domain={}, blocked=[]),
        paths=SimpleNamespace(msot_skill="/msot", mfu_skill=None))


SIG = {"bottleneck_class": "schedule", "hot_ops": [{"op": "aclnnX"}, {"op": "aclnnY"}],
       "mfu_pct": 30}


def test_diagnose_emit_then_throttle(monkeypatch, tmp_path):
    import json
    from engine import agent_gate, state as st
    monkeypatch.setattr(agent_gate, "TASK_FILE", tmp_path / "task.json")
    monkeypatch.setattr(agent_gate, "RESULT_FILE", tmp_path / "result.json")
    monkeypatch.setattr(st, "RUN_STATE", tmp_path / "run_state.json")

    flow, cfg = mk_flow_msot(), mk_cfg()
    state = {"rounds": [], "diagnosis_count": 0, "diagnosis": None,
             "diag_fingerprint": None}

    # 1) 首轮：emit 暂停（返回 2）
    assert al._maybe_diagnose(flow, cfg, state, SIG, "r1") == 2
    # 2) agent 回传 → 消费记账
    agent_gate.RESULT_FILE.write_text(json.dumps(
        {"task_id": "r1-diagnose", "status": "done",
         "result": {"bottleneck_class": "schedule", "summary": "host 同步等待"}}))
    assert al._maybe_diagnose(flow, cfg, state, SIG, "r1") is None
    assert state["diagnosis"]["summary"] == "host 同步等待"
    assert state["diagnosis_count"] == 1
    # 3) 同指纹 → 节流跳过（不再 emit）
    assert al._maybe_diagnose(flow, cfg, state, SIG, "r2") is None
    # 4) 指纹变化（瓶颈换类）→ 重新诊断
    sig2 = dict(SIG, bottleneck_class="compute")
    assert al._maybe_diagnose(flow, cfg, state, sig2, "r3") == 2
    agent_gate.RESULT_FILE.write_text(json.dumps(
        {"task_id": "r3-diagnose", "status": "done", "result": {"summary": "算子瓶颈"}}))
    assert al._maybe_diagnose(flow, cfg, state, sig2, "r3") is None
    # 5) 预算耗尽（≥2 次）→ 即便指纹再变也不诊断
    sig3 = dict(SIG, bottleneck_class="comm")
    assert al._maybe_diagnose(flow, cfg, state, sig3, "r4") is None
    assert state["diagnosis_count"] == 2


def test_diagnose_no_signals_or_no_skill(monkeypatch, tmp_path):
    from engine import agent_gate, state as st
    monkeypatch.setattr(agent_gate, "TASK_FILE", tmp_path / "task.json")
    monkeypatch.setattr(agent_gate, "RESULT_FILE", tmp_path / "result.json")
    monkeypatch.setattr(st, "RUN_STATE", tmp_path / "run_state.json")
    state = {"rounds": [], "diagnosis_count": 0}
    cfg = mk_cfg()
    assert al._maybe_diagnose(mk_flow_msot(), cfg, state, None, "r1") is None
    no_skill = SimpleNamespace(params=SimpleNamespace(domain={}, blocked=[]),
                               paths=SimpleNamespace(msot_skill=None, mfu_skill=None))
    assert al._maybe_diagnose(no_skill, cfg, state, SIG, "r1") is None
