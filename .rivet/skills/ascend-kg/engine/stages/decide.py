"""⑥决策：滞回保留/回滚 + 终态健康兜底 + 基线记账。"""
import shutil
import subprocess
from pathlib import Path

from engine import state as st
from engine.state import SCRIPT_DIR
from engine.stages.apply import rollback
from engine.stages.verify import smoke_test


def _ensure_healthy(cfg, state) -> bool:
    """终态健康兜底：health + smoke。不健康则依次尝试最近 2 份备份脚本（各 120s）。

    L7：最近备份本身不可用时再试次近一份——仅试 1 份在「最后动作恰是劣化源头且
    倒数第二配置健康」的场景下会放弃本可恢复的服务。
    """
    if smoke_test(cfg.backend, cfg.port, cfg.model):
        return True
    bak_scripts = sorted(st.SCRIPTS_BAK.glob("*.sh"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    for script_src in bak_scripts[:2]:
        script = Path(f"/tmp/{cfg.flow_id}_start_vllm_opt_recover.sh")
        shutil.copy2(script_src, script)
        subprocess.run(["bash", str(SCRIPT_DIR / "kill_by_port.sh"), str(cfg.port), "15",
                        cfg.backend.process_pattern(cfg.port) or ""],
                       capture_output=True, text=True)
        subprocess.Popen(["bash", str(script)], stdout=open("/tmp/recover.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        from engine.stages.apply import _wait_health
        if _wait_health(cfg, cfg.port, 120):
            return True
        print(f"[recover] 备份 {script_src.name} 未恢复健康，尝试下一份", flush=True)
    return False


def decide(flow, cfg, state, round_key: str, metrics: dict, baseline: dict,
           candidate: dict, graph_ok: bool | None = None) -> dict:
    """返回 {decision: keep|rollback|no_candidate|rollback_failed|unhealthy, rationale, improved_pct}。

    graph_ok：图捕获校验结果（verify 产出）。False = 配置声称开图但被平台跳过/未捕获，
    配置未按预期落地 → 即使主指标改善也判 rollback（与 agent_loop.try_action 的强制回滚同守）。
    滞回判定（主指标改善 ≥ hyst + 护栏放行）与 agent_loop 共用 tools/decide.keep_verdict，
    保证两条入口判定一致；目标感知见 flow.target。
    """
    from engine.tools.decide import keep_verdict, primary_key

    key = primary_key(flow, cfg.target)
    bb = (baseline or {}).get(key)
    cur = (metrics or {}).get(key) if metrics else None
    if not bb or bb <= 0:
        return {"decision": "unhealthy", "rationale": f"基线 {key}<=0，非法",
                "improved_pct": 0.0, "healthy": False}
    kept, improved_pct, keep_why = keep_verdict(flow, metrics, baseline, cfg.target)

    if candidate.get("candidate") is None:
        decision = "no_candidate"
        rationale = candidate.get("rationale", "无候选")
    elif graph_ok is False:
        # 图捕获未生效：配置没按预期落地（如 cudagraph_mode 被降级/跳过），性能改善不可信 → 判回滚
        decision = "rollback"
        rationale = "图捕获未生效（模式被跳过/未捕获），配置未落地，判回滚"
        if not rollback(flow, cfg, state, round_key):
            rationale += "；⚠️ 回滚脚本缺失"
            decision = "rollback_failed"
    elif kept:
        decision = "keep"
        rationale = keep_why
    else:
        decision = "rollback"
        rationale = keep_why
        if not rollback(flow, cfg, state, round_key):
            rationale += "；⚠️ 回滚脚本缺失"
            decision = "rollback_failed"

    # 终态健康兜底
    healthy = _ensure_healthy(cfg, state)
    if not healthy:
        decision = "unhealthy"
        rationale += "；⚠️ 主服务无法恢复健康"

    return {"decision": decision, "rationale": rationale, "improved_pct": improved_pct,
            "healthy": healthy}


def run(flow, ctx):
    """决策本轮结果，产出写 ctx["verdict"]。"""
    cfg = ctx["cfg"]
    state = ctx["state"]
    round_key = ctx["round_key"]
    metrics = ctx.get("metrics")
    baseline = ctx.get("baseline")
    cand = ctx.get("cand") or {}
    ctx["verdict"] = decide(flow, cfg, state, round_key, metrics, baseline, cand,
                            graph_ok=ctx.get("graph_ok"))
