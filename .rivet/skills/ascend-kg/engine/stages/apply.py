"""④应用：渲染启动脚本 → 备份上轮健康脚本+参数 → 精确 kill → 重启 → 健康门禁 → 失败回滚。

渲染/健康路径/kill 兜底签名经 cfg.backend 分派（DESIGN-multi-backend §3.2），
本模块不再硬编码 vLLM 细节。
"""
import json
import shutil
import subprocess
from pathlib import Path

from engine import state as st
from engine.state import SCRIPT_DIR


def _kill(cfg, port: int, grace: str = "15"):
    """精确杀端口进程（红线：一律 kill_by_port.sh，绝不宽模式 pkill）。

    兜底签名由 backend.process_pattern 注入；None → 传空串禁用兜底。
    """
    pattern = cfg.backend.process_pattern(port)
    return subprocess.run(["bash", str(SCRIPT_DIR / "kill_by_port.sh"),
                           str(port), grace, pattern or ""],
                          capture_output=True, text=True, timeout=60)


def _wait_health(cfg, port: int, timeout: int = 300) -> bool:
    health = cfg.backend.health_url(cfg.flow, port)
    try:
        r = subprocess.run(["bash", str(SCRIPT_DIR / "wait_port.sh"), str(port),
                            str(timeout), health],
                           capture_output=True, text=True, timeout=timeout + 20)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def restart_to_defaults(flow, cfg, state) -> bool:
    """强制把主服务重启回 defaults 配置，用于 round 0 建基线。

    端口上若残留上轮 keep 的候选服务，直接拿它压测会把候选性能误记为 defaults 基线，
    污染后续所有 keep/rollback 判定。这里一律现场渲染 defaults 基线脚本并重启。
    返回服务是否健康；成功时更新 state.current_script / current_params。
    """
    script = cfg.backend.render_start_script(flow, cfg, flow.params.defaults, "baseline")
    _kill(cfg, cfg.port)
    subprocess.Popen(["bash", str(script)],
                     stdout=open(cfg.backend.log_path(flow, "baseline"), "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    state["current_script"] = str(script)
    state["current_params"] = dict(flow.params.defaults)
    if _wait_health(cfg, cfg.port, 300):
        return True
    print(f"FATAL: 基线服务启动失败（见 {cfg.backend.log_path(flow, 'baseline')}）", flush=True)
    return False


def _apply(flow, cfg, state, round_key: str, new_params: dict) -> dict:
    """应用参数：重启主服务。返回 {ok, rolled_back, script, log}。"""
    st.SCRIPTS_BAK.mkdir(parents=True, exist_ok=True)
    script = cfg.backend.render_start_script(flow, cfg, new_params, round_key)

    # 备份上轮健康脚本 + 对应参数（回滚时脚本与参数必须一致恢复）
    prev = Path(state.get("current_script") or str(cfg.backend.script_path(flow, "baseline")))
    if prev.exists():
        bak = st.SCRIPTS_BAK / f"{round_key}_prev.sh"
        shutil.copy2(prev, bak)
        bak_json = st.SCRIPTS_BAK / f"{round_key}_prev.json"
        bak_json.write_text(json.dumps(state.get("current_params", flow.params.defaults),
                                       ensure_ascii=False), encoding="utf-8")

    # 精确杀主服务（端口作用域，绝不宽模式 pkill）。
    # 失败判定看退出码（M4）：kill_by_port 失败即 exit 1；子串匹配 "ERR" 有误判面。
    r = _kill(cfg, cfg.port)
    if r.returncode != 0:
        return {"ok": False, "rolled_back": False, "script": str(script),
                "log": r.stdout + r.stderr}

    # 重启
    log_path = cfg.backend.log_path(flow, round_key)
    p = subprocess.Popen(["bash", str(script)], stdout=open(log_path, "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    state["current_script"] = str(script)
    state["current_pid"] = p.pid
    state["current_params"] = new_params

    if _wait_health(cfg, cfg.port, 300):
        return {"ok": True, "rolled_back": False, "script": str(script), "log": str(log_path)}
    return {"ok": False, "rolled_back": False, "script": str(script), "log": str(log_path)}


def rollback(flow, cfg, state, round_key: str) -> bool:
    """回滚：恢复 SCRIPTS_BAK 里上一轮健康脚本+参数并重启。"""
    bak = st.SCRIPTS_BAK / f"{round_key}_prev.sh"
    script = bak if bak.exists() else cfg.backend.script_path(flow, "baseline")
    if not script.exists():
        # 兜底：现场渲染基线脚本（防历史遗留脚本缺参数）
        script = cfg.backend.render_start_script(flow, cfg, flow.params.defaults, "baseline")
    restored = cfg.backend.script_path(flow, "rollback")
    shutil.copy2(script, restored)
    # 恢复对应参数（脚本与参数必须一起恢复）
    bak_json = st.SCRIPTS_BAK / f"{round_key}_prev.json"
    if bak_json.exists():
        state["current_params"] = json.loads(bak_json.read_text(encoding="utf-8"))
    else:
        state["current_params"] = dict(flow.params.defaults)
    _kill(cfg, cfg.port)
    p = subprocess.Popen(["bash", str(restored)],
                         stdout=open(cfg.backend.log_path(flow, "rollback"), "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    state["current_script"] = str(restored)
    state["current_pid"] = p.pid
    return _wait_health(cfg, cfg.port, 300)


def run(flow, ctx):
    """应用本轮候选参数；无候选则跳过。致命失败写 ctx["fatal"] 交由 run.py 中断。"""
    cand = ctx.get("cand") or {}
    if cand.get("candidate") is None:
        return
    cfg = ctx["cfg"]
    state = ctx["state"]
    round_key = ctx["round_key"]
    app = _apply(flow, cfg, state, round_key, cand["new_params"])
    if not app["ok"]:
        print(f"[apply] 应用失败: {app['log'][-500:]}", flush=True)
        ctx["fatal"] = True
        ctx["fatal_reason"] = f"apply 失败: {app['log'][-200:]}"
        return
    print(f"[apply] 已重启 {app['script']} (pid {state['current_pid']})", flush=True)
    ctx["round_rec"]["changed"] = [cand["candidate"]]
