"""①采集：启 profiling 实例(隔离卡/端口) → 驱动请求 → 精确销毁 → 定位 ASCEND_PROFILER_OUTPUT。

隔离原则：profiling 永远在 profile_device/profile_port，与主服务物理+端口隔离。
"""
import shutil
import subprocess
import time
from pathlib import Path

from engine.state import render, TEMPLATE_DIR, SCRIPT_DIR, params_to_args, params_to_env


def _ensure_kernel_details(raw_dir: Path) -> bool:
    """kernel_details.csv 缺失时补转。本版本 profiler 退出时自动产出 ASCEND_PROFILER_OUTPUT；
    若缺失，探测 torch_npu 转换入口。"""
    out_dir = raw_dir / "ASCEND_PROFILER_OUTPUT"
    if (out_dir / "kernel_details.csv").exists():
        return True
    probes = [
        "import torch_npu.profiler as p; p.analyse('{raw}')",
        "import torch_npu.profiler.analysis as a; a.prof_parse.profiling_analyse('{raw}')",
    ]
    for code in probes:
        try:
            subprocess.run(
                ["python", "-c", code.format(raw=raw_dir)],
                capture_output=True, text=True, timeout=120)
            if (out_dir / "kernel_details.csv").exists():
                return True
        except Exception:
            continue
    return (out_dir / "kernel_details.csv").exists()


def run(flow, ctx):
    """执行一轮采集，产出写 ctx["out_dir"]（失败 None，可降级，不抛）。"""
    cfg = ctx["cfg"]
    state = ctx["state"]
    round_key = ctx["round_key"]
    # profiling 开关（DESIGN-multi-backend §3.1）：flow 声明 profiling.enabled=false
    # 即整体跳过采集（后端无 profiling 通道时不空转）
    prof = getattr(flow, "profiling", None)
    if prof is not None and not getattr(prof, "enabled", True):
        print("[capture] flow profiling.enabled=false，跳过采集（signals=None）", flush=True)
        ctx["out_dir"] = None
        return
    # H4 熔断：连续 2 轮失败后本 run 后续轮直接跳过——profiling 实例起不来时
    # 每轮都固定空等 wait 超时（纯浪费），且失败原因（占卡/资源不足）轮间不会自愈
    if state.get("capture_fail_streak", 0) >= 2:
        print("[capture] 连续 2 轮失败，本轮跳过（熔断，signals=None）", flush=True)
        ctx["out_dir"] = None
        return
    vllm_prof_dir = Path(flow.paths.vllm_prof_dir)

    # 1) 清空采集目录，使"最新"无歧义
    if vllm_prof_dir.exists():
        shutil.rmtree(vllm_prof_dir)
    vllm_prof_dir.mkdir(parents=True, exist_ok=True)

    # 2) 渲染 profiling 实例启动脚本 + 请求脚本（/tmp 名按 flow 前缀隔离，DESIGN §3.4）
    prof_script = Path(f"/tmp/{flow.flow_id}_start_vllm_profile_{round_key}.sh")
    render(TEMPLATE_DIR / "start_vllm_profile.sh.j2", prof_script,
           DEVICE=cfg.profile_device, PORT=cfg.profile_port, MODEL=cfg.model,
           PROF_DIR=vllm_prof_dir,
           VLLM_ARGS=params_to_args(state.get("current_params", flow.params.defaults),
                                    flow.params.domain, cfg.profile_port),
           VLLM_ENV=params_to_env(state.get("current_params", flow.params.defaults),
                                  flow.params.domain))
    req_script = Path(f"/tmp/{flow.flow_id}_prof_req_{round_key}.py")
    wl = cfg.workload
    input_base = wl.get("input_len_min", wl.get("input_len_avg", 184) - 80)
    step = max(10, (wl.get("input_len_max", 264) - input_base) // 4)
    render(TEMPLATE_DIR / "prof_req.py", req_script,
           BASE=f"http://localhost:{cfg.profile_port}/v1/completions", MODEL=cfg.model,
           INPUT_BASE=input_base, INPUT_STEP=step, OUTPUT_MAX=wl.get("output_len_max", 128))

    # 3) 启动 profiling 实例
    log = vllm_prof_dir / f"profile_{round_key}.log"
    subprocess.Popen(["bash", str(prof_script)], stdout=open(log, "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)

    # 4) 等端口就绪（H4：240s——profiling 实例同构冷启 ~90s + 图捕获/compile 余量；
    #    叠加连续失败熔断，最坏浪费被压住）
    try:
        wait = subprocess.run(["bash", str(SCRIPT_DIR / "wait_port.sh"),
                               str(cfg.profile_port), "240",
                               cfg.backend.health_url(cfg.flow, cfg.profile_port)],
                              capture_output=True, text=True, timeout=280)
    except subprocess.TimeoutExpired:
        wait = None
    if wait is None or wait.returncode != 0:
        subprocess.run(["bash", str(SCRIPT_DIR / "kill_by_port.sh"), str(cfg.profile_port),
                        "5", cfg.backend.process_pattern(cfg.profile_port) or ""])
        state["capture_fail_streak"] = state.get("capture_fail_streak", 0) + 1
        print(f"[capture] profiling 实例未就绪（连续失败 {state['capture_fail_streak']}）",
              flush=True)
        ctx["out_dir"] = None
        return

    # 5) 在线触发 profiling：POST /start_profile（torch_npu profiler 需显式 start/stop 才落盘）
    import urllib.request

    def _post(endpoint, timeout=30):
        req = urllib.request.Request(f"http://localhost:{cfg.profile_port}{endpoint}",
                                     data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status

    try:
        print(f"[capture] /start_profile -> {_post('/start_profile', 20)}", flush=True)
    except Exception as e:
        print(f"[capture] /start_profile FAIL: {e}", flush=True)

    # 6) 驱动请求触发计算
    try:
        r = subprocess.run(["python", str(req_script)], capture_output=True, text=True, timeout=200)
        print(f"[capture] prof_req rc={r.returncode}, tail={r.stdout.strip().splitlines()[-2:]}", flush=True)
    except subprocess.TimeoutExpired:
        print("[capture] prof_req timeout", flush=True)

    # 7) 给 profiler 刷 trace 的时间
    time.sleep(8)

    # 8) 停止 profiling（flush trace 到 torch_profiler_dir）。
    #    重负载下 stop 可能 >30s 才返回，超时不代表失败——宽限等待后由 kill 兜底 flush。
    stop_ok = False
    for attempt in range(2):
        try:
            print(f"[capture] /stop_profile -> {_post('/stop_profile', 90)}", flush=True)
            stop_ok = True
            break
        except Exception as e:
            print(f"[capture] /stop_profile attempt{attempt + 1} FAIL: {e}", flush=True)
            time.sleep(10)
    if not stop_ok:
        time.sleep(20)

    # 9) 精确销毁 profiling 实例（触发 profiler 退出自动 analyse）
    subprocess.run(["bash", str(SCRIPT_DIR / "kill_by_port.sh"), str(cfg.profile_port),
                    "10", cfg.backend.process_pattern(cfg.profile_port) or ""],
                   capture_output=True, text=True)

    # 10) 定位输出并补转（H4：成功复位熔断计数，失败累计）
    raw_dir = None
    for _ in range(3):
        cands = sorted(vllm_prof_dir.glob("rank*_ascend_pt"),
                       key=lambda x: x.stat().st_mtime, reverse=True)
        if cands and _ensure_kernel_details(cands[0]):
            raw_dir = cands[0]
            break
        time.sleep(3)
    if raw_dir:
        state["capture_fail_streak"] = 0
        ctx["out_dir"] = str(raw_dir / "ASCEND_PROFILER_OUTPUT")
    else:
        state["capture_fail_streak"] = state.get("capture_fail_streak", 0) + 1
        print(f"[capture] profiling 产出缺失（连续失败 {state['capture_fail_streak']}）",
              flush=True)
        ctx["out_dir"] = None
