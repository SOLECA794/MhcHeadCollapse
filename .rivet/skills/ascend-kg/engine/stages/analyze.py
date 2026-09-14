"""②分析：定位最新 ASCEND_PROFILER_OUTPUT → analyze_prof.py → 解析瓶颈信号。"""
import re
import subprocess
from pathlib import Path

from engine import state as st
from engine.state import TEMPLATE_DIR


def _parse_signals(text: str) -> dict:
    s = {}
    m = re.search(r"估算 MFU:\s*([\d.]+)%", text)
    if m:
        s["mfu_pct"] = float(m.group(1))
    m = re.search(r"采集窗口.*?([\d.]+)s", text)
    if m:
        s["wall_s"] = float(m.group(1))
    m = re.search(r"大 matmul.*?(\d+)个.*?([\d.]+)ms", text)
    if m:
        s["prefill_matmul_cnt"] = int(m.group(1))
        s["prefill_matmul_ms"] = float(m.group(2))
    m = re.search(r"小 matmul.*?(\d+)个.*?([\d.]+)ms", text)
    if m:
        s["decode_matmul_cnt"] = int(m.group(1))
        s["decode_matmul_ms"] = float(m.group(2))
    # 热点算子
    hot = []
    for line in text.splitlines():
        if re.match(r"^aclnnMatmul|^FusedInfer|^SwiGlu|^AddRmsNorm", line) and "Total(ms)" not in line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    hot.append({"op": parts[0], "count": int(parts[1]),
                                "total_ms": float(parts[2]), "pct": float(parts[3])})
                except ValueError:
                    pass
    s["hot_ops"] = hot
    # Wait Top：第一个 wait 总时长（下发瓶颈信号）
    wait_lines = [l for l in text.splitlines() if re.match(r"^\s+(\S+): 总等待", l)]
    if wait_lines:
        s["wait_top_total_ms"] = float(re.search(r"总等待 ([\d.]+)ms", wait_lines[0]).group(1))
    return s


def _classify_bottleneck(signals: dict) -> str:
    """确定性粗分类：wait 高 → schedule；MFU 低 → compute；否则 unknown。

    精诊断（哪个算子、为什么慢、怎么优化）交给 diagnose(LLM)。
    """
    wait = signals.get("wait_top_total_ms", 0)
    if wait and wait > 100:
        return "schedule"
    mfu = signals.get("mfu_pct")
    if mfu is not None and mfu < 50:
        return "compute"
    return "unknown"


def run(flow, ctx):
    """运行 analyze_prof.py 并解析信号；失败写 ctx["signals"]=None（降级 optix-only）。"""
    cfg = ctx["cfg"]
    round_key = ctx["round_key"]
    out_dir = ctx.get("out_dir")
    if not out_dir:
        ctx["signals"] = None
        return
    kd = Path(out_dir) / "kernel_details.csv"
    if not kd.exists():
        ctx["signals"] = None
        return
    st.PER_ROUND_DIR.mkdir(parents=True, exist_ok=True)
    txt = st.PER_ROUND_DIR / f"{round_key}.analyze.txt"
    analyze_script = TEMPLATE_DIR / "analyze_prof.py"
    wl = cfg.workload
    pre_tok = wl.get("input_len_avg", 184) * 5
    dec_tok = wl.get("output_len_avg", 116) * 5
    try:
        r = subprocess.run(["python", str(analyze_script), str(kd), str(txt),
                            str(pre_tok), str(dec_tok)], capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        r = None
    if r is None or r.returncode != 0 or not txt.exists():
        ctx["signals"] = None
        return
    signals = _parse_signals(txt.read_text(encoding="utf-8"))
    # L2：正则失配不静默——关键信号全空时大声告警（上游 analyze_prof.py 输出格式
    # 变化会让信号悄悄变 None，无此告警则质量下滑无人知晓）
    if (signals.get("mfu_pct") is None and not signals.get("wait_top_total_ms")
            and not signals.get("hot_ops")):
        print("[analyze] ⚠️ 信号解析为空：疑似 analyze_prof.py 输出格式变化（正则失配），"
              "本轮按无信号降级", flush=True)
    signals["analysis_available"] = True
    signals["bottleneck_class"] = _classify_bottleneck(signals)
    ctx["signals"] = signals
