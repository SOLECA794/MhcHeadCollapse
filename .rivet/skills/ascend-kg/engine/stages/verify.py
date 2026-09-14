"""⑤验证：并发压测 → 解析 TTFT/TTOT + smoke 测试 + 图捕获校验。

请求面（smoke URL/body、压测模板）与图捕获判据经 cfg.backend 分派
（DESIGN-multi-backend §3.2），本模块只保留编排与解析。
"""
import re
import statistics
import subprocess
from pathlib import Path

from engine import state as st
from engine.state import render, TEMPLATE_DIR


def _mad(samples: list) -> float:
    """median absolute deviation —— 比 std 抗离群，适合小样本离散度估计。"""
    if not samples:
        return 0.0
    med = statistics.median(samples)
    return statistics.median(abs(x - med) for x in samples)


def parse_benchmark(text: str) -> list:
    """解析 'reqN: ttft=Xs total=Ys [out=Ntok]' -> 样本 dict 列表。

    样本含 {ttft_ms, ttot_ms, out_tokens}；out_tokens=None 表示探针未上报 token 数
    （旧模板/非 MindIE 后端）——调用方据此降级仅报 TTFT/TTOT，不算吞吐/TPOT/TPS。
    curl 连接失败（退出码 7）时 -s 吞错、-w 仍输出 ttft=0.000000s，丢弃这类假样本，
    否则死服务会被 is_kept 判成 ttft=0 无限优。
    """
    samples = []
    for m in re.finditer(r"req\d+: ttft=([\d.]+)s total=([\d.]+)s(?: out=(\d+)tok)?", text):
        f_ttft = float(m.group(1)) * 1000
        f_ttot = float(m.group(2)) * 1000
        if f_ttft <= 0:
            continue
        out_tokens = int(m.group(3)) if m.group(3) else None
        samples.append({"ttft_ms": f_ttft, "ttot_ms": f_ttot, "out_tokens": out_tokens})
    return samples


def _run_bench_script(cfg, round_key: str, log: Path) -> str:
    script = Path(f"/tmp/{cfg.flow_id}_concurrency_test_{round_key}.sh")
    # L4：并发数从 workload 注入（缺省 5）；模板按 backend 分派
    render(TEMPLATE_DIR / cfg.backend.bench_template(), script,
           PORT=cfg.port, MODEL=cfg.model,
           OUTPUT_MAX=cfg.workload.get("output_len_max", 128),
           TIMEOUT=cfg.bench_timeout,
           CONC=cfg.workload.get("concurrency", 5),
           PROBE=cfg.backend.probe_script())
    try:
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           timeout=cfg.bench_timeout * 5 + 30)
    except subprocess.TimeoutExpired:
        log.write_text("[bench] timeout", encoding="utf-8")
        return ""  # 空输出 → parse_benchmark 得空 → run_benchmark 返 None → 10x 基线兜底
    log.write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.stdout


def run_benchmark(cfg, state, round_key: str = "baseline") -> dict | None:
    """并发压测：1 次 warmup(丢弃, 对抗偶发慢采样) + reps 次测量。
    返回 median-of-medians 的 {ttft_ms, ttot_ms, ttft_p95, p99_ms, stats}；失败返回 None。
    stats（P0-2 统计功效）= {n_samples, ttft_mad_ms, ttft_rel_noise, ttot_rel_noise,
    thr_rel_noise?} —— MAD/median 相对噪声，keep 判定据此抬门槛（hyst_eff）。
    p99_ms = 端到端延迟（发送→完整接收）的 P99 分位——独立指标，99% 请求耗时低于该值。

    探针上报 out token 数时（MindIE 流式探针）追加三档吞吐/解码指标：
      tpot_ms   = 单请求 (ttot-ttft)/out —— 解码每 token 墙钟，median-of-medians
      tps       = 单请求 out/((ttot-ttft)/1000) —— 解码每秒 token，median-of-medians
      thr_tok_s = 并发批聚合吞吐 = Σout / max(ttot) —— 批吞吐代理，median-of-medians
    无 out 数据 → 缺省上述字段（vLLM/sglang 模板不报 token 数，保持零回归）。"""
    st.PER_ROUND_DIR.mkdir(parents=True, exist_ok=True)
    log = st.PER_ROUND_DIR / f"{round_key}.bench.log"
    # MindIE 等后端：health-ready 早于模型加载完成 → 压测前先预热至稳态（后端可选实现，缺省 no-op）
    warm = getattr(cfg.backend, "warm_until_stable", None)
    if warm:
        try:
            warm(cfg)
        except Exception:
            pass  # 预热失败不阻断：压测自身兜底（失败由 decide 判 unhealthy/rollback）
    # warmup（丢弃，吸收冷态/偶发扰动）
    _run_bench_script(cfg, f"{round_key}_warmup", log)
    medians_ttft, medians_ttot, all_ttft, all_ttot = [], [], [], []
    medians_tpot, medians_tps, medians_thr = [], [], []
    for _ in range(cfg.reps):
        out = _run_bench_script(cfg, round_key, log)
        samples = parse_benchmark(out)
        if not samples:
            return None
        medians_ttft.append(statistics.median(s["ttft_ms"] for s in samples))
        medians_ttot.append(statistics.median(s["ttot_ms"] for s in samples))
        all_ttft.extend(s["ttft_ms"] for s in samples)
        all_ttot.extend(s["ttot_ms"] for s in samples)
        # 吞吐/解码指标仅当整批样本都有 out token 数才可信（缺任一请求则整批不可算）
        if all(s.get("out_tokens") for s in samples):
            pos = [(s, max(s["ttot_ms"] - s["ttft_ms"], 1.0))
                   for s in samples if s["out_tokens"] > 0]
            tpot = [d / s["out_tokens"] for s, d in pos]
            tps = [s["out_tokens"] / (d / 1000) for s, d in pos]
            agg = sum(s["out_tokens"] for s in samples) \
                / (max(s["ttot_ms"] for s in samples) / 1000)
            medians_tpot.append(statistics.median(tpot))
            medians_tps.append(statistics.median(tps))
            medians_thr.append(agg)
    # median-of-medians：抗单轮离群
    ttft_ms = statistics.median(medians_ttft)
    ttot_ms = statistics.median(medians_ttot)
    all_ttft.sort()
    all_ttot.sort()
    n = len(all_ttft)
    ttft_p95 = all_ttft[int(n * 0.95)] if n else ttft_ms
    # P99 延迟（独立指标）：一段时间内所有请求的端到端延迟（发送→完整接收，即 total）
    # 从快到慢排第 99 百分位——99% 请求耗时低于该值。样本量小时≈max，相对比较同口径仍有效。
    nt = len(all_ttot)
    p99_ms = all_ttot[min(int(nt * 0.99), nt - 1)] if nt else ttot_ms
    metrics = {"ttft_ms": ttft_ms, "ttot_ms": ttot_ms, "ttft_p95": ttft_p95,
               "p99_ms": p99_ms}
    if medians_tpot:
        metrics["tpot_ms"] = statistics.median(medians_tpot)
        metrics["tps"] = statistics.median(medians_tps)
        metrics["thr_tok_s"] = statistics.median(medians_thr)
    # 统计功效（P0-2）：点估计之外附带离散度，供 keep 判定做噪声地板
    # （hyst 0.3% 常远低于实测 ±2% 波动——单点比较会把噪声当信号）。
    # rel_noise = MAD/median（相对离散度，无量纲可跨指标比较）。
    stats = {
        "n_samples": len(all_ttft),
        "ttft_mad_ms": round(_mad(all_ttft), 3),
        "ttft_rel_noise": round(_mad(all_ttft) / ttft_ms, 4) if ttft_ms > 0 else None,
        "ttot_rel_noise": round(_mad(all_ttot) / ttot_ms, 4) if ttot_ms > 0 else None,
    }
    if medians_thr:
        thr_med = statistics.median(medians_thr)
        stats["thr_rel_noise"] = round(_mad(medians_thr) / thr_med, 4) if thr_med > 0 else None
    metrics["stats"] = stats
    return metrics


def smoke_test(backend, port: int, model: str, timeout: int = 60) -> bool:
    """单次短输出探测，确认服务可推理。请求面（URL/body）由 backend.smoke_request 提供。"""
    import json
    import urllib.request
    url, body = backend.smoke_request(port, model)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def run(flow, ctx):
    """压测当前服务，产出写 ctx["metrics"] + ctx["graph_ok"]。失败降级为 10x 基线交由 decide 判 rollback/unhealthy。"""
    cfg = ctx["cfg"]
    state = ctx["state"]
    round_key = ctx["round_key"]
    metrics = run_benchmark(cfg, state, round_key)
    baseline = ctx.get("baseline")
    if not metrics and baseline:
        metrics = {"ttft_ms": baseline["ttft_ms"] * 10, "ttot_ms": baseline["ttot_ms"] * 10}
    ctx["metrics"] = metrics
    # 图捕获校验：图模式开启时应确认真正捕获（本轮服务日志；apply 写入 current_params 后本 stage 执行）
    backend = cfg.backend
    ctx["graph_ok"] = backend.graph_probe(backend.log_path(flow, round_key),
                                          state.get("current_params") or {})
