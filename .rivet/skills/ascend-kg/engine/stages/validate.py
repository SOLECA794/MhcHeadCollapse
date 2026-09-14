"""②护栏：capture 后校验 profiling 交付件是否完整可分析。

对照 msagent `ascend-profiler-data-validation` skill 的「关键交付件检查」，把其中
确定性可自动化的部分固化为纯 stdlib 校验（不调 offline_parse —— 那是 torch_npu
离线解析，capture 的 _ensure_kernel_details 已做等价补转）。

- kernel_details.csv 是 analyze 的硬依赖，缺失 → 降级 out_dir（后续 analyze/diagnose 全跳过）。
- 其余交付件缺失仅记 warning，不阻断（analyze 只用 kernel_details.csv）。
"""
from pathlib import Path

_HARD = ("kernel_details.csv",)
_SOFT = ("op_statistic.csv", "trace_view.json")


def run(flow, ctx):
    out_dir = ctx.get("out_dir")
    if not out_dir:
        ctx["validation"] = {"status": "skipped", "reason": "capture 无产出"}
        return
    od = Path(out_dir)
    missing = [f for f in _HARD + _SOFT if not (od / f).exists()]

    if any(f in missing for f in _HARD):
        ctx["validation"] = {"status": "invalid", "missing": missing}
        ctx["out_dir"] = None  # analyze 硬依赖缺失 → 降级
    elif missing:
        ctx["validation"] = {"status": "warning", "missing": missing}
    else:
        ctx["validation"] = {"status": "valid", "missing": []}
