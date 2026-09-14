"""工具①：采集 profiling + 校验交付件 + 解析瓶颈信号。

合并原 capture/validate/analyze 三阶段为一个可独立调用的工具。
薄封装：构造临时 ctx，复用现有 stage 的 run()，再提取结果返回。
"""
from engine.stages import capture as _capture
from engine.stages import validate as _validate
from engine.stages import analyze as _analyze


def capture(flow, cfg, state, round_key: str) -> dict:
    """采集 profiling → 校验交付件 → 解析瓶颈信号。

    返回 {out_dir, validation, signals}。signals=None 表示采集/解析失败（可降级，不抛）。
    """
    ctx = {"cfg": cfg, "flow": flow, "state": state, "round_key": round_key,
           "out_dir": None, "validation": None, "signals": None}
    _capture.run(flow, ctx)
    _validate.run(flow, ctx)
    _analyze.run(flow, ctx)
    return {"out_dir": ctx.get("out_dir"),
            "validation": ctx.get("validation"),
            "signals": ctx.get("signals")}
