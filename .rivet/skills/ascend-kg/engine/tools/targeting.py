"""目标感知基础函数（无依赖，供 decide/objective 共用，消除循环 import）。

阶段 A 偏差 1 的解法：primary_key/_eff_target 原在 tools/decide.py，objective 需用
primary_key 而 decide.keep_verdict 需调 objective.compute_objective，两者互相引用成环。
抽到本模块后，decide 与 objective 都单向 import 本模块，环解开。
"""


def _eff_target(flow, target):
    """生效目标：CLI --target 显式覆盖 flow.target 时以显式值为准（flow JSON 默认 ttft）。"""
    return target or getattr(flow, "target", "ttft")


def primary_key(flow, target=None) -> str:
    """主指标名：生效目标=throughput → thr_tok_s（↑优）；否则 ttft_ms（↓优）。

    target 缺省回退 flow.target，保证零配置（run.py 确定性流水线不传 --target）行为不变。"""
    return "thr_tok_s" if _eff_target(flow, target) == "throughput" else "ttft_ms"
