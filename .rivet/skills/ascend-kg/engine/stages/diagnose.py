"""⑤诊断（LLM）：读 signals → 暂停回传主会话 agent 做热点算子根因诊断 + 算子级建议。

内核层「轻」的落点：不动算子源码，产 tiling/dtype/融合/算子选型建议（AscendC 重写属第二阶段）。
resume 时若 agent 已写回结果则直接消费，不重复 emit。
"""
from engine import agent_gate


def run(flow, ctx):
    signals = ctx.get("signals")
    if not signals:
        ctx["diagnosis"] = None
        return

    # resume：agent 已写回结果 → 直接消费，不重复 emit
    result = agent_gate.consume_result(ctx)
    if result is not None:
        ctx["diagnosis"] = result
        return

    # 提示词型 skill 引用（本地 SKILL.md 路径），agent 读后执行推理
    skill_refs = []
    msot = getattr(flow.paths, "msot_skill", None)
    if msot:
        skill_refs.append({"name": "msot-msopprof-operator-profiler",
                           "path": msot,
                           "note": "读算子 profiler CSV 定位瓶颈 + 给优化建议的方法论"})
    mfu = getattr(flow.paths, "mfu_skill", None)
    if mfu:
        skill_refs.append({"name": "op-mfu-calculator",
                           "path": mfu,
                           "note": "MFU 公式与解读（signals 已含 MFU，用于判 compute bound）"})
    if not skill_refs:
        ctx["diagnosis"] = None  # 配置缺 skill 路径 → 降级跳过，不阻断
        return

    goal = ("定位 profiling 热点算子的性能瓶颈根因，给出算子级优化建议"
            "（tiling/dtype/融合/算子选型），不动算子源码")
    schema = {
        "bottleneck_class": "string: compute|schedule|comm|unknown",
        "hot_ops": [{"op": "string", "root_cause": "string", "suggestion": "string"}],
        "operator_level_suggestions": ["string: tiling/dtype/融合/选型建议"],
        "summary": "string: 一句话诊断结论",
    }
    context = {
        "signals": signals,
        "out_dir": str(ctx.get("out_dir")) if ctx.get("out_dir") else None,
        "note": ("signals 含 MFU/热点算子/wait 时长/bottleneck_class 粗分类；"
                 "kernel_details.csv 在 out_dir 下，如需可读"),
    }
    agent_gate.emit_task(ctx, goal=goal, skill_refs=skill_refs, schema=schema, context=context)
