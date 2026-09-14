"""⑥落地前门槛（Critic）：弱证据候选不值得为它重启一次服务。

对齐 ROCm.AI 决策层的 Critic/PolicyGate——recommend 选型后、apply 落地前，
给「纯启发式且无 profiling 证据」的候选设门槛。否决即 candidate=None，apply 自动跳过。
纯确定性规则：候选证据强弱由 recommend 的 prio 编码（信号命中证据映射=1/2/3，否则=9）。
"""


def run(flow, ctx):
    cand = ctx.get("cand") or {}
    if cand.get("candidate") is None:
        return  # 本就无候选，apply 会跳过

    gate = getattr(flow.decision, "critic_prio_gate", 7)
    prio = cand.get("prio", 9)
    src = cand.get("source", "heuristic")

    if src == "heuristic" and prio >= gate:
        cand["candidate"] = None
        cand["critic_pass"] = False
        cand["critic_rationale"] = (
            f"Critic 否决：heuristic 候选 prio={prio}（无 profiling 证据支撑），"
            f"不值得为此重启服务")
        print(f"[critic] 否决 {cand.get('rationale', '')}", flush=True)
        return

    cand["critic_pass"] = True
    cand["critic_rationale"] = f"Critic 放行：src={src} prio={prio}"
    print(f"[critic] 放行 src={src} prio={prio}", flush=True)
