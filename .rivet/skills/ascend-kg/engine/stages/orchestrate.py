"""⑩决策层（跨轮次）：best 记账 + 下一轮动作。

对齐 ROCm.AI 决策层的 Orchestrator + Robustness：单轮 decide 只判 accept/reject（滞回），
本 stage 管全局——什么时候继续、什么时候停（平台期/穷尽/不健康）、best 记账。
纯确定性规则（决策层是可审计的守门员，不引 LLM）。

终止判定委托 tools/decide.should_terminate（与 agent_loop 共用，保证两入口一致）；
本 stage 只保留 best 记账 + next_action 的上下文回填。

next_action 取值：continue | stop_unhealthy | stop_exhausted | stop_plateau。
"""
from engine.tools.decide import should_terminate, best_candidate, _consec_no_improve


def run(flow, ctx):
    state = ctx["state"]
    verdict = ctx.get("verdict") or {}
    metrics = ctx.get("metrics")
    decision = verdict.get("decision", "no_candidate")

    # 1) best 记账（本轮 keep 且优于历史 best；主指标按 flow.target，见 tools/decide）
    if decision == "keep" and metrics:
        nb = best_candidate(state, flow, ctx["round_rec"]["round"], metrics,
                            ctx["cfg"].target)
        if nb:
            state["best"] = nb

    # 2) 终止判定：委托 should_terminate（纯函数，修复旧实现的 double-count）
    #    本 stage 与 agent_loop 共用同一判定，消除两套准则的分叉。
    term = should_terminate(state, flow, ctx["cfg"].target)
    consec = _consec_no_improve(state.get("rounds", []))  # 仅供日志展示
    if term is not None:
        action = {"exhausted": "stop_exhausted",
                  "unhealthy": "stop_unhealthy",
                  "plateau": "stop_plateau"}[term.kind]
        why = term.detail
    else:
        action, why = "continue", f"连续无改善 {consec}/{getattr(flow.decision, 'consec_no_improve_stop', 2)}"

    ctx["orchestration"] = {"next_action": action, "rationale": why,
                            "consec_no_improve": consec}
