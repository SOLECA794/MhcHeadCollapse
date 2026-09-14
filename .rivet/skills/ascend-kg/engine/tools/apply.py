"""工具②④：应用动作（健康门禁 + 失败自动回滚）+ 独立回滚。

apply_action 内置确定性硬护栏：健康门禁不过 → 自动回滚到上一个健康配置，不依赖 LLM。
rollback_to_last 独立暴露给 LLM 作兜底。
"""
from engine import state as st
from engine.state import save_state
from engine.stages.apply import _apply, rollback


def _action_to_params(state, action: dict, flow) -> tuple:
    """action → new_params（action v2：单参数 / 组合双形态）。

    单参数：{"param": ..., "value": ...}；组合：{"params": {p: v, ...}}（P0-1，
    多参数原子变更——一次 apply 渲染进同一启动脚本，天然原子）。
    布尔 flag（domain 里 flag=true）传 value=True/False 即可，params_to_args 会渲染成 --k。
    返回 (new_params, err)。
    """
    new_params = dict(state.get("current_params", flow.params.defaults))
    params = action.get("params")
    if isinstance(params, dict) and params:
        new_params.update(params)
        return new_params, None
    param = action.get("param")
    value = action.get("value")
    if param is None:
        return None, "action 缺 param"
    new_params[param] = value
    return new_params, None


def apply_action(flow, cfg, state, action: dict, round_key: str) -> dict:
    """应用一个动作：渲染→备份→kill→重启→健康门禁→失败自动回滚。

    返回 {ok, rolled_back, log, current_params}。失败时自动回滚到上一个健康配置。
    """
    new_params, err = _action_to_params(state, action, flow)
    if err:
        return {"ok": False, "rolled_back": False, "log": err,
                "current_params": state.get("current_params")}
    # 施加前持久化「apply 进行中」标记：若进程在此窗口被杀（SIGTERM/KILL/OOM），
    # resume 可据此识别残留候选服务并回滚对齐，消除「服务跑候选、state 却还是基线」的分叉。
    # 标记不在本函数清除——由主循环在「本轮记录落盘」后清除（见 agent_loop.main）。
    state["apply_pending"] = {"round_key": round_key, "params": new_params, "action": action}
    save_state(state, st.RUN_STATE)
    app = _apply(flow, cfg, state, round_key, new_params)
    if app["ok"]:
        return {"ok": True, "rolled_back": False, "log": app["log"],
                "current_params": new_params}
    # 失败自动回滚（确定性硬护栏，不依赖 LLM）
    rolled = rollback(flow, cfg, state, round_key)
    return {"ok": False, "rolled_back": rolled, "log": app["log"],
            "current_params": state.get("current_params")}


def rollback_to_last(flow, cfg, state, round_key: str = "manual") -> dict:
    """恢复到上一个健康配置。返回 {ok}。"""
    ok = rollback(flow, cfg, state, round_key)
    return {"ok": ok}
