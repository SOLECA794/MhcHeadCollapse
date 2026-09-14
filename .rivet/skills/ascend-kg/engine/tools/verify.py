"""工具③：并发压测 + smoke + 输出一致性比对（精度护栏）+ 图捕获校验。

请求面与图捕获判据经 cfg.backend 分派（DESIGN-multi-backend §3.2）。
"""
from engine.stages.verify import run_benchmark, smoke_test
from engine.tools.accuracy import check_accuracy


def verify(flow, cfg, state, round_key: str = "manual") -> dict:
    """压测当前服务。返回 {metrics, smoke_ok, accuracy_match, accuracy_detail, graph_ok}。

    metrics=None 表示压测失败（调用方降级 10x 基线）。
    accuracy_match 为精度对拍（True/False/None，困惑度漂移口径）：False = 语义破坏（调用方
    agent_loop.try_action 按 cls 强制回滚或放行+惩罚）；None = 不判（未配置/无基线/采集失败，不误伤）。
    graph_ok 为图捕获校验（True/False/None）：图模式开启但实际未捕获 → False，
    调用方据此判不 keep 并强制回滚。
    """
    backend = cfg.backend
    metrics = run_benchmark(cfg, state, round_key)
    smoke_ok = smoke_test(backend, cfg.port, cfg.model)
    accuracy_match, accuracy_detail = check_accuracy(flow, cfg, state)
    graph_ok = backend.graph_probe(backend.log_path(flow, round_key),
                                   state.get("current_params") or {})
    return {"metrics": metrics, "smoke_ok": smoke_ok,
            "accuracy_match": accuracy_match, "accuracy_detail": accuracy_detail,
            "graph_ok": graph_ok}
