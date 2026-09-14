"""flow 定义加载与校验：flows/<flow_id>.json → Flow 对象。

纯 stdlib（json + SimpleNamespace），无第三方依赖。
"""
import json
from pathlib import Path
from types import SimpleNamespace

FLOW_DIR = Path(__file__).resolve().parent / "flows"

_REQUIRED = ["flow_id", "name", "mode", "target", "stages", "params"]
_STAGE_REQUIRED = ["id", "module"]


def load_flow(flow_id: str) -> SimpleNamespace:
    """按 flow_id 加载并校验 flow 定义，返回带属性访问的 Flow 对象。"""
    path = FLOW_DIR / f"{flow_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"flow 未定义: {flow_id}（找不到 {path}）")
    raw = json.loads(path.read_text(encoding="utf-8"))
    _validate(raw, flow_id)
    f = _to_flow(raw)
    # domain 来源标注：load 时一律手写；KG 合并由 agent_loop 的 domain 必走节点在
    # 运行时完成（tools/domain_node.py）——load 读缓存会绕过必走节点，故不再覆盖。
    f.domain_source = "handwritten"
    return f


def _validate(raw: dict, flow_id: str) -> None:
    for k in _REQUIRED:
        if k not in raw:
            raise ValueError(f"flow {flow_id} 缺必填字段: {k}")
    for i, s in enumerate(raw["stages"]):
        for k in _STAGE_REQUIRED:
            if k not in s:
                raise ValueError(f"flow {flow_id} stage[{i}] 缺必填字段: {k}")
    # 决策字段上移（DESIGN-multi-backend §3.1）：非 vllm-ascend 后端不得回退 vLLM 默认
    # goal/rules 措辞——措辞里含 vLLM 专属语义，静默回退会误导新后端的决策 agent
    backend = raw.get("backend", "vllm-ascend")
    if backend != "vllm-ascend":
        dec = raw.get("decision") or {}
        if not dec.get("goal") or not dec.get("rules"):
            raise ValueError(
                f"flow {flow_id} backend={backend} 必须显式提供 decision.goal 与 "
                f"decision.rules（不回退 vLLM 默认措辞）")


def _to_flow(raw: dict) -> SimpleNamespace:
    """顶层 + 二级段（paths/perf/profiling/model_dims/params）转 SimpleNamespace；
    params 内的 domain/blocked/defaults 保留为 dict（stage 按 key 查表）。"""
    f = SimpleNamespace(**raw)
    f.paths = SimpleNamespace(**raw.get("paths", {}))
    f.perf = SimpleNamespace(**raw.get("perf", {}))
    f.decision = SimpleNamespace(**raw.get("decision", {}))
    # throughput 目标的决策块（goal/rules 吞吐取向，见 agent_loop._decision_block）
    f.decision_throughput = SimpleNamespace(**raw.get("decision_throughput", {}))
    f.profiling = SimpleNamespace(**raw.get("profiling", {"enabled": True}))
    f.model_dims = SimpleNamespace(**raw.get("model_dims", {}))
    f.params = SimpleNamespace(**raw["params"])
    f.stages = [SimpleNamespace(**s) for s in raw["stages"]]
    return f
