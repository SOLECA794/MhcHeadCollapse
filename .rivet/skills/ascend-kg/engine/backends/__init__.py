"""后端注册表：flow.backend → Backend 实例（DESIGN-multi-backend §3.2 接缝层）。

引擎代码只面向 Backend 接口编程；新推理引擎 = 新增 backend 模块 + 在此注册 +
flow 声明 backend 字段，主循环/工具链不动。缺省 vllm-ascend（老 flow 不带
backend 字段也不破坏）。
"""
from engine.backends.base import Backend
from engine.backends.vllm_ascend import VllmAscend
from engine.backends.sglang import SGLang
from engine.backends.mindie import MindIE

DEFAULT_BACKEND = "vllm-ascend"

_REGISTRY = {
    VllmAscend.name: VllmAscend(),
    SGLang.name: SGLang(),
    MindIE.name: MindIE(),
}


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise ValueError(f"未知 backend: {name}（已注册: {sorted(_REGISTRY)}）")
    return _REGISTRY[name]


def backend_of(flow) -> Backend:
    """flow → Backend。backend 字段缺省 vllm-ascend。"""
    return get_backend(getattr(flow, "backend", None) or DEFAULT_BACKEND)
