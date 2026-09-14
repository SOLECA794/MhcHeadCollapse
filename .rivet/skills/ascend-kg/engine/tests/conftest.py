"""engine 单测公共夹具。

- sys.path 指到 skill 根（engine 包可导入）；
- state 作用域隔离：save_state/record_domain 重定向进 tmp_path，绝不触碰真实
  state/{flow_id}/ 与 domain_cache/（引擎函数有落盘副作用，测试必须兜住）。
"""
import json
import pathlib
import sys

import pytest

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from engine import state as st  # noqa: E402
from engine.registry import _to_flow, _validate  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state_scope(tmp_path):
    """每个用例独立的 STATE_DIR/PER_ROUND_DIR + flow scope（防串写真实运行状态）。"""
    st.set_flow_scope("unit-test")
    st.STATE_DIR = tmp_path / "state"
    st.PER_ROUND_DIR = st.STATE_DIR / "per_round"
    st.STATE_DIR.mkdir(parents=True, exist_ok=True)
    st.PER_ROUND_DIR.mkdir(parents=True, exist_ok=True)
    # domain_node.record_domain 落盘位置按模块常量走，测试一并重定向
    from engine.tools import domain_node
    domain_node.CACHE_DIR = tmp_path / "domain_cache"
    yield


@pytest.fixture(scope="session")
def case5_flow():
    """真实形态 flow（源自 case5 真机案例）：24 参数手写 domain + 2 组交互先验。

    案例文件不入库（flows/case*.json 属机器本地数据，.gitignore 排除）——测试自带
    fixtures 副本，与 load_flow 同链路校验/装配，缺 flows/case5 也能跑全套单测。
    """
    p = pathlib.Path(__file__).parent / "fixtures" / "flow-24params-vllm.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    _validate(raw, raw["flow_id"])
    return _to_flow(raw)
