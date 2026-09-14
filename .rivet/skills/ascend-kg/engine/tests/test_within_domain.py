"""F8 回归：locked 参数必须被硬校验拦截。

核心意图：动作空间文本标「锁定（不可调）」、sampler/refine/priors 排除 locked，
都只是降低误提概率——LLM 硬提议仍会走到 `_parse_decision` 的 within_domain 最后一道闸。
此前该闸对 locked 恒 True（垃圾字符串都放行），提案直接落地真机，靠健康门禁
兜底烧一轮 + 回滚。锁定值由 defaults 承载，提案任何值（含当前值的 no-op）都该拒。
"""
from types import SimpleNamespace

from engine.state import within_domain

DOM = {
    "trust-remote-code": {"locked": True, "flag": True},
    "block-size": {"locked": True, "min": 1, "max": 4, "fixed": 128},
    "seqs": {"min": 1, "max": 512, "step": 32},
}


def test_locked_rejects_all_values():
    """locked 参数任何提案都拒——含翻转布尔、垃圾值、以及 fixed 声明值。

    fixed 在域 schema 里是 flag payload 语义（state.py 渲染 `--flag 'payload'`），
    不是锁定值，不能当作可接受提案。
    """
    assert within_domain("trust-remote-code", False, DOM) is False
    assert within_domain("trust-remote-code", True, DOM) is False
    assert within_domain("trust-remote-code", "whatever", DOM) is False
    assert within_domain("block-size", 128, DOM) is False


def test_locked_rejected_in_decision_chain():
    """端到端：locked 提议经 _parse_decision 整单拒绝（None=按停止处理）。"""
    from engine.agent_loop import _parse_decision
    flow = SimpleNamespace(params=SimpleNamespace(domain=DOM))
    r = _parse_decision(
        {"action": {"param": "trust-remote-code", "value": False}}, flow)
    assert r is None


def test_unlocked_still_works():
    """非 locked 参数取值域行为不变（防修复误伤正常校验）。"""
    assert within_domain("seqs", 256, DOM) is True
    assert within_domain("seqs", 9999, DOM) is False
    assert within_domain("ghost", 1, DOM) is False
