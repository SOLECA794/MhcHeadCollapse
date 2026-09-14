"""P0-1 层 2 数据源：domain_node.validate_interactions（KG 交互先验校验）。

核心意图：先验必须可溯源（evidence 必填防 LLM 编造联动）、参数必须在合并后
动作空间内（防引用不存在的参数）、≥2 个参数才构成交互组。
"""
from engine.tools.domain_node import validate_interactions, validate_kg_domain, merge_domain

MERGED = {"flag-a": {"kind": "env", "flag": True}, "seqs": {"min": 1, "max": 512, "step": 32}}


def test_valid_group_normalized():
    ok, dropped = validate_interactions(
        [{"params": ["flag-a", "seqs"], "evidence": "指南：A 与 B 同开收益叠加"}], MERGED)
    assert ok == [{"params": ["flag-a", "seqs"], "evidence": "指南：A 与 B 同开收益叠加"}]
    assert not dropped


def test_bare_list_dropped():
    """裸列表 [p1, p2] 无 evidence → 丢弃（先验必须可溯源）。"""
    ok, dropped = validate_interactions([["flag-a", "seqs"]], MERGED)
    assert ok == [] and "evidence" in dropped["[0]"]


def test_unknown_param_dropped():
    ok, dropped = validate_interactions(
        [{"params": ["flag-a", "ghost-p"], "evidence": "x"}], MERGED)
    assert ok == []
    assert "不在合并后 domain" in dropped["[0]"]


def test_single_param_dropped():
    ok, dropped = validate_interactions(
        [{"params": ["flag-a"], "evidence": "x"}], MERGED)
    assert ok == [] and dropped


def test_duplicate_params_dropped():
    ok, dropped = validate_interactions(
        [{"params": ["flag-a", "flag-a"], "evidence": "x"}], MERGED)
    assert ok == [] and dropped


def test_empty_evidence_dropped():
    ok, dropped = validate_interactions(
        [{"params": ["flag-a", "seqs"], "evidence": "  "}], MERGED)
    assert ok == [] and "evidence" in dropped["[0]"]


def test_none_passthrough():
    assert validate_interactions(None, MERGED) == ([], {})


# ---------------- KG domain 校验/合并（回归锚点） ----------------


def test_validate_kg_domain_numeric_requires_min_max():
    """手写也没有 min/max 的数值参数 → 丢弃（within_domain 直接下标会炸）。"""
    valid, dropped = validate_kg_domain({"new-p": {"risk": "low"}}, {})
    assert "new-p" not in valid and "new-p" in dropped


def test_validate_kg_domain_json_requires_path():
    valid, dropped = validate_kg_domain(
        {"cfg-p": {"kind": "json"}}, {"cfg-p": {}})
    assert "cfg-p" in dropped


def test_merge_domain_field_inheritance():
    """KG 只覆盖给出字段，kind/flag 等执行元数据从手写继承（防渲染坏）。"""
    merged = merge_domain(
        {"flag-a": {"kind": "env", "flag": True, "risk": "low"}},
        {"flag-a": {"risk": "mid"}})
    assert merged["flag-a"] == {"kind": "env", "flag": True, "risk": "mid"}
