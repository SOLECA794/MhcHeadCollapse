"""P0-1 action v2：sampler 值级 tried（组合键）/ grouping 组合记账与先验交集。

核心意图：组合轮 {"params": {...}} 必须同时进两种记账——frozenset 组合键防同组合
重试、组内 (param, value) 对保持值级视图完整；grouping.kept_params 组合 keep 按
本次生效值记账（组合 ≠ 单参各自最优，final_sweep 对拍兜底）。
"""
from types import SimpleNamespace

from engine.tools.sampler import tried_values, pending_neighborhood
from engine.tools.grouping import kept_params, group, combination_candidates

DOMAIN = {
    "flag-a": {"kind": "env", "flag": True, "risk": "low"},
    "flag-b": {"kind": "env", "flag": True, "risk": "mid"},
    "seqs": {"kind": "cli", "min": 1, "max": 512, "step": 32, "risk": "low"},
    "enum-p": {"kind": "cli", "values": ["a", "b"], "risk": "low"},
}


def mk_flow():
    return SimpleNamespace(params=SimpleNamespace(domain=dict(DOMAIN)))


# ---------------- tried_values：组合键 + 值级视图 ----------------


def test_tried_combo_frozenset_key():
    """组合轮进 tried：frozenset 组合键 + 组内单值对，两形态都可查。"""
    state = {"rounds": [
        {"round": "c1", "action": {"params": {"flag-a": True, "seqs": 128}}, "kept": True},
    ]}
    t = tried_values(state)
    assert frozenset({("flag-a", True), ("seqs", 128)}) in t  # 组合键
    assert ("flag-a", True) in t and ("seqs", 128) in t        # 值级视图


def test_tried_same_combo_different_values_distinct():
    """同参数集不同取值 = 不同组合键（值级语义，不是参数级 tried）。"""
    s1 = {"rounds": [{"round": "c1", "action": {"params": {"flag-a": True, "seqs": 128}}}]}
    s2 = {"rounds": [{"round": "c1", "action": {"params": {"flag-a": True, "seqs": 160}}}]}
    assert tried_values(s1) != tried_values(s2)


def test_tried_single_param_round():
    state = {"rounds": [{"round": "r1", "action": {"param": "seqs", "value": 128}}]}
    assert ("seqs", 128) in tried_values(state)


# ---------------- pending_neighborhood：backbone 准入补扫 ----------------


def test_pending_neighborhood_flags_skipped():
    """布尔/枚举天然全覆盖不补扫；连续参数 keep 后邻域未试 → 补扫建议。"""
    state = {"rounds": [
        {"round": "r1", "action": {"param": "seqs", "value": 160}, "kept": True},
        {"round": "r2", "action": {"param": "flag-a", "value": True}, "kept": True},
    ]}
    out = pending_neighborhood(state, mk_flow())
    assert [p["param"] for p in out] == ["seqs"]
    assert set(out[0]["values"]) == {128, 192}  # 160±32 均未试


def test_pending_neighborhood_tried_neighbor_excluded():
    """邻域一侧已试过 → 只补另一侧。"""
    state = {"rounds": [
        {"round": "r1", "action": {"param": "seqs", "value": 160}, "kept": True},
        {"round": "r2", "action": {"param": "seqs", "value": 192}, "kept": False},
    ]}
    out = pending_neighborhood(state, mk_flow())
    assert out and out[0]["values"] == [128]


def test_pending_neighborhood_combo_kept_params_covered():
    """组合轮 keep 的连续参数同样进补扫判定（action v2 兼容）。"""
    state = {"rounds": [
        {"round": "c1", "action": {"params": {"flag-a": True, "seqs": 160}}, "kept": True},
    ]}
    out = pending_neighborhood(state, mk_flow())
    assert [p["param"] for p in out] == ["seqs"]


# ---------------- kept_params / group / 组合候选 ----------------


def test_kept_params_combo_accounting():
    """组合 keep 按本次生效值记账组内每个参数。"""
    state = {"rounds": [
        {"round": "c1", "action": {"params": {"flag-a": True, "seqs": 96}}, "kept": True},
        {"round": "c2", "action": {"params": {"flag-b": True, "seqs": 128}}, "kept": False},
    ]}
    assert kept_params(state, None) == {"flag-a": True, "seqs": 96}


def test_group_prior_intersection():
    """交互子集 = kept ∩ 先验组（组内 ≥2 个 kept 才圈出）；backbone = 其余 kept。"""
    kept = {"flag-a": True, "flag-b": True, "seqs": 160}
    grp = group(mk_flow(), kept, prior_interactions=[["flag-a", "flag-b"]])
    assert grp["interactive"] == ["flag-a", "flag-b"]
    assert grp["backbone"] == {"seqs": 160}


def test_group_prior_single_hit_no_interaction():
    """先验组只有 1 个参数被 keep → 不圈出（交互检测需两端都有收益）。"""
    grp = group(mk_flow(), {"flag-a": True, "seqs": 160},
                prior_interactions=[["flag-a", "flag-b"]])
    assert grp["interactive"] == []
    assert grp["backbone"] == {"flag-a": True, "seqs": 160}


def test_combination_candidates_flag_dims():
    """flag 参数维度 = [True, False] → 2 参数 4 组合。"""
    combos = combination_candidates(mk_flow(), ["flag-a", "flag-b"],
                                    {"flag-a": True, "flag-b": True})
    assert len(combos) == 4
    assert {"flag-a": False, "flag-b": True} in combos


def test_combination_candidates_numeric_line():
    """连续参数在 kept 最优值附近取 3 档（value_sequence line_search）。"""
    combos = combination_candidates(mk_flow(), ["seqs"], {"seqs": 160})
    assert sorted(c["seqs"] for c in combos) == [128, 160, 192]
