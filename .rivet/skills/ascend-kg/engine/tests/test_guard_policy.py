"""守门画像持久化（#5，2026-08-29 修）。

核心意图（为什么这些行为重要）：--profile/--ttft-budget 是用户显式声明的价值
判断（batch 场景反向指标无意义 / 在线场景愿意用延迟换吞吐）。原先只烘进
flow.perf 不落 state——resume 忘带参数即静默回退 online，同一 run 前后轮
guard 口径不一致（前轮放宽拦下的劣化，后轮按严格口径又放行）。解析三级
优先级 CLI 显式 > state 记录 > online，且全 run 只应用一次（Cfg 不预烘，
否则叠加两次 = 预算翻倍）。
"""
from types import SimpleNamespace

from engine.cli import Cfg, parse_args, _apply_guard_policy, resolve_guard_policy


def _flow():
    return SimpleNamespace(
        flow_id="t", target="ttft", max_rounds=8, backend=None,
        paths=SimpleNamespace(model_default="/m"),
        perf=SimpleNamespace(hyst=0.003, ttft_room=1.10, ttot_room=1.10),
        decision=SimpleNamespace(goal="", rules=[]), decision_throughput=None)


def _args(argv):
    return parse_args(["--flow", "t"] + argv)


# ---------------- resolve_guard_policy：三级优先级 ----------------


def test_priority_cli_explicit_over_saved():
    args = _args(["--profile", "batch"])
    state = {"profile_policy": {"profile": "balanced", "ttft_budget": 0.15}}
    assert resolve_guard_policy(args, state) == ("batch", 0.15)


def test_priority_saved_fills_missing_cli():
    """#5 回归主场景：resume 忘带 --profile → 沿用 state 记录，不回退 online。"""
    args = _args([])
    state = {"profile_policy": {"profile": "balanced", "ttft_budget": 0.15}}
    assert resolve_guard_policy(args, state) == ("balanced", 0.15)


def test_priority_online_when_never_set():
    args = _args([])
    assert resolve_guard_policy(args, {"profile_policy": None}) == ("online", None)


def test_cli_budget_overrides_saved_independently():
    """budget 单独可覆盖：profile 沿用 state 而 budget 用 CLI 新值。"""
    args = _args(["--ttft-budget", "0.2"])
    state = {"profile_policy": {"profile": "balanced", "ttft_budget": 0.05}}
    assert resolve_guard_policy(args, state) == ("balanced", 0.2)


# ---------------- Cfg 不预烘 + 单次应用 ----------------


def test_cfg_does_not_prewrite_flow_perf():
    """#5 回归：Cfg 构造不得预烘 flow.perf——_apply_guard_policy 以当前 room 为
    orig 基准，main 再应用一次 = balanced 预算翻倍 / batch 灾难档覆盖记账 orig。"""
    flow = _flow()
    Cfg(_args(["--profile", "balanced", "--ttft-budget", "0.15"]), flow)
    assert flow.perf.ttft_room == 1.10 and flow.perf.ttot_room == 1.10
    assert not hasattr(flow.perf, "guard_policy")  # 未应用 = 未记账
    assert Cfg(_args(["--profile", "batch"]), _flow()).profile == "batch"


def test_apply_twice_doubles_budget_documents_the_trap():
    """文档化陷阱（Cfg 必须不预烘的原因）：二次应用以已放宽 room 为 orig 再加
    budget——0.15 叠两次 = 1.40 而非 1.25。这不是期望行为，是「只许应用一次」
    纪律的依据；若未来改成幂等实现，本测试应删除。"""
    perf = _flow().perf
    _apply_guard_policy(perf, "balanced", 0.15)
    _apply_guard_policy(perf, "balanced", 0.15)
    assert perf.ttft_room == 1.40  # 1.10+0.15 → 1.25+0.15（陷阱现场）


def test_apply_batch_records_orig_for_audit():
    """batch：room → 灾难档；覆写前原值进 guard_policy.orig 供审计分辨。"""
    perf = _flow().perf
    _apply_guard_policy(perf, "batch", None)
    assert perf.ttft_room == perf.ttot_room == 1.5
    assert perf.guard_policy["orig"] == {"ttft_room": 1.10, "ttot_room": 1.10}
