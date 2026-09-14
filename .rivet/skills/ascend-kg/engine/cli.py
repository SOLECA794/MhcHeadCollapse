"""CLI 参数与运行配置（Cfg）——从 run.py 迁出的唯一权威定义（P1-1 入口统一）。

run.py 退役后 agent_loop.py 是唯一入口；本模块承载两条入口曾各自维护的
args 解析与 Cfg 装配，防再次分叉。纯装配、无副作用，便于单测。
"""
import argparse
import json
import sys
from pathlib import Path

sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from engine.registry import load_flow  # noqa: E402
from engine.backends import backend_of  # noqa: E402


def _apply_guard_policy(perf, profile, budget):
    """业务画像 guard 放宽：装配期烘进 flow.perf 的守门容限（case7 r11 教训）。

    compute_objective 与 keep_verdict 两条 guard 判定路径都从 flow.perf 读
    ttft_room/ttot_room——这里一处覆写全局生效（含影子记账副目标），判定代码
    零改动；覆写前生效值与策略记入 perf.guard_policy，供 objective components
    审计透传。guard_policy 与 P0-A 的 guard_widened 噪声展宽**分开记账**：
    业务放宽是价值判断（用户显式声明画像），统计展宽是测量问题（噪声地板），
    复盘时必须可分辨 keep/拦各依赖了哪一层。

    - online（缺省）：不动，现状严格行为零回归
    - batch：守门指标退化为灾难档（room → catastrophic_room，缺省 1.5，flow
      可覆盖）——批量离线场景反向指标无商业意义，只拦功能性劣化（如 prefill
      路径坏了）；smoke/精度护栏独立，不受影响照常拦
    - balanced：配置容限之上再放宽 budget（--ttft-budget 0.15 = 守门指标可
      再退化 15%），在线场景愿意用延迟换吞吐的显式预算
    """
    if profile == "batch":
        limit = float(getattr(perf, "catastrophic_room", 1.5) or 1.5)
        perf.guard_policy = {"profile": "batch", "limit_room": limit,
                             "orig": {"ttft_room": getattr(perf, "ttft_room", 1.10),
                                      "ttot_room": getattr(perf, "ttot_room", 1.10)}}
        perf.ttft_room = limit
        perf.ttot_room = limit
    elif profile == "balanced":
        b = float(budget or 0)
        orig_t = float(getattr(perf, "ttft_room", 1.10) or 1.10)
        orig_o = float(getattr(perf, "ttot_room", 1.10) or 1.10)
        perf.guard_policy = {"profile": "balanced", "budget": b,
                             "orig": {"ttft_room": orig_t, "ttot_room": orig_o}}
        perf.ttft_room = orig_t + b
        perf.ttot_room = orig_o + b


def resolve_guard_policy(args, state):
    """守门画像解析（#5 修）：CLI 显式 > state 记录 > online。返回 (profile, budget)。

    --profile/--ttft-budget 原先只烘进 flow.perf 不落 state——resume 忘带即静默
    回退 online，前后轮 guard 口径不一致。state["profile_policy"] 由 agent_loop.main
    每次应用后写回，作为下轮 resume 的回退源。
    """
    saved = state.get("profile_policy") or {}
    profile = getattr(args, "profile", None) or saved.get("profile") or "online"
    budget = getattr(args, "ttft_budget", None)
    if budget is None:
        budget = saved.get("ttft_budget")
    return profile, budget


class Cfg:
    """运行参数：CLI 显式值优先，未给则回退 flow 定义。"""

    def __init__(self, args, flow):
        self.model = args.model or flow.paths.model_default
        self.port = args.port
        self.device = args.device
        self.profile_port = args.profile_port
        self.profile_device = args.profile_device
        self.target = args.target or flow.target
        self.max_rounds = args.max_rounds or flow.max_rounds
        self.max_combo_rounds = getattr(args, "max_combo_rounds", 8) or 8
        # 精修轮预算（②a 探索/精修分离）：keep 参数邻域补扫不占 max_rounds
        self.max_refine_rounds = getattr(args, "max_refine_rounds", 4) or 4
        self.reps = args.reps
        self.bench_timeout = args.bench_timeout
        self.workload = (json.loads(args.workload)
                         if isinstance(args.workload, str) else args.workload)
        self.flow_id = flow.flow_id
        # optix 先验注入开关（P1-1b）：off 时 decide context 不含 priors
        self.with_optix = getattr(args, "with_optix", "on") != "off"
        # 粗筛幕开关（③）：off 时跳过值级幕前的全参数单点快筛
        self.screen = getattr(args, "screen", "on") != "off"
        # P1-B 漂移锚点间隔：每 N 个值级轮零重启复测当前配置，|漂移|>5% 重锚（0=关闭）
        self.anchor_every = getattr(args, "anchor_every", 5)
        # 副目标 Pareto 通道上限：值级幕后复测副目标影子赢家的条数（0=关闭）
        self.pareto_max = getattr(args, "pareto_max", 1)
        # 后端接缝（DESIGN-multi-backend §3.2）：渲染/健康路径/兜底签名/请求面分派
        self.backend = backend_of(flow)
        self.flow = flow  # backend.health_url 等按 flow.paths 取后端专属端口/路径
        # 业务画像（#5 修，2026-08-29）：不再在此预烘 flow.perf——守门画像持久化在
        # state（agent_loop.main 解析「CLI 显式 > state 记录 > online」后应用一次并
        # 写回），resume 忘带 --profile 不再静默回退 online；此处预烘会造成二次放宽
        # （_apply_guard_policy 以当前 room 为 orig 基准，叠加两次 = 预算翻倍）。
        self.profile = getattr(args, "profile", None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ascend-kg 编排引擎（agent_loop，LLM 会话委托双循环）")
    p.add_argument("--flow", required=True, help="flow_id（对应 flows/<id>.json）")
    p.add_argument("--model", default=None, help="模型路径（默认取 flow）")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--profile-port", type=int, default=8001)
    p.add_argument("--profile-device", type=int, default=1)
    p.add_argument("--target", default=None, choices=["ttft", "throughput"])
    p.add_argument("--max-rounds", type=int, default=None, help="迭代上限（默认取 flow）")
    p.add_argument("--max-combo-rounds", type=int, default=8,
                   help="组合幕预算上限（P0-1 两幕编排：值级收敛后交互子集组合验证的轮数上限）")
    p.add_argument("--max-refine-rounds", type=int, default=4,
                   help="精修轮预算上限（keep 参数 center±step 邻域补扫，不占 max_rounds 探索预算）")
    p.add_argument("--reps", type=int, default=3, help="每次压测重复次数")
    p.add_argument("--bench-timeout", type=int, default=120, help="单请求超时秒")
    p.add_argument("--with-optix", default="on", choices=["on", "off"],
                   help="optix 先验注入 decide context 开关（P1-1b；off 时不注入）")
    p.add_argument("--screen", default="on", choices=["on", "off"],
                   help="粗筛幕开关（③）：值级幕前对全部可设参数做最大对比度单点快筛"
                        "（reps=1，不占 max_rounds），榜单进 decide context")
    p.add_argument("--anchor-every", type=int, default=5,
                   help="P1-B 漂移锚点：每 N 个值级轮后零重启复测当前运行配置，"
                        "|漂移|>5%% 时重锚基线/水位线（0=关闭；case7 终局才发现 -46.5%% 的教训）")
    p.add_argument("--pareto-max", type=int, default=1,
                   help="副目标 Pareto 通道：值级幕收敛后复测副目标影子赢家（主指标持平+副指标"
                        "真改善则 keep 进交付配置）的条数上限（0=关闭；case8 档C教训：TTFT -16%% "
                        "的配置因吞吐持平从未复测/进组合）")
    p.add_argument("--profile", default=None, choices=["online", "batch", "balanced"],
                   help="业务画像 guard 策略（case7 r11 教训：TTFT +11.4%% 拦下吞吐 +8.0%%，"
                        "守门容限从未按吞吐业务校准）：online=现状严格（零回归）；"
                        "batch=守门退化为灾难档（catastrophic_room 缺省 1.5，只拦功能性劣化）；"
                        "balanced=配置容限上再放宽 --ttft-budget")
    p.add_argument("--ttft-budget", type=float, default=None,
                   help="balanced 画像预算：守门指标在配置容限之上可再退化的比例"
                        "（0.15=+15%%；仅 balanced 生效，必填）")
    p.add_argument("--resume", action="store_true", help="读 state/run_state.json 续跑")
    p.add_argument("--workload", default=json.dumps({
        "input_len_avg": 184, "input_len_max": 264,
        "output_len_avg": 116, "output_len_max": 128}))
    args = p.parse_args(argv)
    if args.profile == "balanced" and not args.ttft_budget:
        p.error("--profile balanced 需要 --ttft-budget（如 --ttft-budget 0.15）")
    return args


def cfg_from_argv(argv=None):
    """parse_args + load_flow + Cfg 一步到位（agent_loop main 入口用）。"""
    args = parse_args(argv)
    flow = load_flow(args.flow)
    return args, flow, Cfg(args, flow)
