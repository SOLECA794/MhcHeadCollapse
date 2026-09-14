#!/usr/bin/env python3
"""【已废弃 DEPRECATED】run.py 双入口 stage 循环 —— 2026-08-27 P1-1 退役归档。

agent_loop.py 是唯一编排入口（LLM 会话委托双循环 + 两幕编排 + domain 必走节点）。
run.py 的 stage 级 resume 语义与 agent_loop 的轮级 resume 已分叉（rounds schema 靠
M1 补丁对齐、domain 节点只接了 agent_loop），长期维护两套不值得。

本文件仅作考古保留，直接执行会被下方 deprecation banner 拦截退出。
历史用法（供对照迁移）：
  python engine/agent_loop.py --flow vllm-serve-optimize [--max-rounds 3] [--resume]
  （Cfg/parse_args 已迁至 engine/cli.py）
"""
import sys

print("=" * 70, file=sys.stderr)
print("[DEPRECATED] engine/run.py 已退役（P1-1 入口统一），归档于 engine/attic/。",
      file=sys.stderr)
print("唯一入口：python engine/agent_loop.py --flow <flow_id> [--resume]",
      file=sys.stderr)
print("CLI 参数与 Cfg 装配：engine/cli.py", file=sys.stderr)
print("=" * 70, file=sys.stderr)
raise SystemExit(2)

# ─── 以下原实现仅存档（import 路径已失效，不可执行） ───
import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from engine.registry import load_flow  # noqa: E402
from engine.state import snapshot_state, save_state, round_keys, STATE_DIR, set_flow_scope  # noqa: E402
from engine import state as st  # noqa: E402
from engine.backends import backend_of  # noqa: E402
from engine.agent_gate import RESULT_FILE  # noqa: E402
from engine.stages.decide import _ensure_healthy  # noqa: E402
from engine.stages.verify import run_benchmark  # noqa: E402
from engine.tools.accuracy import collect_outputs, score_outputs  # noqa: E402
from engine.tools.decide import primary_key  # noqa: E402


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
        self.max_combo_rounds = getattr(args, "max_combo_rounds", 6) or 6
        self.reps = args.reps
        self.bench_timeout = args.bench_timeout
        self.workload = (json.loads(args.workload)
                         if isinstance(args.workload, str) else args.workload)
        self.flow_id = flow.flow_id
        # 后端接缝（DESIGN-multi-backend §3.2）：渲染/健康路径/兜底签名/请求面分派
        self.backend = backend_of(flow)
        self.flow = flow  # backend.health_url 等按 flow.paths 取后端专属端口/路径


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ascend-kg 编排引擎")
    p.add_argument("--flow", required=True, help="flow_id（对应 flows/<id>.json）")
    p.add_argument("--model", default=None, help="模型路径（默认取 flow）")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--profile-port", type=int, default=8001)
    p.add_argument("--profile-device", type=int, default=1)
    p.add_argument("--target", default=None, choices=["ttft", "throughput"])
    p.add_argument("--max-rounds", type=int, default=None, help="迭代上限（默认取 flow）")
    p.add_argument("--max-combo-rounds", type=int, default=6,
                   help="组合幕预算上限（P0-1 两幕编排：值级收敛后交互子集组合验证的轮数上限）")
    p.add_argument("--reps", type=int, default=3, help="每次压测重复次数")
    p.add_argument("--bench-timeout", type=int, default=120, help="单请求超时秒")
    p.add_argument("--resume", action="store_true", help="读 state/run_state.json 续跑")
    p.add_argument("--workload", default=json.dumps({
        "input_len_avg": 184, "input_len_max": 264,
        "output_len_avg": 116, "output_len_max": 128}))
    return p.parse_args(argv)


def _load_stage(module_path):
    """动态 import stage 模块，返回其 run(flow, ctx) 函数。"""
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "run"):
        raise AttributeError(f"stage 模块 {module_path} 缺 run(flow, ctx) 函数")
    return mod.run


def main(argv=None):
    args = parse_args(argv)
    flow = load_flow(args.flow)
    cfg = Cfg(args, flow)
    # 状态按 flow 隔离（DESIGN §3.4）：run_state/per_round/scripts_bak 落 state/{flow_id}/；
    # --target 覆盖 flow.target 时独立 state 目录，不同优化目标各存一份基线/轮次，互不覆盖
    set_flow_scope(flow.flow_id if cfg.target == flow.target else f"{flow.flow_id}-{cfg.target}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    st.PER_ROUND_DIR.mkdir(parents=True, exist_ok=True)

    # 启动时加载全部 stage（早失败，避免跑到一半才报缺模块）
    stage_fns = [(_load_stage(s.module), s) for s in flow.stages]

    state = snapshot_state(st.RUN_STATE)
    if not (args.resume and state.get("baseline")):
        # 全新 run：清掉上轮残留的 agent_result.json（task_id 仅 round_key-stage，
        # 跨 run 同名，残留 done 结果会被 diagnose 误消费为新诊断）
        if RESULT_FILE.exists():
            RESULT_FILE.unlink()
        # 渲染与 current_params 一致的基线脚本，保证 apply/rollback 备份-恢复闭环用真实基线
        baseline_script = cfg.backend.render_start_script(flow, cfg, flow.params.defaults,
                                                          "baseline")
        state = {"rounds": [], "best": None, "status": "running", "kg_notes": [],
                 "current_params": dict(flow.params.defaults),
                 "current_script": str(baseline_script),
                 "port": cfg.port}

    ctx = {
        "cfg": cfg, "flow": flow, "state": state,
        "baseline": state.get("baseline"), "round_key": None, "round_rec": {},
        "stage": None, "stage_idx": None,
        # stage 间传递数据
        "out_dir": None, "validation": None, "signals": None,
        "diagnosis": None, "cand": None,
        "metrics": None, "verdict": None,
    }

    print(f"[flow] {flow.flow_id} ({flow.name}) mode={flow.mode} target={cfg.target} "
          f"max_rounds={cfg.max_rounds}", flush=True)
    print(f"[stages] {' → '.join(s.id for s in flow.stages)}", flush=True)

    # ---------- round 0：基线 ----------
    if not state.get("baseline"):
        print("[round 0] 重启主服务为 defaults 配置 + 采集基线", flush=True)
        from engine.stages.apply import restart_to_defaults
        if not restart_to_defaults(flow, cfg, state):
            print("FATAL: 主服务无法重启为 defaults，无法建立基线", flush=True)
            state["status"] = "fatal"
            save_state(state, st.RUN_STATE)
            return 1
        m = run_benchmark(cfg, state, "baseline")
        if not m:
            print("FATAL: 基线压测失败（服务不可用？）", flush=True)
            state["status"] = "fatal"
            save_state(state, st.RUN_STATE)
            return 1
        # H1 精度护栏：基线生成参考文本（greedy 固定 prompt 集）+ 逐条评分困惑度（基线口径），
        # 供每轮 verify 用「困惑度漂移」对拍（替代逐字符精确匹配，抗 reduce 顺序扰动）。
        outputs = collect_outputs(flow, cfg)
        if outputs is None:
            print("[round 0] ⚠️ 基线参考文本采集失败，本次 run 精度护栏不生效（accuracy 不判）",
                  flush=True)
        outputs_ppl = score_outputs(flow, cfg, outputs) if outputs else None
        state["baseline"] = {"ttft_ms": m["ttft_ms"], "ttot_ms": m["ttot_ms"],
                             "ttft_p95": m.get("ttft_p95"),
                             "p99_ms": m.get("p99_ms"),
                             "tpot_ms": m.get("tpot_ms"),
                             "tps": m.get("tps"),
                             "thr_tok_s": m.get("thr_tok_s"),
                             "params": dict(flow.params.defaults),
                             "outputs": outputs, "outputs_ppl": outputs_ppl}
        ctx["baseline"] = state["baseline"]
        save_state(state, st.RUN_STATE)
        thr = f" 吞吐={m['thr_tok_s']:.1f}t/s" if m.get("thr_tok_s") else ""
        print(f"[round 0] 基线 TTFT={m['ttft_ms']:.1f}ms TTOT={m['ttot_ms']:.1f}ms P99={m['p99_ms']:.1f}ms{thr}", flush=True)

    done_rounds = round_keys(state)

    # resume：从 pending 暂停点继续（stage 级恢复，而非整轮重跑）
    resume_from = None
    if args.resume and state.get("status") == "pending" and state.get("pending"):
        resume_from = state["pending"]  # {round_key, stage_idx, task_id}

    # ---------- 优化轮 ----------
    for r in range(1, cfg.max_rounds + 1):
        if f"r{r}" in done_rounds:
            print(f"[round {r}] 已存在，跳过（resume）", flush=True)
            continue
        ctx["round_key"] = f"r{r}"
        ctx["round_rec"] = {"round": f"r{r}", "changed": [], "decision": "", "rationale": ""}
        ctx["fatal"] = False
        ctx["fatal_reason"] = ""

        # 起点：默认整轮从 0；resume 命中 pending 轮次则从该 stage 续跑并恢复中间产物
        start_idx = 0
        if resume_from and resume_from.get("round_key") == f"r{r}":
            start_idx = resume_from.get("stage_idx", 0)
            snap = state.get("snapshot") or {}
            ctx["out_dir"] = snap.get("out_dir")
            ctx["validation"] = snap.get("validation")
            ctx["signals"] = snap.get("signals")
            ctx["cand"] = snap.get("cand")
            resume_from = None  # 只恢复一次
        else:
            ctx["out_dir"] = None
            ctx["validation"] = None
            ctx["signals"] = None
            ctx["cand"] = None
        ctx["diagnosis"] = None
        ctx["metrics"] = None
        ctx["verdict"] = None
        print(f"\n========== round {r} ==========", flush=True)

        for i in range(start_idx, len(stage_fns)):
            fn, sdef = stage_fns[i]
            ctx["stage"] = sdef.id
            ctx["stage_idx"] = i
            print(f"  [{sdef.id}] {sdef.module}", flush=True)
            try:
                fn(flow, ctx)
            except Exception as e:
                # stage 异常不裸死整个闭环：大声报错 + 标记 fatal，交由下方统一收尾落盘
                traceback.print_exc()
                ctx["fatal"] = True
                ctx["fatal_reason"] = f"stage {sdef.id} 异常: {e}"
            if ctx.get("fatal"):
                print(f"[fatal] {ctx.get('fatal_reason')}", flush=True)
                break
            if ctx.get("pending"):
                # LLM 暂停点：快照中间产物 + 持久化 pending 标记，返回 2 等待 agent 回传
                state["pending"] = {"round_key": ctx["round_key"], "stage_idx": i,
                                    "task_id": ctx.get("pending_task_id")}
                state["snapshot"] = {"out_dir": ctx.get("out_dir"),
                                     "validation": ctx.get("validation"),
                                     "signals": ctx.get("signals"),
                                     "cand": ctx.get("cand")}
                state["status"] = "pending"
                save_state(state, st.RUN_STATE)
                print(f"\n[pending] 等待 agent 执行 {ctx.get('pending_task_id')}："
                      f"读 engine/state/agent_task.json → 加载 skill_refs → "
                      f"写 engine/state/agent_result.json → --resume 续跑", flush=True)
                return 2

        cand = ctx.get("cand") or {}
        for n in cand.get("optix_notes", []):
            print(f"  [optix] {n}", flush=True)
        for n in cand.get("kg_notes", []):
            print(f"  [kg] {n}", flush=True)
            state["kg_notes"].append(n)

        if ctx.get("fatal"):
            state["status"] = "fatal"
            save_state(state, st.RUN_STATE)
            break

        # 回填本轮记录
        rr = ctx["round_rec"]
        verdict = ctx.get("verdict") or {}
        if cand.get("candidate") is not None:
            rr["candidate"] = cand["candidate"]
            # M1：与 agent_loop 的 rounds 记录 schema 对齐（补 action/kept），
            # 混用入口 resume 后 _fmt_history / orchestrate 计数才不失效
            rr["action"] = {"param": cand["candidate"],
                            "value": (cand.get("new_params") or {}).get(cand["candidate"])}
        if ctx.get("metrics"):
            m = ctx["metrics"]
            rr["ttft_ms"] = m["ttft_ms"]
            rr["ttot_ms"] = m["ttot_ms"]
            for k in ("ttft_p95", "p99_ms", "tpot_ms", "tps", "thr_tok_s"):
                if m.get(k) is not None:
                    rr[k] = m[k]
        rr["decision"] = verdict.get("decision", "")
        rr["kept"] = verdict.get("decision") == "keep"
        rr["rationale"] = verdict.get("rationale", cand.get("rationale", ""))
        rr["analysis_available"] = bool(ctx.get("signals"))
        if ctx.get("validation"):
            rr["validation"] = ctx["validation"]
        if ctx.get("diagnosis"):
            rr["diagnosis"] = ctx["diagnosis"]

        state["rounds"].append(rr)
        save_state(state, st.RUN_STATE)

        # 决策层（orchestrate）决定下一轮动作：continue / stop_*
        action = (ctx.get("orchestration") or {}).get("next_action", "continue")
        if action != "continue":
            print(f"[orchestrate] {action}: "
                  f"{(ctx.get('orchestration') or {}).get('rationale', '')}", flush=True)
            if action == "stop_unhealthy":
                state["status"] = "fatal"
                save_state(state, st.RUN_STATE)
            break

    # ---------- 收尾 ----------
    if state["status"] != "fatal":
        state["status"] = "complete"
    save_state(state, st.RUN_STATE)
    healthy = _ensure_healthy(cfg, state)
    print(f"\n========== 闭环完成 ==========", flush=True)
    print(f"主服务健康: {'是' if healthy else '否'}", flush=True)
    best = state.get("best")
    if best:
        key = primary_key(flow, cfg.target)
        print(f"最优: round {best['round']} {key}={best.get(key):.1f}", flush=True)
    for r in state["rounds"]:
        changed = "/".join(r.get("changed", [])) or "(无候选)"
        print(f"  round {r['round']}: {changed} → {r.get('decision', '?')}", flush=True)
    return 1 if state["status"] == "fatal" else 0


if __name__ == "__main__":
    raise SystemExit(main())
