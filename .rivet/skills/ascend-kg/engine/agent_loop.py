#!/usr/bin/env python3
"""LLM 主循环：外层探索（会话委托决策）+ 内层确定性原子操作（方案 C 双循环）。

外层：引擎跑到「决策」暂停点（agent_gate）暂停（退出码 2），主会话 agent 读
engine/state/agent_task.json → 按动作空间/信号/历史决定「下一轮试哪个动作」或「停止」
→ 写 engine/state/agent_result.json；引擎 --resume 续跑。引擎本身不内嵌 LLM、零第三方依赖。

内层：apply → verify → is_kept → 变差自动回滚，单轮不翻车（确定性，不依赖 LLM）。

用法：
  python engine/agent_loop.py --flow vllm-serve-optimize [--max-rounds 3] [--port 8000] [--resume]
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.registry import load_flow  # noqa: E402
from engine.state import (  # noqa: E402
    snapshot_state, save_state, within_domain, round_keys, STATE_DIR, set_flow_scope,
)
from engine import state as st  # noqa: E402
from engine.cli import (Cfg, parse_args, _apply_guard_policy,  # noqa: E402
                        resolve_guard_policy)
from engine.stages.decide import _ensure_healthy  # noqa: E402
from engine.stages.verify import run_benchmark  # noqa: E402
from engine.tools import capture, apply_action, verify, rollback_to_last, is_kept  # noqa: E402
from engine.tools.decide import (best_candidate, primary_key, should_terminate,  # noqa: E402
                                 consec_rejected,
                                 keep_verdict_detail, pareto_verdict)
from engine.tools.objective import compute_objective  # noqa: E402
from engine.tools.accuracy import collect_outputs, score_outputs  # noqa: E402
from engine.tools import grouping, sampler  # noqa: E402
from engine.tools import priors as priors_mod  # noqa: E402
from engine.tools import domain_node  # noqa: E402
from engine import agent_gate  # noqa: E402

# 决策字段已上移到 flow.decision（goal/rules，DESIGN-multi-backend §3.1）；
# 以下仅作老 flow 缺字段时的回退。非 vllm-ascend 后端不回退——registry 已 fail-loud。
_DECIDE_GOAL = (
    "vLLM-Ascend 服务延迟优化：从动作空间里选「下一轮要试的一个参数改动」，"
    "或判断「该停止」。基于 profiling 信号 + 历史轮次 + 风险等级，优先低风险动作，"
    "勿重复已试参数。只把最终决策写进 agent_result.json 的 result 字段。"
)

_VLLM_RULES = [
    "每轮只改一个参数，param 名须与动作空间完全一致",
    "动作空间每个参数标了风险等级（低/中/高）：优先低风险；高风险（改 dtype/图模式）仅当低风险已穷尽且确信不破坏精度才试",
    "布尔开关 value 用 true/false；枚举参数 value 只能从枚举列表里选，不能自造",
    "高风险动作（risk=high）仅当本轮有 profiling 信号（signals 非空）才允许，否则被确定性拒绝并白费一轮",
    "环境变量参数只能显式设值（布尔 False 渲染为 0 而非 unset）；需要撤销已试的 env 改动由回滚恢复",
    "不要重复「已试过」的参数（见 history）",
    "图模式互斥：cudagraph-mode（非 NONE = 开图）与 enforce-eager（开 eager = 关图）不要同时启用；cudagraph-mode=NONE 用于 A/B 关图对照",
    "融合开关（fuse-norm-quant/fuse-qknorm-rope/fuse-muls-add/fusion-gmmswigluquant）默认全开，关闭仅用于隔离劣化 pass；图/融合动作后 verify 会检查图捕获是否生效，未生效判不 keep 并回滚",
    "何时回传无推荐(value=null)：当你判断继续试不会有益时（瓶颈不在动作空间作用域、已有明显最优、或动作空间已穷尽）。引擎另有确定性早停（连续多轮无改善即停），你的语义洞察是提前止损、避免注定白跑的轮次",
]

_DECIDE_SCHEMA = {
    "action": {"param": "string: 动作空间内参数名（单参数形态）",
               "value": "any: 枚举值/数值/布尔，须在取值域内；传 null 表示无推荐（你认为继续无意义时）",
               "params": "object（可选，组合形态）: {参数名: 值}，1..3 个参数一次原子变更，"
                         "每个参数独立过取值域校验，任一越界整单拒绝；组合幕阶段从 "
                         "composition_queue 选择时用此形态"},
    "reason": "string: 一句话理由（含语义洞察，如瓶颈不在动作空间作用域）",
}


def _decision_block(flow, cfg):
    """按生效目标取决策块（委托 tools/decide.decision_block，消除两处分叉）。"""
    from engine.tools.decide import decision_block
    return decision_block(flow, cfg.target)


def action_label(action: dict) -> str:
    """动作 → 展示标签（action v2）：单参数 'p=v'，组合 'p1=v1+p2=v2'。"""
    if isinstance(action.get("params"), dict) and action["params"]:
        return "+".join(f"{p}={v}" for p, v in action["params"].items())
    return f"{action.get('param', '?')}={action.get('value', '?')}"


def action_changed(action: dict) -> list:
    """动作涉及的参数名列表（round 记录 changed 字段 / tried 集合）。"""
    if isinstance(action.get("params"), dict) and action["params"]:
        return list(action["params"])
    p = action.get("param")
    return [p] if p else []


# 失败根因关键行（P1-A：apply 失败原因进 rounds，复盘不再翻 /tmp 日志）
_FAIL_PATTERNS = ("ERROR", "Error", "OOM", "out of memory", "timeout", "Timeout",
                  "Aborted", "Traceback", "FAILED", "failed to", "core dumped")


def _fail_reason(log) -> str:
    """apply 失败日志 → 一句话根因摘录（≤300 字符）。

    log 可能是日志文件路径（_apply 成功渲染但启动失败）或 kill 失败的 stdout 文本。
    命中失败关键词（OOM/超时/崩溃）的最后一处取其行+次行；无命中退回末行。
    """
    text = None
    if isinstance(log, str):
        p = Path(log)
        if p.exists() and p.is_file():
            try:
                text = p.read_text(errors="replace")[-4000:]
            except OSError:
                text = None
        if text is None:
            text = log[-4000:]
    if not text:
        return ""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    hits = [i for i, l in enumerate(lines) if any(t in l for t in _FAIL_PATTERNS)]
    if not hits:
        return lines[-1].strip()[:200]
    i = hits[-1]
    return " | ".join(x.strip() for x in lines[i:i + 2])[:300]


# ---------------- P0-B：apply 连续失败熔断 ----------------
# case7 教训：static-kernel 编译超时留孤儿占 56GB，粗筛幕 scr14-24 连锁 11 轮 apply
# 失败（每轮烧 ~10min 健康等待），引擎毫无察觉空转 1h27m，直到外层会话人工干预。
_FUSE_THRESHOLD = 3

_FUSE_GOAL = (
    "apply 连续失败熔断：连续 {n} 轮参数应用失败（服务起不来/健康门禁超时），"
    "大概率是环境问题（残留孤儿进程占卡/显存不足/端口冲突/宿主过载/编译超时）而非参数问题。"
    "请排查环境（残留 EngineCore / NPU 显存 / recent_failures 里的日志路径），"
    "修复后回传 continue 继续调优；确认无法修复回传 terminate 结束本次 run。"
)

_FUSE_SCHEMA = {"action": "continue | terminate",
                "reason": "string: 排查结论与处置说明"}


def _fuse_check(flow, cfg, state) -> int | None:
    """P0-B 熔断检查（粗筛/值级/组合三循环每轮迭代顶部调用）。

    apply 连续失败 ≥ _FUSE_THRESHOLD → agent_gate 暂停点回传 agent：
    - continue：环境已修复，失败计数清零（再给 _FUSE_THRESHOLD 次机会）；
    - terminate：status=complete 收尾退出。
    task_id = fail{streak}-fuse，幂等（崩溃在清零前重进时 consume 到同一结果）。
    返回 None=无需熔断/已恢复续跑；2=暂停待回传；0=terminate 已收尾。
    """
    streak = state.get("apply_fail_streak") or 0
    if streak < _FUSE_THRESHOLD:
        return None
    # task_id 带 episode 序号（#7 修，2026-08-29）：consume_result 不清结果文件，同 run
    # 第二次同 streak 熔断会消费到上一 episode 的陈旧 continue——agent 根本不知道有
    # 第二次故障就被自动续跑。episode 在「消费成功后」才推进持久化：崩溃在推进前 →
    # 同 task_id 重 consume 同一结果（幂等，保留原设计意图）；推进后崩溃 → 新 task_id
    # 重新暂停（agent 最多被多问一次）。
    ep = state.get("fuse_episode") or 0
    ctx = {"round_key": f"fail{streak}-e{ep}", "stage": "fuse"}
    res = agent_gate.consume_result(ctx)
    if res is None:
        fails = [r for r in state.get("rounds") or []
                 if r.get("metrics") is None and r.get("decision") in ("rollback", "screen")][-3:]
        recent = [{"round": r.get("round"),
                   "action": r.get("action"),
                   "fail_reason": r.get("fail_reason") or (r.get("rationale") or "")[:160]}
                  for r in fails]
        agent_gate.emit_task(
            ctx,
            goal=_FUSE_GOAL.format(n=streak),
            skill_refs=[],
            schema=_FUSE_SCHEMA,
            context={"fail_streak": streak, "recent_failures": recent,
                     "current_params": state.get("current_params"),
                     "note": ("recent_failures 里 fail_reason 是日志摘录，完整日志在 "
                              "/tmp/<flow_id>_start_*_<round>.log。常见根因：残留 "
                              "EngineCore 孤儿占 NPU 显存（engine/scripts/kill_by_port.sh "
                              "清理）、宿主过载、static-kernel 编译超时。continue 后失败"
                              "计数清零（再给 3 次机会）")},
        )
        state["pending"] = {"round_key": f"fail{streak}-e{ep}", "stage": "fuse",
                            "task_id": ctx["pending_task_id"]}
        state["snapshot"] = None
        state["status"] = "pending"
        save_state(state, st.RUN_STATE)
        print(f"\n[pending] ⚠️ apply 连续失败 {streak} 次，熔断暂停 {ctx['pending_task_id']}："
              f"排查环境后写 agent_result.json（continue/terminate）→ --resume", flush=True)
        return 2

    state["fuse_episode"] = ep + 1  # 消费成功：episode 推进（下次熔断换新 task_id）
    state["pending"] = None
    state["snapshot"] = None
    action = res.get("action") if isinstance(res, dict) else None
    reason = res.get("reason", "") if isinstance(res, dict) else ""
    if action == "terminate":
        state["status"] = "complete"
        state["fuse"] = {"terminated": True, "streak": streak, "reason": reason}
        save_state(state, st.RUN_STATE)
        print(f"[fuse] agent 判定 terminate（apply 连续失败 {streak} 次）：{reason}", flush=True)
        print("\n========== 闭环终止（apply 失败熔断） ==========", flush=True)
        return 0
    state["apply_fail_streak"] = 0
    state["fuse"] = {"resumed": True, "streak": streak, "reason": reason}
    save_state(state, st.RUN_STATE)
    print(f"[fuse] agent 判定 continue（失败计数清零续跑）：{reason[:120]}", flush=True)
    return None


def _fmt_action_space(flow) -> str:
    """动作空间 → 一行一参数的文本（含生效机制 kind 与风险等级 risk）。"""
    risk_label = {"low": "低", "mid": "中", "high": "高"}
    lines = []
    for name, d in (flow.params.domain or {}).items():
        if d.get("compose"):
            continue  # 派生参数（compose 合成），不可直接调，由 sub-param 触发
        if d.get("kind") == "env":
            kind = "环境变量"
        elif d.get("kind") == "sub":
            kind = "CLI(合成到组合flag)"
        elif d.get("kind") == "json":
            kind = "config.json"
        else:
            kind = "CLI"
        risk = risk_label.get(d.get("risk"), "低")
        if d.get("locked"):
            lines.append(f"  {name}: 锁定（不可调）")
        elif d.get("values"):
            lines.append(f"  {name}: 枚举 {{{', '.join(map(str, d['values']))}}} [{kind}，风险{risk}]")
        elif d.get("flag"):
            lines.append(f"  {name}: 布尔开关（true/false）[{kind}，风险{risk}]")
        else:
            lines.append(f"  {name}: {d.get('min')}..{d.get('max')} (step {d.get('step')}) [{kind}，风险{risk}]")
    return "\n".join(lines)


def _fmt_history(state, flow, target=None) -> str:
    """历史轮次 → 文本（主指标按生效目标，--target 覆盖 flow.target）。

    action v2：单参数 'p=v'，组合轮渲染 'p1=v1+p2=v2'。"""
    if not state.get("rounds"):
        return "  （无）"
    key = primary_key(flow, target)
    lines = []
    for r in state["rounds"]:
        if r.get("sweep"):
            continue  # 粗筛/终局 sweep 轮：见 screening 榜单/sweep 专报，不混入逐轮史
        a = r.get("action") or {}
        if a.get("params"):
            label = "+".join(f"{p}={v}" for p, v in a["params"].items())
        else:
            param = a.get("param", r.get("changed", ["?"])[0] if r.get("changed") else "?")
            label = f"{param}={a.get('value', '?')}"
        m = r.get("metrics") or {}
        val = m.get(key, "?")
        lines.append(f"  {r.get('round', '?')}: {label} "
                     f"→ kept={r.get('kept', '?')}, {key}={val}")
    return "\n".join(lines)


_COMPOSITION_GOAL = (
    "组合验证阶段：值级探索已收敛，现在验证交互子集的组合收益。从 context."
    "composition_queue（剩余待验证组合，均叠加在 backbone 上）中选择下一个要验证的"
    "组合，以 action.params 形态回传该组合的完整参数取值；若你认为剩余组合都不值得"
    "验证（如 backbone 取值已含某参数最优、组合必劣化），回传 value=null 软终止。"
    "组合与单参数走完全相同的校验/护栏（任一参数越界整单拒绝）。"
)

_COMPOSITION_RULES = [
    "只从 composition_queue 里选组合，不要自造队列外的组合（预算已按队列截断）",
    "回传形态：{\"action\": {\"params\": {参数名: 值, ...}}, \"reason\": \"...\"}",
    "组合 keep/rollback 判定与单参数相同（对比 rolling best + 噪声地板 + 护栏），变差自动回滚",
    "每轮选一个组合；队列耗尽或连续无改善时引擎自动进入终局 SWEEP（backbone+交互最优拼装验证）",
]


def _dual_target_summary(flow, state, main_target: str) -> dict:
    """双目标摘要（decide context / 收尾报告共用）。

    主目标读 state["best"]（权威记账）；副目标读 best_by_target（影子记账）。
    """
    other = OTHER_TARGET.get(main_target)
    out = {"main_target": main_target,
           "main_best": state.get("best") or None,
           "other_target": other,
           "other_best": (state.get("best_by_target") or {}).get(other) if other else None}
    if other and out["other_best"]:
        out["note"] = (f"副目标 {other} 的最优已实测记录（影子记账：主目标回滚的轮若副目标"
                       f"大赢，配置与实测值仍在 other_best）——发现 trade-off（如吞吐大赢/"
                       f"TTFT 小亏）时可作为组合探索或终局选择依据")
    return out


def _build_decision_context(flow, state, signals, round_key: str, rules, target=None,
                            phase=None, queue=None, backbone=None, cfg=None) -> dict:
    """构造决策暂停点的上下文（写进 agent_task.json）。goal/rules 由主循环按目标选定传入。

    P0-1e：常驻注入 pending_neighborhood（已 keep 未确认局部最优的补扫队列——
    ②a 起由引擎 refine 轮自动执行，注入仅供 LLM 可见性 + refine_note 提示勿重复推荐）；
    组合幕（phase=composition）额外注入 composition_queue/backbone。
    P1-1b：cfg.with_optix 时注入 priors（optix/启发式参考先验，值已过域校验）。
    """
    baseline = state.get("baseline") or {}
    best = state.get("best") or {}
    from engine.tools.sampler import pending_neighborhood
    ctx = {
        "round_key": round_key,
        "baseline": {"ttft_ms": baseline.get("ttft_ms"), "ttot_ms": baseline.get("ttot_ms"),
                     "tpot_ms": baseline.get("tpot_ms"), "tps": baseline.get("tps"),
                     "thr_tok_s": baseline.get("thr_tok_s"),
                     "stats": baseline.get("stats")},
        "best": best,
        "signals": signals,
        "history": _fmt_history(state, flow, target),
        "action_space": _fmt_action_space(flow),
        "rules": rules,
        "pending_neighborhood": pending_neighborhood(state, flow, target),
        # ②a：邻域补扫已自动化——LLM 无需把宝贵的探索轮花在这些点上
        "refine_note": ("已 keep 参数的邻域补扫（pending_neighborhood）由引擎精修轮自动执行，"
                        "不占探索预算；除非有明确理由，优先探索未试过的参数/取值"),
        # P1-3：最近有效诊断（结构化根因结论，解释动作空间为何有/无收益；
        # 节流保证它来自 ≤2 次诊断暂停，非本轮必现字段）
        "diagnosis": state.get("diagnosis"),
        # 双目标影子记账（方案 1）：副目标实测最优 + trade-off 提示
        "dual_target": _dual_target_summary(flow, state, target or flow.target),
    }
    # ③ 粗筛榜单：全参数单点快筛效应（实测排序替代信念排序；粗筛轮已从 history 剔除）
    digest = _screen_digest(state, target or flow.target, flow)
    if digest:
        ctx["screening"] = digest
    # P1-B：最近漂移锚点（环境尺度信息——解释改善百分比的参照系是否被重锚过）
    last_anchor = next((a for a in reversed(state.get("anchors") or []) if a.get("ok")), None)
    if last_anchor:
        ctx["env_drift"] = {"last_anchor": last_anchor,
                            "note": ("每 N 值级轮零重启复测当前配置的环境漂移探测；"
                                     "漂移>5% 时基线/水位线已重锚，改善百分比按当前环境"
                                     "尺度解释（终局以 r0 口径复核）")}
    if cfg is not None and getattr(cfg, "with_optix", True):
        # P1-1b：optix 先验（参考性质——历史已试参数在 history 可见，LLM 保留探索自由度）
        priors, priors_notes = priors_mod.build_priors(flow, cfg, state, signals)
        if priors:
            ctx["priors"] = priors
            ctx["priors_note"] = ("参考先验（optix/启发式产出，非指令）；"
                                  "已试过的参数/取值见 history，可结合探索")
        for n in priors_notes:
            print(f"[priors] {n}", flush=True)
    if phase == "composition":
        ctx["phase"] = "composition"
        ctx["composition_queue"] = queue or []
        ctx["backbone"] = backbone or {}
    return ctx


def _coerce_value(param: str, value, domain: dict) -> tuple:
    """把 LLM 回传的 value 收敛到参数声明类型（_parse_decision 在 within_domain 前调用）。

    - 枚举（values）→ 原样返回（枚举值是字符串，不能强转）；
    - 布尔（flag）→ 收敛为 bool（接受 true/false/1/0 及对应字符串）；
    - 数值 → 域 min/max/step 全为 int 时收敛为 int（拒非整数，不静默截断），否则收敛为 float。
    返回 (coerced, ok)；无法收敛返回 (value, False)。
    """
    d = domain.get(param, {})
    if d.get("values"):
        return value, True
    if d.get("flag"):
        if isinstance(value, bool):
            return value, True
        if value in (0, 1):
            return bool(value), True
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1"):
                return True, True
            if s in ("false", "0"):
                return False, True
        return value, False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value, False
    is_int_domain = all(
        isinstance(d.get(k), int) and not isinstance(d.get(k), bool)
        for k in ("min", "max", "step") if d.get(k) is not None
    )
    if is_int_domain:
        if f != int(f):
            return value, False
        return int(f), True
    return f, True


def _parse_decision(result, flow) -> dict | None:
    """校验 agent 回传的决策 dict（action v2：单参数 / 组合双形态）。

    返回：
      {"param":..., "value":...}   —— 单参数动作，进 try_action
      {"params": {p: v, ...}}      —— 组合动作（1 ≤ |params| ≤ 3，P0-1：多参数原子
                                      apply，供组合幕/LLM 主动组合验证）
      {"candidate": None}          —— LLM 软终止（无推荐），记 no_candidate 轮交
                                      should_terminate 判 exhausted（不在此直接停）
      None                         —— 无效决策（非 dict/缺字段/越界），LLM 失误，
                                      直接停（不浪费一轮跑空 apply）

    终止判定主路径在 should_terminate（引擎侧确定性），本函数不再消费 stop 布尔——
    LLM 不决定停止，只决定「试哪个参数」或「我推荐不出去了」(value=null)。
    """
    if not isinstance(result, dict):
        print(f"[llm] 决策结果非 dict，按停止处理: {result!r}", flush=True)
        return None
    action = result.get("action") or {}
    params = action.get("params")
    param = action.get("param")
    value = action.get("value")
    # LLM 软终止：明确回传无推荐（param/params 缺失 + value=null）→ 记 no_candidate 轮，
    # 由 should_terminate 走 exhausted 终止。保留 LLM 的语义停止洞察价值。
    if not param and params is None and value is None:
        print(f"[llm] LLM 无推荐（软终止建议）: {result.get('reason', '')}", flush=True)
        return {"candidate": None}
    # 组合形态（action v2）：逐参数走与单参数完全相同的校验链（域内 + 类型收敛 +
    # 取值域），任一不过整单拒绝——组合不放宽任何护栏。
    if isinstance(params, dict) and params:
        if not 1 <= len(params) <= 3:
            print(f"[llm] 组合动作参数数 {len(params)} 越界（须 1..3），按停止处理",
                  flush=True)
            return None
        domain = flow.params.domain or {}
        coerced = {}
        for p, v in params.items():
            if p not in domain:
                print(f"[llm] 组合内 param={p} 不在动作空间，整单拒绝，按停止处理",
                      flush=True)
                return None
            cv, ok = _coerce_value(p, v, domain)
            if not ok:
                print(f"[llm] 组合内 param={p} value={v!r} 无法收敛到声明类型，"
                      f"整单拒绝，按停止处理", flush=True)
                return None
            if not within_domain(p, cv, domain):
                print(f"[llm] 组合内 param={p} value={cv} 越界，整单拒绝，按停止处理",
                      flush=True)
                return None
            coerced[p] = cv
        return {"params": coerced}
    if not param or value is None:
        print(f"[llm] action 缺 param/value，按停止处理: {result!r}", flush=True)
        return None
    # 安全校验：param 必须在动作空间内 + value 在取值域内（防止 agent 越界）
    domain = flow.params.domain or {}
    if param not in domain:
        print(f"[llm] param={param} 不在动作空间，拒绝，按停止处理", flush=True)
        return None
    # 类型收敛：LLM 可能回传字符串形式（"true"/"8192"），先收敛到域声明类型再校验。
    # 否则 flag 字符串会被 within_domain 误拒（整环停止）、数值字符串会以 str 落进
    # MindIE config.json（daemon 拒收）。
    value, ok = _coerce_value(param, value, domain)
    if not ok:
        print(f"[llm] param={param} value={value!r} 无法收敛到声明类型，拒绝，按停止处理",
              flush=True)
        return None
    if not within_domain(param, value, domain):
        print(f"[llm] param={param} value={value} 越界，拒绝，按停止处理", flush=True)
        return None
    return {"param": param, "value": value}


# ---------------- 双目标影子记账（方案 1：主目标驱动 + 副目标记账） ----------------
# run_benchmark 每轮全量采集 ttft/thr 等指标，单目标 pass 却只按主指标判 keep——
# 第二遍换目标的 pass 把同参数重测一遍（案例6：pass1 r1 的 metrics 里已有
# thr=220.1，pass2 r1 重测出 221.3）。影子记账让一轮测量服务两个目标：
# 主目标照旧驱动 keep/回滚/服务配置（零语义改动）；副目标只记账
# state["best_by_target"][副]——主目标回滚但副目标大赢的 trade-off
# （FULL_DECODE_ONLY 型：TTFT 小亏 / 吞吐 +65%）不丢失，第二遍 pass 整遍省掉。

OTHER_TARGET = {"ttft": "throughput", "throughput": "ttft"}
_METRIC_KEYS = ("ttft_ms", "ttot_ms", "ttft_p95", "p99_ms", "tpot_ms", "tps", "thr_tok_s")


def _baseline_best(flow, baseline: dict, target: str) -> dict | None:
    """从 baseline 构造某目标的初始 best（value 统一最小化口径，与 best.value 同源）。"""
    if not baseline:
        return None
    obj = compute_objective(flow, baseline, baseline, target)
    if obj.get("value") is None:
        return None
    return {"target": target, "value": obj["value"], "round": "r0",
            "params": dict(baseline.get("params") or {}), "improved_pct": 0.0,
            "metrics": {k: baseline.get(k) for k in _METRIC_KEYS
                        if baseline.get(k) is not None}}


def _shadow_update(flow, state, metrics, cfg_params, round_key: str, main_target: str,
                   accuracy_match=None, accuracy_detail=None) -> dict | None:
    """副目标影子记账：metrics 对副目标跑完整判定（含副目标自己的 guard/噪声地板），
    更优则更新 state["best_by_target"][副目标]，返回新 best（未更新返回 None）。

    与主目标 keep/回滚完全解耦；主目标 best 走原 state["best"]，不在此重复
    （防双写漂移）。调用点必须在回滚发生前（current_params 仍为本轮候选配置）。
    """
    other = OTHER_TARGET.get(main_target)
    if not other or not metrics:
        return None
    baseline = state.get("baseline") or {}
    if baseline.get(primary_key(flow, other)) is None:
        return None  # 后端不报该指标（如无 token 数 → 无 thr）→ 影子记账自动不生效
    bbt = state.setdefault("best_by_target", {})
    cur = bbt.get(other) or _baseline_best(flow, baseline, other)
    det = keep_verdict_detail(flow, metrics, baseline, other, best=cur)
    if not det.get("kept"):
        return None
    obj = compute_objective(flow, metrics, baseline, other,
                            accuracy_match, accuracy_detail)
    if obj.get("value") is None:
        return None
    if cur and cur.get("value") is not None and cur["value"] <= obj["value"]:
        return None
    new = {"target": other, "value": obj["value"], "round": round_key,
           "params": dict(cfg_params or {}),
           "improved_pct": det.get("improved_pct"),
           "metrics": {k: metrics.get(k) for k in _METRIC_KEYS
                       if metrics.get(k) is not None}}
    bbt[other] = new
    return new


def try_action(flow, cfg, state, action: dict, round_key: str) -> dict:
    """内层确定性原子操作：apply → verify → is_kept → 变差自动回滚。

    返回本轮记录 dict（落盘到 state["rounds"]）。
    """
    baseline = state.get("baseline") or {}
    app = apply_action(flow, cfg, state, action, round_key)
    if not app["ok"]:
        # P0-B：失败计数 +1（连续 ≥3 触发熔断暂停）；P1-A：根因摘录进 rounds
        state["apply_fail_streak"] = (state.get("apply_fail_streak") or 0) + 1
        reason = _fail_reason(app.get("log"))
        rec = {"round": round_key, "action": action, "changed": action_changed(action),
               "kept": False, "decision": "rollback", "metrics": None,
               "rationale": f"apply 失败: {app['log'][-200:]}",
               "rolled_back": app.get("rolled_back", False)}
        if reason:
            rec["fail_reason"] = reason
        print(f"[try] {action_label(action)} apply 失败 "
              f"(回滚={app.get('rolled_back')}){f'：{reason}' if reason else ''}", flush=True)
        return rec
    state["apply_fail_streak"] = 0  # apply 成功即清零（P0-B 熔断计数）

    ver = verify(flow, cfg, state, round_key)
    metrics = ver["metrics"]
    # 确定性硬护栏：smoke/精度/图捕获任一不过 → 强制回滚，绝不把死服务/精度破坏判成 keep。
    # accuracy_match/graph_ok 的 None = 不判（未配置/不可判），不误伤，只拦 is False。
    if not ver.get("smoke_ok", True):
        rollback_to_last(flow, cfg, state, round_key)
        rec = {"round": round_key, "action": action, "changed": action_changed(action),
               "kept": False, "decision": "rollback", "metrics": metrics,
               "rationale": "smoke 失败，服务不可推理，强制回滚", "rolled_back": True}
        print(f"[try] {action_label(action)} smoke 失败，强制回滚", flush=True)
        return rec
    if ver.get("accuracy_match") is False:
        cls = (ver.get("accuracy_detail") or {}).get("cls")
        if cls != "A":
            # B 类语义破坏（乱码/重复/截断/分叉在开头）：硬否决，强制回滚
            rollback_to_last(flow, cfg, state, round_key)
            rec = {"round": round_key, "action": action, "changed": action_changed(action),
                   "kept": False, "decision": "rollback", "metrics": metrics,
                   "accuracy_detail": ver.get("accuracy_detail"),
                   "rationale": "精度比对失败(B类语义破坏)，强制回滚", "rolled_back": True}
            print(f"[try] {action_label(action)} 精度破坏(B类)，强制回滚", flush=True)
            return rec
        # A 类数值漂移（阶段 A 核心变化）：放行，由 compute_objective 记 accuracy_A 惩罚，
        # 不再因 1 条长序列后段的 token 分叉否决真实性能收益（问题④）。
        print(f"[try] {action_label(action)} A 类数值漂移，放行（记惩罚）", flush=True)
    if ver.get("graph_ok") is False:
        # 图捕获未生效：配置声称开图但实际被平台跳过/未捕获，配置没按预期落地 → 强制回滚
        rollback_to_last(flow, cfg, state, round_key)
        rec = {"round": round_key, "action": action, "changed": action_changed(action),
               "kept": False, "decision": "rollback", "metrics": metrics,
               "rationale": "图捕获未生效（模式被跳过/未捕获），强制回滚", "rolled_back": True}
        print(f"[try] {action_label(action)} 图捕获未生效，强制回滚", flush=True)
        return rec
    if not metrics and baseline:
        # 压测失败降级 10x 基线（沿用原 verify 契约），必判不 keep → 自动回滚
        metrics = {"ttft_ms": baseline["ttft_ms"] * 10, "ttot_ms": baseline["ttot_ms"] * 10}
    # keep 判定：复利水位线（rolling best + ε）+ 噪声地板（P0-2，hyst_eff=max(hyst, k×相对噪声)）
    det = keep_verdict_detail(flow, metrics, baseline, cfg.target, best=state.get("best"))
    remeasured = False
    if det["borderline"]:
        # 临界轮（改善过了配置门槛 hyst 但被噪声地板 hyst_eff 拦下）：重测一次，
        # 两次 median 取均值后按同一 hyst_eff 终判——边界抖动的真改善不该被单次
        # 采样误杀。每轮最多重测 1 次防爆预算。
        m2 = run_benchmark(cfg, state, f"{round_key}_remeas")
        if m2:
            merged = dict(metrics)
            for k in ("ttft_ms", "ttot_ms", "thr_tok_s", "tpot_ms", "tps"):
                if metrics.get(k) is not None and m2.get(k) is not None:
                    merged[k] = (metrics[k] + m2[k]) / 2
            merged["stats"] = m2.get("stats") or metrics.get("stats")
            metrics = merged
            det = keep_verdict_detail(flow, metrics, baseline, cfg.target,
                                      best=state.get("best"))
            remeasured = True
            print(f"[remeas] {action_label(action)} 临界重测（改善 {det['improved_pct']:+.1f}% "
                  f"vs 地板 {det['hyst_eff']*100:.2f}%），两次均值后终判", flush=True)
    kept, rationale = det["kept"], det["rationale"]
    # 阶段 A：objective 标量化——每轮记标量 value，供 Sampler/Pruner/best 消费（阶段 B/C 依赖）
    obj = compute_objective(flow, metrics, baseline, cfg.target,
                            ver.get("accuracy_match"), ver.get("accuracy_detail"))

    # 双目标影子记账（方案 1）：副目标大赢的 trade-off 不因主目标回滚丢失。
    # 调用点在 rollback 之前——current_params 仍为本轮候选配置。
    shadow = _shadow_update(flow, state, metrics, state.get("current_params"),
                            round_key, cfg.target,
                            ver.get("accuracy_match"), ver.get("accuracy_detail"))
    if shadow:
        ok = OTHER_TARGET.get(cfg.target)
        mk = primary_key(flow, ok)
        print(f"[shadow] 副目标({ok})最优更新: {mk}={shadow['metrics'].get(mk)} "
              f"(改善 {shadow['improved_pct']:+.1f}%)，配置={shadow['params']}", flush=True)

    if kept:
        # best 记账（阶段 A：argmin objective value，A 类精度惩罚参与排序）
        nb = best_candidate(state, flow, round_key, metrics, cfg.target, value=obj["value"])
        if nb:
            state["best"] = nb
    else:
        # 变差自动回滚（确定性硬护栏）
        rollback_to_last(flow, cfg, state, round_key)

    rec = {"round": round_key, "action": action, "changed": action_changed(action),
           "kept": kept, "decision": "keep" if kept else "rollback",
           "metrics": metrics, "objective": obj, "rationale": rationale,
           "rolled_back": not kept}
    if remeasured:
        rec["remeasured"] = True
    print(f"[try] {action_label(action)} → kept={kept}, "
          f"{primary_key(flow, cfg.target)}={metrics.get(primary_key(flow, cfg.target), '?')} | {rationale}",
          flush=True)
    return rec


def _init_baseline(flow, cfg, state) -> bool:
    """建立基线（round 0）：强制重启主服务为 defaults 再压测。返回是否成功。

    不复用端口已有服务——那可能是上轮 keep 的候选配置，直接压测会把候选性能
    误记为 defaults 基线。统一走 restart_to_defaults 现场渲染并重启。
    """
    from engine.stages.apply import restart_to_defaults
    print("[round 0] 重启主服务为 defaults 配置 + 采集基线", flush=True)
    if not restart_to_defaults(flow, cfg, state):
        return False
    m = run_benchmark(cfg, state, "baseline")
    if not m:
        print("FATAL: 基线压测失败（服务不可用？）", flush=True)
        return False
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
                         "stats": m.get("stats"),  # P0-2：基线噪声地板的参照
                         "params": dict(flow.params.defaults),
                         "outputs": outputs, "outputs_ppl": outputs_ppl}
    thr = f" 吞吐={m['thr_tok_s']:.1f}t/s" if m.get("thr_tok_s") else ""
    print(f"[round 0] 基线 TTFT={m['ttft_ms']:.1f}ms TTOT={m['ttot_ms']:.1f}ms P99={m['p99_ms']:.1f}ms{thr}", flush=True)
    # 双目标影子记账：两目标都从基线起步（主目标运行期走 state["best"]，不双写）
    state["best_by_target"] = {t: bb for t in ("ttft", "throughput")
                               if (bb := _baseline_best(flow, state["baseline"], t))}
    return True


def _degrade_domain(flow, state, reason: str) -> None:
    """降级使用手写动作空间（agent 明示 use_default / 回传非法 / 校验全灭）。"""
    handwritten = flow.params.domain or {}
    state["domain_ready"] = True
    state["domain"] = {"source": "handwritten-fallback", "reason": reason,
                       "merged_param_count": len(handwritten)}
    domain_node.record_domain(flow.flow_id, handwritten, state["domain"])
    print(f"[domain] 降级使用手写动作空间（{len(handwritten)} 参数）：{reason}", flush=True)


def _domain_stage(flow, cfg, state) -> int | None:
    """KG 构建动作空间必走节点（round 0 前执行，fresh run 必走）。

    流程：确定性 KG 检索（domain_node.search_materials）→ 暂停点（LLM 提取，回传
    ok/use_default/terminate）→ 结构校验 → 与手写 domain 字段级合并（KG 字段覆盖、
    缺失字段手写继承）→ 安装 flow.params.domain + 落盘 domain_cache 记录。

    返回：None = 继续主流程；2 = 暂停待 agent 回传（--resume 续跑）；
    0 = agent 判定 terminate（已收尾）。
    """
    ctx = {"round_key": "r0", "stage": "domain"}
    res = agent_gate.consume_result(ctx)
    if res is None:
        materials = domain_node.search_materials(flow)
        raw_path = domain_node.dump_raw(flow.flow_id, getattr(flow, "backend", ""),
                                        materials)
        agent_gate.emit_task(
            ctx,
            goal=domain_node.DOMAIN_GOAL,
            skill_refs=[],
            schema=domain_node.DOMAIN_SCHEMA,
            context=domain_node.domain_task_context(flow, materials),
        )
        state["pending"] = {"round_key": "r0", "stage": "domain",
                            "task_id": ctx["pending_task_id"]}
        state["snapshot"] = None
        state["status"] = "pending"
        save_state(state, st.RUN_STATE)
        print(f"\n[pending] 等待 agent 构建动作空间 {ctx['pending_task_id']}："
              f"读 engine/state/agent_task.json → 写 agent_result.json → --resume"
              f"（kg_status={materials['kg_status']}，材料落盘 {raw_path.name}）", flush=True)
        return 2

    state["pending"] = None
    state["snapshot"] = None
    action = res.get("action") if isinstance(res, dict) else None
    reason = res.get("reason", "") if isinstance(res, dict) else ""

    if action == "terminate":
        state["domain_ready"] = True
        state["domain"] = {"source": "terminated", "reason": reason}
        state["status"] = "complete"
        save_state(state, st.RUN_STATE)
        print(f"[domain] agent 判定 terminate：{reason}", flush=True)
        print("\n========== 闭环完成（domain 节点终止） ==========", flush=True)
        return 0

    if action == "use_default":
        _degrade_domain(flow, state, reason or "agent 明示 use_default")
        save_state(state, st.RUN_STATE)
        return None

    if action != "ok" or not isinstance(res.get("domain"), dict):
        _degrade_domain(flow, state,
                        f"回传非法（action={action!r}, domain 是否 dict="
                        f"{isinstance(res.get('domain') if isinstance(res, dict) else None, dict)}）：{reason}")
        save_state(state, st.RUN_STATE)
        return None

    handwritten = flow.params.domain or {}
    valid, dropped = domain_node.validate_kg_domain(res["domain"], handwritten)
    if not valid:
        _degrade_domain(flow, state, f"KG 提取条目全部未过结构校验：{dropped}")
        save_state(state, st.RUN_STATE)
        return None

    merged = domain_node.merge_domain(handwritten, valid)
    new_params = sorted(p for p in valid if p not in handwritten)
    overridden = sorted(p for p in valid if p in handwritten)
    flow.params.domain = merged
    flow.domain_source = "kg-merged"
    # 交互先验（P0-1 层 2）：LLM 在同一回传里提取的联动组——与手写 flow 先验并集后
    # 供 plan_composition 消费。与动作空间同生命周期：KG 新增参数的交互知识自动带上，
    # 不需要改 flow JSON（三层数据源设计，见 engine-remediation-plan §1.1）。
    kg_inter, dropped_inter = domain_node.validate_interactions(
        res.get("interaction_priors"), merged)
    state["domain_ready"] = True
    state["domain"] = {
        "source": "kg-merged",
        "reason": reason,
        "kg_param_count": len(res["domain"]),
        "valid_param_count": len(valid),
        "merged_param_count": len(merged),
        "new_params": new_params,
        "overridden_params": overridden,
        "dropped": dropped,
        "kg_interactions": kg_inter,
        "dropped_interactions": dropped_inter or None,
    }
    domain_node.record_domain(flow.flow_id, merged, state["domain"])
    save_state(state, st.RUN_STATE)
    inter_note = (f"，交互先验 {len(kg_inter)} 组"
                  + (f"（丢弃 {dropped_inter}）" if dropped_inter else ""))
    print(f"[domain] source=kg-merged：手写{len(handwritten)} + KG新增{len(new_params)}"
          f" + 覆盖{len(overridden)} → 共{len(merged)}参数{inter_note}"
          f"{f'（丢弃不合格 {dropped}）' if dropped else ''}", flush=True)
    return None


# ---------------- 幕二：组合验证（P0-1 两幕编排） ----------------


def _eff_priors(flow, state) -> list:
    """有效交互先验 = 手写 flow JSON（层1兜底）∪ KG 同源提取（层2，domain 节点落盘）。

    frozenset 去重后返回 [[param, ...], ...]（grouping.group 的 prior_interactions 形态）。
    KG 新增参数的交互知识随 domain 节点自动带上——不需要为新参数改 flow JSON。
    """
    groups = []
    for g in (getattr(flow, "interaction_priors", None) or []):
        groups.append(g.get("params") if isinstance(g, dict) else g)
    for g in ((state.get("domain") or {}).get("kg_interactions") or []):
        groups.append(g.get("params"))
    seen, out = set(), []
    for g in groups:
        k = frozenset(g or [])
        if len(k) >= 2 and k not in seen:
            seen.add(k)
            out.append(sorted(k))
    return out


def _combo_rounds(state: dict) -> list:
    """组合幕轮记录（round_key 以 c 开头），按落盘顺序。"""
    return [r for r in state.get("rounds") or []
            if isinstance(r.get("round"), str) and r["round"].startswith("c")]


def _interactive_best(state: dict, interactive: list) -> dict:
    """交互子集的组合最优取值：kept 的组合轮里 objective value 最小者；
    无 kept 组合轮时退化为各参数的单参数 keep 值（final_sweep 对拍兜底语义）。"""
    kept_combos = [r for r in _combo_rounds(state) if r.get("kept")]
    if kept_combos:
        best_rec = min(kept_combos, key=lambda r: (r.get("objective") or {}).get("value", float("inf")))
        sel = {p: v for p, v in (best_rec.get("action") or {}).get("params", {}).items()
               if p in interactive}
        if sel:
            return sel
    return {p: v for p, v in (grouping.kept_params(state, None) or {}).items() if p in interactive}


def _enter_composition(flow, cfg, state) -> None:
    """幕一（值级探索）收敛后的幕二入口：分组降维 → 组合队列 → state 切 phase。

    composition_enabled=false 或组合候选为空时跳过（phase=done，直接进收尾）。
    预算：组合候选数截断到 max-combo-rounds（截断大声声明——被截掉的部分由
    final_sweep 拼装对拍 + gap 检测兜底，不是静默丢弃）。
    """
    if state.get("phase") == "composition":
        return  # resume 已在组合幕
    if not getattr(flow.perf, "composition_enabled", True):
        state["phase"] = "done"
        print("[composition] composition_enabled=false，跳过组合幕", flush=True)
        return
    try:
        plan = grouping.plan_composition(state, flow, cfg.target,
                                         prior_interactions=_eff_priors(flow, state))
    except Exception as e:  # 分组降维是增值路径，失败不拖垮收尾
        state["phase"] = "done"
        print(f"[composition] 分组降维产出失败（非致命，跳过组合幕）: {e}", flush=True)
        return
    combos = plan["combinations"]
    if len(combos) > cfg.max_combo_rounds:
        print(f"[composition] ⚠️ 组合候选 {len(combos)} 个超预算 {cfg.max_combo_rounds}，"
              f"截断为前 {cfg.max_combo_rounds} 个（截断部分由 final_sweep 对拍 + gap 检测兜底）",
              flush=True)
        combos = combos[:cfg.max_combo_rounds]
    state["composition_plan"] = plan
    state["composition_backbone"] = plan["backbone"]
    if not combos:
        state["phase"] = "done"
        print(f"[composition] 无交互子集组合（backbone={list(plan['backbone']) or '空'}），"
              f"跳过组合幕", flush=True)
        return
    state["phase"] = "composition"
    state["composition_queue"] = combos
    save_state(state, st.RUN_STATE)
    print(f"[composition] 进入组合幕：interactive={plan['interactive']} "
          f"backbone={list(plan['backbone']) or '空'} 队列={len(combos)} 预算={cfg.max_combo_rounds}",
          flush=True)


def _match_queue_combo(action_params: dict, queue: list) -> dict | None:
    """LLM 回传的组合 ↔ 队列成员匹配（严格 frozenset 成员校验，容忍带 backbone）。

    先精确匹配；不中则容忍「把 backbone 参数一起带回」的形态——任一队列组合是
    回传的子集时收窄到该组合（backbone 取值已在 current_params 里，叠加语义不变）。
    队列外自造组合返回 None（预算按队列截断过，不允许绕过）。
    """
    sel = frozenset((p, v) for p, v in (action_params or {}).items()
                    if p is not None and v is not None)
    qks = [frozenset(q.items()) for q in queue]
    for qk in qks:
        if qk == sel:
            return dict(qk)
    for qk in qks:  # 子集收窄
        if qk and qk <= sel:
            return dict(qk)
    return None


def _run_composition_phase(flow, cfg, state) -> int | None:
    """组合幕主循环：c1..cN 轮，每轮 LLM 从剩余队列选一个组合 → try_action 真机验证。

    与值级轮完全同构（_parse_decision 校验 + keep_verdict + 护栏 + 自动回滚），
    差异：跳过 capture（flow.perf.composition_capture=true 可开）；不调
    should_terminate（其 plateau 计数跨幕会把值级收敛尾巴误判成组合幕平台期——
    组合幕的终止条件就是队列耗尽 / LLM 软终止）。
    返回 None=正常走完；2=决策暂停待回传；其他非 None=异常退出码。
    """
    queue = state.get("composition_queue") or []
    backbone = state.get("composition_backbone") or {}
    interactive = (state.get("composition_plan") or {}).get("interactive") or []
    done = round_keys(state)
    tried = sampler.tried_values(state)

    for c in range(1, cfg.max_combo_rounds + 1):
        # P0-B 熔断检查：apply 连续失败 ≥3 → 暂停回传（组合轮 apply 同样会失败）
        fret = _fuse_check(flow, cfg, state)
        if fret is not None:
            return fret
        round_key = f"c{c}"
        if round_key in done:
            continue
        # 剔除已试组合（值级 frozenset 键去重）+ 同值单参数已试过的组合
        queue = [q for q in queue if frozenset(q.items()) not in tried]
        if not queue:
            print("[composition] 组合队列已耗尽（或剩余组合均已试过）", flush=True)
            break
        state["composition_queue"] = queue
        print(f"\n========== round {round_key}（组合幕） ==========", flush=True)

        # capture 默认关（组合幕凭队列+历史即可决策）；开启时暂停恢复走 snapshot 不重采
        signals = None
        pend = state.get("pending") or {}
        if pend.get("round_key") == round_key and state.get("status") == "pending":
            signals = (state.get("snapshot") or {}).get("signals")
        elif getattr(flow.perf, "composition_capture", False):
            cap = capture(flow, cfg, state, round_key)
            signals = cap.get("signals")

        ctx = {"round_key": round_key, "stage": "decide"}
        decision = agent_gate.consume_result(ctx)
        if decision is None:
            agent_gate.emit_task(
                ctx,
                goal=_COMPOSITION_GOAL,
                skill_refs=[],
                schema=_DECIDE_SCHEMA,
                context=_build_decision_context(
                    flow, state, signals, round_key, _COMPOSITION_RULES, cfg.target,
                    phase="composition", queue=queue, backbone=backbone, cfg=cfg),
            )
            state["pending"] = {"round_key": round_key, "stage": "decide",
                                "task_id": ctx["pending_task_id"]}
            state["snapshot"] = {"signals": signals}
            state["status"] = "pending"
            state["composition_queue"] = queue
            save_state(state, st.RUN_STATE)
            print(f"\n[pending] 等待 agent 选择组合 {ctx['pending_task_id']}："
                  f"读 engine/state/agent_task.json → 写 agent_result.json → --resume", flush=True)
            return 2

        action = _parse_decision(decision, flow)
        if action is None:
            print("[composition] 决策无效，组合幕终止", flush=True)
            break
        if "param" not in action and "params" not in action:
            rec = {"round": round_key, "action": {}, "changed": [],
                   "kept": False, "decision": "no_candidate", "metrics": None,
                   "rationale": decision.get("reason", "LLM 放弃剩余组合") if isinstance(decision, dict) else "LLM 放弃剩余组合",
                   "rolled_back": False}
            print(f"[llm] 组合幕软终止: {rec['rationale']}", flush=True)
            state["rounds"].append(rec)
            state["pending"] = None
            state["snapshot"] = None
            state["composition_queue"] = queue
            save_state(state, st.RUN_STATE)
            break

        matched = _match_queue_combo(action.get("params"), queue)
        if matched is None:
            rec = {"round": round_key, "action": action, "changed": action_changed(action),
                   "kept": False, "decision": "rejected_offqueue", "metrics": None,
                   "rationale": "回传组合不在 composition_queue 内（自造组合被拒），本轮不执行",
                   "rolled_back": False}
            print(f"[gate] {action_label(action)} 不在组合队列内，确定性拒绝（不浪费真机轮）",
                  flush=True)
            state["rounds"].append(rec)
            state["pending"] = None
            state["snapshot"] = None
            save_state(state, st.RUN_STATE)
            # #10：组合幕不走 should_terminate（plateau 跨幕误判），但「连拒 ≥3 =
            # 决策面失效」是 phase 局部信号——无测量空转烧组合预算，止损进 final_sweep
            if consec_rejected(state["rounds"]) >= 3:
                print("[composition] ⚠️ 连续 3 轮提议被拒（自造组合），止损进终局 sweep",
                      flush=True)
                break
            continue

        action = {"params": matched}
        print(f"[llm] 组合决策: {action_label(action)}（叠加 backbone）", flush=True)
        rec = try_action(flow, cfg, state, action, round_key)
        state["rounds"].append(rec)
        state["apply_pending"] = None
        state["pending"] = None
        state["snapshot"] = None
        # 记账后同步 tried（本轮组合即刻从队列视角消失，防同轮重选）
        tried.add(frozenset(matched.items()))
        queue = [q for q in queue if frozenset(q.items()) != frozenset(matched.items())]
        state["composition_queue"] = queue
        save_state(state, st.RUN_STATE)

    state["phase"] = "done"
    state["pending"] = None
    state["snapshot"] = None
    save_state(state, st.RUN_STATE)
    return None


def _final_sweep(flow, cfg, state) -> None:
    """终局 SWEEP：backbone + 交互最优组合 拼装成一次完整配置真机对拍（notes §3 兜底）。

    回答「局部最优拼起来是否真最优」：拼装配置 vs rolling best 的 gap > 0.5% 时
    记 state["unmodeled_interactions"] 告警（三层数据源的层 3 gap 反馈——提示先验
    遗漏了真实交互，下轮 run 应补充先验或扩大交互子集）。
    """
    if "sweep" in round_keys(state):
        return
    plan = state.get("composition_plan")
    if not plan:
        return
    backbone = plan.get("backbone") or {}
    interactive = plan.get("interactive") or []
    interactive_best = _interactive_best(state, interactive)
    combo = dict(backbone)
    combo.update(interactive_best)
    cur = state.get("current_params") or {}
    if all(cur.get(p) == v for p, v in combo.items()):
        print("[sweep] 拼装配置与当前运行配置一致，跳过终局对拍", flush=True)
        return
    print(f"[sweep] 终局拼装验证：{'+'.join(f'{p}={v}' for p, v in combo.items()) or '(空)'}",
          flush=True)
    rec = try_action(flow, cfg, state, {"params": combo}, "sweep")
    rec["sweep"] = True
    state["rounds"].append(rec)
    save_state(state, st.RUN_STATE)

    # gap 反馈（层 3）：拼装劣于 rolling best → 未建模交互告警
    key = primary_key(flow, cfg.target)
    best = state.get("best") or {}
    b, s = best.get(key), (rec.get("metrics") or {}).get(key)
    if b and s:
        gap_pct = (b - s) / b * 100 if key == "thr_tok_s" else (s - b) / b * 100
        if gap_pct > 0.5:
            state["unmodeled_interactions"] = {
                "gap_pct": round(gap_pct, 2), "key": key,
                "assembled": combo, "best_round": best.get("round"),
                "note": "拼装配置劣于各部分最优，疑似存在未建模的参数交互——"
                        "下轮 run 应补充 interaction_priors 或扩大交互子集",
            }
            save_state(state, st.RUN_STATE)
            print(f"[sweep] ⚠️ 拼装配置 {key} 劣于 best {gap_pct:.2f}%，"
                  f"记录 unmodeled_interactions（先验遗漏告警）", flush=True)


def _diag_fingerprint(signals: dict) -> list:
    """节流指纹 = [bottleneck_class, top-3 热点算子名]（JSON 可序列化）。"""
    ops = [h.get("op") for h in (signals.get("hot_ops") or [])[:3]]
    return [signals.get("bottleneck_class"), ops]


def _maybe_diagnose(flow, cfg, state, signals, round_key: str) -> int | None:
    """P1-3 诊断→决策联动：可选 rN-diagnose 暂停点（协议复用 agent_gate）。

    触发条件（全部满足才 emit）：
    - signals 非空（capture 成功）且 flow 配了 msot/mfu skill 路径（diagnose 降级链）；
    - 本 run 诊断次数 < 2（主会话 agent 负担上限）；
    - 指纹变化：上次诊断时的 [bottleneck_class, hot_ops top-3] 与本轮 signals 不同
      （同类瓶颈不重复诊断——ERNIE run 已证明根因结论跨轮稳定）。

    消费路径：resume 时 consume 命中 → state["diagnosis"] 更新 + 计数/指纹记账。
    返回 None = 继续（无需诊断/已消费/节流跳过）；2 = 已 emit 暂停待回传。
    """
    if not signals:
        return None
    msot = getattr(flow.paths, "msot_skill", None)
    mfu = getattr(flow.paths, "mfu_skill", None)
    if not (msot or mfu):
        return None  # 配置缺 skill 路径 → 降级跳过（沿用 diagnose.py 的降级语义）

    ctx = {"round_key": round_key, "stage": "diagnose"}
    result = agent_gate.consume_result(ctx)
    if result is None:
        count = state.get("diagnosis_count") or 0
        fp_now = _diag_fingerprint(signals)
        if count >= 2:
            return None  # 预算耗尽
        if state.get("diag_fingerprint") and state.get("diagnosis") \
                and state["diag_fingerprint"] == fp_now:
            return None  # 同指纹不重诊（节流核心）
        skill_refs = []
        if msot:
            skill_refs.append({"name": "msot-msopprof-operator-profiler", "path": msot,
                               "note": "读算子 profiler CSV 定位瓶颈 + 给优化建议的方法论"})
        if mfu:
            skill_refs.append({"name": "op-mfu-calculator", "path": mfu,
                               "note": "MFU 公式与解读（signals 已含 MFU，用于判 compute bound）"})
        agent_gate.emit_task(
            ctx,
            goal=("定位 profiling 热点算子的性能瓶颈根因，给出算子级优化建议"
                  "（tiling/dtype/融合/算子选型），不动算子源码"),
            skill_refs=skill_refs,
            schema={
                "bottleneck_class": "string: compute|schedule|comm|unknown",
                "hot_ops": [{"op": "string", "root_cause": "string", "suggestion": "string"}],
                "operator_level_suggestions": ["string: tiling/dtype/融合/选型建议"],
                "summary": "string: 一句话诊断结论",
            },
            context={
                "signals": signals,
                "history_digest": _fmt_history(state, flow, cfg.target)[-1500:],
                "note": ("signals 含 MFU/热点算子/wait 时长/bottleneck_class 粗分类；"
                         "kernel_details.csv 在 per_round 目录下，如需可读。"
                         "诊断结论将注入后续每轮决策上下文（解释动作空间为何有/无收益）"),
            },
        )
        state["pending"] = {"round_key": round_key, "stage": "diagnose",
                            "task_id": ctx["pending_task_id"]}
        state["snapshot"] = {"signals": signals}
        state["status"] = "pending"
        save_state(state, st.RUN_STATE)
        print(f"\n[pending] 等待 agent 诊断 {ctx['pending_task_id']}："
              f"读 engine/state/agent_task.json（含 skill_refs）→ 写 agent_result.json"
              f" → --resume", flush=True)
        return 2

    # 消费成功：记账（诊断进决策上下文由 _build_decision_context 注入）
    state["pending"] = None
    state["snapshot"] = None
    state["diagnosis"] = result
    state["diagnosis_count"] = (state.get("diagnosis_count") or 0) + 1
    state["diag_fingerprint"] = _diag_fingerprint(signals)
    save_state(state, st.RUN_STATE)
    summ = result.get("summary", "") if isinstance(result, dict) else ""
    print(f"[diagnose] 结论（第 {state['diagnosis_count']} 次）: "
          f"{summ[:120] or result}", flush=True)
    return None


def _refine_used(state: dict) -> int:
    return sum(1 for r in state.get("rounds") or [] if r.get("refine"))


def _drain_refines(flow, cfg, state) -> None:
    """排空邻域补扫队列（②a 探索/精修预算分离）。

    对「已 keep 但未确认局部最优」的参数，把 center±step 的未试值逐点真机补扫
    （每轮一个值，保持单变量归因）。定位是**精修**不是探索：
    - 不占 max_rounds 探索预算（独立 cfg.max_refine_rounds 封顶）；
    - 不走 decide 暂停点（确定性微调无需 LLM，符合内层确定性原子操作定位）；
    - 不做 profiling capture（try_action 的 verify 自带压测，省诊断成本）；
    - rollback 不计探索 plateau（_consec_no_improve 跳过 refine 轮）——补扫
      rollback 是「确认原值已局部最优」的预期结果。
    队列由 pending_neighborhood 按 tried_values 幂等重算 → resume 天然续跑不重扫；
    keep 后 center 移动会链式前进（真 line search），由总预算封顶。
    """
    from engine.tools.sampler import pending_neighborhood
    while True:
        used = _refine_used(state)
        pend = pending_neighborhood(state, flow, cfg.target)
        if not pend:
            return
        if used >= cfg.max_refine_rounds:
            params = [p["param"] for p in pend]
            print(f"[refine] ⚠️ 精修预算耗尽（{used}/{cfg.max_refine_rounds}），"
                  f"剩余未扫邻域: {params}（其局部最优性未确认）", flush=True)
            return
        p, val = pend[0]["param"], pend[0]["values"][0]
        n = used + 1
        round_key = f"f{n}"
        print(f"\n========== refine round {n}（{p}={val}，精修不占探索预算） ==========",
              flush=True)
        rec = try_action(flow, cfg, state, {"param": p, "value": val}, round_key)
        rec["refine"] = True
        state["rounds"].append(rec)
        state["apply_pending"] = None
        state["pending"] = None
        state["snapshot"] = None
        save_state(state, st.RUN_STATE)
        print(f"[refine] {p}={val} → {rec['decision']}", flush=True)


# ---------- 粗筛幕（③：全参数最大对比度单点快筛，值级幕前必经） ----------

def _screen_params(flow) -> list:
    """粗筛资格参数及其哨兵值（值域内必合法），按风险升序返回。

    排除 locked（不可改）与 compose（派生合成，禁止直设）——其余全部入选，
    覆盖保证是粗筛幕的存在理由（LLM 是 exploit 型选择器，冷门参数不会自己登场）。
    哨兵值选取（default 取 flow.params.defaults，未列 = 引擎缺省）：
    - values 枚举：全值展开（default 已知的值 = 基线状态本身，无对比度，跳过）。
      case7 教训：单哨兵 vals[0] 恰好错过 FULL_DECODE_ONLY——枚举各档机制差异大
      （只 decode 整图 vs 含 prefill 分段 vs 关图），单点不构成覆盖
    - flag 布尔：default 已知 → 翻转；未知 → True（开启是典型杠杆方向）
    - 数值：default 已知 → min/max 离 default 较远端；未知 → max（调参以扩资源向为主）

    排序 = 风险升序（low → mid → high，未标注按 low；同风险保持 domain 声明序）：
    易炸参数（编译超时/改 dtype/图模式）排最后，失败或熔断（P0-B）发生时低风险
    参数的画像已到手。case7 教训：static-kernel 无序排在第 14 位，编译超时的
    孤儿显存连锁毒死后面 11 轮，粗筛画像丢 46%。
    """
    dom = flow.params.domain or {}
    defaults = flow.params.defaults or {}
    out = []
    for name, d in dom.items():
        if d.get("locked") or d.get("compose"):
            continue
        if d.get("values"):
            dv = defaults.get(name)
            out.extend((name, v) for v in d["values"] if v != dv)
        elif d.get("flag"):
            dv = defaults.get(name)
            sent = (not dv) if dv is not None else True
            out.append((name, sent))
        else:
            lo, hi = d.get("min"), d.get("max")
            if lo is None or hi is None or hi <= lo:
                continue  # 域不完整（值级轮也设不了），跳过
            dv = defaults.get(name)
            sent = hi if dv is None or (hi - dv) >= (dv - lo) else lo
            out.append((name, sent))
    risk_rank = {"low": 0, "mid": 1, "high": 2}
    return sorted(out, key=lambda pv: risk_rank.get((dom.get(pv[0]) or {}).get("risk"), 0))


def _pct(cur, base, up: bool):
    """改善百分比（up=True 越大越好）；缺数据返回 None。"""
    if cur is None or not base:
        return None
    return (cur - base) / base * 100 if up else (base - cur) / base * 100


def _screen_round(flow, cfg, state, param, value, round_key) -> tuple:
    """粗筛单轮：apply → verify（reps=1）→ 恒回滚（仅还 state，物理重启由幕末统一做）。

    OFAT 归因——每个参数从基线配置出发单变量测量，不与前一轮叠加。恒回滚：
    粗筛只测量不占领（keep 语义留给值级幕的完整判定 reps=3+噪声地板）。
    中间轮不做物理回滚（下一轮 apply 本就会 kill+重启，省一次重启/轮）；
    apply 失败时 apply_action 内部已自动物理回滚自愈。
    返回 (round 记录, 榜单条目)。
    """
    action = {"param": param, "value": value}
    entry = {"param": param, "value": value, "round": round_key, "ok": False, "note": "",
             "applied": False}
    baseline = state.get("baseline") or {}
    prev_params = dict(state.get("current_params") or baseline.get("params") or {})
    import copy
    cfg1 = copy.copy(cfg)  # 浅拷贝只改 reps（backend/flow 引用只读共享）
    cfg1.reps = 1
    app = apply_action(flow, cfg1, state, action, round_key)
    if not app["ok"]:
        # P0-B：失败计数 +1（连续 ≥3 触发熔断暂停）；P1-A：根因摘录进 rounds
        state["apply_fail_streak"] = (state.get("apply_fail_streak") or 0) + 1
        reason = _fail_reason(app.get("log"))
        entry["note"] = f"apply 失败: {app['log'][-120:]}"
        rec = {"round": round_key, "action": action, "changed": action_changed(action),
               "kept": False, "decision": "screen", "metrics": None, "sweep": True,
               "rationale": entry["note"], "rolled_back": app.get("rolled_back", False)}
        if reason:
            rec["fail_reason"] = reason
        return rec, entry
    state["apply_fail_streak"] = 0  # apply 成功即清零（P0-B 熔断计数）
    ver = verify(flow, cfg1, state, round_key)
    entry["applied"] = True  # apply 成功（服务此刻跑哨兵配置），幕末需物理回滚
    metrics = ver["metrics"]
    acc_bad = (ver.get("accuracy_match") is False
               and (ver.get("accuracy_detail") or {}).get("cls") != "A")
    if not ver.get("smoke_ok", True):
        entry["note"] = "smoke 失败（服务不可推理）"
    elif acc_bad:
        entry["note"] = "精度 B 类语义破坏"
    elif metrics:
        # 双目标影子记账：副目标杠杆从粗筛就入账（测量配置=基线+哨兵，回滚前记录）
        _shadow_update(flow, state, metrics, state.get("current_params"),
                       round_key, cfg.target,
                       ver.get("accuracy_match"), ver.get("accuracy_detail"))
        entry["ok"] = True
        if ver.get("accuracy_match") is False:
            entry["note"] = "A 类数值漂移（放行，值级幕复测终判）"
        entry["ttft_pct"] = _pct(metrics.get("ttft_ms"), baseline.get("ttft_ms"), up=False)
        entry["thr_pct"] = _pct(metrics.get("thr_tok_s"), baseline.get("thr_tok_s"), up=True)
        entry["main_pct"] = (entry["thr_pct"] if cfg.target == "throughput"
                             else entry["ttft_pct"])
    else:
        entry["note"] = "压测失败（无有效样本）"
    # 还原 state 侧基线（服务物理回滚由幕末统一执行一次）
    state["current_params"] = prev_params
    rec = {"round": round_key, "action": action, "changed": action_changed(action),
           "kept": False, "decision": "screen", "metrics": metrics, "sweep": True,
           "rolled_back": True,
           "rationale": entry["note"] or "粗筛单点快测（reps=1），恒回滚保 OFAT 归因"}
    return rec, entry


def _run_screening(flow, cfg, state) -> int | None:
    """③ 粗筛幕：值级幕前对全部可设参数做最大对比度单点快筛。

    - 覆盖保证：每个可设参数至少 1 次真机登场，消灭「LLM exploit 型选择导致
      冷门参数永不登场」（案例6：24 参数只实测 3 个 ≈12% 覆盖）；
    - 榜单（state["screening"]）进 decide context——LLM 选参从信念排序变实测排序，
      副目标杠杆也全指标画像可见；
    - 不占 max_rounds、不走 decide 暂停点（确定性快筛无需 LLM）、不计 plateau
      （sweep 语义，_consec_no_improve 跳过）；
    - reps=1 只排序不裁决——假阳性由值级幕完整判定（reps=3+噪声地板）证伪；
    - 幂等续跑：按参数名查已完成条目；--screen off 关闭；幕末统一一次物理回滚
      还原基线服务（比逐轮回滚省 N-1 次重启）；
    - P0-B 熔断：每轮顶部检查 apply 连续失败（case7 scr14-24 连锁 11 轮空转的直接
      教训）——暂停/终止时穿透返回（2/0），dirty 幂等改从 state 派生，恢复后幕末
      回滚不丢。
    返回 None=正常走完；2=熔断暂停待回传；0=熔断 terminate 已收尾。
    """
    if not getattr(cfg, "screen", True) or state.get("phase", "value") != "value":
        return None
    # 幂等查重按 (param, value) 粒度：枚举参数全值展开后同参数有多个待测值，
    # 按参数名查重会漏测未出场档位；旧 run state（单哨兵形态）条目天然兼容。
    done = {(e.get("param"), e.get("value")) for e in (state.get("screening") or [])}
    todo = [(p, v) for p, v in _screen_params(flow) if (p, v) not in done]

    def _finalize_screening() -> None:
        """幕末物理还原（#1 修，2026-08-29）：dirty 从持久化 entries 派生（熔断暂停-
        恢复后局部变量会丢）。回滚目标 = scr1_prev（施加第一个哨兵**前**的脚本 =
        真基线）——_apply 备份的是施加前正在跑的脚本，末轮 scrN_prev 是
        「baseline+倒数第二个哨兵」，用它恢复 = 回滚错脚本（state 侧 current_params
        正确、服务侧脏，分叉）。还原成功才清 applied（幂等；失败保留标志 resume 重试）。"""
        if not any(e.get("applied") for e in state.get("screening") or []):
            return
        print("[screen] 粗筛幕收尾：物理回滚还原基线服务（供值级幕 r1 profiling 干净地基）",
              flush=True)
        rb = rollback_to_last(flow, cfg, state, "scr1")
        if rb.get("ok"):
            for e in state["screening"]:
                e["applied"] = False
        else:
            print("[screen] ⚠️ 幕末回滚未恢复健康（applied 标志保留，resume 重试）", flush=True)
        save_state(state, st.RUN_STATE)

    if not todo:
        # 幕已完成的 resume：上次进程可能恰好死在「循环结束~幕末回滚」之间——
        # 这里是唯一的还原时机（todo 非空时循环后的正常收尾会做，无需入口多做一次）
        _finalize_screening()
        return None
    if not state.get("screening"):
        state["screening"] = []
    print(f"[screen] 粗筛幕：{len(todo)} 个参数 × 最大对比度单点（reps=1，不占探索预算）",
          flush=True)
    for p, v in todo:
        fret = _fuse_check(flow, cfg, state)
        if fret is not None:
            return fret
        rk = f"scr{len(state['screening']) + 1}"
        print(f"\n========== screen {rk}: {p}={v} ==========", flush=True)
        rec, entry = _screen_round(flow, cfg, state, p, v, rk)
        state["rounds"].append(rec)
        state["screening"].append(entry)
        state["apply_pending"] = None
        state["pending"] = None
        state["snapshot"] = None
        save_state(state, st.RUN_STATE)
        sign = f"{entry.get('main_pct'):+.1f}%" if entry.get("main_pct") is not None else entry.get("note")
        print(f"[screen] {p}={v} → {sign}", flush=True)
    _finalize_screening()  # 幕末正常收尾：还原基线服务
    ok_rows = sorted((e for e in state["screening"] if e.get("ok")
                      and e.get("main_pct") is not None),
                     key=lambda e: abs(e["main_pct"]), reverse=True)
    print(f"[screen] 榜单（按主目标效应绝对值降序，前 5）：", flush=True)
    for e in ok_rows[:5]:
        bits = []
        if e.get("ttft_pct") is not None:
            bits.append(f"ttft {e['ttft_pct']:+.1f}%")
        if e.get("thr_pct") is not None:
            bits.append(f"吞吐 {e['thr_pct']:+.1f}%")
        print(f"[screen]   {e['param']}={e['value']}: {' | '.join(bits)}", flush=True)


def _screen_digest(state, target, flow=None) -> str:
    """粗筛榜单 → decide context 文本摘要（含值级复测置信反馈）。

    置信反馈（case7 教训：粗筛 4 个正向条目值级复测 3 个是噪声）：用已完成的
    值级轮对粗筛条目做同 (param, value) 匹配——主目标增量（vs 当时水位线）同号且 |增量|≥1% 为保真，异号或 |增量|<1% 为证伪不可兑现
    （reps=1 噪声或与已 keep 骨干负交互，两种成因都不值得追）。头部给汇总，
    证伪条目行尾标记，让 LLM 拿榜单时自带误差棒，不再把榜单正向当预验证结论。
    """
    entries = state.get("screening") or []
    if not entries:
        return ""
    rechecked, falsified = _screen_recheck(state, target, flow)
    lines = ["[粗筛榜单] reps=1 单点快测，仅排序参考（未经噪声地板确认，值级幕复测终判）；"
             "按主目标效应绝对值降序："]
    pos = [k for k, e in rechecked.items() if e["main_pct"] > 0]
    n_falsified_pos = sum(1 for k in pos if k in falsified)
    if pos:
        warn = ("——⚠️ 正向条目多数已被复测证伪为噪声，榜单只配排序，勿当预验证结论"
                if n_falsified_pos * 2 >= len(pos) and n_falsified_pos else
                "；正向条目复测证伪率见行尾标记")
        lines[0] += f"（正向已复测 {len(pos)} 项，{n_falsified_pos} 项证伪不可兑现{warn}）"
    ok_rows = sorted((e for e in entries if e.get("ok") and e.get("main_pct") is not None),
                     key=lambda e: abs(e["main_pct"]), reverse=True)
    for e in ok_rows:
        bits = []
        if e.get("ttft_pct") is not None:
            bits.append(f"ttft {e['ttft_pct']:+.1f}%")
        if e.get("thr_pct") is not None:
            bits.append(f"吞吐 {e['thr_pct']:+.1f}%")
        mark = "（复测已证伪→不可兑现）" if (e["param"], e["value"]) in falsified else \
               "（复测保真）" if (e["param"], e["value"]) in rechecked else ""
        lines.append(f"  {e['param']}={e['value']}: {' | '.join(bits)} {mark}".rstrip())
    for e in entries:
        if not e.get("ok"):
            lines.append(f"  {e['param']}={e['value']}: ✗ {e.get('note') or '失败'}")
    return "\n".join(lines)


def _screen_recheck(state, target, flow=None) -> tuple:
    """值级轮 → 粗筛条目的复测匹配。返回 ({(p,v): entry_ref}, falsified 集)。

    匹配规则：单参数 action（组合轮无 param/value 对，跳过）、非 sweep、有
    metrics 的值级轮；同 (p,v) 多次复测取首轮（值级幕勿重复已试参数，重复
    本身是异常，首轮即可代表）。

    复测口径（2026-08-29 修正）：**该轮相对当时水位线（rolling best）的增量**，
    与 keep_verdict 同口径——值级轮叠在已 keep 骨干上跑，vs 基线的 pct 里大头
    是骨干收益。case8 教训：r2 叠 FULL_DECODE_ONLY 骨干 +35.2%（vs 基线）被标
    「复测保真」，实际增量 -1.5%（rollback）——误差棒失真直接喂错了停机决策。
    水位链按轮序重构（kept 且主指标严格更优才推进，与 best_candidate 一致）。
    """
    entries = state.get("screening") or []
    if not entries:
        return {}, set()
    try:
        key = primary_key(flow, target) if flow is not None else None
    except Exception:
        key = None
    if not key:
        return {}, set()
    base = (state.get("baseline") or {}).get(key)
    if not base:
        return {}, set()
    up = (key == "thr_tok_s")  # 主指标改善方向（thr 越大越好，ttft 越小越好）
    scr = {(e.get("param"), e.get("value")): e for e in entries
           if e.get("ok") and e.get("main_pct") is not None}
    rechecked, falsified = {}, set()
    ref = base  # 水位线：从基线起，按轮序推进（= keep_verdict 的 rolling best）
    for r in state.get("rounds") or []:
        if r.get("sweep"):
            continue  # 粗筛/终局轮不参与
        a = r.get("action") or {}
        p, v = a.get("param"), a.get("value")
        kv = (p, v)
        cur = (r.get("metrics") or {}).get(key)
        if p is not None and kv not in rechecked and kv in scr and cur is not None:
            pct = (cur - ref) / ref * 100 if up else (ref - cur) / ref * 100
            rechecked[kv] = scr[kv]
            if abs(pct) < 1.0 or (pct > 0) != (scr[kv]["main_pct"] > 0):
                falsified.add(kv)  # 复测增量近零或与粗筛方向相反 → 粗筛效应不可兑现
        if r.get("kept") and not r.get("pareto") and isinstance(cur, (int, float)) and (
                (cur > ref) if up else (cur < ref)):
            ref = cur  # kept 且严格更优 → 水位线推进；pareto keep 一律不推进
            # （与 best_candidate 同口径——主目标水位线只认值级/精修/组合轮）
    return rechecked, falsified


def _drift_check(flow, cfg, state) -> None:
    """P0-2c 收尾漂移核验：run 结束时若服务仍跑 defaults 配置，复测一次主指标
    对比基线——|偏差|>5% 大声告警（环境漂移：同配置不同性能，本轮结论可信度存疑）。
    只告警不自动重评（重评需要重跑全量轮次，交人工决策）。非 defaults 配置不适用。
    """
    if state.get("drift_check"):
        return
    base_params = (state.get("baseline") or {}).get("params") or dict(flow.params.defaults)
    cur = state.get("current_params") or {}
    if any(cur.get(p) != v for p, v in base_params.items()):
        state["drift_check"] = {"applicable": False,
                                "note": "结束配置非 defaults（保留最优候选），漂移核验不适用"}
        save_state(state, st.RUN_STATE)
        return
    m = run_benchmark(cfg, state, "drift_check")
    if not m:
        state["drift_check"] = {"applicable": True, "ok": None, "note": "复测压测失败"}
        save_state(state, st.RUN_STATE)
        return
    key = primary_key(flow, cfg.target)
    b, c = (state.get("baseline") or {}).get(key), m.get(key)
    if not b or not c:
        state["drift_check"] = {"applicable": True, "ok": None, "note": "主指标缺失"}
        save_state(state, st.RUN_STATE)
        return
    pct = (c - b) / b * 100
    ok = abs(pct) <= 5.0
    state["drift_check"] = {"applicable": True, "ok": ok, "key": key,
                            "baseline": b, "remeasured": c, "drift_pct": round(pct, 2)}
    save_state(state, st.RUN_STATE)
    if ok:
        print(f"[drift] 漂移核验通过：{key} 复测 {c:.1f} vs 基线 {b:.1f}（{pct:+.2f}%）", flush=True)
    else:
        print(f"[drift] ⚠️ 环境漂移：defaults 复测 {key}={c:.1f} vs 基线 {b:.1f}"
              f"（{pct:+.2f}%，超 ±5%）——同配置性能显著变化，本轮改善结论可信度存疑，"
              f"建议排查环境后重跑（不自动重评）", flush=True)


def _maybe_anchor(flow, cfg, state) -> None:
    """P1-B 中途漂移锚点：每 anchor_every 个值级轮后复测当前运行配置（零重启）。

    case7 教训：漂移只在终局 drift_check 测（-46.5% 为时已晚）——run 后半段的候选
    对比的是数小时前的基线，host 劣化被记成参数劣化（r12 的「病态 -14.6%」混入漂移）。
    锚点测「当前运行配置」而非重启 defaults：服务正在跑的就是它（零重启成本，一次
    bench 即得漂移比；不变式：current_params 恒等于最近一次 keep 轮的配置或基线）。
    判定链：|drift|>5%（与 drift_check 同阈值）→ 重锚——baseline 全指标与 rolling
    best 按漂移比缩放、stats 换锚点新鲜估计（min 口径地板随新噪声走）；r0 基线
    归档 baseline_r0 供收尾报告对照。锚点轮 decision=anchor + sweep=True
    （plateau/history 透明）；记录 state["anchors"]，decide context 注入 env_drift。
    """
    k = getattr(cfg, "anchor_every", 5) or 0
    if k <= 0 or (state.get("anchor_since") or 0) < k:
        return
    state["anchor_since"] = 0
    # schema 默认值是 None（TOP_LEVEL_DEFAULTS），setdefault 拿不到 [] —— 显式归一
    if not state.get("anchors"):
        state["anchors"] = []
    anchors = state["anchors"]
    rk = f"a{len(anchors) + 1}"
    m = run_benchmark(cfg, state, rk)
    if not m:
        anchors.append({"round": rk, "ok": False, "note": "锚点压测失败（服务不可用？）"})
        save_state(state, st.RUN_STATE)
        return
    key = primary_key(flow, cfg.target)
    # 参考点（#2 修，2026-08-29）：锚点链接力仅在「配置未变」时成立——两锚之间发生
    # keep 后 current_params 已变，跨配置对比会把 keep 改善误判为漂移（假重锚把基线
    # /水位线按 keep 收益比例缩放，水位线推到不可达 → 之后全 rollback）。配置已变时
    # 回退「当前配置的上次测量」= 最近 keep 轮实测（同配置对称参照）。
    last_anchor = next((a for a in reversed(anchors) if a.get("ok")), None)
    if last_anchor and last_anchor.get("config") == (state.get("current_params") or {}):
        prev = last_anchor["cur"]
    else:
        ref_rec = next((r for r in reversed(state.get("rounds") or []) if r.get("kept")), None)
        prev = ((ref_rec or {}).get("metrics") or state.get("baseline") or {}).get(key)
    cur = m.get(key)
    if not prev or not cur:
        anchors.append({"round": rk, "ok": False, "note": "主指标缺失"})
        save_state(state, st.RUN_STATE)
        return
    pct = (cur - prev) / prev * 100
    anchors.append({"round": rk, "ok": True, "key": key, "prev": prev, "cur": cur,
                    "drift_pct": round(pct, 2),
                    "config": dict(state.get("current_params") or {})})
    state["rounds"].append({"round": rk, "action": {}, "changed": [], "kept": False,
                            "decision": "anchor", "metrics": m, "sweep": True,
                            "rationale": f"漂移锚点：{key} {prev:.1f}->{cur:.1f}（{pct:+.2f}%）"})
    print(f"[anchor] {rk} 漂移锚点：{key} {prev:.1f}->{cur:.1f}（{pct:+.2f}%）", flush=True)
    if abs(pct) <= 5.0:
        save_state(state, st.RUN_STATE)
        return
    # 重锚：基线/水位线按漂移比缩放（比较回到当前环境尺度），r0 归档供报告对照
    ratio = cur / prev
    if not state.get("baseline_r0"):
        state["baseline_r0"] = dict(state.get("baseline") or {})
    b = state.get("baseline") or {}
    for mk in _METRIC_KEYS:
        if isinstance(b.get(mk), (int, float)):
            b[mk] = b[mk] * ratio
    if m.get("stats"):
        b["stats"] = m["stats"]
    best = state.get("best")
    if best and isinstance(best.get(key), (int, float)):
        best[key] = best[key] * ratio
        if isinstance(best.get("value"), (int, float)):
            best["value"] = best["value"] * ratio
    save_state(state, st.RUN_STATE)
    print(f"[anchor] ⚠️ 漂移 {pct:+.2f}% 超 ±5%，重锚基线/水位线（×{ratio:.4f}）；"
          f"r0 基线归档 baseline_r0（终局改善率建议对照 r0 口径复核）", flush=True)


def _pareto_recheck(flow, cfg, state) -> None:
    """副目标 Pareto 通道（值级幕后、组合幕前的确定性节点，不走 decide 暂停点）。

    case8 档C教训：副目标赢家（吞吐 run 里 TTFT -16% 的 TRANSPOSE_KV_CACHE）只在
    影子账记了账——主目标持平 → 过不了 keep_verdict → 从未复测、从未进组合幕
    （组合准入 = kept-only）。但「同主指标 + 更好副指标」对最终配置是严格 Pareto
    占优，不该沉底。本节点把影子赢家升格为可信证据：在其来源轮可归因单参数时，
    于当前运行配置（= 主目标最优）上 reps=3 复测 → pareto_verdict 双轴判定 →
    keep 进 kept_params（自然进 backbone/final_sweep，交付配置含它）或回滚。
    主目标水位线 state["best"] 不更新（防副目标轮污染主目标记账）；
    收尾报告双列主目标纯净版对照。上限 --pareto-max（默认 1，0=关闭）；
    幂等：pN 轮已存在或参数已被 keep 同值即跳过。apply 失败同 P0-B/P1-A 口径
    计数/摘录（熔断由后续组合幕循环顶部检查兜底）。
    """
    cap = getattr(cfg, "pareto_max", 1) or 0
    if cap <= 0:
        return
    other = OTHER_TARGET.get(cfg.target)
    if not other:
        return
    shadow = (state.get("best_by_target") or {}).get(other) or {}
    # "r0" = _baseline_best 种子（副目标从未实测改善）——无事可做是正常态，静默返回，
    # 不走 loud skip（r0 不是 rounds 里的真实轮，也谈不上「非单参数来源」）。
    if not shadow.get("round") or shadow.get("round") == "r0":
        return
    # 影子条目来源轮必须可追溯且是单参数动作（多参数组合来源无法归因单参效应）
    rec0 = next((r for r in state.get("rounds") or []
                 if r.get("round") == shadow["round"]), None)
    a = (rec0 or {}).get("action") or {}
    param, value = a.get("param"), a.get("value")
    if not param or value is None:
        print(f"[pareto] 副目标({other})影子赢家 {shadow['round']} 非单参数来源，跳过 Pareto 复测",
              flush=True)
        return
    if (grouping.kept_params(state, flow) or {}).get(param) == value:
        return  # 已 keep 同值，通道无事可做
    p_rounds = [r for r in state.get("rounds") or []
                if str(r.get("round", "")).startswith("p") and r.get("pareto")]
    if len(p_rounds) >= cap or any((r.get("action") or {}).get("param") == param
                                   for r in p_rounds):
        return  # 预算已满 / 该参数已走过 Pareto 轮（幂等）
    key_main, key_sec = primary_key(flow, cfg.target), primary_key(flow, other)
    ref_rec = next((r for r in reversed(state.get("rounds") or []) if r.get("kept")), None)
    ref_m = ((ref_rec or {}).get("metrics") or state.get("baseline") or {})
    main_ref, sec_ref = ref_m.get(key_main), ref_m.get(key_sec)
    if not main_ref or not sec_ref:
        print(f"[pareto] 当前配置参照值缺失（{key_main}/{key_sec}），跳过 Pareto 复测", flush=True)
        return

    rk = f"p{len(p_rounds) + 1}"
    action = {"param": param, "value": value}
    print(f"[pareto] {rk} 副目标({other})影子赢家 {shadow['round']}（{param}={value}）"
          f"在当前最优配置上复测（reps={cfg.reps}）", flush=True)
    baseline = state.get("baseline") or {}
    app = apply_action(flow, cfg, state, action, rk)
    if not app["ok"]:
        state["apply_fail_streak"] = (state.get("apply_fail_streak") or 0) + 1
        reason = _fail_reason(app.get("log"))
        rec = {"round": rk, "action": action, "changed": action_changed(action),
               "kept": False, "decision": "rollback", "metrics": None, "pareto": True,
               "rationale": f"apply 失败: {app['log'][-200:]}",
               "rolled_back": app.get("rolled_back", False)}
        if reason:
            rec["fail_reason"] = reason
        state["rounds"].append(rec)
        save_state(state, st.RUN_STATE)
        print(f"[pareto] {rk} apply 失败{f'：{reason}' if reason else ''}", flush=True)
        return
    state["apply_fail_streak"] = 0
    ver = verify(flow, cfg, state, rk)
    metrics = ver["metrics"]
    # 确定性硬护栏（与 try_action 同口径）：smoke/精度B/图捕获任一不过 → 强制回滚
    if not ver.get("smoke_ok", True):
        hard = "smoke 失败，服务不可推理"
    elif ver.get("accuracy_match") is False and \
            (ver.get("accuracy_detail") or {}).get("cls") != "A":
        hard = "精度比对失败(B类语义破坏)"
    elif ver.get("graph_ok") is False:
        hard = "图捕获未生效（模式被跳过/未捕获）"
    else:
        hard = None
    if hard or not metrics:
        rollback_to_last(flow, cfg, state, rk)
        rec = {"round": rk, "action": action, "changed": action_changed(action),
               "kept": False, "decision": "rollback", "metrics": metrics, "pareto": True,
               "rationale": hard or "压测失败（无有效样本）", "rolled_back": True}
        state["rounds"].append(rec)
        save_state(state, st.RUN_STATE)
        print(f"[pareto] {rk} {rec['rationale']}，强制回滚", flush=True)
        return
    det = pareto_verdict(flow, metrics, baseline, main_ref, sec_ref, cfg.target)
    # 副目标影子账更新（复测配置的副目标值；不优则不更新）
    _shadow_update(flow, state, metrics, state.get("current_params"), rk, cfg.target,
                   ver.get("accuracy_match"), ver.get("accuracy_detail"))
    if det["ok"]:
        rec = {"round": rk, "action": action, "changed": action_changed(action),
               "kept": True, "decision": "keep", "metrics": metrics, "pareto": True,
               "rationale": det["rationale"], "rolled_back": False}
        state["rounds"].append(rec)
        save_state(state, st.RUN_STATE)
        print(f"[pareto] {rk} {param}={value} Pareto keep：{det['rationale']}"
              f"（进 backbone/final_sweep，交付配置含它）", flush=True)
        return
    rollback_to_last(flow, cfg, state, rk)
    rec = {"round": rk, "action": action, "changed": action_changed(action),
           "kept": False, "decision": "rollback", "metrics": metrics, "pareto": True,
           "rationale": det["rationale"], "rolled_back": True}
    state["rounds"].append(rec)
    save_state(state, st.RUN_STATE)
    print(f"[pareto] {rk} {param}={value} 回滚：{det['rationale']}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    flow = load_flow(args.flow)
    cfg = Cfg(args, flow)
    # 状态按 flow 隔离（DESIGN §3.4）：run_state/per_round/scripts_bak 落 state/{flow_id}/；
    # --target 覆盖 flow.target 时独立 state 目录，不同优化目标各存一份基线/轮次
    set_flow_scope(flow.flow_id if cfg.target == flow.target else f"{flow.flow_id}-{cfg.target}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    st.PER_ROUND_DIR.mkdir(parents=True, exist_ok=True)

    state = snapshot_state(st.RUN_STATE)
    if not (args.resume and (state.get("baseline") or state.get("run_started"))):
        # 全新 run：清陈旧备份（跨 run 端口可能不同）+ 陈旧 agent_result
        # （跨 run 的 task_id 可能撞 "rN-decide"，避免首轮 consume 误读上轮结果）。
        # run_started 置位后即视为 run 已启动——domain 节点暂停后的 --resume 不再
        # 重置 state（否则暂停-恢复链路会把 pending/domain 记录冲掉）。
        if st.SCRIPTS_BAK.exists():
            shutil.rmtree(st.SCRIPTS_BAK)
        if agent_gate.RESULT_FILE.exists():
            agent_gate.RESULT_FILE.unlink()
        state = {"rounds": [], "best": None, "status": "running", "kg_notes": [],
                 "current_params": dict(flow.params.defaults),
                 "port": cfg.port, "run_started": True,
                 # 两幕编排（P0-1）：value=幕一值级探索 → composition=幕二组合验证 → done
                 "phase": "value", "composition_queue": None}
        save_state(state, st.RUN_STATE)

    # 守门画像持久化（#5 修，2026-08-29）：--profile/--ttft-budget 原先只烘进
    # flow.perf 不落 state——resume 忘带 CLI 参数即静默回退 online，前后轮 guard
    # 口径不一致（半程换尺子）。解析顺序：CLI 显式 > state 记录 > online；应用
    # 一次后写回 state（下轮 resume 的回退源）。Cfg 不再预烘（防二次放宽）。
    saved_policy = state.get("profile_policy") or {}
    eff_profile, eff_budget = resolve_guard_policy(args, state)
    if eff_profile == "balanced" and not eff_budget:
        eff_profile = "online"  # saved 残缺兜底（正常记录必带 budget）
        print("[profile] ⚠️ balanced 缺预算（记录残缺？），回退 online 严格口径", flush=True)
    if saved_policy.get("profile") not in (None, eff_profile):
        print(f"[profile] ⚠️ resume 画像 {saved_policy['profile']} 被 CLI --profile "
              f"{args.profile} 显式覆盖", flush=True)
    _apply_guard_policy(flow.perf, eff_profile, eff_budget)
    cfg.profile = eff_profile
    state["profile_policy"] = {"profile": eff_profile, "ttft_budget": eff_budget}
    save_state(state, st.RUN_STATE)

    # —— domain 必走节点：KG 检索 → 暂停（LLM 提取）→ 与手写动作空间合并 ——
    # 每个 fresh run 必走（baseline 之前）；旧 run_state（有 baseline、无本节点标记）
    # resume 时跳过，保持历史行为不变。
    if not state.get("baseline") and not state.get("domain_ready"):
        ret = _domain_stage(flow, cfg, state)
        if ret is not None:  # 2=暂停待回传 / 0=terminate 已收尾
            return ret

    # resume 回装 KG 合并域：_domain_stage 只把 merged 装进当次进程内存
    # （flow.params.domain），--resume 重启进程后 load_flow 读的是手写 flow JSON——
    # 不回装则跨暂停续跑的 decide/apply 退回手写域（KG 覆盖丢失、新增参数消失）。
    # 用 merge_domain 叠加而非整体替换：flow JSON 后续新增的手写参数保留，
    # 缓存里的 KG 字段覆盖同名手写字段（与首装语义一致）。fresh run 刚装过，
    # 同条目重叠加是幂等的。
    if state.get("domain_ready") and (state.get("domain") or {}).get("source") == "kg-merged":
        cached = domain_node.load_cached_domain(flow.flow_id)
        if cached:
            flow.params.domain = domain_node.merge_domain(flow.params.domain or {}, cached)
            flow.domain_source = "kg-merged"

    if not state.get("baseline"):
        if not _init_baseline(flow, cfg, state):
            state["status"] = "fatal"
            save_state(state, st.RUN_STATE)
            return 1
        save_state(state, st.RUN_STATE)

    # 中断恢复：上次 run 若在 apply 窗口被打断（apply_pending 标记仍在），运行中的服务
    # 可能是残留候选配置，与 persisted state 分叉。此处回滚对齐到上一健康配置（= current_params）。
    if state.get("apply_pending"):
        pr = state["apply_pending"]
        rk = pr.get("round_key", "resume")
        print(f"[resume] 检测到中断的 apply（round {rk}），回滚对齐到上一健康配置", flush=True)
        if rollback_to_last(flow, cfg, state, rk).get("ok"):
            # 补记一条「中断回滚」记录：避免 resume 后 agent 重试已试参数
            if rk not in round_keys(state):
                state["rounds"].append({
                    "round": rk, "action": pr.get("action") or {}, "kept": False,
                    "metrics": None, "rolled_back": True,
                    "rationale": "中断恢复：apply 未落盘即被回滚，本轮不计改善"})
            state["apply_pending"] = None
            save_state(state, st.RUN_STATE)
        else:
            # 对齐失败（如残留 EngineCore 占卡）：保留标记，交由收尾 _ensure_healthy 兜底
            print("[resume] 回滚对齐失败，保留 apply_pending 标记，进入收尾健康兜底", flush=True)

    print(f"[flow] {flow.flow_id} LLM 主循环（会话委托） target={cfg.target} "
          f"max_rounds={cfg.max_rounds}", flush=True)

    # ③ 粗筛幕（值级幕前必经；resume 幂等——已完成参数跳过；P0-B 熔断可中断返回 2/0）
    ret = _run_screening(flow, cfg, state)
    if ret is not None:
        return ret

    # 暂停恢复：上次暂停在「决策」点时，从该轮续跑（signals 走 snapshot，不重跑 profiling）
    resume_decide = None
    if state.get("status") == "pending" and state.get("pending"):
        resume_decide = state["pending"]  # {round_key, stage, task_id}

    done = round_keys(state)
    # ---------- 幕一：值级探索（resume 已在组合幕时整段跳过，不重跑值级轮） ----------
    value_phase = state.get("phase", "value") == "value"
    for r in range(1, cfg.max_rounds + 1):
        if not value_phase:
            break
        # P0-B 熔断检查：apply 连续失败 ≥3 → 暂停回传（环境故障不空转烧轮）
        fret = _fuse_check(flow, cfg, state)
        if fret is not None:
            return fret
        # ②a 精修排空：上轮 keep 产生的邻域补扫队列先清（不占本轮探索预算）
        _drain_refines(flow, cfg, state)
        round_key = f"r{r}"
        if round_key in done:
            print(f"[round {r}] 已存在，跳过（resume）", flush=True)
            continue
        print(f"\n========== round {r} ==========", flush=True)

        # capture（暂停恢复时从 snapshot 取 signals，不重跑 profiling）
        signals = None
        if resume_decide and resume_decide.get("round_key") == round_key:
            signals = (state.get("snapshot") or {}).get("signals")
            resume_decide = None
        else:
            cap = capture(flow, cfg, state, round_key)
            signals = cap.get("signals")

        # P1-3 诊断联动：capture 后 decide 前可选 rN-diagnose 暂停点（节流：每 run
        # ≤2 次、bottleneck_class/hot_ops top-3 指纹不变不重诊）。返回 2 = 暂停待回传。
        diag_ret = _maybe_diagnose(flow, cfg, state, signals, round_key)
        if diag_ret is not None:
            return diag_ret

        # 决策暂停点：会话委托（复用 agent_gate 回传主会话 agent）
        dec = _decision_block(flow, cfg)
        ctx = {"round_key": round_key, "stage": "decide"}
        decision = agent_gate.consume_result(ctx)
        if decision is None:
            agent_gate.emit_task(
                ctx,
                goal=getattr(dec, "goal", None) or _DECIDE_GOAL,
                skill_refs=[],
                schema=_DECIDE_SCHEMA,
                context=_build_decision_context(
                    flow, state, signals, round_key,
                    getattr(dec, "rules", None) or _VLLM_RULES, cfg.target, cfg=cfg),
            )
            state["pending"] = {"round_key": round_key, "stage": "decide",
                                "task_id": ctx["pending_task_id"]}
            state["snapshot"] = {"signals": signals}
            state["status"] = "pending"
            save_state(state, st.RUN_STATE)
            print(f"\n[pending] 等待 agent 决策 {ctx['pending_task_id']}："
                  f"读 engine/state/agent_task.json → 写 agent_result.json → --resume", flush=True)
            return 2

        action = _parse_decision(decision, flow)
        if action is None:
            # 无效决策（非 dict/越界/类型错）—— LLM 失误，直接停，不浪费一轮跑空 apply
            print("[loop] 决策无效，停止", flush=True)
            break

        # LLM 软终止（无推荐 value=null）→ 记 no_candidate 轮，不直接停：
        # 交由下方 should_terminate 走 exhausted 终止，统一终止判定路径。
        # 判据 = _parse_decision 的软终止标记 {"candidate": None}（无 param/params 键）。
        # ⚠️ 有效动作返回 {"param":...} 或 {"params": {...}}（action v2）不含
        # candidate 键，必须用 param/params 双键判别，否则有效决策会被误判成软终止。
        if "param" not in action and "params" not in action:
            rec = {"round": round_key, "action": {}, "changed": [],
                   "kept": False, "decision": "no_candidate", "metrics": None,
                   "rationale": decision.get("reason", "LLM 无推荐") if isinstance(decision, dict) else "LLM 无推荐",
                   "rolled_back": False}
            print(f"[llm] 软终止（无推荐）: {rec['rationale']}", flush=True)
            state["rounds"].append(rec)
            state["apply_pending"] = None
            state["pending"] = None
            state["snapshot"] = None
            save_state(state, st.RUN_STATE)
            term = should_terminate(state, flow, cfg.target, max_rounds=cfg.max_rounds)
            if term:
                print(f"[terminate] {term.kind}: {term.detail}", flush=True)
                break
            continue

        print(f"[llm] 决策: {action_label(action)}", flush=True)

        # 确定性动作门槛（M3）：high-risk 动作需本轮有 profiling 信号，否则拒绝。
        # 不 apply 不压测（白费一轮，规则已告知 LLM；history 可见促其换动作）——
        # LLM 遵守规则是软约束，引擎兜底拦截才是硬保证。
        # action v2：组合动作内任一参数为 high-risk 即触发（组合不放宽护栏）。
        rec = None
        doms = [(flow.params.domain or {}).get(p, {}) for p in action_changed(action)]
        if (any(d.get("risk") == "high" for d in doms)
                and getattr(dec, "high_risk_requires_signals", True)
                and not signals):
            rec = {"round": round_key, "action": action, "changed": action_changed(action),
                   "kept": False, "decision": "rejected", "metrics": None,
                   "rationale": "高风险动作无 profiling 证据，确定性拒绝（不 apply 不压测）",
                   "rolled_back": False}
            print(f"[gate] {action_label(action)} 含高风险参数且本轮无 signals，确定性拒绝",
                  flush=True)
        if rec is None:
            rec = try_action(flow, cfg, state, action, round_key)
        state["rounds"].append(rec)
        state["apply_pending"] = None
        state["pending"] = None
        state["snapshot"] = None
        # P1-B：值级轮计数（漂移锚点触发依据）——锚点每 anchor_every 轮零重启复测
        state["anchor_since"] = (state.get("anchor_since") or 0) + 1
        save_state(state, st.RUN_STATE)
        _maybe_anchor(flow, cfg, state)

        # 确定性终态判定（Pruner，DESIGN §4.4 层1）：不依赖 LLM。
        # 停止判定的主路径在此——LLM 的软终止只是提前止损，硬终止确保它判断差时
        # 不会无限跑（修旧版「stop 全靠 LLM 自觉」的职责错放）。
        term = should_terminate(state, flow, cfg.target, max_rounds=cfg.max_rounds)
        if term:
            print(f"[terminate] {term.kind}: {term.detail}", flush=True)
            break

    # ---------- 幕二：组合验证（值级收敛 → 分组降维 → 队列组合真机轮 → 终局拼装） ----------
    if value_phase:
        _drain_refines(flow, cfg, state)  # backbone 准入：进组合幕前确认已 keep 参数局部最优
        _pareto_recheck(flow, cfg, state)  # 副目标 Pareto 通道：影子赢家复测（keep 进交付配置）
        _enter_composition(flow, cfg, state)
    if state.get("phase") == "composition":
        ret = _run_composition_phase(flow, cfg, state)
        if ret is not None:
            return ret
    _final_sweep(flow, cfg, state)

    # 收尾
    state["status"] = "complete"
    state["phase"] = "done"
    save_state(state, st.RUN_STATE)
    healthy = _ensure_healthy(cfg, state)
    # P0-2c：漂移核验（defaults 结束配置复测 vs 基线，|偏差|>5% 大声告警）
    _drift_check(flow, cfg, state)
    # P1-B：锚点曲线（中途漂移史 + 重锚记录）
    for a in (state.get("anchors") or []):
        if a.get("ok"):
            print(f"[anchor] {a['round']}: {a['key']} {a['prev']:.1f}->{a['cur']:.1f}"
                  f"（{a['drift_pct']:+.2f}%）"
                  + ("，已重锚基线/水位线" if abs(a["drift_pct"]) > 5 else ""), flush=True)
        else:
            print(f"[anchor] {a.get('round')}: 失败（{a.get('note')}）", flush=True)
    if state.get("baseline_r0"):
        print("[anchor] 注意：run 中途发生过重锚，r0 基线归档于 state.baseline_r0"
              "（终局改善率建议对照 r0 口径复核）", flush=True)
    print(f"\n========== 闭环完成 ==========", flush=True)
    print(f"主服务健康: {'是' if healthy else '否'}", flush=True)
    if state.get("unmodeled_interactions"):
        g = state["unmodeled_interactions"]
        print(f"[gap] ⚠️ 未建模交互告警：拼装配置 {g['key']} 劣于 best {g['gap_pct']}%"
              f"（best_round={g['best_round']}），先验疑似遗漏真实交互", flush=True)
    plan = state.get("composition_plan") or {}
    if plan:
        print(f"[composition] 计划：backbone={list(plan.get('backbone') or [])} "
              f"interactive={plan.get('interactive')} "
              f"组合轮={[r['round'] for r in _combo_rounds(state)]}"
              f"（补扫建议：{[p['param'] for p in plan.get('pending_neighborhood') or []]}）",
              flush=True)
    best = state.get("best")
    if best:
        key = primary_key(flow, cfg.target)
        print(f"最优: {best.get('round')} {key}={best.get(key):.1f}", flush=True)
    # 双目标报告（方案 1）：副目标实测最优（可能 ≠ 主目标最优配置，均为已实测数据）
    ob = (state.get("best_by_target") or {}).get(OTHER_TARGET.get(cfg.target) or "")
    if ob:
        ok = primary_key(flow, ob["target"])
        ov = ob["metrics"].get(ok)
        ovs = f"{-ov:.1f}t/s" if ok == "thr_tok_s" and ov is not None else ov
        print(f"副目标最优（{ob['target']}，影子记账，实测于 {ob['round']}）："
              f"{ok}={ovs}（基线起改善 {ob['improved_pct']:+.1f}%），"
              f"配置={ob['params']}", flush=True)
        if ob["params"] != (state.get("current_params") or {}):
            print(f"  ↳ 该配置非当前运行配置（主目标取向未采纳）；切换用它重启即得，无需重测",
                  flush=True)
    for r in state["rounds"]:
        a = r.get("action") or {}
        m = r.get("metrics") or {}
        key = primary_key(flow, cfg.target)
        label = (f"{a['param']}={a.get('value', '?')}" if a.get("param")
                 else "+".join(f"{p}={v}" for p, v in (a.get("params") or {}).items()) or "?")
        print(f"  {r.get('round')}: {label} "
              f"{key}={m.get(key, '?')} → kept={r.get('kept')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
