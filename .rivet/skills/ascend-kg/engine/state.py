"""状态持久化 + 模板渲染 + 参数域工具（平移自 vllm_ascend_opt/machine.py）。

纯 stdlib。函数显式接收路径/参数域；engine 目录路径常量在此统一定义。
state schema（P1-2a）：文件格式不变（JSON 可手读），读写路径加验证与迁移——
SCHEMA_VERSION 落盘，未来迁移按版本号分派而非发明兼容标志位。
"""
import json
from dataclasses import dataclass, fields
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = ENGINE_DIR / "templates"
SCRIPT_DIR = ENGINE_DIR / "scripts"
# state 根目录保持全局（agent_task.json/agent_result.json 是会话级协议文件，不分 flow）；
# run_state/per_round/scripts_bak 按 flow 隔离（DESIGN-multi-backend §3.4），由入口在
# 加载 flow 后经 set_flow_scope() 重定向。调用方必须以属性访问（state.RUN_STATE）取值——
# from-import 在 import 时绑定旧路径，拿不到重定向后的值。
STATE_DIR = ENGINE_DIR / "state"
FLOW_STATE_DIR = STATE_DIR
PER_ROUND_DIR = FLOW_STATE_DIR / "per_round"
SCRIPTS_BAK = FLOW_STATE_DIR / "scripts_bak"
RUN_STATE = FLOW_STATE_DIR / "run_state.json"


def set_flow_scope(flow_id: str) -> None:
    """状态目录切换到 state/{flow_id}/（多 flow 并跑互不覆盖 run_state/备份/逐轮产物）。"""
    global FLOW_STATE_DIR, PER_ROUND_DIR, SCRIPTS_BAK, RUN_STATE
    FLOW_STATE_DIR = STATE_DIR / flow_id
    PER_ROUND_DIR = FLOW_STATE_DIR / "per_round"
    SCRIPTS_BAK = FLOW_STATE_DIR / "scripts_bak"
    RUN_STATE = FLOW_STATE_DIR / "run_state.json"


def render(template_path, out_path, **vars) -> None:
    """极简模板渲染：@KEY@ 占位符替换，避免 jinja2 依赖。"""
    s = Path(template_path).read_text(encoding="utf-8")
    for k, v in vars.items():
        s = s.replace(f"@{k}@", str(v))
    out = Path(out_path)
    out.write_text(s, encoding="utf-8")
    out.chmod(0o755)


def params_to_args(params: dict, domain: dict, port: int, base_args: str = None) -> str:
    """params dict → CLI 启动参数串。

    - base_args：通用前缀（host/port 等固定项）。缺省为 vllm 前缀（向后兼容）；
      非 vllm 后端传自己的前缀（如 sglang）。
    - kind=env 的参数跳过（它们走 params_to_env 渲染成 export 行，不进 CLI）。
    - kind=sub 的参数跳过（只作 compose 输入，不单独渲染成独立 flag）。
    - kind=json 的参数跳过（走 params_to_json_config 渲染进配置文件，不进 CLI）。
    - 带 compose 的参数从显式设置过的 sub-param 组装内联 JSON 渲染；未设置的省略 → 引擎默认。
    """
    if base_args is None:
        # vllm 前缀；block-size 必须显式带上（防引擎默认与纠偏值不一致）——由 defaults 承载
        base_args = f"--tensor-parallel-size 1 --host 0.0.0.0 --port {port}"
    parts = [base_args]
    for k, v in params.items():
        if v is None:
            continue
        d = domain.get(k, {})
        if d.get("kind") in ("env", "sub", "json") or d.get("compose"):
            continue  # env→VLLM_ENV；sub 只作 compose 输入；json→配置文件；compose 由下方单独渲染
        if d.get("flag"):
            if v:
                # 布尔开关：True 才渲染，False 省略（vLLM 布尔 flag 均为正向启用）
                fixed = d.get("fixed")
                if fixed:
                    parts.append(f"--{k} '{fixed}'")  # 布尔 flag 带固定非布尔 payload（如 JSON）
                else:
                    parts.append(f"--{k}")
        else:
            if isinstance(v, str):
                parts.append(f"--{k} '{v}'")  # 字符串值单引号包裹，防 shell 分词/花括号展开
            else:
                parts.append(f"--{k} {v}")
    # compose 参数不在 params dict（只有 defaults + 显式设置项），从 domain 单独渲染。
    for k, d in domain.items():
        if not d.get("compose"):
            continue
        payload = _compose_json(d["compose"], params, domain)
        if payload:
            # 内联 JSON（vllm --additional-config/--compilation-config 经 union_dict_and_str/validate_json 解析）
            parts.append(f"--{k} '{json.dumps(payload)}'")
    return " \\\n  ".join(parts)


def _compose_json(mapping: dict, params: dict, domain: dict) -> dict:
    """compose 映射 → JSON payload。只含 params 里显式设置过的 sub-param（未设置省略 → vllm 默认）。

    mapping 两种形态：
      {json_key: sub_param}          → 扁平（如 compilation-config: {"cudagraph_mode": "cudagraph-mode"}）
      {json_group: {json_key: sub}}  → 分组（如 additional-config 的 ascend_compilation_config）
    """
    out = {}
    for key, val in mapping.items():
        if isinstance(val, dict):
            group = {}
            for jk, sub in val.items():
                if sub in params and params[sub] is not None:
                    group[jk] = _json_scalar(params[sub], domain.get(sub, {}))
            if group:
                out[key] = group
        else:
            sub = val
            if sub in params and params[sub] is not None:
                out[key] = _json_scalar(params[sub], domain.get(sub, {}))
    return out


def _json_scalar(v, d: dict):
    """sub-param 值 → JSON 标量。布尔 flag 显式转 true/false（False 也要 emit，区别于独立 flag 的 False 省略）。"""
    if d.get("flag"):
        return v in (True, 1, "1", "true", "True")
    return v


def params_to_json_config(base_config: dict, params: dict, domain: dict) -> dict:
    """base config.json + 显式设置过的 kind=json 参数 → 每轮完整 config（MindIE 形态，DESIGN §3.3）。

    - domain[k]["path"] = 嵌套 json 键路径列表（如 ["ScheduleConfig", "maxQueueDelayMicroseconds"]）。
    - 只 patch params 里显式设置过的项（含 defaults 内的初始值）；未设置的沿用 base —— base 是
      从镜像实读的完整样例，引擎不发明字段（防 daemon 拒绝未知/缺失键）。
    - 深拷贝 base，不改原 dict；路径逐级 mkdir（缺失的中间层补 dict）。
    """
    import copy
    out = copy.deepcopy(base_config)
    for k, v in params.items():
        if v is None:
            continue
        d = domain.get(k, {})
        if d.get("kind") != "json" or not d.get("path"):
            continue
        node = out
        path = d["path"]
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = v
    return out


def params_to_env(params: dict, domain: dict) -> str:
    """params dict → 环境变量 export 行（仅 kind=env 的参数）。

    布尔环境变量（flag=true）渲染成 1/0；否则按原值。空字符串表示无环境变量。
    """
    lines = []
    for k, v in params.items():
        if v is None:
            continue
        d = domain.get(k, {})
        if d.get("kind") != "env":
            continue
        if d.get("flag"):
            val = "1" if v in (True, 1, "1", "true", "True") else "0"
            lines.append(f"export {k}={val}")
        else:
            lines.append(f"export {k}={v}")
    return "\n".join(lines)


def within_domain(param: str, value, domain: dict) -> bool:
    d = domain.get(param)
    if not d:
        return False
    if d.get("compose"):
        return False  # 派生参数（compose 合成），禁止直接设置，防 agent 越界触发数值分支
    if d.get("locked"):
        return False  # 锁死参数禁止提案（F8，2026-08-29）：此前恒 True，LLM 越界提议
        # 会通过校验落地真机、靠健康门禁兜底烧一轮。锁定值由 defaults 承载，提案
        # 当前值也是 no-op——任何值一律拒。提示层「锁定（不可调）」只防误提，此处是硬校验。
    if d.get("values"):
        return value in d["values"]  # 枚举取值
    if d.get("flag"):
        return value in (True, False, 0, 1)  # 布尔开关
    try:
        return d["min"] <= float(value) <= d["max"]
    except (TypeError, ValueError):
        return False


def snapshot_state(run_state: Path) -> dict:
    state = None
    if Path(run_state).exists():
        try:
            state = json.loads(Path(run_state).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            # 损坏/半写不致命：大声报错并按全新状态初始化，避免 resume 直接 crash 废掉续跑。
            print(f"[state] {run_state} 解析失败（损坏或半写），按全新状态初始化", flush=True)
    if state is None:
        state = {"rounds": [], "best": None, "status": "init", "kg_notes": []}
    return validate_and_migrate(state)


# ---------------- state schema（P1-2a：渐进式 schema 化，只加护栏不重写） ----------------

SCHEMA_VERSION = 2

# 顶层键默认值：读路径统一补齐（替代散落各处的 setdefault）。None = 必须存在但不补默认。
TOP_LEVEL_DEFAULTS = {
    "rounds": [], "best": None, "status": "init", "kg_notes": [],
    # 两幕编排（P0-1）：旧 state 缺省按值级阶段处理（resume 收尾路径不受影响）
    "phase": "value", "composition_queue": None,
    # 双目标影子记账（方案 1）：{target: best}，_init_baseline 起步、try_action 更新副目标
    "best_by_target": None,
    # 粗筛幕（③）：[{param, value, round, ok, note, ttft_pct, thr_pct, main_pct}]——
    # 全参数单点快筛榜单，decide context 注入、幂等续跑按参数名查重
    "screening": None,
    # P0-B apply 连续失败熔断：连续失败计数（≥3 触发 fail{N}-fuse 暂停点）+ 处置记录；
    # fuse_episode = 已消费的熔断暂停次数（#7：task_id 带序号防陈旧结果跨 episode 复用）
    "apply_fail_streak": 0, "fuse": None, "fuse_episode": 0,
    # 守门画像持久化（#5）：CLI 显式 > 此记录 > online——resume 忘带 --profile 不回退
    "profile_policy": None,
    # P1-B 中途漂移锚点：距上次锚点的值级轮计数 + 锚点记录 + 重锚前的 r0 基线归档
    "anchor_since": 0, "anchors": None, "baseline_r0": None,
    "current_params": {}, "baseline": None, "pending": None, "snapshot": None,
}


@dataclass
class RoundRecord:
    """rounds 条目的字段单一权威定义（防 drift：读时校验 + 单测断言依据）。

    写路径仍 dump dict（文件格式不变）；本 dataclass 不参与运行时序列化，
    新增字段必须同步改 REQUIRED_ROUND_KEYS / OPTIONAL_ROUND_KEYS 与单测。
    """
    round: str                 # 轮键 "rN"/"cN"/"sweep"/"scrN"/"aN"/"pN"（int 为旧入口遗留，读时归一）
    action: dict               # {"param":..,"value":..} | {"params":{..}} | {}
    kept: bool
    decision: str              # keep | rollback | rejected | no_candidate | rejected_offqueue | screen | anchor...
    metrics: dict | None = None
    objective: dict | None = None
    rolled_back: bool = False
    rationale: str = ""
    changed: list | None = None
    fail_reason: str | None = None  # P1-A：apply 失败根因摘录（OOM/超时/崩溃关键行）
    remeasured: bool = False
    refine: bool = False           # ②a 精修轮（keep 邻域补扫）：不占探索预算、不计 plateau
    sweep: bool = False
    pareto: bool = False           # 副目标 Pareto 通道轮（pN）：双轴判定 keep，不更新主目标 best
    accuracy_detail: dict | None = None
    diagnosis: dict | None = None
    validation: dict | None = None


# rounds 条目必填键（缺了大声报错，不静默跳过）——与 RoundRecord 无默认值字段一致
_REQUIRED_ROUND = {"round", "action", "kept", "decision"}

# v1 指标键 → 现名（#3）：ttft 化改名（Task 8：ttfb→ttft 全链路）晚于这批 run 落盘。
# 实测旧 state 分布：metrics/baseline/best 的 ttfb_ms、metrics 的 ttfb_p95；
# ttfb_pct（粗筛榜单）/裸 "ttfb"（best_by_target 目标键）为防御性收录。
_TTFB_RENAMES = {"ttfb_ms": "ttft_ms", "ttfb_p95": "ttft_p95",
                 "ttfb_pct": "ttft_pct", "ttfb": "ttft"}


def _rename_keys(obj, renames: dict, hits: list, path: str):
    """递归改 dict 键（#3 迁移用）：返回改写后的新结构，命中处记 "path.key" 进 hits。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = renames.get(k, k)
            if nk != k:
                hits.append(f"{path}.{k}")
            out[nk] = _rename_keys(v, renames, hits, f"{path}.{nk}")
        return out
    if isinstance(obj, list):
        return [_rename_keys(x, renames, hits, f"{path}[{i}]") for i, x in enumerate(obj)]
    return obj
# 允许出现在 rounds 条目里的全部键（超出 = 未知键，大声告警防 schema 漂移）
KNOWN_ROUND_KEYS = {f.name for f in fields(RoundRecord)} | {"candidate"}


def validate_and_migrate(state: dict) -> dict:
    """读路径统一入口：补默认键、类型归一、v0/v1 旧 state 迁移。

    原则（plan §4.1）：只做补默认/类型归一，不做语义拒绝——旧 state 必须无感续跑。
    校验失败大声报错（带 key 路径），不裸抛 KeyError。#3/#4（2026-08-29）把两条
    原 raise 降级为迁移：ttfb* 指标键改名、rounds 缺必填键补默认——残留 v1 state
    的机器上 resume 不再起不来；round 键本身缺失仍属结构损坏，照 raise。
    """
    if not isinstance(state, dict):
        raise ValueError(f"state 非法（{type(state).__name__}，期望 dict）")
    ver = state.get("schema_version", 1)
    if not isinstance(ver, int):
        raise ValueError(f"state.schema_version 非法: {ver!r}（期望 int）")
    if ver > SCHEMA_VERSION:
        raise ValueError(f"state.schema_version={ver} 高于本引擎 {SCHEMA_VERSION}"
                         f"（用旧引擎 resume 新 state 会丢语义，拒绝）")

    # #3：ttfb* → ttft* 指标键迁移（v1 时代指标名，ttft 化改名晚于这些 run）。
    # 全容器递归改键（baseline/best/rounds[].metrics/screening/best_by_target 键），
    # 值不动——rationale 里的 "ttfb_ms 44.1->40.6" 是历史文案，机器只读键。
    hits = []
    state = _rename_keys(state, _TTFB_RENAMES, hits, "state")
    if hits:
        shown = ", ".join(hits[:8]) + (" …" if len(hits) > 8 else "")
        print(f"[state] 迁移 ttfb→ttft 指标键 {len(hits)} 处：{shown}", flush=True)

    # v1 → v2：rounds 条目的 round 键 int → str 归一（run.py 旧入口遗留）
    rounds = state.get("rounds")
    if rounds is None:
        state["rounds"] = []
        rounds = state["rounds"]
    if not isinstance(rounds, list):
        raise ValueError("state.rounds 非法（期望 list）")
    for i, r in enumerate(rounds):
        path = f"rounds[{i}]"
        if not isinstance(r, dict):
            raise ValueError(f"{path} 非法（期望 dict，得到 {type(r).__name__}）")
        rk = r.get("round")
        if isinstance(rk, int) and not isinstance(rk, bool):
            r["round"] = f"r{rk}"  # 归一写回（round_keys 兼容两种，此处固化 str）
            rk = r["round"]
        if not isinstance(rk, str) or not rk:
            raise ValueError(f"{path}.round 非法: {rk!r}（期望非空 str 或 int）")
        missing = _REQUIRED_ROUND - set(r)
        if missing:
            # #4：v1 老轮缺必填键 → 补默认大声告警，不再拒载（schema 加严不该
            # 卡死旧 run 的 resume）。decision 从 kept 推导（v1 无 decision 词表，
            # kept 是当时的唯一真相源）；round 键无法发明身份，仍 raise。
            fixes = []
            if "action" in missing:
                r["action"] = {}
                fixes.append("action={}")
            if "kept" in missing:
                r["kept"] = False
                fixes.append("kept=False")
            if "decision" in missing:
                r["decision"] = "keep" if r.get("kept") else "rollback"
                fixes.append(f'decision={r["decision"]}(由kept推导)')
            print(f"[state] ⚠️ {path}({rk}) 缺必填键 {sorted(missing)}——补默认 "
                  f"{'，'.join(fixes)}", flush=True)
            missing = _REQUIRED_ROUND - set(r)
            if missing:
                raise ValueError(f"{path} 缺必填键 {sorted(missing)}（round={rk}）")
        unknown = set(r) - KNOWN_ROUND_KEYS
        if unknown:
            print(f"[state] ⚠️ {path}({rk}) 含未知键 {sorted(unknown)}——"
                  f"schema 漂移？请同步 RoundRecord 定义", flush=True)

    # 顶层补默认（不覆盖已有值；pending/snapshot 等键 None 语义保留）。
    # 可变默认值必须逐 state 拷贝——共享模块级对象会被 apply 链原地改写，
    # 串污下一次 snapshot_state 的补默认结果。
    for k, v in TOP_LEVEL_DEFAULTS.items():
        if k in state:
            continue
        state[k] = (dict(v) if isinstance(v, dict)
                    else list(v) if isinstance(v, list) else v)
    state["schema_version"] = SCHEMA_VERSION
    return state


def save_state(state: dict, run_state: Path) -> None:
    Path(run_state).parent.mkdir(parents=True, exist_ok=True)
    # 原子写：先写临时文件再 rename，避免进程被 SIGKILL 时留下半写 JSON（废掉 resume）
    state.setdefault("schema_version", SCHEMA_VERSION)
    tmp = Path(str(run_state) + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(run_state)


def round_keys(state: dict) -> set:
    """state["rounds"] 各轮 round 键，归一化为字符串 "rN"。

    两个入口曾混用整数（run.py）与字符串（agent_loop），跨入口 resume 会失配重跑；
    这里把两种格式统一成 "rN"，供 done 集合去重比较。
    """
    keys = set()
    for r in state.get("rounds") or []:
        rk = r.get("round")
        if isinstance(rk, int):
            rk = f"r{rk}"
        if rk:
            keys.add(rk)
    return keys
