# ascend-kg 引擎 v2 落地方案：双循环（LLM 探索 + 确定性单轮兜底）

> 状态：已实现（阶段 1–3 落地，`agent_loop.py` + `tools/` 已交付）；遗留 `verify` 的「输出一致性比对」精度护栏为恒真 stub（阶段 2），见 §4.2
> 日期：2026-08-14
> 前身：`vllm-serve-optimize` 10 阶段确定性流水线（见 `CHANGES.md`）

---

## 1. 背景与问题

### 1.1 现状

引擎 `vllm-serve-optimize` 是一条 **10 阶段确定性流水线**（`flows/vllm-serve-optimize.json`）：

```
capture → validate → analyze → diagnose(LLM暂停点) → recommend → critic → apply → verify → decide → orchestrate
```

- 参数域只有 8 个**普通 vLLM 参数**（`max-model-len`/`max-num-seqs`/`max-num-batched-tokens`/`gpu-memory-utilization`/`block-size`/`max-num-partial-prefills`/`long-prefill-token-threshold`/`enforce-eager`）。
- 决策层是**确定性硬编码**：`critic`（部署前门禁 `critic_prio_gate=7`）、`decide`（滞回 `hyst=0.003`/`ttot_room=1.10`）、`orchestrate`（连续无改善即停 `consec_no_improve_stop=2`）。
- LLM 只在 `diagnose` 一个**暂停点**介入：引擎退出码 2 → 主会话 agent 读 `agent_task.json` 写 `agent_result.json` → `--resume` 续跑（半自动）。

### 1.2 效果差距的根因（已诊断）

对标 ROCm.AI 优化主干（分析 + 内核重写 + 参数 + 护栏），三个结构性缺口：

1. **动作空间窄**：8 个普通参数里**不含昇腾真解**。KG 已确认 vLLM-Ascend 的 schedule 瓶颈（wait 主导 / MFU 仅 2%）真解是**昇腾专属维度**：图模式、算子融合、调度策略、dtype——全都不在当前 domain 内。决策层再聪明，也是在错误的搜索空间里空转。
2. **缺「深优化」层**：ROCm.AI 效果大头来自「内核重写」；昇腾上内核闭源不可重写，但昇腾有自己的深优化维度（图模式/融合/调度/dtype），一直被当成「几个 flag」散落，从未系统化成一个独立支柱。
3. **决策主体是确定性规则，不是 LLM**：整个优化策略（何时停、何时换方向、试什么）是人写死的，LLM 只当诊断参谋。这不是 ROCm.AI 的「LLM 自主探索」。

### 1.3 目标

- 把决策主体从「确定性规则」反转为「LLM 自主探索」；
- 把探索空间从「8 参数」升级到「深优化层 + 参数」；
- 同时保住引擎已验证的「不翻车」能力（健康门禁 / 回滚 / 端口作用域杀进程）。

---

## 2. 核心设计思想：自主与可靠分离

一句话：**「往哪探索」交给 LLM（智能，从零打磨），「怎么安全地试」交给引擎（确定性，已验证）。**

- **外层（LLM 探索循环）** = 智能。LLM 持有优化上下文（baseline / 历史轮次 / 当前瓶颈），自主决定「下一步试哪个维度、哪个动作」「何时停、何时换方向」。
- **内层（确定性执行原语）** = 安全。引擎把「应用 → 压测 → 比对 → 变差自动回滚」做成**原子操作**，单轮绝不翻车，LLM 的坏决策最多浪费一轮，不会把服务挂在坏状态。

这解耦了「智能」与「安全」两个正交的轴：智能可以随 prompt/经验迭代提升，安全不依赖智能。

---

## 3. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  外层：探索循环（agent_loop.py，决策经 agent_gate 暂停点回传主会话 agent） │
│                                                             │
│   loop:                                                     │
│     signals = capture()            # 看当前瓶颈             │
│     action  = LLM.decide(          # 自主决策下一步          │
│                 signals, history, best, 护栏约束)            │
│     if action == STOP: break                                │
│     result  = apply(action)        # 应用（失败自动回滚）     │
│     metrics = verify()             # 压测 + 精度比对          │
│     history.append(action, metrics)                         │
│                                                             │
│   └──────────────┬──────────────────────────────────────────┘
│                  │ 调用（subprocess / function tool）        │
│   ┌──────────────▼──────────────────────────────────────────┐
│   │  内层：确定性执行原语（引擎瘦身后的工具）                 │
│   │                                                         │
│   │  capture  → 采集 profiling + 解析瓶颈信号                │
│   │  apply    → 应用动作（健康门禁 + 失败自动回滚）           │
│   │  verify   → 并发压测 + 输出一致性比对（精度护栏）         │
│   │  rollback → 恢复到上一个健康配置                         │
│   └─────────────────────────────────────────────────────────┘
```

**LLM 决策载体**：`agent_loop.py` **不内嵌 LLM、零单独配置**——外层「下一轮试哪个动作」的决策经 `engine/agent_gate.py` 的「decide」暂停点回传主会话 agent（退出码 2 + `agent_task.json`/`agent_result.json`，与 `run.py` 的 `diagnose` 同一约定）。引擎跑到决策点写任务、暂停，主会话 Claude 读动作空间/信号/历史/规则推理、写回决策，引擎 `--resume` 续跑。不整合 msagent（用户决定），也**不重造** deepagents 那么重的框架——决策智能由 Claude Code 自身的 Agent/Subagent 体系提供，引擎只保留确定性内层原语。工具 schema 即下节 4 个原语的输入输出契约。

---

## 4. 组件详设

### 4.1 外层：LLM 探索循环

**文件**：`engine/agent_loop.py`（新）

**循环伪码**：

```
ctx = load_context()                     # baseline / best / history / 当前瓶颈
while not exceed(max_rounds):
    signals = capture()                  # 工具①：读当前瓶颈
    decision = agent_gate.decide(        # 暂停点回传主会话 agent 决策
        goal, ctx.action_space,
        signals, ctx.history, ctx.best)
    if decision.action == "STOP":
        break
    ctx.current_action = decision.action
    result = apply(decision.action)      # 工具②：应用（失败自动回滚）
    metrics = verify()                   # 工具③：压测 + 精度比对
    ctx.history.append(decision, result, metrics)
    ctx.best = update_best(ctx.best, metrics)
    persist_context(ctx)                 # 每轮落盘，可中断续跑
```

**LLM 的职责**（system prompt 约束）：
- 读结构化瓶颈信号 + 历史，选「下一个最可能有效」的动作；
- 判断「该继续 / 换维度 / 停止」；
- 遵守动作空间白名单 + 危险动作约束（见 4.4 软护栏）。

**LLM 明确不做**（内层确定性兜底）：
- 不做进程管理（kill/重启）——由 `apply` 原子操作执行；
- 不做回滚判断——`apply` 内置「变差自动回滚」；
- 不做延迟测量——`verify` 原子操作执行。

### 4.2 内层：确定性执行原语（引擎瘦身为工具）

引擎从「10 阶段流水线」瘦身为 **4 个可独立调用的工具**，每个有明确输入输出契约。红线不变：进程管理一律 `kill_by_port.sh`（端口作用域），严禁宽模式 `pkill`。

#### 工具① `capture`
- **输入**：无（或可选 profiling 配置覆盖）。
- **输出**：`{out_dir, signals}`，`signals = {mfu_pct, wait_top_total_ms, bottleneck_class, hot_ops}`。
- **作用**：合并原 `capture` + `analyze` 两阶段——隔离卡采集 profiling → 解析瓶颈信号（MFU / wait / 热点算子 / 粗分类）。
- **失败**：返回 `signals=None`，不抛（LLM 据此走「无分析信号的降级策略」）。

#### 工具② `apply`
- **输入**：`action`（一个动作：改参数 / 开图模式 / 换融合算子 / 改 dtype，见 4.3）。
- **输出**：`{ok, rolled_back, log, current_config}`。
- **作用**：渲染启动脚本 → 备份当前健康配置（脚本 + 参数）→ 端口作用域 kill → 重启 → 健康门禁 → **失败自动回滚**。
- **内置硬护栏**（确定性，不依赖 LLM）：
  - 健康门禁：`wait_port`（300s）；
  - 超时兜底：kill/重启 subprocess 全程 try/except（继承 P0/P1 修复）；
  - 失败自动回滚：门禁不过 → 恢复备份的健康脚本 + 参数。

#### 工具③ `verify`
- **输入**：无。
- **输出**：`{ttft_ms, ttot_ms, ttft_p95, accuracy_match}`。
- **作用**：并发压测（原 `verify` 逻辑）+ **输出一致性比对（新增精度护栏）**。
- **精度比对**：同一批固定 prompt，比对「优化后输出 vs 基线输出」，阈值内判 `accuracy_match=true`；不一致则本轮视为「精度回归」，即使延迟更优也不接受。**这是当前缺失的护栏维度**——改 dtype/图模式可能变数值，只测延迟会漏掉正确性破坏。
- **实现状态**：`accuracy_match` 已在 `agent_loop.try_action` 接线（为 False 强制回滚），但比对逻辑仍是恒真 stub（`tools/verify.py` 暂置 True），输出一致性对拍留待阶段 2。
- **失败**：压测失败降级为 10× 基线（沿用现有契约），`accuracy_match` 单独标记。

#### 工具④ `rollback`
- **输入**：无。
- **输出**：`{ok}`。
- **作用**：恢复到上一个健康配置（`state/scripts_bak` 里的脚本 + 参数）。
- **用途**：`apply` 内部失败时调用；也独立暴露给 LLM 作兜底（LLM 判断历史轮次该回退时）。

### 4.3 探索空间：深优化层 + 参数（核心升级）

动作空间从「8 参数」升级为 **四个昇腾深优化维度 + 参数**。动作 = 单一维度的一次改动（`max_changes_per_round=1` 语义保留，由 LLM 遵守）。

> ⚠️ 下表 flag 名为 KG 已确认的方向，**落地时逐个以 KG `/search` 核实精确名与默认值**（昇腾 API/配置细节不凭记忆）。

| 维度 | 优化动作类型 | 代表性手段（待 KG 核实） | 对应昇腾瓶颈 |
|---|---|---|---|
| **图模式** | 开/关图执行器 | npugraph_ex / aclGraph 图模式 | 下发/调度瓶颈（wait 主导） |
| **算子融合** | 换融合算子 | FusedInferAttention、SwiGlu 等 | 计算瓶颈（逐算子开销） |
| **调度策略** | 开/关调度优化 | `TASK_QUEUE_ENABLE`、`--async-scheduling`、`VLLM_ASCEND_ENABLE_PREFETCH_MLP` | Host 下发瓶颈 |
| **dtype** | 换精度 | BF16 → INT8 / FP8 | 计算带宽瓶颈 |
| **参数** | 现有域 + 昇腾专属参数 | 现有 8 个 + 昇腾扩展参数 | 常规调参 |

**落地约束**：
- `params_to_args` 需扩展以渲染图模式 / 融合 / dtype / 调度 flag（当前只支持 `--k v` 和 `--k` 布尔）；
- 每个动作需带「风险等级」（低/中/高），供 LLM 决策与软护栏使用（见 4.4）。

### 4.4 护栏（三层）

#### 层 1：工具层硬护栏（确定性，不依赖 LLM，不可绕过）
- `apply` 内置健康门禁 + 超时 + 失败自动回滚（见 4.2）；
- `verify` 内置压测超时降级 + 精度比对；
- 进程管理走 `kill_by_port.sh` 端口作用域。

#### 层 2：LLM 层软护栏（system prompt，约束决策）
- **动作白名单**：LLM 只能从 4.3 动作空间选动作，不得越界；
- **危险动作约束**：高风险动作（如 dtype 变更）需满足前置条件（如精度比对通过）才允许；
- **连续失败熔断提示**：连续 N 轮无改善时，prompt 显式要求 LLM 换维度或停；
- **禁止项**：严禁宽模式 kill、严禁改非目标端口/卡、严禁绕过 apply 直接操作进程。

#### 层 3：环境护栏
- ATB 环境加载顺序（`source conda.sh && conda activate ascend-dev && source ~/Ascend/nnal/atb/set_env.sh`）；
- profiling 隔离卡/端口（`profile_device`/`profile_port` 与主服务物理+端口隔离）。

### 4.5 状态与中断续跑

**持久化对象**（`state/run_state.json` 扩展）：
- `baseline`（基线延迟 + 参数）、`best`（最优轮次 + 配置 + 延迟）；
- `history`（每轮：动作、结果、metrics、精度比对、LLM 决策理由）；
- `current_config`（当前健康配置的脚本 + 参数）；
- `agent_ctx`（LLM 循环上下文，含最后一步状态）。

**中断续跑**：
- 每轮原子操作后立即落盘（`persist_context`）；
- agent 重启后 `agent_loop.py --resume` 读 `run_state.json`，从「上一个已完成的原子操作」继续，不重跑已完成的轮次；
- 取代现有「暂停点 + 主会话回传」机制——`agent_gate.py` / `agent_task.json` / `agent_result.json` **废弃**（不再是半自动等待，LLM 在循环内自主决策）。

---

## 5. 与现有引擎的映射（改造清单）

| 现有 stage | 处置 | 去向 |
|---|---|---|
| `capture` | 保留 | 并入工具① `capture` |
| `validate` | 保留 | 并入工具②/③的确定性校验（不再单独成阶段） |
| `analyze` | 保留 | 并入工具① `capture`（采集后即解析信号） |
| `diagnose` | **删除** | 诊断融入 LLM 外层循环本身 |
| `recommend` | **删除决策** | optix 脚本降级为 LLM 的「可选建议工具」，决策权归 LLM |
| `critic` | **删除** | 部署前门禁改为 LLM 软护栏（层 2）+ apply 硬护栏（层 1） |
| `apply` | 保留（核心） | 工具② `apply`，内置回滚不变 |
| `verify` | 保留（核心） | 工具③ `verify`，新增精度比对 |
| `decide` | **拆解** | 单轮回滚并入 `apply`；跨轮「接受/拒绝」交还 LLM |
| `orchestrate` | **删除** | 停止条件交还 LLM + 「连续无改善」降级为软护栏提示 |

**文件改动**：
- 新增：`engine/agent_loop.py`（LLM 主循环）、`engine/tools/`（4 个原语的工具化封装）。
- 改造：`state.py`（扩展 `run_state.json` schema）、`flows/vllm-serve-optimize.json`（扩展 `params.domain` 深优化维度 + 动作空间定义）。
- 删除/废弃：`stages/diagnose.py`、`stages/critic.py`、`stages/decide.py`、`stages/orchestrate.py`（决策逻辑交还主会话 agent）。`agent_gate.py` 保留，作为 `run.py` 的 `diagnose` 与 `agent_loop.py` 的 `decide` 共用的决策回传约定文件。
- 保留不动：`scripts/`（kill_by_port/wait_port/recover_service）、`templates/`（启动/压测/profiling 脚本）。

---

## 6. 落地步骤（分阶段）

### 阶段 1：引擎瘦身为确定性原语（工具化，不引入 LLM）
1. `capture` + `analyze` 合并成工具①，暴露 `{signals}` 契约；
2. `apply` 改造成原子操作（现有回滚逻辑已是原子，微调接口）；
3. `verify` 加「输出一致性比对」精度护栏（新增固定 prompt 对拍脚本）；
4. `rollback` 独立成工具；
5. 删除 `diagnose`/`critic`/`decide`/`orchestrate`/`recommend` 决策逻辑（optix 保留为独立脚本）。
   - **验收**：4 个工具可独立 subprocess 调用，单轮「应用→验证→回滚」闭环跑通，服务不挂。

### 阶段 2：探索空间升级（深优化层）
1. 逐维核实昇腾专属 flag 名（KG `/search`，见 4.3）；
2. `params.domain` 扩展四个深优化维度 + 动作风险等级；
3. `params_to_args` 扩展渲染图模式/融合/dtype/调度 flag。
   - **验收**：`apply` 能正确渲染并应用图模式/调度/dtype 动作。

### 阶段 3：LLM 主循环（外层）
1. 写 `agent_loop.py`（决策经 `agent_gate` 暂停点回传主会话 agent，4 个工具 schema）；
2. 写 system prompt（探索策略 + 护栏约束 + 停止条件）；
3. 状态持久化 + `--resume` 续跑。
   - **验收**：给定 baseline，LLM 无人值守跑完 ≥3 轮优化并正确停止/收敛。

### 阶段 4：验证
1. **单工具对拍**：apply 失败自动回滚、verify 精度比对、rollback 恢复；
2. **端到端无人值守**：从 baseline 自动优化到优于人工调优（或收敛停止），全程无人工介入；
3. **中断续跑**：中途 kill agent 进程 → `--resume` 从断点继续；
4. **回归**：服务全程健康、进程管理无宽模式 kill。

---

## 7. 成功标准

| 维度 | 标准 |
|---|---|
| 无人值守 | LLM 自主完成 ≥3 轮优化并正确收敛/停止，无人工介入 |
| 不翻车 | 单轮任何动作失败都自动回滚到健康配置，服务全程可用 |
| 有效 | 探索空间含深优化层，能选到昇腾真解（图模式/调度 flag），而非在 8 参数里空转 |
| 可审计 | 每轮「试什么、结果、为何继续/停」落盘可回溯 |
| 精度安全 | 延迟优化不得以破坏输出正确性为代价（精度比对护栏兜底） |

---

## 8. 风险与边界

| 风险 | 应对 |
|---|---|
| LLM 决策质量从零打磨，可能低效/发散 | 动作空间白名单 + 软护栏提示约束；先用最小动作空间验证，再扩深优化层 |
| 昇腾 flag 名/默认值记不准 | 逐维 KG 核实，落地时以 KG 为准，不凭记忆 |
| 精度比对引入额外压测开销 | 精度对拍用短固定 prompt 集，与延迟压测分开，可控成本 |
| dtype/融合变更精度风险高 | 归为高风险动作，软护栏前置条件 + verify 精度比对硬兜底 |
| 内核重写支柱在昇腾结构性缺失（闭源算子） | **明确不做**：以「图模式/融合/调度/dtype」的执行方式重写替代，天花板天然低于 ROCm.AI，属生态位差异而非缺陷 |

**边界（本次不做）**：
- AscendC 用户算子重写（对标 GEAK）——依赖用户算子工程 + 闭源算子栈，单独立项；
- 训练场景（NaN 检测、RL 一致性）——本引擎只面向 vLLM 推理服务延迟优化；
- 不整合 msagent（用户决定）。
