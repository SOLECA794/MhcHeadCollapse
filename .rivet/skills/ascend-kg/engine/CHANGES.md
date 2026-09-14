# ascend-kg 引擎决策层改造梳理

> 交付文档。对照 `README.md`（单一事实来源）阅读，本文聚焦「本次改造改了什么、为什么、怎么验证」。
> 只呈现最终形态，不保留迭代过程。

---

## 一、改造目标与结论

把 ascend-kg 引擎从「参数扫描闭环」升级为对齐 ROCm.AI（Hyperloom）优化主干的**完整闭环**，补齐缺失的决策层——即 ROCm.AI 的 **Orchestrator + Critic + Robustness** 所承担的能力：

> 什么时候继续、值不值得做、该回环到哪、循环是否在劣化。

同时补上 LLM 步骤的承载机制：引擎本身保持**纯确定性、零 LLM 依赖**，需要 LLM 的环节经「暂停点」回传主会话 agent。

**结论**：端到端 3 轮闭环验证通过（`EXIT_CODE=0`），决策层三层全部命中预期行为。

---

## 二、最终形态：十阶段闭环

```
capture → validate → analyze → diagnose → recommend → critic → apply → verify → decide → orchestrate
（采集）（护栏）  （分析）  （LLM诊断） （推荐）  （门槛）（应用）（验证）（单轮决策）（跨轮决策）
```

| 阶段 | 职责 | 本次状态 |
|---|---|---|
| capture | 启 profiling 实例（隔离卡/端口）→ 驱动请求 → 销毁 → 定位 `ASCEND_PROFILER_OUTPUT` | 已有（out_dir 改 str） |
| validate | 护栏：校验 profiling 交付件是否完整可分析 | **新增** |
| analyze | `analyze_prof.py` → 解析瓶颈信号 + 确定性粗分类 | 已有（增强 bottleneck_class） |
| diagnose | 读 signals → 暂停回传主会话 agent 做算子级根因诊断 | **新增** |
| recommend | optix 推荐 → 证据映射 → KG 纠偏 → 选单参数候选 | 已有（验证 KG 纠偏） |
| critic | 落地前门槛：否决无 profiling 证据的弱候选 | **新增** |
| apply | 渲染脚本 → 备份 → 精确 kill → 重启 → 健康门禁 | 已有 |
| verify | warmup + reps 压测 → median-of-medians | 已有 |
| decide | 单轮滞回判定：keep / rollback | 已有（验证） |
| orchestrate | 跨轮决策：best 记账 + 平台期/发散/穷尽检测 | **新增** |

---

## 三、决策层设计（本次核心）

决策层是**可审计的确定性守门员**，三层各司其职，全部是纯规则、不引 LLM。

### 3.1 critic —— 落地前门槛（值不值得做）

`recommend` 选型后、`apply` 重启服务前。给「纯启发式且无 profiling 证据」的候选设门槛，否决即 `candidate=None`，apply 自动跳过——**不为弱证据候选重启一次服务**。

- 门槛：`decision.critic_prio_gate = 7`
- 规则：`source == "heuristic"` 且 `prio >= 7` → 否决；否则放行
- 证据强弱由 recommend 的 `prio` 编码：**信号命中证据映射 = 1/2/3，无信号 = 9**（数字小 = 证据强）

### 3.2 decide —— 单轮滞回判定（这轮要不要）

对「keep / rollback」做滞回裁决，不把噪声当收益：

- **keep**：`ttft ≤ baseline × (1 − hyst)` 且 `ttot ≤ baseline × ttot_room`（`hyst = 0.003`、`ttot_room = 1.10`）
- **rollback**：未达 0.3% 改善门槛，或 TTOT 退化超 10% 容限 → 恢复上一轮健康脚本 + 参数并重启
- 终态健康兜底 `_ensure_healthy`：smoke 不通过则尝试回滚到最近健康脚本

### 3.3 orchestrate —— 跨轮决策（该回环到哪、是否劣化）

单轮 decide 只判 accept/reject，本层管全局：

1. **best 记账**：本轮 keep 且优于历史 best → 更新 `state.best`
2. **连续无改善计数**：从最近一轮往回数非 keep 的连续轮数
3. **next_action**（`run.py` 据此决定继续或 break）：

| next_action | 触发条件 |
|---|---|
| `stop_unhealthy` | decide 判 `unhealthy` / `rollback_failed`（主服务无法恢复健康） |
| `stop_exhausted` | `no_candidate`（候选已穷尽或被 Critic 否决） |
| `stop_plateau` | 连续 `consec_no_improve_stop = 2` 轮无改善 |
| `continue` | 以上皆非 |

`stop_unhealthy` 置 `state.status = fatal`（退出码 1）；其余 `stop_*` 干净收束（退出码 0）。

---

## 四、LLM 暂停点机制（agent_gate.py）

引擎红线「不内嵌 LLM、不编排 msAgent agent」，需要 LLM 的 `diagnose` 阶段经文件约定回传主会话 agent。

### 4.1 文件约定（`engine/state/` 下）

| 文件 | 写入方 | 内容 |
|---|---|---|
| `agent_task.json` | 引擎 | `{task_id, stage, round_key, goal, context, skill_refs, expected_output_schema, status:"pending"}` |
| `agent_result.json` | agent | `{task_id, result, status:"done"}` |

### 4.2 流程

```
diagnose 读 signals 非空 → emit_task 写 agent_task.json + 置 ctx["pending"]
  → run.py 检测 pending：快照中间产物(out_dir/validation/signals/cand)
      + 持久化 state.pending{round_key, stage_idx, task_id} → 退出码 2
主会话 agent：读 agent_task.json → 按 skill_refs 加载本地 SKILL.md → 推理
  → 按 expected_output_schema 写 agent_result.json
  → python engine/run.py --flow vllm-serve-optimize --resume
run.py resume：读 state.pending → 定位 (round_key, stage_idx) → 恢复快照
  → 从该 stage 续跑（不重跑 capture/analyze）
```

### 4.3 关键设计点

- **task_id 幂等**：`task_id = round_key + "-" + stage`（如 `r1-diagnose`）；`consume_result` 校验 task_id 匹配，防上一轮 stale 结果被误读（`emit_task` 同时清掉旧 result）。
- **stage 级 resume**：从 pending 的 stage 续跑，而非整轮重跑；已完成轮次跳过。
- **退出码**：`fatal = 1` / `pending = 2` / `complete = 0`。

---

## 五、护栏与分析层增强

### 5.1 validate（护栏）

capture 后校验 profiling 交付件：

- **硬依赖** `kernel_details.csv` 缺失 → `validation = invalid` + 降级 `out_dir=None`（后续 analyze/diagnose 全跳过）
- **软依赖** `op_statistic.csv` / `trace_view.json` 缺失 → 仅 `warning`，不阻断（analyze 只用 kernel_details.csv）

### 5.2 analyze（瓶颈粗分类）

`_classify_bottleneck` 确定性粗分类（精诊断交 diagnose/LLM）：

- `wait_top > 100ms` → `schedule`（下发/同步等待主导）
- `MFU < 50%` → `compute`
- 否则 → `unknown`

信号含：`mfu_pct`、`hot_ops`（热点算子 TopN）、`wait_top_total_ms`、prefill/decode matmul 计数与耗时。

---

## 六、关键正确性约定

- **`ctx["out_dir"]` 统一为 str**（capture 产出 `str(raw_dir / "ASCEND_PROFILER_OUTPUT")`），避免 `Path` 对象泄漏进 JSON 序列化（`agent_task.json` 的 context、`run_state.json` 的 snapshot 两处都会写 out_dir）。
- **进程管理红线**：销毁一律走 `engine/scripts/kill_by_port.sh`（端口作用域），严禁宽模式 pkill。
- **ATB 环境顺序**：启动 vllm serve 前 `source atb/set_env.sh`，且必须在 conda activate 之后（模板已内置）。

---

## 七、文件改动清单

### 新增

| 文件 | 职责 |
|---|---|
| `engine/agent_gate.py` | 暂停点：`emit_task` / `consume_result` |
| `engine/stages/validate.py` | 护栏：profiling 交付件校验 |
| `engine/stages/diagnose.py` | LLM 诊断暂停点 |
| `engine/stages/critic.py` | 落地前门槛 |
| `engine/stages/orchestrate.py` | 跨轮决策层 |

### 修改

| 文件 | 改动 |
|---|---|
| `engine/run.py` | pending 检测 + `return 2` + stage 级 resume + best 记账 + `stop_*` 断点 |
| `engine/stages/capture.py` | `out_dir` 改为 str（根因修复） |
| `engine/stages/analyze.py` | `_classify_bottleneck` 粗分类并入 signals |
| `engine/flows/vllm-serve-optimize.json` | stages 序列插 validate/diagnose/critic/orchestrate；新增 `decision` / `perf` 段、`paths.msot_skill`/`mfu_skill`；domain 扩 6 sub + 2 compose 参数（图模式/融合开关） |
| `engine/state.py` | `params_to_args` compose 渲染 + `within_domain` compose 守卫 |
| `engine/agent_loop.py` | 动作空间隐藏 compose/标注 sub + 图模式互斥规则 + `graph_ok=False` 强制回滚 |
| `engine/stages/verify.py` | `_check_graph_capture` 图捕获校验 + `ctx["graph_ok"]` |
| `engine/stages/decide.py` | `graph_ok=False` → 判 rollback（确定性流水线同守） |
| `engine/tools/verify.py` | `verify()` 返回 `graph_ok` 字段 |

### 文档/指引

| 文件 | 改动 |
|---|---|
| `engine/README.md` | pending 机制 + stage 契约 + 验证记录 |
| `SKILL.md` §8.4 | 引擎 pending 时 agent 的职责 |
| `agents/ascend-kg-worker.md` | 同上，一段 |
| `claude-memory.md` | 「引擎 pending 非失败，agent 应回传」一句 |
| `evals/evals.json` | 新增 id=7/8/9（引擎路由 + pending 回传）+ id=10（agent_loop 双循环） |

---

## 八、端到端验证（2026-08-13）

Qwen2.5-7B / TP=1 / 910B3 BF16，`target=ttft`，`max-rounds=3`，hyst=3%。

> ⚠️ 本节验证于 2026-08-13、当时 `hyst=3%`。2026-08-14 已按需把门槛下调至 `0.3%`（`flow.perf.hyst=0.003`），
> 故下表「+1.2%/+2.1% 判 rollback」的结论在 0.3% 阈值下会反转为 keep——此处仅作历史记录，不代表当前行为。

| 轮次 | 推荐参数 | TTFT | decide | orchestrate |
|---|---|---|---|---|
| round 0 | —（基线） | 3168.9ms | — | — |
| round 1 | `max-num-batched-tokens` | 3132.0ms | rollback（+1.2%） | continue |
| round 2 | `max-num-seqs` | 3102.7ms | rollback（+2.1%） | **stop_plateau** |

**决策层三层命中**：

- **critic**：两轮均 `放行`（`src=optix prio=9` → 有证据，未误拦截）
- **decide**：两轮改善均低于 3% 门槛，正确 rollback（未把噪声当收益）
- **orchestrate**：连续 2 轮无改善 → `stop_plateau`，干净收束，`EXIT_CODE=0`

**暂停点链路**（两轮各验证一次）：diagnose 暂停（退出码 2）→ 主会话 agent 写 `agent_result.json`（r1/r2 区分 task_id）→ `--resume` 从 diagnose 续跑，**未重跑 capture/analyze**（日志 `[round N] 已存在，跳过（resume）` + 直接从 diagnose 开始）。

**诊断结论**：`bottleneck_class=schedule`，MFU 仅 2.0%、wait 1212ms 主导，热点算子 `aclnnMatmul` 占 79%（低并发小 shape 搬运墙）。参数层已无可榨取空间，真正瓶颈在算子融合 / AscendC 算子重写（第二阶段）。

**KG 增强**：orchestrate 阶段命中 `cann-recipes-infer` 的 `block_size=128` 强制确认（score 0.91），佐证诊断结论。

---

## 九、图优化 + 融合算子落进动作空间（DESIGN-v2 阶段2 partial）

把 `params.domain` 从「纯参数扫描」扩展到**图优化 + 融合算子**两个重优化维度（DESIGN-v2 阶段2 探索空间升级）。
默认行为与基线一致：融合开关全开、图模式 `FULL_AND_PIECEWISE`（v1 默认），未设置时基线不渲染任何新 flag。

### 9.1 新参数面（`flows/vllm-serve-optimize.json`）

| 参数 | kind | 取值 | 生效 |
|---|---|---|---|
| `cudagraph-mode` | sub | `FULL_AND_PIECEWISE`/`FULL_DECODE_ONLY`/`PIECEWISE`/`NONE` | 合成到 `--compilation-config` |
| `enable-static-kernel` | sub flag | true/false | 合成到 `--additional-config`（需 npugraph_ex） |
| `fuse-norm-quant` / `fuse-qknorm-rope` / `fuse-muls-add` | sub flag | true/false | 合成到 `--additional-config`（融合 pass 开关） |
| `fusion-gmmswigluquant` | sub flag | true/false | 合成到 `--additional-config`（算子级融合） |

sub 参数（`kind:"sub"`）只作 compose 输入、不单独渲染成独立 flag；由 compose 参数组装为内联 JSON：
`--compilation-config '{"cudagraph_mode": ...}'`；`--additional-config '{"ascend_compilation_config": {...}, "ascend_fusion_config": {...}}'`。

排除项：`fuse_allreduce_rms`/`enable_sp`（需 TP>1）；xlite 图模式（实验性，切 XliteWorker）；`cudagraph_capture_sizes`（`last==max_cudagraph_capture_size` 断言易碎）。

### 9.2 渲染与守卫（`engine/state.py`）

- `params_to_args`：`kind=sub` 跳过（同 kind=env）；compose 参数从 params 里**显式设置过**的 sub 组装 JSON，非空才渲染；布尔 sub 显式转 true/false（**False 也要 emit，区别于独立 flag 的 False 省略**）；未设置的 sub 省略 → vllm 默认。
- `within_domain`：compose 是派生参数，禁止直接设置（防 agent 越界触发数值分支 TypeError）。

### 9.3 决策与护栏

- `agent_loop` 动作空间隐藏 compose 派生参数、sub 参数标注「CLI(合成到图/融合flag)」；决策规则新增：图模式互斥（cudagraph-mode 非 NONE 与 enforce-eager 不同时开）、融合开关默认全开、图/融合动作后校验图捕获。
- **图捕获校验**（补 README §八「graph capture 卡死无程序化校验」痛点）：verify 读本轮服务日志 `Graph capturing finished` → 图已生效；`Skipping CUDA graph capture` → 配置声称开图但被平台跳过/未捕获；`enforce-eager` 开 → 不适用；日志不可判 → None 不误伤。
- 图捕获未生效 → `agent_loop.try_action` 与确定性 `decide` **双路径强制回滚**（对齐 smoke/精度硬护栏），杜绝「配置没落地却记成收益」。

### 9.4 边界（本次不做）

新增 torch.fx 融合 pass 源码（需改 site-packages，独立阶段）；xlite 图模式；capture-sizes 细调；AscendC 核函数重写（另立 track）。

---

## 十、边界（本次明确不做）

- **AscendC 算子重写（重）**——需用户算子工程 + 精度风险，第二阶段单独验证；本次仅 `diagnose` 产算子级建议（tiling/dtype/融合/选型）作为其输入。
- **不接 msprof_mcp MCP**——改用本机确定性脚本 + 回传 agent 加载提示词型 skill。
- **代码审查护栏（Minos）**——轻阶段无代码变更，不接。
- **NaN 精度护栏**——训练场景，推理服务不适用。
- **无 router / 经验层 / guided 模式**——单 flow、纯 auto、无人工审批暂停点。
- **不新建第二条 flow**——仍只维护 `vllm-serve-optimize`。
