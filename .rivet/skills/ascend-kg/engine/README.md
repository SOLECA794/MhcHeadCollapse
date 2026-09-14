# ascend-kg 编排引擎（engine/）

> 状态：**已交付**（2026-08-13）。单一事实来源；接手者先读本文件。
> 目标读者：后续扩展/维护引擎的实现者，以及需触发引擎的 agent。

---

## 一、引擎是什么

把 ascend-kg 从「检索引擎 + 方法论手册」升级为**按 flow 自动执行昇腾任务**的编排层。

当前支持**一类场景**：ROCm.AI（Hyperloom）式的推理服务延迟优化闭环，三个后端（vLLM-Ascend / SGLang-NPU / MindIE，均真机实测）。其他昇腾任务**不走**这条自动化路径，仍走 SKILL.md §0-§7 的检索/加载 Skill 流程。

**闭环**：静态 flow 定义（JSON）声明参数域 + 路径常量，`agent_loop.py` 读 flow → 两幕编排（值级幕 r1..N 每轮 decide 暂停点 → apply → capture → verify → keep/回滚；组合幕 cN 从已 keep 参数的交互子集组合验证）→ final_sweep 对拍 → drift_check 收尾 → 状态持久化。

```
r0-domain → baseline(r0) → 粗筛幕 scr1..K → [refine 排空 → decide → apply → capture → verify → keep/rollback] × N
（KG动作空间） （defaults 重启）  （全参数单点快筛）   ↑ LLM 暂停点，退出码 2，回传 agent 决策
→ 组合幕 c1..M（composition_queue 子集组合对拍） → final_sweep → drift_check → 收尾
旁路：值级轮每 anchor_every 轮 → 漂移锚点 aN（零重启复测当前配置，|漂移|>5% 重锚基线/水位线）；
     apply 连续失败 ≥3（粗筛/值级/组合任一循环）→ fail{N}-fuse 熔断暂停，回传 continue/terminate
     值级幕收敛后 → 副目标 Pareto 通道 pN（影子赢家在当前最优配置上复测，占优则 keep 进交付配置）
```

**精修轮（fN，探索/精修预算分离）**：每轮 decide 前排空上轮 keep 产生的邻域补扫队列（center±step 未试值，逐点真机轮）——不占 `--max-rounds`（独立 `--max-refine-rounds` 默认 4 封顶）、不走 decide 暂停点（确定性微调）、rollback 不计 plateau、进组合幕前最终排空（backbone 局部最优准入）。

**双目标影子记账（方案 1，2026-08-28）**：每轮实测全量采集 ttft/吞吐指标，主目标照旧驱动 keep/回滚；副目标跑完整判定（复用自己的 guard/噪声地板）后只记账 `state["best_by_target"][副]`（不写主目标 best，防双写漂移）。主目标回滚但副目标大赢的 trade-off 配置（案例6 FULL_DECODE_ONLY：TTFT -6.6% 回滚 / 吞吐 +58%）不丢失；decide context 的 `dual_target` 字段与收尾报告双列最优可见，换 `--target` 的第二遍 pass 整遍省掉。基线不含副目标指标的后端自动禁用。

**粗筛幕（scr 轮，2026-08-28）**：值级幕前对全部可设参数（排除 locked/compose）快筛——数值=离 default 最远端 1 点、布尔=翻转/开启、**枚举=全值展开**（case7 教训：单哨兵 vals[0] 错过 FULL_DECODE_ONLY 档；default 已知的值=基线状态跳过）——reps=1 快测、恒回滚（OFAT 从基线出发单变量归因）、不占 max_rounds、不计 plateau（sweep 语义）；**执行序=风险升序**（low→mid→high，未标注按 low）——易炸参数（编译超时/改 dtype）殿后，失败或熔断（P0-B）发生时低风险参数画像已到手（case7 教训：static-kernel 无序排第 14，连锁毒死后面 11 轮丢 46% 画像）。中间轮只还 state 不物理回滚（下一轮 apply 本就重启服务），幕末统一一次物理回滚还原基线（省 N-1 次重启）。实测效应榜单（双指标画像+**值级复测置信反馈**：复测口径=该轮相对当时水位线的**增量**（2026-08-29 修正，case8 教训：叠骨干轮按 vs 基线算把 -1.5% 增量误标「复测保真」）；证伪条目行尾标记「复测已证伪→不可兑现」（reps=1 噪声或与骨干负交互）、正向多数证伪时头部警示——case7 教训：4 个正向条目复测 3 个是噪声，reps=1 榜单需要误差棒）进 decide context `screening` 字段——LLM 选参从信念排序变实测排序，冷门参数不再零登场（案例6 教训：24 参数只实测 3 个）。apply 失败/smoke 失败/精度 B 类记入榜单失败项，不中断幕；幂等续跑按 (param, value) 查重；`--screen off` 关闭。

**统计判定加固（P0-A/P0-B/P1-A/P1-B，2026-08-28 case7 复盘落地）**：

- **噪声地板 min 化（P0-A）**：`hyst_eff = max(hyst, noise_k × min(基线相对噪声, 本轮相对噪声))`——原 max 口径把地板锁死在最差一次测量（case7 基线池化 17.9% vs 轮内 2-6%，≤5% 真收益全判噪声、影子记账零触发）；min 取测量更稳一侧。配套 **unstable 轮确认**：本轮噪声 >2× 基线且 >2% → 过线 keep 也标 borderline 触发重测（min 放松假阴性防护后，幸运轮假 keep 由一次重测兜底）。跨时段系统性漂移由锚点 + 终局 drift_check 兜底。
- **guard 噪声展宽（P0-A）**：守门指标自身噪声超过配置容限时 `limit_room = max(room, 1+地板)`（case7 r11：吞吐 +8.0% 被 TTFT +11.4% 拦，而同配置 TTFT 摆动 10pt+——阈值小于噪声的守门是随机拦截）；展宽只放宽退化上限，catastrophic 仍拦；`guard_widened` 记 components（放行依赖展宽/展宽后仍拦两侧都记）。
- **业务画像 guard 放宽（--profile，同日第三轮）**：统计展宽管「退化是否超出噪声可解释」，业务放宽管「代价是否商业可接受」——正交两层。CLI `--profile batch|online|balanced [--ttft-budget X]`：`batch`=守门退化为灾难档（room → `catastrophic_room` 缺省 1.5，flow 可覆盖；批量离线反向指标无商业意义，只拦功能性劣化，smoke/精度护栏不受影响）；`balanced`=配置容限上再放宽显式预算（`--ttft-budget 0.15`=守门指标可再退 15%）；`online`（缺省）=现状严格零回归。装配期 `cli._apply_guard_policy` 把放宽后 room 烘进 `flow.perf`（compute_objective/keep_verdict/影子记账全路径生效，判定代码零改动），原值与策略记 `perf.guard_policy`；components 里 `guard_policy` 与 `guard_widened` **分开记账**——keep/拦各依赖了业务契约还是统计展宽，复盘可分辨。case7 r11 镜像推演：batch 下吞吐 +8.0%/TTFT +11.4% 从被拦变 keep → 组合幕激活。
- **apply 连续失败熔断（P0-B）**：连续 ≥3 次 apply 失败 → `fail{N}-fuse` 暂停点（context 带 recent_failures 根因摘录与孤儿清理提示）；continue 清零再给 3 次机会 / terminate 收尾。case7 教训：scr14-24 连锁 11 轮失败空转 1h27m。
- **失败不计 plateau + 根因入账（P1-A）**：metrics=None 的轮不计连续无改善（无测量 ≠ 无改善证据，环境故障不烧探索名额）；apply 失败的 OOM/超时/崩溃关键行摘录进 rounds 的 `fail_reason`（复盘不再翻 /tmp 大日志）。
- **中途漂移锚点（P1-B）**：`--anchor-every N`（默认 5，0=关闭）每 N 个值级轮零重启复测当前运行配置（服务正跑的就是它）；|漂移|>5% → baseline 全指标与 rolling best 按比例缩放 + stats 换锚点新鲜估计 + r0 归档 `baseline_r0`（收尾报告对照）。锚点轮 decision=anchor + sweep 语义（plateau/history 透明），decide context 注入 `env_drift`。
- **副目标 Pareto 通道（pN 轮，2026-08-29 case8 档C教训）**：值级幕收敛后、组合幕前的确定性节点（不走 decide 暂停点）——副目标影子赢家（如吞吐 run 里 TTFT -16% 的参数）因主目标持平永远过不了 keep_verdict（只看主指标）、组合准入又是 kept-only，过去从未复测从未进组合；但「同主指标 + 更好副指标」对最终配置是**严格 Pareto 占优**。通道把影子赢家升格为可信证据：来源轮可归因单参数时，在当前运行配置（=主目标最优）上 reps=3 复测 → `pareto_verdict` 双轴判定（主指标 vs 最近 keep 轮实测退化 ≤ 主地板；副指标改善 ≥ max(副目标 hyst, 副地板)）→ keep 进 kept_params（自然进 backbone/final_sweep，**交付配置含它**，final_sweep 终局对拍兜底联合劣化）或回滚；主目标水位线 `best` 不更新（防副目标轮污染主目标记账）。上限 `--pareto-max`（默认 1，0=关闭）；smoke/精度B/图捕获硬护栏与 try_action 同口径不放宽。

**复审修复 8 项（2026-08-29，外部 review + 自审落地）**：

- **#1 粗筛幕末回滚错脚本**：幕末收尾曾回滚到 `scr{N}_prev`（= 上一个哨兵叠加态），非基线——成功路径必触发，值级幕起跑在脏配置上。修为 `_finalize_screening()` 统一回滚到 `scr1`（= 基线脚本），幂等可 resume（回滚失败保留 applied 标志重试）。影响面校准：值级轮每轮从 current_params 重渲染，裁决不受脏脚本污染；脏的是 r1 profiling signals、诊断轮与全回滚 run 的终局 drift_check。
- **#2 锚点接力把 keep 改善误判为漂移（灾难级）**：两锚点之间发生 keep（配置已变）时，锚点参照仍取上一锚点实测（跑在旧配置上）——keep 收益被当成环境漂移，基线/水位线按收益比例假重锚，水位线推到不可达、之后全 rollback。修为锚点记 `config`，接力仅在配置未变时成立，否则回退「当前配置的上次测量」= 最近 keep 轮实测。
- **#3 v1 state ttfb 指标键迁移**：ttft 化改名晚于一批 run 落盘，残留旧 state 的机器 resume 后 ttft 键全读不到。`validate_and_migrate` 加递归改键 walker（`ttfb_ms/ttfb_p95/ttfb_pct/best_by_target 目标键 ttfb` → `ttft*`，全容器；rationale 历史文案值不动），命中大声打印。
- **#4 v1 rounds 缺必填键降级**：实测旧 state rounds 无 `decision` 键（schema 加严后 resume 直接 raise）。修为 action/kept 补保守默认、decision 从 kept 推导（v1 无 decision 词表，kept 是唯一真相源）+ 大声告警；round 键缺失仍拒载（无身份可发明）。
- **#5 守门画像持久化**：`--profile/--ttft-budget` 原先只烘 flow.perf 不落 state，resume 忘带即静默回退 online，前后轮 guard 口径不一致。修为 `resolve_guard_policy` 三级解析（CLI 显式 > `state["profile_policy"]` > online）+ main 应用一次写回；Cfg 不再预烘（`_apply_guard_policy` 以当前 room 为 orig 基准，预烘+main 再烘 = 预算翻倍）。
- **#6 TTOT 噪声键映射**：`key="ttot_ms"` 的噪声地板/borderline 此前落到 `ttft_rel_noise`——TTOT 守门一直拿 TTFT 的离散度当地板。objective/decide 两处映射 `ttot_ms→ttot_rel_noise`（verify 侧分母磁盘上本已正确）。
- **#7 熔断 task_id episode 序号**：`fail{N}-fuse` 固定 task_id，同 run 第二次熔断会自动消费上一 episode 的陈旧 continue（agent 没被重新问过）。修为 `fail{N}-e{episode}`，消费成功才推进 episode 持久化——崩溃在同 episode 内重 consume 幂等保留。
- **#10 连拒终止**：被拒提议轮（高风险无信号/组合幕队列外，decision=rejected*）不 apply 不压测——熔断够不着（apply 没跑）、plateau 不计（无测量 ≠ 无改善证据），LLM 反复提高风险可无限空转烧满 max_rounds。修为 `consec_rejected ≥3 → exhausted` 终止（sweep/refine 透明、apply 失败轮打破连拒链），组合幕 offqueue 拒绝点同步局部 break 止损。

四类 LLM 暂停点（统一经 `agent_gate`，退出码 2）：`r0-domain`（KG 检索→agent 提取→与手写 domain 合并）、每轮 `decide`（下一轮试什么/停止）、`rN-diagnose`（瓶颈深诊，节流：每 run ≤2 次 + 瓶颈指纹不变不重诊）、`fail{N}-fuse`（apply 连续失败熔断，continue/terminate）。

**唯一入口**（P1-1 入口统一，2026-08-27；状态按 flow 隔离存 `state/{flow_id}/run_state.json`，`agent_task/agent_result.json` 为会话级协议文件留在 `state/` 根）：

- `agent_loop.py` —— **LLM 会话委托双循环**（方案 C，见 `DESIGN-v2.md`）。外层决策经 `agent_gate` 暂停点回传主会话 agent（不内嵌 LLM、不依赖 `ANTHROPIC_*` 环境变量）+ 内层确定性原子操作（`tools/`：apply→verify→is_kept→变差自动回滚）。
- ~~`run.py` 确定性十阶段流水线~~ —— **已退役**，归档 `attic/run.py`（启动即打印废弃横幅并退出）；其 Cfg/参数解析迁至 `cli.py`，optix 证据产出迁至 `tools/priors.py` 注入 decide context。`stages/` 十阶段仅存档供考古（§五契约对应已退役入口）。

## 二、目录结构

```
engine/
├── README.md            # 本文件
├── agent_loop.py        # 唯一 CLI 入口（LLM 会话委托双循环）：python engine/agent_loop.py --flow <id> [flags] [--resume]
├── cli.py               # parse_args/Cfg/cfg_from_argv（两条旧入口的参数装配合一，P1-1）
├── registry.py          # load_flow(flow_id) → 校验 + SimpleNamespace
├── state.py             # render/snapshot_state/save_state/round_keys/params_to_args/params_to_env/within_domain + validate_and_migrate（P1-2a schema）+ 路径常量与 set_flow_scope
├── backends/            # 推理引擎后端接缝（DESIGN-multi-backend §3.2）
│   ├── __init__.py      # get_backend/backend_of 注册表（flow.backend 缺省 vllm-ascend）
│   ├── base.py          # Backend 六方法接口 + OpenAI 兼容默认实现
│   ├── vllm_ascend.py   # vLLM-Ascend：渲染/图捕获校验/kill 兜底签名
│   ├── sglang.py        # SGLang-NPU：CLI flag 载体 + launch_server 启动
│   └── mindie.py        # MindIE 3.0.0：config.json 载体 + 容器内 daemon 重启
├── kg.py                # kg_available/kg_search/kg_correct（纯 stdlib + curl；kg_correct 遗留兼容，新链路勿用）
├── agent_gate.py        # checkpoint 暂停点：emit_task/consume_result（domain/decide/diagnose 三类暂停点共用，task_id 按 "{round_key}-{stage}" 幂等）
├── setup_domain.py      # KG 检索独立摸底薄包装（检索逻辑在 tools/domain_node.py）
├── attic/run.py         # 已退役的确定性十阶段流水线（启动即废弃横幅退出，存档考古）
├── flows/               # flow 定义（JSON）
│   ├── vllm-serve-optimize.json   # 已真机闭环
│   ├── sglang-serve-optimize.json # 已真机实测（conda env sglang-dev 卡6；环境搭建见记忆 sglang-npu-facts）
│   └── mindie-serve-optimize.json # 已真机实测（容器常驻 mindie-opt 卡7；基线 config 在 ~/mindie-opt/）
├── stages/              # 十阶段（统一签名 run(flow, ctx)）——仅存档：服务已退役的 run.py，agent_loop 不调用
│   ├── capture.py  validate.py  analyze.py  diagnose.py
│   ├── recommend.py  critic.py  apply.py  verify.py  decide.py  orchestrate.py
├── tools/               # 内层确定性原子操作（agent_loop 调用）
│   ├── apply.py  verify.py  decide.py  capture.py  objective.py  accuracy.py
│   ├── sampler.py  grouping.py  targeting.py   # 候选生成/分组交互先验/目标感知
│   ├── domain_node.py  priors.py               # KG 动作空间节点 / optix 先验（P1-1b）
├── tests/               # 单元测试（162 例：编排/组合/统计功效/schema/渲染矩阵/priors/诊断节流/gate 协议/精修轮/影子记账/粗筛幕/熔断/锚点/画像 guard/Pareto 通道/v1 迁移/连拒终止）
├── scripts/             # 运维原语（红线工具）
│   ├── kill_by_port.sh  wait_port.sh  recover_service.sh
└── templates/           # 渲染模板 + 分析脚本
    ├── start_vllm_opt.sh.j2  start_vllm_profile.sh.j2  concurrency_test.sh.j2
    ├── prof_req.py  analyze_prof.py
```

## 三、CLI

```bash
python engine/agent_loop.py --flow vllm-serve-optimize \
  [--port 8000] [--device 0] [--profile-port 8001] [--profile-device 1] \
  [--target ttft|throughput] [--max-rounds 50] [--max-combo-rounds 8] \
  [--max-refine-rounds 4] [--reps 3] [--bench-timeout 120] \
  [--with-optix on|off] [--screen on|off] [--anchor-every 5] [--pareto-max 1] \
  [--profile online|batch|balanced --ttft-budget 0.15] [--resume] [--model <路径>] [--workload <JSON>]
```

- `--flow` 必填；`--model/--target/--max-rounds` 未给时回退 flow 定义（参数装配在 `cli.py`，两条旧入口合一）。
- `--resume` 读 `state/{flow_id}/run_state.json` 续跑（跳过已完成轮次；读路径过 `validate_and_migrate`，旧 schema 无感迁移）。
- **round 0 强制把主服务重启为 defaults 配置再采基线**——无需预启动；端口上若有上轮残留服务会被清场重起，避免把上轮 keep 的候选性能误当 defaults 基线。fresh run 在此之前先走 `r0-domain` 必走节点。
- 每轮「决策」是 `agent_gate` 暂停点（退出码 2）：引擎写 `agent_task.json`（含动作空间/信号/历史/规则/priors）→ 主会话 agent 决策 → 写 `agent_result.json` → `--resume` 续跑。**不依赖 `ANTHROPIC_*` 环境变量、不内嵌 LLM**。
- 决策主体是主会话 agent，`tools/` 是确定性兜底（apply→verify→is_kept→变差自动回滚）。

## 四、flow 定义格式（JSON）

`flows/vllm-serve-optimize.json` 为完整实例（10 阶段全量）；`sglang/mindie-serve-optimize` 为裁剪实例（无 diagnose/recommend/critic，主入口 `agent_loop.py`，动作空间由 KG 核实的参数域驱动）。结构：

```jsonc
{
  "flow_id": "vllm-serve-optimize",
  "name": "…", "mode": "auto", "target": "ttft", "max_rounds": 3,
  "paths": { "atb_env": "…", "model_default": "…", "optix_script": "…", "…" },
  "model_dims": { "n_layers": 28, "hidden": 3584, "…" },
  "perf": { "mfu_peak_tflops": 294.91, "hyst": 0.003, "ttot_room": 1.10 },
  "decision": { "consec_no_improve_stop": 2, "critic_prio_gate": 7 },  // 跨轮/落地门槛
  "params": {
    "defaults": { "max-model-len": 8192, "…" },          // round 0 基线参数
    "domain": { "block-size": {"min":128,"max":128,"locked":true}, "…" },  // 参数域护栏
    "blocked": { "block-size": "910B chunked prefill 强制 128" }  // 禁止推荐
  },
  "stages": [
    {"id":"capture","module":"engine.stages.capture"},
    {"id":"recommend","module":"engine.stages.recommend","kg_correct":true},
    {"id":"apply","module":"engine.stages.apply","gate":{"rollback":true}},
    {"id":"decide","module":"engine.stages.decide","gate":{"hysteresis":true}}  // … 共 10 阶段（validate/analyze/diagnose/critic/verify/orchestrate 略，见 §二）
  ]
}
```

- **确定性引擎纯 stdlib**（JSON + SimpleNamespace），无第三方依赖；入口（`agent_loop.py`）不内嵌 LLM——LLM 步骤统一经 `agent_gate` 回传主会话 agent（见 §一）。
- `params.domain/blocked/defaults` 保留为 dict（stage 按 key 查表）；其余段转 SimpleNamespace。
- `paths`/`model_dims` 段为本机绝对路径（`/home/z60124773/...`），**单机部署假设**：迁移到别的机器只需改 `flows/<flow>.json` 的 `paths`（模板经 `@KEY@` 渲染，无需改）。

## 五、stage 契约（遗留：仅服务已退役的 attic/run.py）

统一签名 `run(flow, ctx)`：`flow` 是加载后的 Flow 对象，`ctx` 是共享 dict 命名空间。**agent_loop 不走 stages/**（它用 `tools/` 原子操作 + 自有两幕编排）；本节保留供考古与对照——暂停点协议已被 agent_loop 泛化复用（task_id 按 round-stage 幂等）。

| 约定 | 说明 |
|---|---|
| stage 间传数据 | 经 `ctx`：`out_dir`（capture）→ `validation`（validate）→ `signals`（analyze）→ `diagnosis`（diagnose）→ `cand`（recommend→critic）→ `metrics`（verify）→ `verdict`（decide）→ `orchestration`（orchestrate） |
| 降级不抛 | capture 失败 → `out_dir=None`；validate 见缺失 → `validation=skipped/invalid`；analyze 见 out_dir 缺失 → `signals=None`；diagnose 见 signals 缺失 → `diagnosis=None`；recommend 无候选或 critic 否决 → `candidate=None`（apply 跳过） |
| 致命错误 | `ctx["fatal"]=True` + `ctx["fatal_reason"]`，`run.py` 检查后中断闭环（退出码 1） |
| LLM 暂停点 | `diagnose` 调 `agent_gate.emit_task` 写 `agent_task.json` 并置 `ctx["pending"]=True`；`run.py` 检测后快照中间产物 + 存 `state.pending` + **退出码 2**；agent 回传 `agent_result.json` 后 `--resume` 从该 stage 续跑（不重跑 capture/analyze） |
| 分支内化 | 线性 stage 序列内，把「无候选/采集失败/回滚失败」推入 stage 内部处理，不破坏 run.py 的 for 循环 |
| 跨轮次决策 | `orchestrate` 产 `orchestration.next_action`（`continue`/`stop_unhealthy`/`stop_exhausted`/`stop_plateau`）；`run.py` 读到 `stop_*` 即 break（`stop_unhealthy` 置 fatal） |

## 六、红线（进程管理）

- **进程销毁一律走 `engine/scripts/kill_by_port.sh`（端口作用域）**，严禁宽模式 `pkill`（会连主服务一起杀）。
- 启动 vllm serve 前必须 `source /home/z60124773/Ascend/nnal/atb/set_env.sh`（模板已内置）。
- KG 纠偏：block-size 强制 128（910B chunked prefill）——统一走 flow domain 的 `locked` 字段（`within_domain` 拒改）；`kg_correct` 已摘除出决策链（P1-1c，遗留兼容见 `kg.py`）。

## 七、明确不做（边界）

- **无 router 多场景路由**——只一条 flow，无任务分诊/关键词匹配。
- **无经验层**——不做经验沉淀/复用/上传 KG。
- **无 guided 半自动模式**——纯 auto，无人工审批暂停点。"暂停"只有 LLM 暂停点（`agent_loop.py` 的 `r0-domain`/`decide`/`rN-diagnose`，均回传 agent，非人审）。
- **无 smart_points / adapter 降级链**——确定性内核（`agent_loop.py` + `tools/`）不内嵌 LLM、不编排 msAgent agent：LLM 步骤经 `agent_gate` 回传主会话 agent（读本地 SKILL.md 推理），KG 检索仅经 `kg.py`/`domain_node.py` 的可插拔调用。外层决策同样经 `agent_gate` 回传主会话 agent（**不内嵌 LLM、不依赖 `ANTHROPIC_*`**），不破坏内核零 LLM 依赖。
- **内核层只做「轻」**——`diagnose` 产算子级建议（tiling/dtype/融合/选型），不动算子源码；AscendC 算子重写（重）属第二阶段，不在本次。
- **不新建第二、三条 flow**——已突破：vLLM 样板之外新增 sglang/mindie 两条 flow（后端接缝 `backends/`，见 DESIGN-multi-backend）。
- **安装副本同步**：`install.sh` 已改为全量拷贝（SKILL.md + evals/ + engine/ + references/ + experience/），`--force` 覆盖重装，见仓库根 `install.sh`。

## 八、验证记录

- 纯逻辑对拍（`parse_benchmark`/`_parse_signals`/`_evidence_score`/`params_to_args`/`within_domain`/`kg_correct`）：13 断言通过，与 vllm_ascend_opt 同输入同输出。
- 端到端冒烟（2026-08-13）：六阶段前四阶段（capture→analyze→recommend→apply）在真实 vllm 服务上跑通，状态持久化正确；verify/decide 因共享机 root unittest 进程占满 182 核 CPU 导致 graph capture 卡死而未走完（环境负载，非引擎缺陷）。
- 决策层（2026-08-13）：`critic` 门槛 + `orchestrate` 跨轮次决策（best 记账 / 连续无改善 stop / stop_exhausted / stop_unhealthy）纯逻辑验证通过；`run.py` 主循环 stop_* 断点 + best 记账迁移 + pending→resume 回归均通过。
