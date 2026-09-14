# 案例库：11 个算子开发任务速览

> 完整复盘在各项目 `docs/任务经验复盘.md`（truncatediv 根 docs/ 缺失，在 `truncate_deliverable/docs/`）。
> 此处每任务 1-2 行精华 + 链接，**不重复拷贝复盘正文**。

| 任务 | 项目路径 | 复盘文档 | 精华（1-2 行） |
|---|---|---|---|
| arange | `~/projects/arange-task-one` | `docs/任务经验复盘.md` | f32_large 7.06µs = MTE3 写带宽上界；CAST_TRUNC 对齐 numpy；bf16 向量 Cast 管线污染修复（Duplicate 填满再 Cast） |
| bincount | `~/projects/bincount-task-two` | `docs/任务经验复盘.md` | A2 原算子走 AICPU 回退 → AICore 向量化 10X（5/5 场景 10.06X~47.9X）；直方图按桶遍历+向量比较而非散射 |
| gcd | `~/projects/gcd-task-one` | `docs/任务经验复盘.md` | 910B 无 SIMT，性能不可追平 TBE（~1.83×）四条优化全证伪回滚；Euclid + Stein golden 51 用例交叉验证 |
| inplacersqrt | `~/projects/inplacersqrt-task-one` | `docs/任务经验复盘.md` | 双→三 buffer 反超 TBE ~6%（13.3µs vs 14.14µs）；UB 系数数学推导 5→11（模板 latent bug）；inplace = aclnn 绑定 y=selfRef |
| inplacesigmoid | `~/projects/inplacesigmoid-task-one` | `docs/任务经验复盘.md` | Reciprocal 仅 INTRINSIC（~2⁻¹⁰）致 fp32 FAIL → fp32 保留 Div 其余 Reciprocal，tileDataNum +50~100% 全达标；int8→fp32 直转不存在 |
| layernorm | `~/projects/layernorm-task-two` | `docs/任务经验复盘.md` | AR-FullLoad TwoPass；大 M 1.19~1.31× 快于原生、小 M 0.48×（双归约开销）；Subs/ReduceSum 设计稿意图 API 实施前查 KG |
| logspace | `~/projects/logspace-task-one` | `docs/任务经验复盘.md` | 561103 全 dtype 失败 = IsAiCoreSupport 运行期关卡（非 dlopen 链路）；A2 float→int8 两步经 half |
| rightshift | `~/projects/rightshift-task-one` | `docs/任务经验复盘.md` | Ands/Select 无 int32 → IntSelect mask trick；软链致 561103（真实目录 cp -r）；unsigned 逻辑修正掩码；≥70X（139ms vs AICPU >10s） |
| selugrad | `~/projects/selugrad-task-one` | `docs/任务经验复盘.md` | CAST_TRUNC 对齐 TBE（INT ±1 → 0 differ）；内置算子当 golden；bf16 必升 f32；64M 全核 FP32 175%/INT8 211% |
| truncatemod | `~/projects/truncatemod-task-one` | `docs/任务经验复盘.md` | 先读后写：已合入版不支持广播不默认达标；广播在 op_api 层 BroadcastTo；aclnn_exclude 防 autogen 冲突；`--experimental` 才编译 experimental/ |
| truncatediv | `~/projects/truncatediv-task-one` | `truncate_deliverable/docs/self_verification_report.md`（根 docs/ 缺失） | 全新独立实现替换贡献者版；10 组 dtype 含混合浮点广播；全链路 aclnn（Contiguous+BroadcastTo 物化+ViewCopy）；kernel UT 7/7 + tiling UT 7/7 + ST 过，性能 169%~356%（torch_npu 基线） |

## 推理部署任务速览（vLLM-Ascend）

> 项目 runbook 已随项目目录删除，原始记录在会话 JSONL。精华见 `inference/deploy-practices.md`，此处不重复。

| 任务 | 原始会话 | 精华（1-2 行） |
|---|---|---|
| Qwen2.5-7B · vLLM-Ascend | `~/.claude/projects/-home-z60124773-projects-ascend-kg-optimize-llm-deploy-vllmascend/f443e4e2-1866-4e52-bf87-d19a36d40aea.jsonl` | CANN 归属红线（用户自有优先）；libatb.so 缺失→NNAL 组件；glibc 2.38 vs 2.35；方法论"环境就绪≠推理可用 / 抽象成思想" |
| Qwen3-0.6B · vLLM-Ascend | `~/.claude/projects/-home-z60124773-projects-qwen-deploy/ef979bab-edab-419f-a09b-bd4c008b39ad.jsonl` + `91f3b660-62e0-48c0-adbf-2f7233391e73.jsonl` | torch ABI 不匹配→`std::bad_alloc`（wheel `Requires-Dist` 钉死）；setuptools<81 / fastapi<0.124；507033 释放竞态 |

## 跨任务共性提炼（8+ 任务验证）

1. **msprof 实测驱动**：全部任务用 msprof 定位瓶颈，"凭直觉优化"几乎必错。
2. **KG 铁律 + subagent 隔离**：依赖 KG/头文件核实 API，用 ascend-kg-worker 隔离长上下文。
3. **dtype 支持矩阵以头文件 static_assert 为准**（KG 噪声多）。
4. **整型舍入对齐**：CAST_TRUNC + unsigned 逻辑修正掩码。
5. **构建坑高度相似**：软链 561103 / `--make_clean` 只 clean / 增量不重编 header / stable 符号冲突 —— 共性 = 全量 `rm -rf build_out` + `--experimental` + 真实目录拷贝。
6. **性能公平性**：超 L2 大 shape、L1 非 L2 profiling、>10X 用端到端计时。
7. **共享 NPU**：选 HBM 最低卡 + 重试 + 交叉验证。
8. **诚实纠偏**：harness 假象、bf16 管线污染、性能未达标都如实记录。
9. **工具链适配是隐藏成本**：AscendOpTest 工具层 gap 需阶段 0 跑通，patch 是环境适配非交付件。
10. **架构代差提前定性**：bincount（A2 无 AICore kernel）、gcd（910B 无 SIMT）—— 必须 KG 架构文档定性，不能凭 header/编译结果推断。
