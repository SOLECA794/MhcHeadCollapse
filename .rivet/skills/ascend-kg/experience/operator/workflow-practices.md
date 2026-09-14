# 算子开发工作流实战经验（程序性）

> 治根：解决"忽略 KG 适配 Skill、凭记忆重复试错"。**核心铁律：先查 KG / 头文件 / 本机源码，后行动，不凭记忆不靠猜测。**

## 1. 阶段 × Skill 路由表

算子开发任务**按阶段**加载对应 KG Skill，不要从头到尾一个流程走到底。每个阶段切换时先问"本阶段该加载哪个 Skill？"

| 阶段 | 加载 Skill（node_id） | 说明 |
|---|---|---|
| 需求理解 | 先 `/search` 锚定同名算子 + golden 语义 | TBE 源码 + aclnn 头文件 + op-info JSON **三源核实**；任务书路径可能版本漂移 |
| 开发 / 核函数 / Tiling | `ascendc-operator-dev` + `ascendc-tiling-design` | **设计阶段就查头文件 dtype 矩阵**（static_assert），不等编译期 |
| 写 kernel 前 | 同仓 grep 已验证同构参考 | 复用其 tiling 模板 / cast 分支框架 |
| 测试 UT / ST | `ascendc-ut-develop` | CPU 仿真 UT + 真机 ST 双层 |
| 精度不对齐 | `ascendc-crash-debug`（现象定位）+ `ascendc-operator-precision-eval`（精度测试集） | FP32 中间计算 / Cast 边界 / 多 dtype 交叉验证 |
| 性能调优 | `ascend-profiling` / `ops-profiling` | msprof 实测驱动 |
| 代码审查 | `ascendc-code-review` | 交付前过一遍 |

> ⚠️ 实测缺口（truncatediv 复盘）：**测试 / 性能 / 审查阶段未加载对应 Skill 是最常见缺口**，导致 3 个 build 坑各浪费 1-2 轮完整构建。把"构建前过 API 黑名单（api-pitfalls.md）"设为强制检查点。

## 2. 目标驱动循环（TDD 思想）

**先跑现状 → 诊断 → 查证（KG/头文件）→ 改 → 重跑**，不一刀切大改。例：CAST_RINT 修复（selugrad）——先跑发现 INT ±1 差异 → 诊断舍入模式 → 查 KG 确认 RoundMode → 改 → 重跑全过。

## 3. 最小模板先编译

先复制**已验证骨架**跑通端到端（`__global__` 入口、tiling、`DataCopyPad` 尾块、aclnn 直调 host main），**只改 compute 主体，先跑通再优化**。别从零即兴发挥。

## 4. 双通路复用核函数

同一 kernel 核函数复用于两条通路：**direct-invoke**（单文件 `.asc` 直调 harness）+ **注册算子**（op_api/aclnn）。先直调验证算法，再包 aclnn 交付；PyTorch/numpy 作 golden 对照。直调 harness 的 tiling 必须与注册版**逐字一致**，否则测的不是算子。

## 5. 构建 / 验证节奏

- 改 header 必 `rm -rf build_out` **全量重编**，不依赖 `--make_clean`（只 clean 不 build）
- 本地验证 `--experimental`；kernel UT(CPU sim) 比 binary build 更可靠
- 再编译 ST；最后 perf 对比 torch_npu 基线

## 6. 八阶段编排 + 检查点门控

Phase0 环境 → 1 设计 → 2 用例 → 3 编码 → 4 精度 → 5 性能 → 6 文档 → 7 自验证，**每级通过才进下一级**。关键转折问用户（认领冲突、交付分工由用户定，不默默执行）。

## 7. 复盘沉淀

- 每个重要步骤后建检查点（已完成 / 已验证 / 剩余）
- 踩过的坑写 memory + 项目 `docs/任务经验复盘.md`
- 长任务派 `ascend-kg-worker` subagent 隔离上下文（并发 ≤5），主会话只收密集结论

## 8. 复用要点速查

- 复用开源 aclnn 源码要**逐行核对架构相关分支**（bincount A2 全走 AICPU 的坑）
- 不盲信继承模板的 UB 系数——用 `总UB=Σ buffer` 数学推导（fp16 模板 coef=5 不够，修正 11）
- 性能对齐任务第一步读官方源码列"性能来源清单"，识别架构代差（gcd 910B 无 SIMT 提前止损）
- 多 dtype 算子**固定内部计算 dtype + op_api 层 Cast** 适配，kernel 不背所有 dtype 分支

## 9. 复用已验证实现（SkillOpt 训练沉淀，全流程 5 阶段通用）

> 来源：SkillOpt 对 ascend-kg 做算子开发**全流程**训练（设计/开发/编译集成/功能验证/精度/性能）时，被 gate 接受的唯一 patch。在该 harness 上验证 selection 0.80→1.0、held-out test 0.5→1.0。

- **同任务（或极相似任务）在本机已被验证通过**（编译成功、NPU 实跑、maxdiff 0.00e+00）→ 优先复用该已知可用实现作地基，再按新任务语义差异做**最小改动**——比从零推导更快更稳，天然对齐已通过的 I/O 契约与精度基线。
- **直接调用场景尤其如此**：优先改造 bit-exact 验证过的单 `.asc` 模板（含 `__global__` kernel + host main），而非从零写。这会同时拿到正确的工程契约（I/O 路径、main() 签名、CMake）与精度基线——一条纪律同时覆盖精度+性能，不必分开学。
- **同算子的优化**也基于先前 verified 范式迭代，不另起炉灶。
- 与 §3「最小模板先编译」的区别：§3 强调"先复制骨架跑通再优化"，本节强调"复用**本机已实测通过**的具体实现（而非任意骨架）"，并把复用扩展到性能优化与后续迭代。
