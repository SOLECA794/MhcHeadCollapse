# ascend-kg 实战经验库（experience）

> 本机 **11 个真实 AscendC 算子开发任务**（`~/projects/*-task-one/*-task-two`）+ **2 个推理部署任务**（Qwen2.5-7B / Qwen3-0.6B，vLLM-Ascend）的实战经验沉淀。
> 定位：**本机 harness / 工程特定经验**，与 KG 通用知识（`ascendc-api-best-practices`、`ascendc-operator-dev`、`vllm-ascend-deploy` 等）**互补、不重复**。
> 通过 ascend-kg SKILL.md **§1.0.3 最小指针**钩入：按触发条件按需加载，不占主文件体积。

## 为什么单独建库（而非写进 SKILL.md）

SKILL.md 是**程序性指令**（检索流程 / 触发条件 / 路由），本库是**声明性知识 + 案例**（避坑事实 / 方法论 / 任务复盘）。按业界 skill 设计规范（渐进披露 + 程序性/声明性分离 + 案例库模式），主文件只放触发钩子，细节外置按需读——避免全量进上下文稀释（长任务里"先查 KG"的约束就是这样被冲掉的）。

## 目录结构

| 文件 | 类型 | 何时读 |
|---|---|---|
| `_index.md` | 主入口 | 任何算子开发 / 推理部署任务开始 / 卡壳时的第一站 |
| `operator/workflow-practices.md` | 程序性 | 开始新算子开发任务 / 阶段切换时 |
| `operator/api-pitfalls.md` | 声明性 | 编译 / 链接 / API 报错，同一报错 >2 次时 |
| `operator/precision-perf-practices.md` | 声明性 | 精度不对齐 / 性能不达标时 |
| `inference/deploy-practices.md` | 声明性 | 推理部署（vLLM-Ascend）环境检查 / 启动失败 / 服务起停时 |
| `cases/_index.md` | 案例 | 需要"某任务当时怎么解决的"完整上下文时 |

## 维护规范（新增一条经验）

1. **记录**：任务复盘时把"触发条件→症状→根因→修复→教训"写进项目 `docs/任务经验复盘.md`。
2. **提炼**：从复盘抽出**可复用规则**（进 `operator/api-pitfalls.md` / `operator/precision-perf-practices.md`）与**一次性的任务结论**（进 `cases/_index.md`）。
3. **入档**：规则遵循五段式 `触发→症状→根因→修复→教训`；案例遵循 cases 表格格式。
4. **更新索引**：`_index.md` 的触发条件表要能命中新条目。
5. **指针不动**：SKILL.md §1.0.3 是稳定指针，新增内容只动本库文件，不改 SKILL.md。
