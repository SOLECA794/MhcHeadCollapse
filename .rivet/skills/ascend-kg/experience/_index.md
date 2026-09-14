# ascend-kg 实战经验库 · 索引

> 使用：进入算子开发 / 推理部署任务后，按下方触发条件取对应文件。**先加载对应 KG Skill（SKILL.md §1.0 路由表），本库是 KG 的本地补充**——卡壳时读这里，多数坑本库已有答案。
> 按域组织：`operator/`（算子开发）、`inference/`（推理部署）、`cases/`（任务案例）。

## 触发条件 → 对应文件

| 触发条件（遇到以下情况） | 读 | 关键内容 |
|---|---|---|
| 开始新算子开发任务 / 阶段切换 | `operator/workflow-practices.md` | 阶段×Skill 路由表、目标驱动循环、最小模板、双通路、复盘节奏、复用已验证实现 |
| 编译 / 链接 / API 报错，同一报错 >2 次 | `operator/api-pitfalls.md` | 27 条避坑（签名 / dtype / 构建），🔥 高频优先 |
| 精度不对齐（数值错、bit-exact 失败） | `operator/precision-perf-practices.md` §A | FP32 中间计算、Cast 边界、golden 方法学、容差阈值 |
| 性能不达标（慢、未达倍数目标） | `operator/precision-perf-practices.md` §B | msprof 驱动、多核 tiling、双/三缓冲、测量口径 |
| 推理部署（vLLM-Ascend / 大模型服务启动 / NPU 部署起停） | `inference/deploy-practices.md` | 部署流程、环境检查清单、10 坑（torch ABI / libatb / glibc / 507033）、服务启停与资源释放 |
| 推理性能调优 | 先加载 KG `vllm-ascend-performance-optimization` | 性能 flag 走 KG 标准答案；现场坑看 `inference/deploy-practices.md` |
| 需要"某任务当时怎么解决的" | `cases/_index.md` | 算子 11 任务 + 推理部署 2 任务速览 |

## 最高频 10 坑速查（🔥 详情见 operator/api-pitfalls.md）

1. **CAST_TRUNC vs CAST_RINT** — int 取整用 TRUNC（对齐 numpy/TBE），half/bf16 用 RINT；KG"推荐"≠内置行为
2. **bf16 向量 Cast 元素未初始化** — 污染向量管线，输出错乱
3. **软链致 kernel .o 缺失** — 561103，必须真实目录 `cp -r`
4. **IsAiCoreSupport 运行期关卡** — 561103 全 dtype 失败先查它
5. **A2 float→int8 无直转** — 两步经 half（CAST_RINT→CAST_TRUNC）
6. **snake_case 命名** — 目录 / 核入口小写，op type CamelCase
7. **Ands/Select 无 int32** — 用 IntSelect mask trick
8. **Subs/ReduceSum 签名** — 设计稿意图 API 实施前必查 KG 落定
9. **aclnn_exclude** — 手写 op_api 防 autogen 符号冲突（multiple definition）
10. **--experimental + 全量重编** — ops-math 编译 experimental 算子；改 header 必 `rm -rf build_out`

## 数据来源

- 11 项目 `docs/任务经验复盘.md`（`~/projects/*-task-one/*-task-two`，truncatediv 在 `truncate_deliverable/docs/`）
- 聚合文档 `/home/z60124773/projects/算子开发任务经验总结.md`（36KB，覆盖其中 8 个任务）
- 会话历史 `~/.claude/projects/`（347MB，逐坑还原的现场记录）→ 算子开发：`operator/api-pitfalls` / `operator/precision-perf-practices.md`；推理部署：`inference/deploy-practices`
