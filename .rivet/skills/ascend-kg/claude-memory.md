<!-- ascend-kg:START -->
## ascend-kg 准则（昇腾知识图谱）

任何涉及昇腾/鲲鹏生态的任务（Ascend/CANN/AscendC/NPU/Atlas/MindSpeed/MindIE/算子/Tiling/模型迁移/性能调优/错误码/部署运维、鲲鹏/Kunpeng/TaiShan/毕昇/BOOM/BoostKit/ARM 服务器等），**先查 KG 或加载对应 Skill，后行动**——不凭记忆、不靠猜测（你的训练数据可能已过时；KG 是实时权威数据源，含跨仓库关联、配置映射、错误码→源码映射）。

- **会话一开始就判断**这是不是昇腾/鲲鹏任务；是 → 立即调用 ascend-kg skill（Skill tool，`skill: ascend-kg`；或用户用 `/ascend-kg`）走 `/search` 检索或加载对应 Skill，拿到结果再回答/写代码。
- **复杂昇腾/鲲鹏子任务**优先分派给 `ascend-kg-worker` subagent（Agent tool，`subagent_type: ascend-kg-worker`；它自带本准则，上下文独立、抗压缩，并发≤5）。
- **推理服务性能调优**是执行型任务且命中 flow → 走编排引擎唯一入口 `python <skill>/engine/agent_loop.py --flow vllm-serve-optimize`（LLM 会话委托双循环，决策经暂停点回传主会话，不依赖 `ANTHROPIC_*` 环境变量，见 SKILL.md §8.5；run.py 确定性流水线已退役归档 `engine/attic/`）；不手动一步步调。引擎 pending（退出码 2）非失败——读 `<skill>/engine/state/agent_task.json` 回传结果（task_id 同 task）再 `--resume`。
- 凭记忆回答昇腾/鲲鹏 API 参数/配置默认值/版本兼容性/错误码 = 禁止；不确定该不该查 KG 时，就查。
<!-- ascend-kg:END -->
