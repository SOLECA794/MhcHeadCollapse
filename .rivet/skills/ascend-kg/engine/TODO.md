# 引擎遗留待办（P2，暂不修）

> 记录 2026-08-14 健壮性审查确认的 P2 级缺口。P0/P1 已修（subprocess 超时降级，见 CHANGES.md）。
> 这些是防御性 / 工具缺口：当前不触发、或仅影响运维工具的可用性，不阻塞闭环，故只记不修。

## 1. `decide.py` 的 `metrics` 无 None 防御
- 位置：`engine/stages/decide.py` `cur = metrics["ttft_ms"]`
- 触发条件：仅当 verify 不再遵守「压测失败 → 降级 10× 基线」契约时才 KeyError；当前不会。
- 建议修法：`metrics` 判空走 `unhealthy`，或兜底 `baseline * 10`。

## 2. ~~`recover_service.sh` 硬编码端口 8000~~（已修复 2026-08-14）
- 已改为读 `run_state.json` 的 `state["port"]`（缺失回退 8000）；同时补现场重渲染 defaults 基线兜底、`kill_by_port` 返回值检查、`setsid`（见报告 §三四连弱点其余三项）。

## 3. resume snapshot 字段窄
- 位置：`engine/run.py` 的 `state["snapshot"]`（只存 `out_dir/validation/signals/cand`）
- 触发条件：仅当未来新增「apply 之后」的暂停点时才不一致；当前唯一暂停点 `diagnose` 在 apply 前，安全。
- 建议修法：新增第二个暂停点时，snapshot 补充 `current_params` / `current_script`。
