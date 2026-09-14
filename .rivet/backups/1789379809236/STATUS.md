# STATUS — 滚动状态板

> **每会话结束必更新本文件**（5 分钟）。这是新会话判断"现在在干什么"的第一入口。
> 更新纪律：完成的划掉并留一行结果；进行中的写明卡在哪；新增的追加。
> 格式：`[状态] 事项 —— 关键事实/下一步动作`
> 状态枚举：`▶ 进行中` `○ 待办` `✅ 完成` `⛔ 阻塞`

**最后更新**：2026-09-14（会话：环境迁移 + 8.5 双环境建成）

---

## 当前主线：本地 vs OJ 对拍，测出真实 case shape

用户核心目标（换到本 NPU 环境的原因）：通过本地实测与 OJ 数据对拍，
反推 5 个 case 的真实 (n, nH, outer)。Case3/4/5 的 outer 是唯一未探维度。

### ▶ 本地迭代循环已打通（本会话完成，可开工）

- 8.5 双环境建成：`/tmp/cann85/cann-8.5.0/`（OJ 同版本），9.0 在 `/opt/conda/Ascend/cann-9.0.0/`
- v117 在 8.5 下编译链全通 + 正确性 7/8 PASS（详见 `.rivet/ENVIRONMENT.md §八`）
- 361001 根因已解：**必须 export ASCEND_HOME_PATH**（conda 环境无 /etc/ascend_install.info）

### ○ 下一步（按优先级）

1. **修 h=1024 MTE 越界 bug**（507035，详见下）——Case5 对应 shape，对拍前置
2. **建 ACL-event 计时 harness**：对齐 OJ 端到端口径（FACTS F2.1g），复用 `tests/mhc_bench*.cpp` 改造
3. **shape 扫描对拍**：本地 (n,nH,outer) 计时矩阵 vs OJ 逐 case 时间
4. **NOP 标定五连提交**：搞清 OJ 计时口径（方法论见 `docs/METHODOLOGY.md` §三）
5. Group sigmoid 精度 bug：先换硬件 Exp 原语 A/B（差分对照法，`docs/METHODOLOGY.md` §二）

### ⛔ 已知 bug：v117 在 nH=1024 稳定崩溃

n=8/nH=1024/任意 outer/fp16 → 507035 vector core exception，
设备侧报 **"The write address of the MTE instruction is out of range"**（MTE 写越界）。
8.5/9.0 双版本复现，非环境问题。历史 verify.sh 从未测过 h=1024，故潜伏至今。
**OJ Case5 正是 n=8/nH=1024**——此 bug 与 Case5 性能问题可能同源。排查入口：
kernel 的 tiling/workspace 尺寸计算 + `plog` 报错在 `~/ascend/log/debug/plog/`。

---

## 背景速览（详见 .rivet/HANDOFF.md）

- 赛题：星辰杯 C 组 MhcHeadCollapse，计分 `100/(1+log₁.₅(t/T))` × 5 case 均值
- 在榜：v118（5/5 Pass），历史最优提交保分（非最新覆盖）
- 优化重心：Case1 缺口最大（2.70×），Case5 双峰次之（mte2 驱动）
- 未解卡点：OJ 计时口径未复刻；Group 路径向量 sigmoid 精度 bug（3.203e-03）

## 资产清单（本会话新增）

- 参考实现库：`.rivet/scratch/cann-ops-competitions/`（官方赛事高分开源，165M）
  - ★ Tangefly GeluV2：sigmoid `x/(1+exp(-inner))` 除法形式 + Mins 钳制（修 Group 精度 bug 首选参考）
  - ★ dhltat submissions：GoogleTest + tiling_context_faker UT 框架（对拍 harness 模板）
- ascend-kg 已接入（`.rivet/skills/ascend-kg/` + key 在 ~/.bashrc）
- 8.5 安装包留存：`/tmp/cann85-pkg/`（1.1GB；/tmp 可能被清，重置后从 OBS 重下，URL 见 ENVIRONMENT.md）
