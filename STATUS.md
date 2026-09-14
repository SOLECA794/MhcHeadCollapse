# STATUS — 滚动状态板

> **每会话结束必更新本文件**（5 分钟）。这是新会话判断"现在在干什么"的第一入口。
> 更新纪律：完成的划掉并留一行结果；进行中的写明卡在哪；新增的追加。
> 格式：`[状态] 事项 —— 关键事实/下一步动作`
> 状态枚举：`▶ 进行中` `○ 待办` `✅ 完成` `⛔ 阻塞`

**最后更新**：2026-09-14（会话：环境迁移 + 8.5 双环境 + 观测基础设施建成）

---

## 当前主线：本地 vs OJ 对拍，测出真实 case shape

用户核心目标（换到本 NPU 环境的原因）：通过本地实测与 OJ 数据对拍，
反推 5 个 case 的真实 (n, nH, outer)。Case3/4/5 的 outer 是唯一未探维度。

### ▶ 本地迭代循环已打通（本会话完成，可开工）

- 8.5 双环境建成：`/tmp/cann85/cann-8.5.0/`（OJ 同版本），9.0 在 `/opt/conda/Ascend/cann-9.0.0/`
- v117 在 8.5 下编译链全通 + 正确性 7/8 PASS（详见 `.rivet/ENVIRONMENT.md §八`）
- 361001 根因已解：**必须 export ASCEND_HOME_PATH**（conda 环境无 /etc/ascend_install.info）
- **h=1024 UB 溢出已修复（v119-fix）**：weight 流式读 + 192KB 预算守卫，16/16 PASS 零回归
  （F3.6；真实根因是 nH>4096 超 910B 单核 UB 192KB 上限，非 nH=1024）
- 观测基础设施：`docs/METHODOLOGY.md` + `scripts/msprof_report.py`（已实测）
- OJ 提交工具入仓：`cannjudge-submit-plaintext/`（含凭据，链路待首次实测）

### ○ 下一步（按优先级）

1. ~~修 h=1024 MTE 越界 bug~~ ✅ 已修（v119-fix，F3.6，16/16 PASS）
2. ~~建 ACL-event 计时 harness~~ ✅ 已建成（F3.7：mhc_shape_bench 双口径，17 shape 矩阵，**outer=4 发现单核瓶颈**）
3. **shape 对拍分析**：矩阵数据已有（shape_scan_v119_8.5.tsv），待 NOP 标定锁口径系数后拟合 OJ 逐 case 时间
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

- ★ **观测基础设施已建成并实测**（`docs/METHODOLOGY.md` + `scripts/msprof_report.py`）：
  msprof 显微镜一次采集即产出瓶颈分项（实测 v117 8/512：scalar 34.4% ⚠️ / mte2 34.0% / vec 27.1%，
  与 FACTS 历史结论互印证）。**今后每次优化先跑显微镜再动手**
- 参考实现库：`.rivet/scratch/cann-ops-competitions/`（官方赛事高分开源，165M）
  - ★ Tangefly GeluV2：sigmoid `x/(1+exp(-inner))` 除法形式 + Mins 钳制（修 Group 精度 bug 首选参考）
  - ★ dhltat submissions：GoogleTest + tiling_context_faker UT 框架（对拍 harness 模板）
- ascend-kg 已接入（`.rivet/skills/ascend-kg/` + key 在 ~/.bashrc）
- 8.5 安装包留存：`/tmp/cann85-pkg/`（1.1GB；/tmp 可能被清，重置后从 OBS 重下，URL 见 ENVIRONMENT.md）
- `建议分析.md`：三短板破局方案（本文档体系的方法论源头，可信性已验证——见 METHODOLOGY.md 验证记录）
