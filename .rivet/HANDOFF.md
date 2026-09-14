# HANDOFF — MhcHeadCollapse 算子优化（2026-09-14 交接）

> 读者：无上下文的新会话（agent/人类）。本文自包含，所有事实落字。
> 赛题：中国电信星辰杯 C 组 MhcHeadCollapse（mHC四路归一）算子性能赛，OJ 平台 cannjudge.cn
> 工作区：`/workspace/notebook13/MhcHeadCollapse/`（本 NPU 机器本地，非远端 notebook）
> NPU 设备：`/dev/davinci2`，CANN 驱动：`/usr/local/Ascend/driver/`
> 代码：`./code/`（`op_host/` + `op_kernel/`）

## 任务目标 — 我们在做什么

用户原话级目标："探测出真实形状 + 能够本地复现出 OJ 计时规则 + 本地环境写代码、迭代优化、编译验证能够等同于 OJ 提交的效果（降低对 OJ 的依赖性）"，终极目标是算子优化拿高分。

非目标：Metax 六课课程线已挂起（见 `D:\Desktop\OP-Learning\Metax\Transformer-on-Metax\课程交接文档.md`，勿动）；CATLASS 库已裁决不可引入（编译期硬门，详见 $MHC/FACTS.md）。

用户已授权"免死金牌"：**一天最多 50 次 OJ 任意提交无顾虑**（2026-09-13 起），超限可换账号。这解除了"失败提交覆盖在榜分"的保守约束。

## 已完成 — 带证据

### 1. OJ 计分公式已实测验证（本会话最关键成果）

公式 `得分 = 100/(1+log₁.₅(t/T))`，T=该 case 全榜历史最小时间，t=当前提交时间，**总分=5 个 case 得分均值**。

验证方法：拉全榜 76 行（`https://cannjudge.cn/api/problems/6a7c23a6a52e0f540a8a1779/ranking?page=N&size=20`），取全榜逐 case 最小时间 [2.12,2.12,2.2,2.48,3.88] 作 T，kkio- 当前 times [4.70,5.84,5.10,4.94,8.64]，算得均值 33.10 vs 榜上实际 33.05（差 0.05，浮点舍入级）。**这同时证明计分就是这 5 个 case**——$MHC/CANNJudge计时与得分规则.md 中"15 个测试点"的说法与实测矛盾，需要修正该文档（比赛附件写 15，但实测 5 点均值精确等于总分，若有 10 个隐藏 case 不可能这么吻合）。

榜单状态：kkio- 33.05 分第 20 名（曾 33.82，被他人刷新 T 后自然下降）。榜首 wangyue ~80.97。

### 2. 三大文档已建成（$MHC/ 下，均以提交）

- `实测结果汇总.md`（145 行）：全部实测数据——OJ 轨迹/msprof 分项/探针阶梯/热身曲线/正确性误差表/带宽口径/硬件一致性。commit 3411584
- `优化方法与事实结论.md`（117 行）：三级探针阶梯法等发明方法、OJ 提交三律、F1-F10 事实结论。同 commit
- `Ascend910B硬件性能模型与延迟参考.md` 第十一章（实测补充）：框架 scalar 地板 ~4μs、DMA 启动 1.7μs、单核带宽 109GB/s 等。commit 391e785
- 跨会话记忆已写 4 条（probe 阶梯法/OJ 探测闭环/实测事实），新会话用 memory recall "MhcHeadCollapse" 可命中

### 3. OJ shape 探测器已建成（未真实运行）

`$MHC/scripts/probe_oj.py`（134 行，commit 391e785）：`build <case> <field> <value>` 注入断言到 op_host InferShape 并提交；`poll <sid>` 轮询；`sweep <field> <case> <v1,v2,...>` 批量。注入点已对齐实际 InferShape（`versions/v118_path2vecz/code/op_host/mhc_head_collapse.cpp:74` 的 `return GRAPH_SUCCESS` 前，outer 用循环乘积非 GetDim(1)），干跑验证注入产物正确。**尚未做真实提交**——首次运行可能暴露提交/轮询解析的小问题。

已知 shape 现状（★ F3.9 实测钉死 2026-09-14）：Case1/2 n=4/nH=16/outer=2；Case3/4 n=8/nH=512/outer=8；Case5 n=8/nH=1024/outer=32。探测手段：TilingFunc 断言（InferShape 无效，F3.9 记录）。

### 4. 当前在榜代码

`$MHC/versions/v118_path2vecz/`（commit 4a5188e）：Group 路径= v116 标量 sigmoid（回退保正确），Row 路径=向量多项式 sigmoid（已验证位级等价），TinyH4= v117 向量化。本地 9/9 PASS 误差与 v116 基线逐位一致。OJ 5/5 Pass。

### 5. GitCode Notebook 链路（旧，可能已被 SSH 替代）

`Ascend/ascend-npu-console/scripts/npu.py`（CDP→WS→远端执行），环境快照见 `docs/session-handoff.md`（910B4,/dev/davinci6,CANN 8.5.0,单核CPU）。**上会话末 notebook 被重置过一次**，远端工作区 /opt/atomgit/mhc 已重建并验证。

## 当前卡点 — 卡在哪、已排除什么

### 卡点 1：本地 NPU 环境已就绪，SSH 远端链路不再是核心依赖（本会话已切换到本地 NPU 环境）

我们已直接在 `/dev/davinci2` 机器上工作，远端 SSH 跳板（`113.47.8.48:2234` → `root@199.98.55.200`）不再需要用于日常迭代。SSH 远端链路仍可用于：OJ 提交自动化、GitCode push 等场景，但当前最高优先改为本地编译验证循环。

**已排除**：无 NPU 环境（否，`/dev/davinci2` 存在）；CANN 工具链缺失（`npu-smi` CLI 在 `/usr/local/bin/npu-smi`，驱动在位）。

### 卡点 2：OJ 计时口径未完全确定

已排除：per-call 同步、本地 streamed、每次 H2D（三者都与 OJ 数字矛盾，见 $MHC/FACTS.md F2.1k）。已知：OJ 时间与本地 kernel 时间量级吻合（v031 kernel 4.18μs vs OJ 4.48μs），文档推测是"ACL event 端到端打点"（$MHC/CANNJudge计时与得分规则.md §4.1），**但未在 NPU 上实测复刻**——这正是"本地复现 OJ 计时"目标的核心待办。

### 卡点 3：v118 Group 路径向量化精度 bug（根因未解，已回退保榜）

Group 路径（outer≥4 走的分支）向量 sigmoid 版误差恒定 3.203e-03（基线 2.331e-04）。四轮修复全证伪：多项式 Horner 修正（数学孤立验证 4e-11 无辜）/pat block-0/parity scratch/V_V barrier——误差一字不变。dump 对账显示所有行 fold 输出均有 ~1-3e-3 偏差（非某行全错）。下一步手段：UB 级 dump（red 原始 ReduceSum 值）对账 phase2 输出 vs 向量链输入。详见 $MHC/FACTS.md F2.1t 与 commit 4a5188e 的诊断史。

## 下一步 — 按优先级

1. **本地编译验证循环（最高优先）**：我们在 NPU 本地环境，先跑通 `cd code && mkdir -p build && cd build && cmake .. && make -j` 验证 CANN 工具链正常。这是后续所有优化的基础。
2. **本地 kernel 时间打点**：用 ACL event（`aclrtCreateEvent/RecordEvent/SynchronizeEvent`）在 NPU 上对已知 shape 逐个打点，建立本地性能基线。
3. **OJ shape 探测首跑**：在确认本地环境可用后，跑 `python scripts/probe_oj.py sweep outer 3 1,2,4,8,16` 补完 Case3/4/5 的 outer 维度。

## 坑 — 绝对不要再踩

1. **排行榜只保留最新提交**——Compile Error 也覆盖在榜分数（曾因此榜上挂 0 分）；现在有 50 次/日免死金牌可放开，但每次提交后仍必须轮询到终态并实时拉排行榜核验分数。
2. **远端增量编译掩盖错误**——redefinition 级错误远端 BUILD_OK 平台报错；远端验证必须 `rm -rf build` 全清重编。
3. **remote push 工具对中文路径过敏**——`Ascend/ascend-npu-console/scripts/npu.py push` 遇中文目录会 tar 错路径；先拷到 `D:\Desktop\OP-Learning\.rivet\scratch\` 无中文无空格的 staging 再推。
4. **远端命令必须单行**——heredoc/多层引号会卡死远端 shell；复杂逻辑走 push 文件。
5. **`$HOME` 在本地 Git Bash 被提前展开**——远端命令用绝对路径。
6. **worker 产出的实验数据可能编造**——星流模式的 team worker 曾虚构 1220 次采样数据（远端无痕迹）；进 FACTS 的数字必须主会话手跑或核验远端产物。
7. **msprof 单次采样是热身曲线切片**——NPU 频率 800→1650MHz 热身约 25 次调用（2~3 倍时间差）；对比实验必须多采样（≥8 次）取稳态/中位。
8. **本地 streamed 口径含 ~4.4μs 主机底噪**（单核 CPU）——本地绝对值不可预测 OJ 分数，只做 A/B 相对比较。
9. **GitCode Notebook 随时可能重置**——远端 /opt/atomgit 工作区非持久，重要产物即拉回；重置后按 `Ascend/ascend-npu-console/docs/session-handoff.md` 恢复。
10. **CANNJudge 文档的时间单位标注 ms 实为 μs**——所有平台时间数值按 μs 理解。
