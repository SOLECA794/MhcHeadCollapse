# MhcHeadCollapse 算子（mHC四路归一）— 工作导航

**赛题**：中国电信星辰杯高校AI算子开发挑战赛 C组中等题
**平台**：CANNJudge | Problem ID `6a7c23a6a52e0f540a8a1779`
**目标**：5 个固定 case 上取得尽可能短的运行时间（per-call 端到端计时）

> **当前工作环境**：本 NPU 机器（910B4, /dev/davinci2），CANN 8.5（OJ 同版本）+ 9.0 双环境。
> 环境事实与初始化命令见 `.rivet/ENVIRONMENT.md`（跑任何 NPU 命令前必读）。
> 当前活跃工作见 `STATUS.md`（滚动状态板，每会话结束更新）。

---

## 🚀 从这里开始

| 我想… | 读这个 |
|---|---|
| **知道现在在干什么** | [`STATUS.md`](STATUS.md) ← **滚动状态板（会话第一入口）** |
| **新会话完整上手** | [`.rivet/HANDOFF.md`](.rivet/HANDOFF.md) ← 唯一权威交接文档 |
| **配环境跑 NPU** | [`.rivet/ENVIRONMENT.md`](.rivet/ENVIRONMENT.md) ← CANN 8.5/9.0 双环境 |
| **了解什么是真的**（避免被过时结论误导） | [`FACTS.md`](FACTS.md) ← **唯一权威事实源** |
| **知道哪些坑已经踩过** | [`PITFALLS.md`](PITFALLS.md) |
| **知道下一步做什么** | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| **读历史全过程** | [`迭代优化记录.md`](迭代优化记录.md) ⚠ 见下方警告 |

> 根目录的 `HANDOFF.md` 是指路牌（旧版已过期），真身在 `.rivet/HANDOFF.md`。

---

## ⚠ 三个必须知道的警告

### 1. 旧文档有已知错误，且错误在正文、勘误在文末

`迭代优化记录.md` 的用例表（`:75-79`）写的时间单位是 `ms`、shape 是 `n=4,h=4`。
**两处都是错的。** 正确的勘误写在同一文件 `:1211`（单位）和 `:1220`（shape）——
**在 900 行之后**。读正文不会读到勘误。

→ **以 `FACTS.md` 为准。** 该文件每个结论都带 `file:line` 证据。

### 2. 两个"新版本"其实没有代码改动

`v116_vsync_batched` 和 `v116_batched_reducesum` 与 `v111_path2_group4` **md5 完全相同**。
文档所称"v116 实机验证通过"，验证的是 v111 的代码。

→ 声明"某版本验证通过"前，先 `md5sum` 对比上一版确认有实质改动（`PITFALLS.md P13`）。

### 3. 一条核心结论建立在空转的代码上

v110/v111 宣称"批处理改造"，据此得出"同算法重排无效"。实际因 `block_dim` 公式，
Case5 的批处理**从未生效**（`FACTS.md F2.3`）。

→ 下"这个优化无效"的结论前，先验证该路径**真的被执行**（`PITFALLS.md P14`）。

---

## 📁 文件地图

```
MhcHeadCollapse/（本 NPU 机器工作区，非旧 Windows 工作区）
│
├── README.md                 ← 你在这里（导航）
├── STATUS.md                 ← ★ 滚动状态板（会话第一入口，每会话结束更新）
├── HANDOFF.md                ← 指路牌（真身 .rivet/HANDOFF.md）
│
├── FACTS.md                  ← ★ 单一事实源（改动代码前必读）
├── PITFALLS.md               ← 已踩过的坑
├── EXPERIMENTS.md            ← 待执行实验 + 方法论沉淀
│
├── 赛题.md                    ← 题目原文
├── 迭代优化记录.md             ← ⚠ 历史过程（140K，含已知错误，只追加不重写）
├── CANNJudge计时与得分规则.md   ← 平台规则
├── Ascend910B硬件性能模型与延迟参考.md ← 硬件参考
│
├── docs/
│   ├── METHODOLOGY.md         ← ★ 观测基础设施方法论（msprof 显微镜/差分对照/NOP 标定，已实测）
│   ├── OP_TEAM_DESIGN.md     ← 团队协作设计（未落地）
│   ├── plans/                ← 计划草稿
│   └── archive/              ← 过程性文档归档（优化全记录09-08/知识参考/方法结论/实测汇总）
│
├── .rivet/
│   ├── HANDOFF.md            ← ★ 权威交接文档（2026-09-14 版）
│   ├── ENVIRONMENT.md        ← ★ NPU/CANN 8.5+9.0 双环境事实（含 361001 根因）
│   └── scratch/              ← 探针与暂存（含 v117_stage 完整代码快照、参考实现库）
│
├── scripts/
│   ├── verify.sh             ← 实机验证（⚠ 会改写 target，见 PITFALLS P8；旧环境版）
│   ├── submit.sh             ← OJ 提交（旧环境路径，需更新）
│   ├── probe_oj.py           ← OJ shape 断言探测器
│   └── ...
│
├── code/                     ← 空壳模板（真实最新代码在 .rivet/scratch/v117_stage/）
├── mhc_head_collapse_torch.py ← PyTorch 参考实现
├── cannbot-skills/           ← CANNBot skills（gitignore，单独管理）
└── AGENTS.md / .rivet.md     ← agent 约定
```

> ⚠ 本工作区**没有** `versions/`（142 个版本目录留在旧 Windows 工作区）；
> 版本演进靠 `迭代优化记录.md` + git history 追溯。

---

## 🔢 5 个测试用例（★ 权威版：F3.9 实测钉死，2026-09-14 TilingFunc 断言探测）

| case | n | h | nH | outer | 执行路径 | v119 OJ 时间 |
|---|---|---|---|---|---|---|
| Case1 | 4 | 4 | 16 | **2** | PATH3 (TinyH4) | 4.86μs |
| Case2 | 4 | 4 | 16 | **2** | PATH3 (TinyH4) | 5.68μs |
| Case3 | 8 | 64 | 512 | **8** | PATH2 (group) | 4.46μs |
| Case4 | 8 | 64 | 512 | **8** | PATH2 (group) | 4.34μs |
| Case5 | 8 | 128 | 1024 | **32** | PATH2 (group) ← 决定排名 | 8.58μs |

dtype 全 fp16；`weight`/`base`/`scale` 保持 fp32。

> ⚠ 本表 outer 列 2026-09-14 起以 F3.9 实测为准。旧记载（Case1=1、Case2=4、
> Case3=1、Case4=2、Case5=8）是历史推断值，已被 8 发 TilingFunc 断言探测推翻
> （outer=2→C1/C2 Pass；outer=8→C3/C4 Pass；outer=32→C5 Pass）。
> Case1/Case2 完全同 shape——其 OJ 时间差纯属评测噪声。

---

## 📊 当前状态

> 滚动状态以 [`STATUS.md`](STATUS.md) 为准，本节只留稳定锚点。

| 项目 | 值 | 证据 |
|---|---|---|
| 在榜提交 | v118（5/5 Pass），历史最优保分 | `FACTS.md F2.14`、`.rivet/HANDOFF.md` |
| 优化重心 | Case1 缺口最大（2.70×），Case5 双峰次之 | `FACTS.md F2.1c` |
| 本地环境 | 910B4 + CANN 8.5（OJ 同版本）双环境，7/8 PASS | `.rivet/ENVIRONMENT.md §八` |
| ⚠ 已知 bug | v117 在 nH=1024 MTE 写越界崩溃（507035），双版本复现 | `STATUS.md` |

**下一步**：见 [`STATUS.md`](STATUS.md) 的优先级列表。

---

## 🛠 怎么跑

### 实机验证（本机无 CANN/NPU，必须走远程）

```powershell
# 连接开发环境
pwsh -File D:\Desktop\OP-Learning\Ascend\atomgit-devspace-tools\Connect-AtomGitDevEnv.ps1

# 跑单个远端命令
pwsh -File ...\Connect-AtomGitDevEnv.ps1 -RemoteCommand 'pwd'

# 验证某个版本
pwsh -File ...\Connect-AtomGitDevEnv.ps1 -RemoteCommand 'bash ~/mhc_scripts/verify.sh vXXX'
```

### OJ 提交（注意限流）

```bash
python "D:/Desktop/OP-Learning/Ascend/cann-learing-hub/skills/cannjudge-submit/cannjudge_cli.py" \
  submit --problem-id 6a7c23a6a52e0f540a8a1779 --project-dir "versions/vXXX/code"
```
⚠ 需 `CANNJUDGE_EMAIL` / `CANNJUDGE_PASSWORD`（本目录内无 `.env`）。
⚠ 429 限流冷却 70-80s。提交前先实机排掉 CE。

### FACTS.md 引用校验

```bash
bash scripts/check-facts.sh
```

---

## ✅ 协作纪律（三条铁律）

1. **新事实写进 `FACTS.md`**，带 `file:line` 证据。**不要**只写 `迭代优化记录.md`。
2. **推翻旧结论时改原条目**，不要追加到文末——追加式记录会让勘误失效（`PITFALLS.md P12`）。
3. **声明"已验证"前**，先确认被验证的代码确实包含所声称的改动（`PITFALLS.md P13`）。
