# MhcHeadCollapse 算子（mHC四路归一）— 工作导航

**赛题**：中国电信星辰杯高校AI算子开发挑战赛 C组中等题
**平台**：CANNJudge | Problem ID `6a7c23a6a52e0f540a8a1779`
**目标**：5 个固定 case 上取得尽可能短的运行时间（per-call 端到端计时）

---

## 🚀 从这里开始

| 我想… | 读这个 |
|---|---|
| **了解什么是真的**（避免被过时结论误导） | [`FACTS.md`](FACTS.md) ← **唯一权威事实源** |
| **知道哪些坑已经踩过** | [`PITFALLS.md`](PITFALLS.md) |
| **知道下一步做什么** | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| **看某个版本怎么来的** | [`versions/README.md`](versions/README.md)（版本索引） |
| **读历史全过程** | [`迭代优化记录.md`](迭代优化记录.md) ⚠ 见下方警告 |

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
【C组中等题】MhcHeadCollapse 算子（mHC四路归一）/
│
├── README.md                 ← 你在这里（导航）
├── FACTS.md                  ← ★ 单一事实源（改动代码前必读）
├── PITFALLS.md               ← 已踩过的坑（19 条）
├── EXPERIMENTS.md            ← 待执行实验（A/B/C/D）
│
├── 赛题.md                    ← 题目原文
├── 迭代优化记录.md             ← ⚠ 历史过程（140K，含已知错误）
├── MhcHeadCollapse_优化全记录_2026-09-08.md  ← 阶段性总结（相对可靠）
├── MhcHeadCollapse优化知识参考.md            ← 背景知识
├── OP_TEAM_DESIGN.md         ← 团队协作设计（未落地）
│
├── versions/                 ← 142 个版本目录（85 唯一 + 52 副本）
│   └── README.md             ← 版本索引
│
├── scripts/
│   ├── verify.sh             ← 实机验证（⚠ 会改写 target，见 PITFALLS P8）
│   ├── submit.sh             ← OJ 提交
│   ├── make_probes.py        ← WA 差分探针生成器
│   ├── make_v112.py
│   ├── balance_check.py      ← 括号配平检查
│   └── check-facts.sh        ← FACTS.md 引用校验（防漂移）
│
├── code/                     ← 当前工作副本（28 行的 stub，非最新）
├── mhc_head_collapse_torch.py ← PyTorch 参考实现
└── build_v058/               ← 构建产物（应 gitignore）
```

---

## 🔢 5 个测试用例（权威版，见 `FACTS.md F2.2`）

| case | n | h | nH | outer | 执行路径 |
|---|---|---|---|---|---|
| Case1 | 4 | 4 | 16 | 1 | PATH3 (TinyH4) |
| Case2 | 4 | 4 | 16 | 4 | PATH3 (TinyH4) |
| Case3 | 8 | 64 | 512 | 1 | PATH2 (向量) |
| Case4 | 8 | 64 | 512 | 2 | PATH2 (向量) |
| Case5 | 8 | 128 | 1024 | 8 | PATH2 (向量) ← 决定排名 |

dtype 全 fp16；`weight`/`base`/`scale` 保持 fp32。

---

## 📊 当前状态

| 项目 | 值 | 证据 |
|---|---|---|
| 在榜最佳（真实） | `v093_polyexp_fix` 41.53 分，第 5 名 | `迭代优化记录.md:173` |
| ⚠ 伪成绩 | `v090_min_host` 41.72 分（UB bug 下的伪 Pass） | `迭代优化记录.md:170` |
| 待验证的最新版本 | `v116_batched_reduce`（唯一含真实改动的 v116） | `FACTS.md F2.4` |
| 榜首参照 | 2.94 / 2.44 / 3.24 / 3.30 / 5.46（μs） | `迭代优化记录.md:1216` |

**下一步**：执行 `EXPERIMENTS.md` 的实验 C（5 分钟，零成本）→ B（15 分钟）→ A（20 分钟）。

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
