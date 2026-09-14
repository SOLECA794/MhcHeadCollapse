# EXPERIMENTS — 待执行的实验

> 每个实验必须能**证伪**某个假设。执行前先读 `FACTS.md`，执行后把结果写回 `FACTS.md`。
> **不要**把结果只写进 `迭代优化记录.md`（那是矛盾产生的根源，见 `PITFALLS.md P12`）。
>
> 所有实验在**实机离线**完成，不消耗 OJ 提交次数（限流 70-80s，见 `PITFALLS.md P9`）。
> 实机连接: `pwsh -File D:\Desktop\OP-Learning\Ascend\atomgit-devspace-tools\Connect-AtomGitDevEnv.ps1`

---

## 优先级与依赖

```
实验 C (5min, 零成本)  ← 先做，钉死计量口径
      │
      ├─→ 实验 B (15min) ── 验证 group-of-4 是否真无效
      │
      └─→ 实验 A (20min) ── 验证 floor 是否可削
                │
                └─→ 决策：优化方向是"换归约原语"还是"消除重复 DMA"
```

**为什么这个顺序**：实验 C 零成本且是其余实验的前提（若单位是 ms，所有性能模型都要重算）；
实验 B 只需改一行；实验 A 依赖 B 的结论解释。

---

## 实验 C — 钉死计时单位（零成本）

**假设 H-C**：平台上报的时间单位是微秒（μs）。

**为什么怀疑**：`迭代优化记录.md:75-79` 写 `ms`，而同文件 `:1211` 的勘误写"一直是 μs"。
两处直接矛盾（`FACTS.md F2.1`）。所有性能模型都建立在这个口径上。

**预期**：
- 若单位是 μs：Case5 处理 1024 个 fp16 × 8 行 ≈ 16KB 数据，耗时 8.2μs → 带宽 ≈ 2GB/s（偏低但合理，因为是小 shape + 固定开销）
- 若单位是 ms：同样数据要 8.2ms → 带宽 ≈ 2MB/s（**荒谬**，AIV 不可能这么慢）

**证伪标准**：若实测带宽落在 MB/s 量级 → 单位是 ms，本文件及所有性能分析需重做。

**最小验证手段**（**不需要跑 kernel、不需要 OJ 提交**）：
```bash
# 查提交工具如何解析平台返回的时间字段
grep -n "time\|unit\|ms\|us" \
  "D:/Desktop/OP-Learning/Ascend/cann-learing-hub/skills/cannjudge-submit/cannjudge_cli.py"
```
**若无结论**，退一步：提交一个已知计算量的最小 kernel，用上报时间反推带宽。

**预期耗时**：5 分钟

**状态**：✅ **已执行（2026-09-11）**

**结果**：单位确证为 **μs**。三重独立证据：

1. **工具链不判定单位** —— `cannjudge-submit/SKILL.md:104` 明文写"time 单位以页面/评测说明为准"，
   客户端只是透传平台原值。这解释了 ms/μs 混乱的来源：**没有任何一处代码做过单位判定**。
2. **物理反推（决定性）** —— Case5 数据量 50KB（x 16KB + w 32KB + out 2KB）。
   若单位是 ms → 耗时 8.2ms → 带宽 **6.2 MB/s**，比任何 NPU 慢 5 个数量级。
3. **启动开销量级** —— 若单位是 ms，Case1（仅 16 个元素）耗时 2.94ms，同样荒谬。

**结论**：H-C 成立，未被证伪。已回写 `FACTS.md F2.1`。

---

## 实验 B — group-of-4 是否真的无效（一行修改）

> ⚠ **2026-09-11 修正**：本实验的假设已由静态探针（`scripts/probe_blockdim.py`）部分回答，
> 且**推翻了一个更早的判断**。请先读 `FACTS.md F2.3` 的追加证据再看本节。

**假设 H-B（修正版）**：`kGroupRows=4` 的批处理因 `block_dim` 公式从未生效。
**但"未生效"不等于"改了就有效"** —— 探针揭示了一个结构性约束：

```
若 outer=8 且 blocks=8，无论怎么切分，每 block 平均只有 1 行。
要让批处理生效，必须让 blocks < outer（牺牲并行度换批处理）。
```

**探针结果**（`scripts/probe_blockdim.py`，可复现）：

| 方案 | Case5 | 代价 |
|---|---|---|
| 现状 `bd=outer` | blocks=8，各收 1 行 → **组空转** | — |
| `bd=ceil(outer/4)` | blocks=2，各收 **4 行** → 组生效 | 并行度 8→2 |
| 解耦（blocks=outer, 连续切分） | blocks=8，仍各 1 行 → **无效** | 无 |

**因此实验 B 的真正问题不是"批处理有没有生效"，而是**：
> 在 outer=8 这个规模下，8 block × 1 行 vs 2 block × 4 行，哪个更快？

**静态侧的证据倾向后者**（计算量核算）：
- Case5 总计算量 ≈ 0.15 MFLOP，AIV 单核 ~百 GFLOP/s → 计算仅需 **~0.2μs**
- 实测 8.2μs → **Case5 是 DMA/同步主导，不是计算主导**
- blocks 8→2 时：weight DMA 256KB→64KB（省 ~0.98μs @200GB/s），组同步 8 次→2 次
- 代价：并行度 8→2，但每行计算仅 ~0.03μs，损失可忽略

**预期**：blocks 8→2 后 Case5 时间**下降**（不是"无改善"——这是本实验最初写错的地方）。

**证伪标准**：Case5 时间**不变或上升** → 说明并行度损失 > DMA/同步节省，批处理在这个规模不划算。

**改动**（`versions/v116_vsync_batched/code/op_host/mhc_head_collapse.cpp:33`）：
```cpp
// 现状
uint32_t block_dim = outer <= 4U ? 1U : (outer < kMaxCores ? outer : kMaxCores);
// 改为
uint32_t block_dim = outer <= 4U ? 1U
                   : ((outer + kGroupRows - 1) / kGroupRows < kMaxCores
                      ? (outer + kGroupRows - 1) / kGroupRows : kMaxCores);
// Case5: outer=8 → block_dim = 2
```
⚠ 注意 `block_dim` 同时是 `SetBlockDim`（`op_host:50`）——这正是"并行度与分组粒度耦合"的根源。
**这可能是真正的优化，不只是诊断探针**（见上方核算）。

**预期耗时**：15 分钟

**状态**：⬜ 本地探针已完成，**实机验证阻塞**（SSH forward 不可用）

**阻塞详情**：`Connect-AtomGitDevEnv.ps1` 报
`AtomGit plugin did not establish a usable SSH forward. Open AtomGit Dev Space and verify the environment is running.`
→ 需在 AtomGit Dev Space 手动启动环境后才能跑。

---

## 实验 A — floor 里有多少是可消除的重复 DMA

**假设 H-A**：`FACTS.md F2.6` 的推算成立——Case5 的 3.38μs "floor" 中，
约 1μs 来自 weight 的**重复搬运**（每 block 搬一次，`F2.5`），而非不可压缩的启动开销。

**为什么关键**：若成立，"冠军把 kernel 藏进 floor"（`FACTS.md A1`）这个归因就**部分错误**，
而正确方向变成"消除重复 DMA"——与"换归约原语"是完全不同的路线。

**改动**：让 weight 只被搬一次。两种做法，先试简单的：
- **做法 1（诊断用）**：把 `block_dim` 临时设为 1 跑 Case5，观察时间变化
  （block_dim=1 时只有 1 个 block 搬 weight，但仍会分组循环）
- **做法 2（正式方案）**：block 间用 workspace 做一次全局广播，或让 block 0 搬完后同步

**预期**：
- 若 H-A 成立：Case5 在 block_dim=1 时，floor 应从 3.38 降至 ≈2.2（与 Case3/4 齐平）
- 若 H-A 不成立：floor 不随 block_dim 变化 → 说明 3.38 确实是启动开销

**证伪标准**：`block_dim=1` 时 Case5 时间**不变** → H-A 错，floor 是真实固定开销。

**最小验证手段**：同实验 B 的实机流程，只改 `:33` 的公式为 `block_dim = 1`。

**预期耗时**：20 分钟（含对照）

**状态**：⬜ **实机验证阻塞** —— 同实验 B（SSH forward 不可用）

---

## 实验 D — 确认 Case3/4/5 的真实 shape（若要彻底钉死）

**假设 H-D**：Case3/4/5 是 n=8/h=64 与 h=128（`FACTS.md F2.2`）。

**为什么**：`迭代优化记录.md:75-79` 说是 n=4/h=4（全走 PATH3），
但探针证据（`make_probes.py:1` 的注释 + 探针实测有效）反证它错了。
:1220 的勘误也只说"很可能错误（待探针确认）"——**从未确认**。

**验证手段**：造一个"若 shape 是 X 则必然崩溃/越界"的断言探针，提交到 OJ 看是否 RE。
（这会消耗 1-2 次 OJ 提交，所以放最后。）

**证伪标准**：探针在 Case3 上 Pass → shape 假设错。

**预期耗时**：30 分钟（含 OJ 限流等待）

**状态**：⬜ 未执行（优先级最低——静态证据已足够支撑 F2.2）

---

## 实验结果记录表

| 实验 | 执行日期 | 结果 | 结论 | 已回写 FACTS.md |
|---|---|---|---|---|
| C | 2026-09-11 | 单位确证为 μs（三重独立证据） | H-C 成立，未证伪 | ✅ |
| B | ⬜ | | | ⬜ |
| A | ⬜ | | | ⬜ |
| D | ⬜ | | | ⬜ |

---

## 执行纪律（每次实验前自检）

1. **改动前**：从**完整 copy 的 code tree** 开始（`PITFALLS.md P6`），不要只 copy 单文件
2. **提交前**：确认 `op_host` 里是 `.AddConfig("ascend910b")`，**不是** `ascend910_93`
   （`verify.sh:10` 会改写它，见 `PITFALLS.md P8`）
3. **测之前**：确认该优化路径**真的被执行**（加计数器或从探针反推）——`PITFALLS.md P14` 的教训
4. **下结论前**：检查数字跨 case 是否自洽——`PITFALLS.md P15` 的教训
5. **有结果后**：写回 `FACTS.md`，**不是**只写 `迭代优化记录.md`


---

## E-v117_vecz（2026-09-12）— 向量化 z 计算：正确但无收益（负结果）

**假设**（源自 F2.1n 静态定位）：TinyH4 步骤 5 每行 9 次 UB 标量访问
（1 rms 读 + 4 gate 读 + 4 SetValue）是 scalar 46% 的主因，向量化后可大幅削减。

**改动**：sigmoid 原地做在 ReduceSum 的 stride-8 块布局上——
Muls(-k) → Add(-base pattern) → Exp → Adds → Reciprocal，全向量。
每行标量访问从 9 次降到 1 次（只剩 rms 读）。负号折进 -k 与 -base 常量
（IEEE754 取负精确，位级等价）。

**过程**：
1. 首版丢负号 bug（lg*k+(-base) 应为 (-lg)*k+(-base)）→ TinyH4 全 FAIL（RED）
2. 修正 -k 后 7/7 PASS，误差与基线完全一致（2.567e-04/1.455e-03）→ 位级等价证实
3. msprof 同口径对比（outer=4, shape 1,4,16，各 6 次采样）：

| 指标 | v116 | v117 |
|---|---|---|
| scalar | 4.34~5.16（中位 5.13） | 4.72~5.32（中位 5.22） |
| vec | 1.59 | 1.75 |
| Task Dur | 7.5~10.9 | 7.5~12.2 |

**结论：证伪**。scalar 持平，vec 略升（+0.16，多做的向量活）。
"每行 9 次标量访问"不是 scalar 开销的主因。

**关键新认知（比原假设更重要）**：
1. **v116 在 outer=4 口径下 scalar 高达 5.16μs**（F2.1e 的 outer=1 口径是 3.32）
   ——scalar 随 outer 增长，说明大部分标量开销在**每行的 fold 段**
   （MhcUnpack2×8 + y 64-bit 打包写），不在步骤 5
2. fold 段每行有 8 次 64-bit GetValue + 16 次浮点乘加 + 1 次 GM SetValue——
   这是下一个真正的标量削减目标（y 输出向量化 + x 读取向量化）
3. 平台 Task Duration 本身抖动 ±2μs（7.5↔10.9 交替），单次对比不可信，
   必须多采样中位

**遗留**：v117 代码保留在 versions/v117_vecz/（正确性已验证，位级等价），
可与后续 fold 段改造叠加。
