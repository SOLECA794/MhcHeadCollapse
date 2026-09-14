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

**假设 H-B**：`kGroupRows=4` 的批处理因 `block_dim` 公式从未生效
（`FACTS.md F2.3`），所以 v110/v111 的"同算法重排无效"证伪**不成立**。

**为什么关键**：这条结论支撑着"必须换归约原语才能突破"（`FACTS.md A2`）。
若 B 成立，则一条主要优化路径被错误关闭。

**改动**（`versions/v116_vsync_batched/code/op_host/mhc_head_collapse.cpp:33`）：
```cpp
// 现状
uint32_t block_dim = outer <= 4U ? 1U : (outer < kMaxCores ? outer : kMaxCores);

// 改为（让每 block 至少能收满一组）
uint32_t block_dim = outer <= 4U ? 1U : ((outer + kGroupRows - 1) / kGroupRows);
// 即 Case5: outer=8 → block_dim = 2，每 block 收 4 行
```
⚠ 注意：`block_dim` 同时是 `SetBlockDim` 的值（`:50`），改小会让并行度下降。
**这不是最终方案，只是诊断探针**——目的是回答"批处理有没有用"，不是刷分。

**预期**：
- 若批处理有用：Case5 时间下降（事件/同步开销被 4 行摊薄）
- 若批处理无用：Case5 时间不变或略升（并行度损失抵消收益）

**证伪标准**：Case5 时间**无改善** → "同算法重排无效"侥幸正确，A2 成立。

**最小验证手段**：
```powershell
# 1) 打包改动后的版本
tar czf /tmp/vB.tgz -C versions/vB_groupfix code
# 2) 上传实机并跑验证矩阵
pwsh -File Connect-AtomGitDevEnv.ps1 -RemoteCommand 'bash ~/mhc_scripts/verify.sh vB'
```
对比 `verify.sh` 输出的 bench 段（`8 128 8 fp16` 那一行）与 v100/v111 的基线。

**预期耗时**：15 分钟

**状态**：⬜ 未执行

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

**状态**：⬜ 未执行

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
| C | ⬜ | | | ⬜ |
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
