# FACTS — MhcHeadCollapse 单一事实源

> **本文件是唯一权威事实源。** 其他文档（迭代优化记录.md / 优化全记录.md）含历史过程，
> 但其中的结论可能已被推翻——**冲突时以本文件为准**。
>
> **准入纪律**：每条事实必须带 `证据:`（file:line 或命令输出）。无证据的一律进 §A 假设区并标置信度。
> **漂移防护**：`bash scripts/check-facts.sh` 会校验本文件所有 file:line 引用是否仍有效。
>
> 最后核对：2026-09-11（由只读侦察得出，未实机复现的条目已标注）

---

## §0 一句话现状

算子已完成基本实现并有在榜成绩，但**两个核心性能归因结论建立在被证伪的前提上**，
且文档中存在多处互相矛盾的记载。当前最该做的不是继续调优，而是先验证 §B 中的两条待验证事实。

---

## §1 算子与赛题

**F1.1** 算子做四步融合：RMS 归一 → 门控线性投影 → sigmoid → 加权折叠。输出维度由 `nH` 变为 `H`。
证据: `赛题.md:1-40`（算子说明与数学定义）

**F1.2** 算子声明为纯 SIMD（AIV_ONLY），**无任何 matmul 调用**。
证据: `versions/v116_vsync_batched/code/op_kernel/mhc_head_collapse.cpp:486`(`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`)

**F1.3** 评测为 per-call 端到端计时，5 个固定 case，`ranking_submission_mode=latest`（后提交覆盖前次）。
证据: `迭代优化记录.md:71`（平台信息表）

---

## §2 事实（已验证）

### F2.1 计时单位是微秒（μs），不是毫秒

**证据**: `MhcHeadCollapse_优化全记录_2026-09-08.md:49`（明确论证）、`迭代优化记录.md:1211`（勘误小节）

⚠ **但 `迭代优化记录.md:75-79` 的用例表至今仍写 `ms`**，与 :1211 的勘误直接矛盾。
读到 :75-79 的人会得到数量级错误的结论。**以本文件为准：单位是 μs。**

### F2.2 Case3/4/5 的真实 shape 是 n=8，不是 n=4

| case | n | h | nH | outer | 路径 |
|---|---|---|---|---|---|
| Case1/2 | 4 | 4 | 16 | 1 / 4 | PATH3 (TinyH4) |
| Case3/4 | 8 | 64 | 512 | 1 / 2 | PATH2 (向量) |
| Case5 | 8 | 128 | 1024 | 8 | PATH2 (向量) |

**证据（两条独立）**：
1. `MhcHeadCollapse_优化全记录_2026-09-08.md:33-39`（正确的用例表）
2. **逻辑反证**：`scripts/make_probes.py:1` 注释写明 "PATH2 kernel only"，`:14` 只清空 `ProcessVectorRow`；
   而该探针在 Case3 上使时间从 4.42 降至 2.2（`迭代优化记录.md:92-93`）。若 Case3 真走 PATH3，探针不该有效。

⚠ **`迭代优化记录.md:75-79` 与 :84 的记载（"5个Case全部走PATH3"）是错的。**
:1220 的勘误已承认"shape 假设很可能错误"，但 :75-79 的主表从未同步。

### F2.3 block_dim 公式使 group-of-4 批处理完全失效

`op_host` 的公式（`versions/v116_vsync_batched/code/op_host/mhc_head_collapse.cpp:33`）：
```cpp
uint32_t block_dim = outer <= 4U ? 1U : (outer < kMaxCores ? outer : kMaxCores);  // kMaxCores=20, :32
```
而 `op_kernel` 的行收集步长是 `block_dim_`（`.../op_kernel/mhc_head_collapse.cpp:346`），
上限 `kGroupRows=4`（`:591`）。

代入实际参数：

| case | outer | block_dim | 实际收集行数 cnt |
|---|---|---|---|
| Case3 | 1 | 1 | 1 |
| Case4 | 2 | 1 | 2 |
| **Case5** | **8** | **8** | **1** |

**Case5 每 block 只收 1 行，`kGroupRows` 从未达到。** 为批处理准备的 parity 双缓冲
（`:407` 附近的 `work[(j & 1U) * nhF]`）、整组单次 `MTE2_V` 同步（`:390` 附近）在 Case5 下全部空转。
UB 预算检查在位（`:296-302`，上限 150KB），但预算被浪费。

### F2.4 `v116_vsync_batched` / `v116_batched_reducesum` 是 `v111_path2_group4` 的逐字节副本

命令: `md5sum versions/{v111_path2_group4,v116_batched_reducesum,v116_vsync_batched}/code/op_kernel/mhc_head_collapse.cpp`
→ 三者均为 `8d7b5512030d...`

只有 `v116_batched_reduce` 有真实改动（`fa273c0a360a...`，新增了 `SetFlag/WaitFlag<V_S>(0)`）。

**推论**：文档所称"v116 实机验证通过"实际验证的是 v111 的代码。
`V_S` 事件在 `ProcessVectorRow` 内**不存在**（全文仅 `:177-178, :204-205`，均在 TinyH4 类内）。

### F2.5 weight 被每个 block 重复搬运一次

`.../op_kernel/mhc_head_collapse.cpp:332`: `DataCopy(w_cache, weight_gm_, n_ * nH_);` 位于 `Process()` 内，每个 block 执行一次。

| case | weight 字节 | block_dim | 总搬运 |
|---|---|---|---|
| Case3 | 16 KB | 1 | 16 KB |
| **Case5** | **32 KB** | **8** | **256 KB** |

### F2.6 floor 差异可归因于重复 DMA（★ 关键推论）

探针实测 floor（`迭代优化记录.md` 探针数据节）：Case3/4 = 2.2μs，Case5 = 3.38μs。
榜首 Case3/4 实测 2.84/2.92μs —— **低于 Case5 的 floor**，这在"floor 是常量"的模型下自相矛盾。

**解释**：P1 探针只清空 `ProcessVectorRow`，而 weight DMA 在 `Process()` 内仍执行（F2.5）。
按 F2.5 的搬运量差 (256−16)KB = 240KB，若有效带宽 ≈200GB/s：
`240KB ÷ 200GB/s ≈ 1.23μs`，与实测 floor 差 `3.38 − 2.2 = 1.18μs` **高度吻合**。

**含义**：所谓"per-call floor"里有约 1μs 是**可消除的重复 DMA**，不是不可压缩的启动开销。
这直接影响"kernel 藏进 floor"这条归因是否成立。

### F2.7 fp16 count 版 Mul/ReduceSum 在 dav_c220 不可用

证据: `迭代优化记录.md:1597`（编译级证伪，实机复现，报错行号与 OJ 逐字一致）

### F2.8 `ReduceDataBlock`/`ReducePairElem` 确实可用 —— 但被架构门控拦住

历史记录称其"不是公开 API"（`迭代优化记录.md:1597` 等）。**这个理由不准确**：

- 它们**在公开头文件中**：`asc-devkit/include/basic_api/kernel_operator_vec_reduce_intf.h:51,71`
  （`kernel_operator_intf.h:54` 会链式包含它）
- 真正的拦截是 `__ASC_USE_RESERVED_UBUF__(3510, ...)` 宏：`asc-devkit/impl/utils/sys_macros.h:110-120`
  只白名单 `2201`/`3510` 两个 arch，而 OJ 是 910B（`dav_c220`）

**结论虽对，理由记错**——这会导致后续在其他 API 上重蹈误判。

### F2.9 CATLASS 不可直接引入本赛题

- `CATLASS_ARCH` 仅识别 `2201`(AtlasA2) / `3510`(Ascend950)：`catlass/CMakeLists.txt:46-48`
- 32B 对齐为 `static_assert` 硬约束，且被 `#if CATLASS_ARCH == 2201` 门控：
  `catlass/include/catlass/gemm/block/block_mmad_pingpong_tla.hpp:133-138`
- 语义层不匹配：CATLASS 是 Cube/MMAD 库，本算子是最重的计算仅 8 次长度 1024 的**向量点积**（F1.2）

### F2.10 eps 属性形同虚设

OpDef 声明了 `eps_norm`/`eps_hc`（`op_host:106-107`），但 host 硬编码为 `1e-6f`（`op_host:47-48`），从不读用户传值。

### F2.11 潜在 DMA 越界读

`.../op_kernel/mhc_head_collapse.cpp:334-335`：
```cpp
AscendC::DataCopy(cst, base_gm_, 8U);        // base 实际只有 n 个元素（Case1/2 时 n=4）
AscendC::DataCopy(cst[8], scale_gm_, 8U);    // scale 实际只有 1 个元素
```
CANN 下此类越界通常静默。**当前能跑不等于安全**，换 shape 或对齐方式可能触发问题。

### F2.12 死符号

- `weight_queue_` 声明未使用：`.../op_kernel/mhc_head_collapse.cpp:578`
- `kOutputTile` 定义未使用：`:592`
- `ScalarToFloat` 及 bfloat16 特化无调用点：`:16, :21`

### F2.13 版本目录中 52/142 是纯副本

命令: 对 `versions/*/code/op_kernel/mhc_head_collapse.cpp` 逐个 md5sum
→ 142 个目录，**85 个唯一内容，52 个重复**。例：`v002/v003/v004/v005/v006` == `v001`。

### F2.14 在榜最佳版本

| 版本 | 成绩 | 真实性 |
|---|---|---|
| `v090_min_host` | 41.72 分 | ⚠ **伪 Pass** —— `迭代优化记录.md:170` 自承"实为 UB 索引 bug 下的伪 Pass" |
| `v093_polyexp_fix` | 41.53 分 | ✅ 首次真正 5/5 Pass —— `迭代优化记录.md:173` |

**证据**: `迭代优化记录.md:170,173`

---

## §A 假设区（未验证，需实验确认）

**A1** 「冠军把 kernel 藏进了 per-call floor」—— 若 F2.6 成立，此断言可能不成立。
置信度: **低**。验证: `EXPERIMENTS.md` 实验 A。

**A2** 「必须换归约原语才能突破」—— 建立于"同算法重排无效"（v110/v111）之上，
但 v110/v111 恰因 F2.3 从未真正生效。置信度: **低**。验证: `EXPERIMENTS.md` 实验 B。

**A3** Case3/4 的具体 outer 值（1/2 还是 2/4）—— 两份文档不一致，
`迭代优化记录.md:85-86` 的推断过程被 :1220 自承不可靠。置信度: **中**。可用 OJ 探针确认。

---

## §B 本文件的使用方式

1. **改动代码前**：先读本文件 §2，避免基于已推翻的结论做决策。
2. **发现新事实**：加条目 + `证据:` 行；**不要**只写进 `迭代优化记录.md`（那正是矛盾产生的根源）。
3. **推翻旧事实**：直接修改本条目并保留「原结论 + 推翻理由」，不要追加到文末。
4. **提交前**：跑 `bash scripts/check-facts.sh` 校验引用未失效。

---

## §C 未验证声明清单（诚实边界）

- 本文件所有内容来自 2026-09-11 的**静态代码阅读 + 文档比对**，**未在本机实机运行**（本机无 CANN、无 NPU）。
- F2.6 的带宽推算（200GB/s）是**估算**，未经实测标定。
- F2.2 的结论依赖"探针生效"这一文档记载；探针的实际运行日志未在仓库中找到。
- F2.10/F2.11/F2.12 来自静态阅读，未编译验证。
