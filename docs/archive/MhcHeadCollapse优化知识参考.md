# MhcHeadCollapse 性能优化知识参考

> 本文档基于对 `e:\Desktop\AscendC算子学习开发\` 下各代码仓库（asc-devkit、ops-transformer、cannbot-skills 等）的挖掘整理，针对 MhcHeadCollapse 算子当前最佳版本 **v031（4.18us / 5.10us）** 到榜首 **2.66us / 2.82us** 的约 1.5us 差距，提供可执行的优化方向、知识索引与参考代码位置。
>
> 整理时间：2026-08-24。所有绝对路径均已核实存在。

---

## 1. 当前状态与差距分析

| 指标 | 数值 |
| --- | --- |
| 当前最佳（v031） | 测试点1: 4.18us，测试点2: 5.10us |
| 平台榜首 | 2.66us / 2.82us |
| 差距 | 约 1.5us / 2.3us |
| 已知瓶颈 | Launch/调度开销（硬件下限约 2.5us）+ 内核执行 |

v031 已实现：全输入 DMA 预载入 UB、标量 0 GM 访存、全生命周期单次 V_S 屏障、5 向量融合流、4 阶 Horner Exp、2 步 Newton Rsqrt、1D 极简 TilingKey、`#pragma GCC optimize("O3,fast-math,unroll-loops")`。

**剩余差距的构成**：头开销（核启动/取址/初始化）+ 尾开销（DCache 回刷等）+ 内核内归约指令延迟。下面按优先级给出仓库中挖掘到的对应优化手段。

---

## 2. 高优先级优化点（针对 v031 的明确缺口）

### 2.1 【缺口A】核函数未设置 KERNEL_TYPE_AIV_ONLY ⭐ 最高优先级

**v031 现状**：`versions/v031_all_dma_ub_zero_scalar_gm/code/op_kernel/mhc_head_collapse.cpp` 的核函数入口（第 300-318 行）**没有** `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。

**问题**：本算子为纯 Vector 标量算子。不显式设置 Kernel 类型时，调度器默认按 AIV:AIC = 1:2 配比下发任务，会同时启动**无任何计算指令的 Cube 核**，白白产生核启动与初始化头开销。官方文档明确说明头开销随启动核数线性增加，且"对于整体耗时在微秒级别且单核计算量耗时较少的算子"收益显著——与本算子场景完全吻合。

**修复方法**（参照工业实现 `e:\Desktop\AscendC算子学习开发\ops-transformer\mhc\mhc_post\op_kernel\mhc_post.cpp` 第 28 行）：

```cpp
template <typename DT_X>
__global__ __aicore__ void mhc_head_collapse(
    GM_ADDR x, GM_ADDR weight, GM_ADDR hc_base, GM_ADDR hc_scale,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);   // <-- 新增：纯向量算子免启动 Cube 核
    REGISTER_TILING_DEFAULT(MhcHeadCollapseTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcHeadCollapseTilingData, tiling_data, tiling);
    ...
}
```

**知识来源**：
- `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\设置合适的核数和算子Kernel类型.md`
- `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\api\SIMD-API\basic_api\Kernel-Tiling\set_Kernel_type.md`
- 实例：`e:\Desktop\AscendC算子学习开发\ops-transformer\mhc\mhc_post\op_kernel\mhc_post.cpp`

### 2.2 【缺口B】TPipe 在 Kernel 类对象内创建 ⭐ 高优先级

**v031 现状**：`KernelMhcHeadCollapseTinyH4` 类中 `AscendC::TPipe pipe_` 是类成员（对象内创建并初始化）。

**问题**：TPipe 对象初始化会设置全局 TPipe 指针，导致 Kernel 类对象内存有被外部污染的风险，编译器将采取保守策略，**不对类内 Scalar 变量做常量折叠和常量传播**，Scalar 指令耗时增加。官方实测该优化使平均时间下降 17%、scalar_time 占比从 21% 降至 17%。本算子 Scalar 单元承担 sigmoid/折叠计算，属于 Scalar 敏感场景。

**修复方法**（TPipe 移到核函数入口创建，类内只存指针）：

```cpp
template <class DT_X>
class KernelMhcHeadCollapseTinyH4 {
public:
    __aicore__ inline void Init(..., AscendC::TPipe* pipeIn) {
        pipe_ = pipeIn;
        pipe_->InitBuffer(ub_buf_, 2432U);
        ...
    }
private:
    AscendC::TPipe* pipe_;   // 指针而非对象
    ...
};

// 核函数入口：
AscendC::TPipe pipe;
KernelMhcHeadCollapseTinyH4<DT_X> op;
op.Init(..., &pipe);
```

**知识来源**：`e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\避免TPipe在对象内创建和初始化.md`

### 2.3 【缺口C】TilingData 结构精简与 8 字节对齐

**问题**：`GET_TILING_DATA` 会把 TilingData 从 GM 拷贝到 AI 处理器栈空间，拷贝耗时为 **us 级**，小 shape 场景收益明显。当前 TilingData 含 `outer/n/h/nH/block_dim/inv_nH/eps_norm/eps_hc` 等字段，需检查：
1. 是否有可由 Kernel 侧推导的冗余字段（如 `n`、`h` 在 Tiny 路径固定为 4，`block_dim` 可用 `GetBlockNum()` 获取）；
2. 字段类型是否最小化（`outer` 不会超 uint16_t 范围时不要用 uint32_t/uint64_t）；
3. 字段排布是否 8 字节对齐（小类型排一起，避免编译器补零膨胀）。

**参考**：`e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\限制TilingData结构大小.md`（含正反例与字节补齐分析）

### 2.4 【缺口D】ReduceSum 替换为 ReduceDataBlock + ReduceRepeat 低延迟组合

**v031 现状**：每行 5 个 `ReduceSum`（1 个 RMS + 4 个 gate 点积），outer=2 时共 10 个。`ReduceSum` 高阶 API 内部由多条指令组合实现。

**官方实测数据**（256 float 输入、循环 100 次）：

| 方案 | 耗时 |
| --- | --- |
| 两次 ReduceRepeat | 13.00us |
| 三次 ReduceDataBlock | 13.94us |
| **一次 ReduceDataBlock + 一次 ReduceRepeat** | **8.44us** |

单指令 `ReduceDataBlock` 执行效率优于 `ReduceRepeat`，而二者组合优于各自单独使用。对本算子 16 元素（2 个 DataBlock）的归约：**一次 `ReduceDataBlock`（16→2 个 float）+ 标量一次加法（或 `ReduceRepeat`）** 预计比 `ReduceSum` 快。同时二分累加方案（Add 折半 + 末尾 ReduceRepeat）在大数据量时更优——归约指令延迟约为 Add 的 2-5 倍。

**关键代码片段**（ReduceDataBlock + ReduceRepeat 组合）：

```cpp
constexpr uint32_t BLK_LEN = 32;
constexpr uint32_t c0Count = BLK_LEN / sizeof(float);  // 8
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;  // 16 -> 2
AscendC::SetMaskCount();
AscendC::SetVectorMask<float>(0, totalLength);
AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, float, float, false>(
    tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
    AscendC::DEFAULT_BLK_STRIDE, AscendC::DEFAULT_BLK_STRIDE, AscendC::DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<AscendC::PIPE_V>();
AscendC::SetVectorMask<float>(0, blockNum0);
AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float, float, false>(
    zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
    AscendC::DEFAULT_BLK_STRIDE, AscendC::DEFAULT_BLK_STRIDE, AscendC::DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<AscendC::PIPE_V>();
AscendC::SetMaskNorm();
```

**知识来源**：`e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\矢量计算\选择低延迟指令-优化归约操作性能.md`
**完整可运行样例**：`e:\Desktop\AscendC算子学习开发\asc-devkit\examples\01_simd_cpp_api\03_basic_api\01_memory_vector_compute\reduce\`（ReduceCustom）

### 2.5 【缺口E】Sigmoid 向量化候选方案：Exp LUT 查表模式

**现状**：v031 用标量 `MhcScalarSigmoid`（4 次/行）。若想移到向量单元批量算 4 个 gate 的 sigmoid（把 4 个 mixes 连续排放后一次 `Exp`），可用高阶 API `Exp<T, expandLevel, isReuseSrc>`，其中 **`expandLevel = 0` 表示不使用泰勒展开（内部走 LUT 查表 + 插值）**，计算量比 Taylor 展开少一个数量级。对超越函数占比高的 VEC bound 场景，这是官方推荐的"减少计算量"核心策略。

```cpp
// expandLevel=0 → LUT 模式；需要 stackBuffer 临时空间
AscendC::Exp<float, 0, false>(dstLocal, srcLocal, stackBuffer, calCount);
```

**知识来源**：
- 样例：`e:\Desktop\AscendC算子学习开发\asc-devkit\examples\01_simd_cpp_api\04_advanced_api\10_math\exp\exp.asc`
- 策略说明：`e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\single-core-pipeline\vec.md`（策略 6：LUT + 插值）

### 2.6 【缺口F】DCI 编译选项减少尾开销（需平台验证）

算子结束时默认插入 DCCI（DataCacheCleanAndInvalid）指令将 DCache 置无效，包含 Clean 回刷 GM 的开销。编译选项 `--cce-no-dcache-flush` 可改为插入更轻量的 DCI 指令。**适用前提**：算子用 `GlobalTensor::SetValue` 正确写 GM（v031 正是），不依赖框架自动插入的 DCCI 保证一致性。

⚠️ 注意：官方文档标注该建议适用于 Ascend 950PR/950DT，910B 平台是否生效需要以编译日志/平台实测确认，建议作为独立小版本验证。

**知识来源**：`e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\设置DCI编译选项来减少算子尾开销.md`

---

## 3. 小 case Scalar Bound 优化策略（本算子典型场景）

本算子（tiny shape n=4,H=4，outer≤2）总计算量极小，属于官方定义的"Scalar Bound / 小 case"场景。cannbot-skills 给出的策略表：

| 策略 | 操作 | v031 状态 |
| --- | --- | --- |
| 性能友好 API | **用 set_flag/wait_flag 代替 Queue，用 LocalTensor 代替 Tbuffer，去除 Tpipe** | 部分达成（TBuf+HardEvent 已用，但仍保留 TPipe） |
| 循环展开 | 展开小循环减少分支代价 | 已用 `#pragma unroll 2` |
| 减少循环轴 | 按 tiling 最简化循环轴 | outer 循环仍在 |
| 指令选择 | 高效 scalar 指令替代低效序列 | 已用 Horner/Newton |
| 减少标量-向量转换 | 避免不必要的 Scalar↔Vector 搬移 | GetValue 读取 16 个 x（可评估） |
| 减少 launch 开销 | 与其他 kernel 融合 | 不适用（单算子赛题） |

"去除 TPipe"的激进方向可参考 v016 的无 TPipe 实现与 v023-v031 的 TBuf 路线折衷：TPipe 本身的构造/InitBuffer 有固定成本，若能用 `GetTPipePtr()` 复用全局 pipe（mamba2 系列做法，见 §5.1）或彻底手写 UB 偏移，可再省一段初始化。

**知识来源**：`e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\single-core-pipeline\scalar.md`

---

## 4. 性能知识地图（按主题索引）

### 4.1 头尾开销优化（Launch Overhead —— 当前主要矛盾）

| 文档 | 核心内容 | 优先级 |
| --- | --- | --- |
| `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\设置合适的核数和算子Kernel类型.md` | 头开销随核数线性增加；微秒级算子应减核增单核算力；AIV 算子须设 AIV_ONLY | 高 |
| `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\限制TilingData结构大小.md` | TilingData GM→栈拷贝 us 级开销；字段最小类型 + 8 字节对齐排布 | 中 |
| `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\避免TPipe在对象内创建和初始化.md` | TPipe 外置触发编译器 Scalar 常量折叠；实测 -17% | 中 |
| `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\核函数内删除Workspace相关冗余操作.md` | 删除 SetSysWorkspace/GetUserWorkspace 冗余判断 | 中 |
| `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化\头尾开销优化\设置DCI编译选项来减少算子尾开销.md` | --cce-no-dcache-flush：DCCI→DCI 减尾开销（950 标注，910B 待验证） | 高(950) |

### 4.2 矢量计算

| 文档 | 核心内容 |
| --- | --- |
| `...\矢量计算\选择低延迟指令-优化归约操作性能.md` | ReduceDataBlock+ReduceRepeat 组合（8.44us vs 13us）；二分累加方案；归约指令延迟 = Add 的 2-5 倍 |
| `...\矢量计算\通过Unified-Buffer融合实现连续vector计算.md` | 前一计算输出驻 UB 直接喂下一计算，GM 搬运从 2n 次降到 2 次 |
| `...\矢量计算\Vector算子灵活运用Counter模式.md` | Counter 掩码模式 |
| `...\矢量计算\基于全局掩码复用的计算性能优化.md` | 掩码复用 |
| `...\矢量计算\VF性能优化\VF融合优化.md` / `VF循环优化.md` / `指令双发优化.md` | VF 范式指令融合、循环与双发 |
| `e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\single-core-pipeline\vec.md` | VEC bound 分级（>80% 深度 bound 靠算法减计算量）；Exp LUT；Cast>20% 须批量化；融合指令 VMULA/VMADD |

（`...` 为 `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\SIMD算子性能优化` 的缩写）

### 4.3 内存访问

| 文档 | 核心内容 |
| --- | --- |
| `...\内存访问\GM地址尽量512B对齐.md` | 32B 对齐只有 512B 对齐 70% 带宽（A2 生效，即 910B） |
| `...\内存访问\尽量一次搬运较大的数据块.md` | 单次搬运 ≥16KB 才能打满带宽（大 shape 场景） |
| `...\内存访问\高效的使用搬运API.md` | 用 DataCopyParams(blockCount/blockLen/srcStride/dstStride) 一次搬完，杜绝 for 循环逐行搬运 |
| `...\内存访问\避免UB的bank冲突\avoid_bank_conflict_npu_arch_2201.md` | 2201 架构（910B）UB bank 结构与冲突场景 |
| `...\内存访问\避免UB的bank冲突\avoid_bank_conflict_npu_arch_3510.md` | 3510 架构（950）UB bank 结构：8 bank group × 2 bank，低位交织 |
| `e:\Desktop\AscendC算子学习开发\asc-devkit\examples\01_simd_cpp_api\05_best_practices\04_memory_access\bank_conflict_ub\bank_conflict_ub.asc` | UB bank conflict 优化可运行样例 |
| `e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\single-core-pipeline\memory.md` | 访存 bound：L2 复用、基本块选取、合并小块搬运 |

### 4.4 流水编排与无 Bound 优化

| 文档 | 核心内容 |
| --- | --- |
| `e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\single-core-pipeline\no-bound.md` | PingPong 双缓冲、Preload 预取、指令提早发射、一次性搬入多次复用 |
| `...\流水编排\使能DoubleBuffer.md` | 搬入/计算 overlap |
| `...\流水编排\使能Iterate或IterateAll异步接口避免AIC-AIV同步依赖.md` | 异步接口消除同步依赖 |

### 4.5 Tiling 与归约类算子建模

| 文档 | 核心内容 |
| --- | --- |
| `e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\tiling\reduction\fallback\example\softmax\experience.md` | **RMSNorm/LayerNorm 与 Softmax 同构**：归约→逐元素→归约→逐元素四步流水；AR 族模板选型（SmallR/FullLoad/Recompute）；UB 预算公式；多核切分 |
| `e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\tiling\reduction\fallback\script\reduction_tiling.py` | 归约类 tiling 参考实现脚本 |
| `e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\references\tiling\elewise\` | 逐元素算子 tiling 模板 |
| `...\Tiling策略\核间负载均衡.md` | 多核负载均衡（大 shape outer 行并行时参考） |

### 4.6 性能分析方法

| 文档 | 核心内容 |
| --- | --- |
| `e:\Desktop\AscendC算子学习开发\asc-devkit\docs\zh\guide\算子实践参考\性能分析\分析性能数据.md` | msOpProf 采集与 OpBasicInfo.csv / PipeUtilization.csv 解读，定位 bound |
| `e:\Desktop\AscendC算子学习开发\cannbot-skills\ops\ascendc-perf-optimize\SKILL.md` | 4 步优化流程：Tiling 建模 → 卡间 → 核间 → 单核流水；bound 诊断 |

---

## 5. 可直接借鉴的参考算子实现

### 5.1 mamba2_rmsnormgated —— 结构最相似的融合算子 ⭐⭐⭐

**路径**：`e:\Desktop\AscendC算子学习开发\ops-transformer\experimental\mamba\mamba2_rmsnormgated\op_kernel\CustVec.h`

该算子 = **RMSNorm + Sigmoid 门控 + 加权**（silu 门控 + RMS 归一化 + 权重乘），与本算子的"RMS 逆 + sigmoid 门控 + 加权折叠"结构高度同构。值得借鉴的工业化技巧：

1. **DBuff 双缓冲 + DEvent 双事件软件流水**：手工 MTE2→V→MTE3 三级流水，`in_ready/in_empty/out_ready/out_empty` 四事件控制，实现搬入/计算/搬出完全 overlap（本算子数据量小收益有限，但 outer 较大的隐藏测试点可参考）。
2. **归约 + Brcb 广播模式**：`ReduceSum` 得到每 group 均值后，用 `Brcb` 指令把标量均值广播回整个向量，再 `Muls` 完成归一化——避免标量逐元素乘。
3. **sigmoid 向量化序列**：`Muls(-1) → Exp → Adds(1) → Div`（比 exp 后手动除更规整）。
4. **常量复用**：`Duplicate` 生成全 1 向量供 Div 使用；`custparam.src0RepStride=0` 实现广播操作数。

配套工具头（**强烈建议通读**）：`e:\Desktop\AscendC算子学习开发\ops-transformer\experimental\mamba\common\tensorutils.h`
- `DBuff`/`TBuff`：双/三缓冲封装；
- `SEvent`/`DEvent`：基于 `AllocEventID` + `set_flag/wait_flag` 的硬件事件封装；
- `GM2UB`/`UB2GM`/`GM2UBPad`：`DataCopyParams` 参数化搬运封装；
- `GetHardEventByPipe`：pipe 对 → HardEvent 枚举映射表。

### 5.2 mhc 家族算子 —— mHC 架构官方工业实现 ⭐⭐

**路径**：`e:\Desktop\AscendC算子学习开发\ops-transformer\mhc\`

华为官方 mHC（Manifold-Constrained Hyper-Connections）系列算子，与本赛题同一架构体系（赛题即 mHC 的折叠算子）。可参考其 Host/Kernel 工程规范：

| 算子 | 说明 | 关键文件 |
| --- | --- | --- |
| `mhc_post` | mHC Post Mapping（x_{l+1} = H_res^T·x_l + h_out⊗H_post） | `op_kernel\mhc_post.cpp`（KERNEL_TYPE_AIV_ONLY 标准写法）、`op_kernel\arch35\mhc_post_regbase.h`（RegBase 范式） |
| `mhc_pre` | mHC Pre Mapping，含 Cube/Vector 分核 | `op_kernel\arch35\mhc_pre_vector_compute.h` 等 |
| `mhc_post` 的 tiling | arch22/arch35 双架构 tiling 分治 | `op_host\op_tiling\` 下多版本 tiling |
| `mhc_sinkhorn` | 双随机矩阵变换 | `op_kernel\arch35\mhc_sinkhorn.h` |

注：mhc 家族面向 950（arch35）/部分 A2（arch22），910B 上不能直接照搬指令（如 RegBase），但**工程结构与 Host 侧写法**值得对照。

### 5.3 其他高相关融合算子

| 算子 | 相关性 | 路径 |
| --- | --- | --- |
| compressor（含 RMSNorm/softmax/rope 融合） | RMSNorm 向量化实现（arch22 即 910B 架构） | `e:\Desktop\AscendC算子学习开发\ops-transformer\experimental\attention\compressor\op_kernel\arch22\rms_norm.h` |
| quant_compressor vf 系列 | vf_add/vf_mul/vf_softmax 基础向量算子 | `...\experimental\attention\quant_compressor\op_kernel\arch35\vf\` |
| asc-devkit reduce 样例 | 归约指令组合性能对比可复现 | `e:\Desktop\AscendC算子学习开发\asc-devkit\examples\01_simd_cpp_api\03_basic_api\01_memory_vector_compute\reduce\` |
| asc-devkit exp 样例 | Exp LUT/Taylor 两种模式 | `e:\Desktop\AscendC算子学习开发\asc-devkit\examples\01_simd_cpp_api\04_advanced_api\10_math\exp\exp.asc` |

---

## 6. 已验证经验与反模式汇总（v001-v033 迭代 + 官方文档互证）

### 已验证有效（保留）

| 技术 | 验证版本 | 效果 |
| --- | --- | --- |
| 1D TilingKey（仅 DT_X） | v021→v022 | Launch 底线 4.7us→2.5us 量级 |
| 全输入 DMA 预载入 + 标量 0 GM 访存 | v031 | 4.94→4.18us |
| 全生命周期单次 V_S 硬件屏障 | v030 | 多次屏障→1 次 |
| 5 向量融合流连续发射 | v027 | 首破 5us |
| 32 字节严格对齐（向量指令） | v029 | 规避 v028 的 RE 507035 |
| 64 float 跨度 UB 拓扑（0 bank conflict） | v031 | 规避 v032 冲突 |
| GlobalTensor::SetValue 输出 | v031/v033 | 裸指针写 GM 会引发总线同步等待（v032 教训） |
| HardEvent 精确同步 | v025 | 解决 v023 异步竞态 WA |

### 反模式（勿再踩）

| 反模式 | 后果 | 出处 |
| --- | --- | --- |
| 多维 TilingKey 模板矩阵 | 动态符号解析加重下发延迟 | v021 |
| 裸 `__gm__` 指针直写 GM | 绕过硬件空间描述符，总线同步等待 | v032 |
| 核函数分裂多个 Process 特化 | 机器码膨胀 → ICache 冷启动 miss | v032 |
| UB 紧缩布局（32B 边界紧贴） | UB crossbar bank conflict | v032 |
| 向量指令操作非 32B 对齐地址 | 硬件异常 RE 507035 | v028 |
| 无同步的裸 TBuf DMA | MTE2/V/S 异步竞态，读脏数据 | v023 |
| TPipe+TQue 固定同步成本（微小数据量） | 建立成本 > 收益 | v018 |
| 运行时 for 循环 + 动态基址 | 循环跳转与基址计算开销（但完全静态展开也因 ICache 失败，需平衡） | v031/v032 |

---

## 7. 建议行动清单（按预期收益排序）

1. **【立即】v033 基础上新增 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`**（§2.1）——文档高优先级 + 工业实现标配 + v031 明确缺失，预期压缩 Cube 核启动头开销。
2. **【立即】TPipe 外置到核函数入口**（§2.2）——触发编译器 Scalar 常量折叠，官方实测 -17% scalar 时间。
3. **【短期】TilingData 字段最小化 + 8 字节对齐重排**（§2.3）——消减 GM→栈 us 级拷贝。
4. **【短期】TinyH4 路径的 5 个 ReduceSum 换成 ReduceDataBlock(+标量加/ReduceRepeat)**（§2.4）——官方数据组合方案快 ~35%。
5. **【实验】4 个 gate 的 sigmoid 尝试向量化**：归约结果连续排放 → 向量 `Muls/Exp(expandLevel=0 LUT)/Adds/Div`（§2.5）——需重排 UB 布局，注意 32B 对齐。
6. **【实验】评估 `--cce-no-dcache-flush`** 在 910B 平台是否生效（§2.6）。
7. **【隐藏大 shape 点】若平台含 outer 较大或 H 较大的隐藏测试点**：参考 mamba2_rmsnormgated 的 DBuff/DEvent 软件流水（§5.1）与 reduction tiling 模板（§4.5），恢复多核行并行（当前 block_dim=1）。
8. **【分析】上传前后用 msOpProf 采集 PipeUtilization**（§4.6），确认 Scalar/Vector/MTE 占比，用数据决定下一步方向。

> 按版本管理规范，以上每项改动均应从 v033（或 v031）复制出新版本目录（如 `v034_aiv_only_kernel_type/`）单独验证，禁止覆盖历史版本。
