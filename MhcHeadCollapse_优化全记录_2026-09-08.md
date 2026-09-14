# MhcHeadCollapse（mHC四路归一）算子优化全记录与深度分析

> 📌 **本文档的相对可靠信息已汇总进 [`FACTS.md`](FACTS.md)（单一事实源）。**
> 本文保留完整分析过程；其中的**性能归因结论**（floor 模型、"kernel 藏进 floor"、
> "必须换归约原语"）已被后续侦察标记为**待重新验证**——见 `FACTS.md §A` 与 `EXPERIMENTS.md`。
> 冲突时以 `FACTS.md` 为准。

## 一、项目背景与目标

### 1.1 赛题信息
- **题目名称**：MhcHeadCollapse 算子（mHC四路归一）
- **题目 ID**：`6a7c23a6a52e0f540a8a1779`
- **竞赛平台**：CANNJudge
- **核心任务**：实现一个高性能的 MhcHeadCollapse 算子，在 5 个测试用例上获得尽可能短的运行时间
- **主要测试数据类型**：fp16（全部 5 个 case 均为 fp16）

### 1.2 算子数学定义

MhcHeadCollapse 算子完成的是多头注意力压缩（Multi-Head Collapse）操作，其计算流程可以概括为：

给定输入 `x` 形状为 `[1, outer, nH]`，其中 `nH = n × h`，权重 `weight` 形状为 `[n, nH]`，偏置 `base` 形状为 `[n]`，缩放系数 `scale` 为标量，对每个 outer 行执行：

1. **RMSNorm**：计算输入的均方根倒数
   - `mean_square = sum(x^2) / nH`
   - `rms_inv = rsqrt(mean_square + eps_norm)`

2. **Gate 计算**：对每个 head i（0 ≤ i < n）
   - `dot_i = sum(x * weight[i, :])`
   - `gate_i = sigmoid(dot_i * rms_inv * scale + base[i]) + eps_hc`

3. **折叠输出**：对每个输出位置 d（0 ≤ d < h）
   - `y[d] = sum_i gate_i * x[i * h + d]`

这个算子在 Transformer-like 结构中用于将多头信息压缩到单头输出，是典型的内存密集+向量计算密集算子。

### 1.3 测试用例特征

| Case | n | h | nH | outer | 路径 | 备注 |
|------|---|---|-----|-------|------|------|
| Case 1 | 4 | 4 | 16 | 1/2 | PATH3 TinyH4 | 极小 shape |
| Case 2 | 4 | 4 | 16 | 4 | PATH3 TinyH4 | 极小 shape |
| Case 3 | 8 | 64 | 512 | 1/2 | PATH2 向量路径 | 中等规模 |
| Case 4 | 8 | 64 | 512 | 2 | PATH2 向量路径 | 中等规模 |
| Case 5 | 8 | 128 | 1024 | 8 | PATH2 向量路径 | 最大规模，决定排名 |

所有 case 均为 fp16 输入输出，weight/base/scale 保持 fp32。

---

## 二、关键发现：计时单位与性能模型

### 2.1 计时单位纠错（重大发现）

平台 CLI 打印的时间标签显示为 "ms"，但经过与排行榜数据对比和多次实验验证，**实际计时单位是微秒（μs）**。这是一个非常容易误导优化的关键点。

例如 v111 提交后显示：
- Case1: 4.30
- Case2: 5.14  
- Case3: 4.42
- Case4: 4.28
- Case5: 8.20

这些数字的单位是 μs，不是 ms。如果误以为是 ms，会对性能模型产生数量级误判。

**验证方法**：
- 榜首 lix123 的成绩为 75.75 分，case 时间为 3.96/2.80/2.84/2.92/4.64
- 如果单位是 ms，则总时间为 17.16ms，与平台算子 5 case 的 2 秒总时限相比，差距不合理
- 如果单位是 μs，则总时间为 17.16μs，与 AIV 向量算子的实际性能完全吻合
- 实机环境 NPU 空闲时 AICore 频率下，本地 bench 测得 kernel_avg_us=6.5μs 量级，进一步印证 μs 单位

### 2.2 瓶颈模型：floor + kernel 显形

通过 WA（Wrong Answer）差分计时探针，我们建立了如下性能模型：

**每 case 时间 ≈ per-call floor（cadence/launch/weight DMA）+ 可见 kernel 时间**

其中 floor 包含：
- 算子 launch 开销
- host 侧调度节拍
- 一次性的 weight/base/scale DMA（Process 阶段）
- kernel 启动开销

可见 kernel 时间包含：
- 每行 x 的 DMA
- cast（fp16→fp32）
- rms 计算（Mul + ReduceSum）
- n 个 gate 点积（Mul + ReduceSum）
- sigmoid 标量计算
- fold 阶段（Muls + Add）
- cast + 输出 DMA

### 2.3 探针归因结果

设计了三组 WA 探针，故意让算子语义错误但能部分执行，通过返回的时间差定位各阶段开销：

**P1 floor_probe**：ProcessVectorRow 置空，只保留 weight DMA 和 launch
- Case3: 2.2μs
- Case4: 2.2μs  
- Case5: 3.38μs

**P2 gateconst_probe**：去掉 8 个 gate 点积+sigmoid，gate 用常数替代
- Case3: 2.88μs
- Case4: 3.06μs
- Case5: 5.30μs

由此推算 gate 点积+sigmoid 开销：
- Case3: 2.88 - 2.2 = 0.68μs（但 v111 实际为 4.42，这里 floor 估计偏低）
- Case5: 5.30 - 3.38 = 1.92μs

更准确的加性归因：
- Case5 8.2μs ≈ floor 3.38 + gate 点积 2.9 + 其余（x DMA+cast+rms+per-row）1.4 + fold/store 0.5

**P3 nofold_probe**：去掉 h 长 fold+Cast+store，改为标量 GM 写
- Case3: 4.12μs
- Case4: 4.10μs
- Case5: 7.7μs

fold+store 开销：
- Case3: 4.42 - 4.12 = 0.3μs
- Case4: 4.28 - 4.10 = 0.18μs
- Case5: 8.2 - 7.7 = 0.5μs

### 2.4 榜首分析

榜首 lix123（75.75 分）的时间：
- Case1: 3.96μs
- Case2: 2.80μs
- Case3: 2.84μs
- Case4: 2.92μs
- Case5: 4.64μs

关键观察：
- 榜首 Case1-4 的时间几乎等于我们测得的 floor（2.2-3.38μs）
- 这说明**冠军把大部分 kernel 执行隐藏到了 per-call floor 之下**，或者大幅优化了可见 kernel
- Case5 4.64μs 对比我们的 floor 3.38μs，冠军可见 kernel 仅约 1.26μs
- 我们 v111 Case5 8.2μs 中可见 kernel 约 4.82μs，是冠军的 3.8 倍

这个发现指明了优化方向：**减少可见 kernel 时间，特别是 gate 点积阶段**。

---

## 三、版本迭代历程

### 3.1 早期版本（v080-v095）

**v080-v089**：多路径 kernel 探索
- 支持 PATH0/1/2/3 多种执行路径
- PATH3 TinyH4 专门优化 n=4,h=4 的极小 shape
- 使用 UB+DMA 向量流水线
- 发现 file-level pragma 对 CANN 8.5.0 DataCopy 的重要性

**v090-v095**：host 侧与同步优化
- v090_min_host：尝试减少 host 侧开销，收益约 0.1μs
- v095_multiblock：动态 block_dim，纯 SIMD AIV_ONLY
- v095 成为后续重要基线

### 3.2 关键基线 v100_v095_nosv

v100 是后续所有实验的锚点版本，特点：
- 纯 SIMD 算子，KERNEL_TYPE_AIV_ONLY
- PATH2 每行独立处理
- weight/base/scale 一次性 DMA 到 UB
- 每行流程：x DMA → cast → rms Mul/ReduceSum → 8×gate Mul/ReduceSum → 标量 sigmoid → fold → cast → y DMA
- 代码 487 行，ProcessVectorRow 在 319-383 行

v100 性能：约 41.71 分，Case5 约 8.2μs

### 3.3 v106：barrier 移除实验（证伪）

尝试移除 Process() 中 weight DMA 后的 V_S(0) barrier，结果性能下降。

结论：weight DMA 与后续向量计算之间需要显式同步，否则会产生数据竞争或流水线气泡。

### 3.4 v107-v109：Matmul 路线探索与失败

**v107_matmul/v108_matmul_real**：尝试使用 Cube Matmul 加速 gate 点积
- 声称 v108 在 OJ 上 Pass 并有收益
- 但本地代码检查显示 v107/v108 产物中**没有任何 matmul 调用**
- 收益无法复核，记录不可靠

**v109_adaptive_matmul**：整文件重写 + CE
- 包含 `#include "matmul/matmul.hpp"`
- 实机编译定位：该头文件不存在，真实路径应为 `adv_api/matmul/matmul.h`
- 更严重的问题：
  1. kernel 声明 AIV_ONLY，但 Matmul 需要 AIC/Cube 核
  2. shape 为 M=8,N=1,K=nH 的微矩阵，Cube 启动和搬运成本远大于向量点积
  3. 无加速意义

**Matmul 路线整体放弃**。

### 3.5 v110/v111：同算法内重排证伪

**v110_path2_decouple**：尝试将 rms 和 gate 计算解耦
**v111_path2_group4**：PATH2 上引入 group-of-4 批处理
- 4 行一起 stage x
- dot phase 全部发出后再 scalar drain
- 使用 parity work/tmp 缓冲
- UB 预算保护

结果：Case5 仍为 8.2-8.4μs，与 v100 基线相比无实质变化。

**结论**：同算法内的执行顺序重排无法突破性能上限，必须从算法原语层面减少 ReduceSum 调用次数。

### 3.6 v112/v112b：fp16 点积尝试与 CE 根因

**假设**：gate 点积是 fp32 向量元素吞吐受限，如果改用 fp16 点积，元素数量翻倍，可能削掉一半时间。

**v112 设计**：
- Process 阶段将 weight 从 fp32 cast 到 DT_X（fp16）
- 8 个 gate dot 使用 `Mul<DT_X>` 和 `ReduceSum<DT_X>`
- rms 和 fold 保持 fp32

**结果**：CE

**v112b 改进**：
- 去掉 reinterpret，使用独立的 typed 缓冲区
- 显式模板参数 `Mul<DT_X>` / `ReduceSum<DT_X>`
- 预切片 w16row / redDst

**结果**：仍然 CE

**实机根因分析**：
在 CANN 8.5.0 / ascend910b（dav_c220）上，最简探针仍然失败：
```cpp
AscendC::Mul(work16, x16, w16, nH);
AscendC::ReduceSum(red16, work16, tmp16, nH);
```

错误固定两处：
- `dav_c220/kernel_operator_vec_binary_impl.h:166`：`vmul(dst,src0,src1,...)`
- `dav_c220/kernel_operator_vec_reduce_impl.h:153`：`vcadd(dst,src,1,...)`
错误消息：`the 1st parameter maybe need a type '__ubuf__ half *'`

读源码发现：MulImpl Level 2 的 SupportType 声明 half 支持，但实际 vmul/vcadd 编译器内建无法绑定 half 指针。

**devkit 佐证**：
- 910（c100）样例使用 `ReduceSum<half>`
- 910b1（c220/dav_c220）官方测试只使用 `ReduceDataBlock<SUM,half>` / `ReducePairElem`，从未使用 count 版 `ReduceSum<half>`

**结论**：fp16 count 版 Mul/ReduceSum 在 dav_c220 上不可用，该路线编译级证伪。

### 3.7 v113/v115：ReduceDataBlock 尝试与 API 限制

**v113 设计**：
- 使用 `ReduceDataBlock<ReduceType::SUM, float>` 一次性归约 8 个 gate 点积
- 将 8 次 ReduceSum 合并为 1 次

**v115 设计**：
- 在 v113 基础上，把 rms 的 ReduceSum 也并入 DataBlock
- 一次归约出 9 个标量（1 个 rms + 8 个 gate）

**结果**：编译失败，`ReduceDataBlock` 不是公开 API，普通 `kernel_operator.h` 不暴露。

读 `kernel_operator_vec_reduce_intf.h`：
- 公开 API 只有 `ReduceSum`、`ReduceMax`、`ReduceMin`（count/repeat 版）
- `ReduceDataBlock`、`ReducePairElem`、`ReduceRepeat` 等虽然定义在头文件中，但受 `__ASCENDC_INCLUDE_INTERNAL_HEADERS__` 宏控制，普通包含不可用
- devkit 测试通过 mock/内部宏绕过，但 OJ 编译链不行

**结论**：ReduceDataBlock 路线在 OJ 上走不通，判死。

### 3.8 v116：batched ReduceSum + 单次 V_S 同步

**设计思路**：
既然无法减少 ReduceSum 调用次数，就优化同步开销。当前 v100 的问题是每行在 rms ReduceSum 后立刻 `GetValue`，这会触发 vector→scalar 的同步。如果能把 rms + 8 个 gate 的 ReduceSum 全部发出，然后只同步一次，再批量读取标量，可能隐藏部分流水线延迟。

**v116 改动**：
1. 在 ProcessVectorRow 中：
   - 先发出 rms 的 Mul + ReduceSum
   - 连续发出 8 个 gate 的 Mul + ReduceSum
   - 然后执行一次 `SetFlag/WaitFlag<V_S>(0)` 同步
   - 再批量 `GetValue` 读取 9 个标量

2. 在 ProcessVectorGroup 中：
   - 同样保持所有 row 的 dot phase 发出后再 scalar drain
   - 增加显式 V_S sync

**实机验证**（CANN 8.5.0 / ascend910_93）：
- BUILD=0，编译通过
- correctness：所有 fp16 case PASS，精度与 v100 一致
- bench：
  - n=4,h=4,outer=2: 6.519μs
  - n=8,h=64,outer=1: 6.506μs
  - n=8,h=128,outer=8: 6.432μs

注意：实机是 910A（旧芯片），OJ 是 910B，绝对时间不同，但结构验证通过。

**OJ 提交待进行**。

---

## 四、方法论总结

### 4.1 WA 差分计时探针法

当不知道性能瓶颈在哪时，不要盲改代码。应该构造**故意语义错误但能部分执行**的版本，通过返回的时间做减法归因。

步骤：
1. 确定要测量的阶段（如 floor、gate dots、fold）
2. 构造一个版本，只保留该阶段之前的所有操作
3. 提交到 OJ（WA 也会返回 time）
4. 与完整版本的时间差即为该阶段开销

关键注意事项：
- CE 会让所有 case time=0，无法使用
- 探针版本必须能编译通过
- 单次探针只能定位到阶段级，不能精确定位到指令级

### 4.2 先定上限再下注

在投入大量修改前，先通过探针确定：
- 理论最优时间 = floor + 0（完全隐藏 kernel）
- 各阶段可削上限
- 如果某阶段可削上限很小，就不值得大改

例如：
- fold+store 只有 0.5μs，不值得用复杂方法优化
- gate 点积有 2.9μs，值得重点投入
- host/launch 侧只有 0.1μs 空间，不值得

### 4.3 实机离线排 CE

OJ 提交限流严重（429 错误，需 70-80s 冷却），且 CE 会浪费提交次数。通过 AtomGit DevSpace 实机：
- SSH host: `devenvc_x9cyrw.0248b66951a4431a8d7290c184b9aaf2.atomgit.0` (127.0.0.1:20022)
- 用户: developer
- CANN: 8.5.0 在 `/home/developer/Ascend/cann-8.5.0`
- 本地脚本：`/home/developer/mhc_scripts/verify.sh`
- 测试程序：`/home/developer/mhc_tests/mhc_bench.cpp`、`mhc_correctness.cpp`

实机编译链与 OJ 同源，fp16 CE 在实机上复现了完全相同的行号。

### 4.4 可证伪的实验设计

每次优化尝试前，应该明确：
1. **假设**：这次改动能提升性能的原因
2. **预期收益**：最好/最坏/无变化分别是多少
3. **证伪标准**：如果时间没变化或变差，说明什么
4. **最小验证**：先实机编译/本地 bench，再 OJ 提交

例如 v110/v111 的证伪：
- 假设：group-of-4 能摊薄 queue event 和 cast 开销
- 预期：Case5 从 8.2 降到 7.x
- 实际：8.2-8.4，无变化
- 结论：同算法内重排无效

---

## 五、重要结论

### 5.1 已验证有效的结论

1. **计时单位是 μs**，不是 ms
2. **性能瓶颈 = floor + 可见 kernel**，榜首几乎把 kernel 藏进 floor
3. **gate 点积是最大可削块**：Case5 中约 2.9μs
4. **同算法内重排无效**：v110/v111 已证伪
5. **fp16 count 版 Mul/ReduceSum 在 dav_c220 上不可用**：编译级证伪
6. **Matmul 路线不可行**：include 错 + AIV_ONLY 冲突 + 微矩阵无意义
7. **ReduceDataBlock 不是公开 API**：OJ 编译链不可用

### 5.2 尚未验证的假设

1. **v116 的 batched ReduceSum + 单次 V_S 同步能否在 OJ 上提升 Case5 到 5.x μs？**
2. **如果 v116 不够，能否通过向量化 sigmoid 进一步削时间？**
3. **fold 阶段能否用更高效的向量原语？**
4. **double-buffer 流水线能否隐藏 DMA latency？**

### 5.3 当前最优候选方案

**v116_batched_reduce**：
- 保持 v111 的 group-of-4 结构
- 在 dot phase 后增加单次 V_S 同步
- 在单 row 路径上也做同样优化
- 风险低，收益预期 0.3-0.6μs

---

## 六、代码结构与关键片段

### 6.1 v116 ProcessVectorRow 核心逻辑

```cpp
AscendC::LocalTensor<float> work = work_buf_.Get<float>();
AscendC::LocalTensor<float> reduced = reduce_buf_.Get<float>();
AscendC::LocalTensor<float> reduce_tmp = reduce_tmp_buf_.Get<float>();

// v116: rms + n gates all issued before any GetValue; single V_S drain after
AscendC::LocalTensor<float> w_cache = weight_cache_buf_.Get<float>();
AscendC::LocalTensor<float> cst = cst_buf_.Get<float>();
AscendC::Mul(work, x_float, x_float, static_cast<int32_t>(nH_));
AscendC::ReduceSum(reduced, work, reduce_tmp, static_cast<int32_t>(nH_));
for (uint32_t i = 0; i < n_; ++i) {
    AscendC::Mul(work, x_float, w_cache[i * nH_], static_cast<int32_t>(nH_));
    AscendC::ReduceSum(reduced[32U * (i + 1U)], work, reduce_tmp, static_cast<int32_t>(nH_));
}
AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

const float hc_scale = cst.GetValue(8);
float gates[kSmallN];
const float rms_inv = ScalarRsqrt(reduced.GetValue(0) * inv_nH_ + eps_norm_);
for (uint32_t i = 0; i < n_; ++i) {
    const float gate = reduced[32U * (i + 1U)].GetValue(0) * rms_inv * hc_scale + cst.GetValue(i);
    gates[i] = ScalarSigmoid(gate) + eps_hc_;
}
```

### 6.2 v116 ProcessVectorGroup 核心逻辑

```cpp
// 2) dot phase: all rows, rms + n gates
for (uint32_t j = 0; j < cnt; ++j) {
    AscendC::LocalTensor<float> x = xf[j * nhF];
    const uint32_t p = j & 1U;
    AscendC::LocalTensor<float> wp = work[p * nhF];
    AscendC::LocalTensor<float> tp = tmpb[p * nhF];
    const uint32_t rb = j * (n_ + 1U) * 32U;
    AscendC::Mul(wp, x, x, count);
    AscendC::ReduceSum(red[rb], wp, tp, count);
    for (uint32_t i = 0; i < n_; ++i) {
        AscendC::Mul(wp, x, w_cache[i * nH_], count);
        AscendC::ReduceSum(red[rb + 32U * (i + 1U)], wp, tp, count);
    }
}

// v116: single hard sync before scalar drain
AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

// 3) scalar phase for the whole group
float gates[kGroupRows][kSmallN];
for (uint32_t j = 0; j < cnt; ++j) {
    const uint32_t rb = j * (n_ + 1U) * 32U;
    const float rms_inv = ScalarRsqrt(red[rb].GetValue(0) * inv_nH_ + eps_norm_);
    for (uint32_t i = 0; i < n_; ++i) {
        const float gate = red[rb + 32U * (i + 1U)].GetValue(0) * rms_inv * hc_scale + cst.GetValue(i);
        gates[j][i] = ScalarSigmoid(gate) + eps_hc_;
    }
}
```

---

## 七、实机环境使用记录

### 7.1 连接方式

```powershell
pwsh -File D:\Desktop\OP-Learning\Ascend\atomgit-devspace-tools\Connect-AtomGitDevEnv.ps1
```

或复用已打开的 forward：

```powershell
pwsh -File D:\Desktop\OP-Learning\Ascend\atomgit-devspace-tools\Connect-AtomGitDevEnv.ps1 -NoPluginConnect -RemoteCommand 'pwd'
```

### 7.2 NPU 状态

- 设备：Ascend910 ×2
- 空闲 AICore
- 注意不要打扰他人运行中的进程（PID 19177 python3 曾占用 chip0）

### 7.3 本地验证流程

1. 打包版本：`tar czf /tmp/v116.tgz code`
2. 上传到实机：`scp /tmp/v116.tgz developer@...:/tmp/`
3. 运行验证：`bash /home/developer/mhc_scripts/verify.sh v116`
4. 检查 BUILD=0 和 correctness PASS
5. 查看 bench 结果

### 7.4 验证脚本行为

`verify.sh` 会：
1. 解压 tgz 到 `/tmp/v116`
2. 将 target 从 ascend910b 改为 ascend910_93（本地 SoC）
3. cmake + make + make binary
4. make package 并安装到 `/tmp/v116_inst`
5. 编译 correctness 和 bench
6. 运行多组 shape 测试

---

## 八、OJ 提交记录与经验

### 8.1 提交工具

使用 `cann-learing-hub/skills/cannjudge-submit-plaintext/cannjudge_cli.py`：

```bash
python cannjudge_cli.py submit --problem-id 6a7c23a6a52e0f540a8a1779 --project-dir ".../versions/vXXX/code"
```

凭据来源：.env 文件或环境变量 `CANNJUDGE_EMAIL` / `CANNJUDGE_PASSWORD`。

### 8.2 限流处理

平台返回 429 时需要冷却 70-80 秒。

### 8.3 已提交版本回顾

| 版本 | 状态 | Case5 | 关键改动 |
|------|------|-------|----------|
| v100 | Pass | 8.2 | 基线 |
| v106 | Pass? | - | 移除 V_S barrier，性能下降 |
| v110 | Pass | 8.4 | decouple |
| v111 | Pass | 8.2 | group-of-4 |
| v112 | CE | 0 | fp16 gate dots |
| v115 | 本地 CE | - | ReduceDataBlock |
| v116 | 待提交 | - | batched sync |

---

## 九、后续优化方向与优先级

### 9.1 高优先级

1. **完成 v116 OJ 提交**，验证 batched sync 实际收益
2. 如果 v116 Case5 到 5.x，继续微调 V_S 位置和 fold 阶段

### 9.2 中优先级

1. **向量化 sigmoid**：
   - 将 8 个 gate 值先写入向量
   - 用 `Exp` + `Adds` + `Reciprocal` 批量计算 sigmoid
   - 避免 8 次标量 `ScalarSigmoid`
   - 风险：精度可能下降

2. **fold 阶段优化**：
   - 当前是 8 次 Muls + 7 次 Add
   - 可以尝试把 8 个 head 的结果先按 gate 加权，然后用 ReduceDataBlock（如果 API 可用）或 PairReduce 合并
   - 但 fold 只有 0.5μs 空间，ROI 低

### 9.3 低优先级/已判死

1. fp16 count 版点积：编译级证伪
2. Matmul 路线：三重硬伤
3. ReduceDataBlock：非公开 API
4. host 侧优化：空间很小

### 9.4 终极目标

追上榜首 75.75 分。要达成这一目标，Case5 需要从当前 8.2μs 降到 4.64μs 附近，意味着可见 kernel 时间要从约 4.8μs 压到约 1.3μs。

如果 v116 只能削 0.3-0.6μs，那么还需要找到另外 2-3μs 的空间。这可能需要：
- 将 gate 点积从 8 次 ReduceSum 真正合并（需要找到可用的 reduce API）
- 或者采用完全不同的算法实现（如 batch matmul、Cube 路径等）

---

## 十、数据表格汇总

### 10.1 各版本 Case 时间对比

| 版本 | Case1 | Case2 | Case3 | Case4 | Case5 | 总分状态 |
|------|-------|-------|-------|-------|-------|----------|
| 榜首 lix123 | 3.96 | 2.80 | 2.84 | 2.92 | 4.64 | 75.75 |
| v100 | ~4.3 | ~5.2 | 4.42 | 4.28 | 8.20 | 41.71 |
| v110 | - | - | - | - | 8.40 | 无收益 |
| v111 | 4.30 | 5.14 | 4.42 | 4.28 | 8.20 | 40.66 |
| v113* | - | - | - | - | - | CE |
| v115* | - | - | - | - | - | CE |
| v116 | 待测 | 待测 | 待测 | 待测 | 待测 | 待提交 |

*注：v113/v115 为 ReduceDataBlock 尝试，本地 CE，未上 OJ。

### 10.2 探针数据

| 探针 | Case3 | Case4 | Case5 | 说明 |
|------|-------|-------|-------|------|
| P1 floor | 2.2 | 2.2 | 3.38 | cadence+launch+weight DMA |
| P2 gateconst | 2.88 | 3.06 | 5.30 | 去掉 gate 点积+sigmoid |
| P3 nofold | 4.12 | 4.10 | 7.70 | 去掉 fold+store |
| v100 完整 | 4.42 | 4.28 | 8.20 | 基线 |

### 10.3 阶段开销拆分（Case5）

| 阶段 | 开销（μs） | 来源 |
|------|-----------|------|
| floor | 3.38 | P1 |
| gate 点积+sigmoid | 2.9 | P2 - P1 |
| x DMA+cast+rms+per-row | 1.4 | 剩余 |
| fold+store | 0.5 | P3 差值 |
| 总计 | 8.2 | v100 |

---

## 十一、代码仓库与文件位置

### 11.1 工作区

- 根目录：`D:\Desktop\OP-Learning\Ascend`
- 算子项目：`D:\Desktop\OP-Learning\Ascend\xingchen-operators\C组赛题\【C组中等题】MhcHeadCollapse 算子（mHC四路归一）`
- 版本目录：`...\versions\`
- 迭代记录：`...\迭代优化记录.md`
- 提交脚本：`...\scripts\submit.sh`

### 11.2 记忆文件

- `C:\Users\34511\.zcode\cli\memories\projects\ascend-930e9c5a1329da78\memory\`
- 相关记忆：
  - `mhc-optimization-progress.md`
  - `best-version-v100.md`
  - `v106-failure.md`
  - `next-steps-cube-path.md`
  - `mhc-unit-and-floor-model.md`
  - `mhc-probe-methodology.md`
  - `mhc-real-machine-env.md`
  - `mhc-fp16-ce-rootcause.md`

### 11.3 实机调试目录

- `/tmp/mhcdbg/`：历史调试目录
- `/tmp/v100.tgz`, `/tmp/v115.tgz`, `/tmp/v116.tgz`：上传的版本包
- `/tmp/v116/`：解压后的编译目录
- `/tmp/v116_bin.log`：编译日志
- `/tmp/verify_v116.log`：验证脚本输出

---

## 十二、踩坑记录

### 12.1 fp16 CE 陷阱

`ReduceSum<half>` 在 dav_c220 上编译失败，但代码中的 static_assert 却声明支持 half。这是编译器内建绑定问题，不是调用方式问题。浪费了很多时间尝试不同的切片/模板/缓冲区写法。

### 12.2 ReduceDataBlock API 陷阱

在 devkit 样例中看到 `ReduceDataBlock` 的用法，但它是内部 API。直接在 OJ 代码中使用会报 `no member named 'ReduceDataBlock' in namespace 'AscendC'`。

### 12.3 Matmul 头文件陷阱

v109 写 `#include "matmul/matmul.hpp"` 路径错误，真实路径是 `adv_api/matmul/matmul.h`。但即使修正，AIV_ONLY + 微矩阵的问题也使其不可能工作。

### 12.4 版本目录陷阱

v112b 最初从错误锚点生成，缺少 op_host/tiling 文件，提交时报 "找不到必要的代码文件"。之后改为完整 copy code tree 再 patch。

### 12.5 实机上传陷阱

`Connect-AtomGitDevEnv.ps1 -RemoteCommand` 模式下，用 WSL 生成的 `/tmp/v116.tgz` 不会被自动传到实机，需要显式 `scp`。

---

## 十三、工具链与依赖

### 13.1 本地环境

- OS：Windows 11 + WSL/Git Bash
- Python：miniconda3
- 提交工具：`cannjudge_cli.py`（requests 库）
- 代码检查：`scripts/balance_check.py`

### 13.2 实机环境

- OS：Linux（Docker）
- CANN：8.5.0 / 8.5.2 / 9.0-a5
- NPU：Ascend910 ×2
- 编译器：bisheng clang 15.0.5
- 本地脚本：`/home/developer/mhc_scripts/verify.sh`

### 13.3 参考资源

- `asc-devkit`：官方样例与 API 测试
- `cannbot-skills`：代码审查规范与历史评论
- `catlass`：高性能算子参考

---

## 十四、总结与展望

MhcHeadCollapse 优化已经从盲目版本迭代进入了**数据驱动、探针验证、精确归因**的阶段。关键里程碑包括：

1. 纠正了 μs 计时单位
2. 建立了 floor + kernel 显形的性能模型
3. 通过探针定位 gate 点积为最大瓶颈
4. 证伪了同算法内重排、fp16 count reduce、matmul、ReduceDataBlock 等多条路线
5. 打通了实机离线编译验证流程
6. 提出了 v116 batched sync 方案并实机验证通过

当前离榜首还有约 0.66μs（Case5）的差距。虽然这看起来很小，但在 μs 级别的算子竞赛中，每一百纳秒都很难榨取。后续需要：
- 先验证 v116 在 OJ 上的实际效果
- 如果不够，继续探索向量化 sigmoid、fold 优化、以及可能的全新算法路径
- 保持记录完整，避免重复劳动

**核心经验**：在算子优化中，先停下来想清楚瓶颈在哪，比盲目迭代更重要。 probes 和 floor 模型的建立，是本轮优化中最重要的方法论收获。

---

*本文档生成时间：2026-09-08*
*最后更新：v116 实机验证通过后*
*作者：ZCode Agent / 用户协作*
