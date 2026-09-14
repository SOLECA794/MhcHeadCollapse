# 精度与性能实战配方

> **触发**：精度不对齐 → 读 §A；性能不达标 → 读 §B。每条均为真实任务实测结论。

---

## A. 精度配方

### A1 低精度 / 整数升 fp32 中间计算
- **触发**：fp16/bf16/int 的 N 元累加或算术。
- **症状**：溢出（fp16 max=65504 易溢出）、累积误差、舍入不齐。
- **修复**：先 `Cast→fp32` 算再 Cast 回。**升精度路径必须同步调整 tiling UB 预算**（多计 f32 中间 buffer）。
- **教训**：910B 向量单元不支持 bf16 直算，bf16 必升 f32。

### A2 Cast 边界（RoundMode 选择）
| 场景 | RoundMode |
|---|---|
| float→int（对齐 numpy.astype / TBE） | `CAST_TRUNC`（向零截断） |
| float→half / bf16 | `CAST_RINT`（就近偶数） |
| 浮点保持 | `CAST_RINT` |

> KG 给"推荐"但内置行为可能不同（如 KG 推 CAST_FLOOR/ROUND，实测内置 TBE 走 truncate-toward-zero）——**以内置实测为准**。

### A3 golden 方法学
- 整型 / 量化 exact-match 优先**用内置算子输出当 golden**（bit-exact 基准），把语义猜测风险降到接近 0；内置不支持 dtype 用 **f64 闭式 golden**（golden 算法与 kernel **不同**才能交叉验证）。
- **golden 输入与被测同源、精度高于被测**：从 kernel 实际接收的 dtype 出发、用 f64 算（否则 result≈0 处分支翻转产生 ~40% 假性误差）。
- 功能对齐 ≠ bit-exact。

### A4 空张量三层防御
aclnn `IsEmpty()` 早退（不 launch）+ tiling `inputNum==0→coreNum=1` + kernel `coreDataNum==0` 守卫空转。不能只在一层处理。

### A5 bf16 向量 Cast 全初始化
bf16 标量读取禁止只 `SetValue` 单元素后向量 Cast；`Duplicate<bfloat16_t>` 填满再 Cast（详见 api-pitfalls B 表）。

### A6 unsigned 高位逻辑掩码
unsigned dtype 升 int32 计算时，右移结果必须 AND 逻辑修正掩码剥高位符号填充；`ShiftRight<uint32>` 的 dst 显式 ReinterpretCast。

### A7 精度 bug 定位流程
1. **先做"误差分布分析 + 语义可行性论证"缩小范围再插桩**（例：线性公式不可能产生超指数序列 → bug 在 Cast/写回，搜索面缩小 2/3）。
2. 跨 dtype 测试**先核对输入文件 dtype/值域**再查 kernel（曾因 np.int32 读 int8 残留字节误诊"巨大差异"）。
3. CPU 仿真 UT **不可替代真机 bit-exact** 验证。

### A8 容差阈值（CANN Judge）
| dtype | 阈值 |
|---|---|
| FP16 | 2⁻¹⁰ |
| BF16 | 2⁻⁷ |
| FP32 | 2⁻¹³ |

判定：`MERE<thr 且 MARE<10×thr`。**MERE=mean、MARE=max**（曾记反）。

---

## B. 性能配方

### B1 msprof 实测驱动（最重要，凭直觉几乎必错）
- **触发**：性能不达标 / 要对齐 TBE。
- **做法**：`msprof --application=... --export=on` 导出 op_summary，先回答"基线慢在哪一阶段"、瓶颈类型（compute / memory / 流水线）。
- **教训**：gcd 双 buffer 直觉被 `aiv_vec 98.7%、copy 仅 0.07%` 证伪——流水最多消除 0.07% 串行，对 1.83× gap 无帮助。compute-bound 加流水无效，memory-bound 减计算无效。

### B2 多核 tiling（elementwise 模板）
- **simple-split**：核间满核均分 + 余数给前 tailBlockNum 核 + 32B 对齐 + 核内按 UB 容量 tile；host 视为一维向量按 totalLength 切分。
- 小 shape **收缩限核避免空转**（`if (tileDataNum >= inputNum) coreNum = 1`）。
- **直调 harness 的 tiling 必须与注册版逐字一致**，否则测的不是算子（曾因简化 tiling 测出假性"不达标"）。
- 广播 tiling **从右对齐**（trailing dims），覆盖 `(4,4)×(1,4)` 等；复用开源基线时 broadcast 语义 bugs 是高发区。

### B3 双缓冲 → 三缓冲（内存受限 elementwise 最直接杠杆）
- `BUFFER_NUM` 2→3 使 MTE2/V/MTE3 三段流水完全重叠，带宽利用率 86%→~100%，且**不损精度**（inplacersqrt 13.3µs 反超 TBE 14.14µs ~6%）。

### B4 测量口径（公平性）
- 端到端（易实现）+ **msprof Task Duration（kernel 级）** 两组数据都给；量级差异 >10X 时端到端即可，不必 profiling。
- 算子级对比用 **L1 `aic_metrics=PipeUtilization`**，非 L2（L2 带 stack+memory 失真）。
- elementwise 对比**必须超 L2 大 shape**（如 64M 元素），否则 cache 命中高估（8M 曾测出 5.6TB/s 假象超 HBM 带宽）。
- 性能**多轮中位数**抗噪声（NPU 温度/频率/功率 Warning 致单次超标）。

### B5 带宽上界论证
- 写密集算子（无输入读取）性能上界 = **MTE3 写带宽**（arange f32_large 7.06µs，`mte3_ratio≈0.66` 主导）。
- 性能基线对比前先确认基线 kernel 可用性；不可用转向任务书允许的分析结论。

### B6 归约必须向量化
- 标量逐元素扫描在 10⁵ 量级吃掉全部性能预算（bincount 负数 offset 标量扫描 +60us 超 10X 红线，改向量比较+ReduceSum 修复）。
- `ReduceMin`/`ReduceMax` A2 仅支持 float（int32 需 Cast）。

### B7 优化实验纪律
- 优化用**独立 commit**，实测证伪后**诚实回滚**，不留半成品（gcd 四条优化全证伪，每条 `git checkout` 回滚）。
- 提前退出类优化先**采样真实输入收敛分布**再调参（perf 输入不代表验收数据）。
- 性能 gap 未缩小就标未达标，不把"卡配置"当待办。

### B8 共享 NPU 环境
- `npu-smi` 选 **HBM 占用最低**卡 + `ASCEND_RT_VISIBLE_DEVICES` 隔离 + 测试重试 + 多套交叉验证；先排除环境竞争再定 bug。
- msprof 输出目录用项目内私有目录 + `chmod 700`。
