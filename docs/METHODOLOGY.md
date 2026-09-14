# 观测基础设施方法论 — msprof 显微镜 / 差分对照 / NOP 标定

> 来源：《建议分析.md》三短板破局方案，**2026-09-14 已实测验证可行性**（验证记录见文末）。
> 本文档是方法论权威版；`scripts/msprof_report.py` 是短板一的落地工具。

## 一、msprof 显微镜（瓶颈归因，替代探针差分法）

### 采集命令（固定用法）

```bash
# 8.5 环境（OJ 同版本）
export ASCEND_HOME_PATH=/tmp/cann85/cann-8.5.0  # + ENVIRONMENT.md §八 的完整初始化
MSPROF=/tmp/cann85/cann-8.5.0/tools/profiler/bin/msprof
$MSPROF --output=./out --ai-core=on --aic-metrics=PipeUtilization \
  --application="<测试程序> <args>"
```

### 关键 CSV 与字段（实测存在的 45+ 列中的核心 7 个）

| CSV | 字段 | 含义 |
|---|---|---|
| op_summary_*.csv | `aiv_time(us)` | AIV 总耗时 |
| | `aiv_vec_ratio` / `aiv_scalar_ratio` / `aiv_mte2_ratio` / `aiv_mte3_ratio` | 各流水 cycle 占比 |
| | `aiv_icache_miss_rate` | 指令缓存缺失率 |
| op_statistic_*.csv | 按 op 聚合统计 | |
| task_time_*.csv | task 级时间线 | |

### 瓶颈判定阈值表

| 指标 | 阈值 | 结论 | 优化动作 |
|---|---|---|---|
| `aiv_scalar_ratio` | ≥20% | **Scalar 受限** | 精简 TilingData、标量展开、减少循环 |
| `aiv_mte2_ratio` | ≥50% | **MTE2 Bound** | 对齐、连续化、蹭 L2、双缓冲 |
| `aiv_mte3_ratio` | ≥30% | MTE3 写受限 | 写合并、减少回写 |
| `aiv_vec_ratio` | ≥60% | 计算充分 | 向量已吃满，优化在别处 |
| `aiv_icache_miss_rate` | ≥10% | 指令缓存缺失 | 代码瘦身、减少分支 |
| mte2+mte3 | ≥70% | 访存受限 | 数据布局、减少搬运 |

### 使用纪律

- **每次优化前先跑显微镜，再动手**；探针法降级为"精确定位"的验证手段
- 注意 msprof 单次采样是热身曲线切片（PITFALLS/ENVIRONMENT 已记录）——对比实验 ≥8 次取中位
- 解析：`python3 scripts/msprof_report.py <msprof输出目录>`

## 二、差分对照法（精度 bug 定位，替代盲猜）

修"向量化 vs 标量位级不一致"类 bug 的标准流程：

1. 写标量参考版（对齐 PyTorch 参考）
2. 写向量化版（被测实现）
3. 每步计算后 dump 中间量（UB → DataCopy → GM → host 侧落盘）
4. 逐元素对比，找**第一个出现差异的位置**——那里就是根因

位级差异只有三个来源（按概率排序）：

| 来源 | 机制 | 检测 |
|---|---|---|
| 舍入模式不同 | 硬件原语固定舍入 vs 手写 Horner 每步路径不同 | 差异是否在第一次乘法后出现 |
| 累加顺序不同 | 向量树状累加 vs 标量线性累加，FP16 大数吃小数 | 拆归约步骤逐元素比部分和 |
| 快速数学近似差异 | 多项式系数/阶数不同，FP16 尾数 10 位放大 | 换硬件原语看误差是否消失 |

**Group sigmoid bug（3.203e-03）首选动作**：把手写 Horner exp 换成硬件 `Exp` 原语——
若误差消失则是近似路径问题；若仍在，用差分对照法定位。参考 Tangefly GeluV2 的
`x/(1+exp(-inner))` 除法形式 + `Mins(x²,100)` 钳制（cann-ops-competitions 官方仓）。

## 三、NOP 标定法（OJ 计时口径探测，5 次提交换一个确定性答案）

kernel 里插入可配置 NOP 循环（空转 N cycle），提交 N=0/100/500/1000/5000 五个版本：

| OJ 时间表现 | 结论 |
|---|---|
| 严格线性 = 基线 + N×cycle | OJ 测纯 kernel 时间，口径无猫腻 |
| 固定偏移但不随 N 变 | 偏移是 launch 开销，N 被过滤 |
| 无线性关系 | OJ 测的不是 kernel 时间（端到端/有过滤） |

热身探测（顺带）：kernel 内静态计数器，首次调用多空转 1000 cycle——
OJ 时间含此开销 → 无热身；不含 → 有热身（首次被丢弃）。

⚠ 本方法论的修正认知：**"计时口径对所有人一致所以只是全局偏移"不成立**——
FACTS F2.1g 实测 Case5 本地 4.48 vs OJ 7.74（差 3.26μs），口径与 shape/路径交互，
NOP 标定是裁决手段而非可跳过项。榜分差距大头仍是 kernel 性能，但口径系数决定
对拍换算是否可信。

### ★ 实测结果（2026-09-14 五连提交，详见 FACTS F3.8）

Case3/4/5：R²=0.998~1.000 完美线性（4.9ns/iter）——OJ 测纯 kernel 时间，口径已解。
Case1/2：斜率≈0——NOP 插入点漏了 TinyH4 独立 kernel 类，标定盲区。
**方法论修正：NOP 插入点必须覆盖所有 kernel 类/所有 tiling path 分支。**
单次评测噪声 ±0.6μs——<1μs 的优化需多次提交取中位。

## 可信性验证记录（2026-09-14，本会话）

1. **msprof 字段实测**：`--aic-metrics=PipeUtilization` 采集 v117 (8/512/2/fp16)
   → op_summary 真实产出 `aiv_vec_ratio=0.271 / aiv_scalar_ratio=0.344 /
   aiv_mte2_ratio=0.340 / aiv_mte3_ratio=0.014 / aiv_icache_miss_rate=0.042`，
   与 FACTS 历史结论（标量 39~46%、mte2 显著）**互相印证**
   （产物：`.rivet/scratch/msprof-verify/out/PROF_000001_*`）
2. **msprof_report.py 实测**：解析上述产物正常输出瓶颈判定
   （scalar 34.4% ⚠️ Scalar 受限）
3. ** NOP 标定/差分对照**：未实测（需 OJ 提交配额与 kernel 改动，列入待办）

## 待办（推进方向）

- [ ] msprof_report.py 加 op_statistic 聚合 + 热身曲线自动剔除
- [ ] NOP 标定五连提交（需先修 h=1024 崩溃 + 走通 OJ 提交链）
- [ ] Group sigmoid：先换硬件 Exp 原语做 A/B（差分对照法第一步）
- [ ] 精度语义卡片库（VecExp/VecSigmoid/ReduceSum 的 FP16 精度/周期/陷阱）——
      每踩一个坑记一张，攒 10 张即成"指令级直觉"

## 四、TilingFunc 断言探测（OJ shape 黑盒反推，2026-09-14 实战定型）

**原理**：op_host 的 `TilingFunc` 在每次 `aclnnGetWorkspaceSize` 都执行（host 侧）。
注入 `if (field != V) return ge::GRAPH_FAILED;` → 断言不成立的 case 直接 RE。
一次提交对 5 case 同时生效，Pass/RE 模式直接解出各 case 的 shape 字段。

**关键教训（实测两发验证）**：
- ⚠ **InferShape 注入无效**——平台评测不走 InferShape（outer=999 全 Pass 证明）。
  v081 历史探针若注在 InferShape，其"实测"结论需复核。
- ✅ **TilingFunc 注入有效**（outer=999 全 RE 对照验证机制）
- 工具：`scripts/probe_oj_v2.py`（自动生成变体+提交+限流退避）

**效率技巧**：断言值选择按二分/分组策略——一发同时测所有 case，Pass 的 case 集合
即该值的命中集。5 个 case 的 outer 用 4 发（2/8/32 + 999 对照）钉死。
