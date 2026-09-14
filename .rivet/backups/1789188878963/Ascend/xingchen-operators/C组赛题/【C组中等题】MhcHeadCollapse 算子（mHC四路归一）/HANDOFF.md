# HANDOFF — MhcHeadCollapse 算子性能优化

> **读这一份就够上手**。稳定事实在 `FACTS.md`，经验教训在 `PITFALLS.md`，环境/连接在
> `Ascend/ascend-npu-console/docs/session-handoff.md`。
> **最后更新**：2026-09-11

---

## 一、一句话现状

算子功能正确（5/5 Pass），在 OJ 排 **第 14 名 / 60 人，37.97 分**。
榜首 wangyue **80.97 分**。**全榜最优 14.26μs vs 我们 26.62μs（1.87×）**。

瓶颈已定位：**标量指令占 39~46%**，是最大单项，且与 shape 无关。

---

## 二、必须知道的三条认知修正（本轮推翻旧结论）

### 1. ⚠️ Case3/4/5 的 shape 从未被实测过（F2.1l）

只有 **Case1/Case2** 的 shape 是 `v038~v040` 探针**实测**的（断言反推）。
`Case3/4/5` 来自 `优化全记录:33-39` 的**推断表**。

证据：`迭代优化记录.md:162` 写「5/5 Pass 锁定FP16」，但同一行的描述是
「证实平台**两个**测试点」——**"5/5 Pass" 是编译单元都过，被误读成"5 个点全解密"**。

**赛题侧**：`赛题.md:128` 只给范围 n ∈ {2,4,8}，从未公布 5 个 testcase 各用哪组。

→ **下一步最高价值动作**：用 v038 同款断言探针反推 Case3/4/5（约 10 分钟/轮）。

### 2. ⚠️ 真实带宽是 109 GB/s，不是 200（F2.1f）

原 `F2.6` 基于估算的 200GB/s，其"240KB=1.23μs ≈ floor差1.18μs"的论证
按 109GB/s 重算是 2.25μs，**不吻合（差 1.9 倍）**。F2.6 已推翻。

### 3. ⚠️ 瓶颈是标量，不是 DMA（F2.1e / F2.1n）

| case | scalar | mte2 | vec |
|---|---|---|---|
| Case1 | **3.319 (46%)** | 1.721 | 0.554 |
| Case5 | **5.473 (39%)** | 3.611 | 2.231 |

历史文档盯 DMA/归约原语，**方向有偏**。换归约原语只影响 vec（第三位）→ ROI 最低。

---

## 三、当前战场（按 ROI 排序）

### P1：削标量开销 ← **最高 ROI**
热点已定位（F2.1n）：`mhc_head_collapse.cpp` L184/L206 内层循环**每行 9~13 次
UB 标量读写**（`GetValue`/`SetValue`）。

具体改造点：
- **L193-196**：`-(gate*scale+base)` 可用 `Muls`/`Adds` 向量指令替代 4 次 SetValue
- **L210-212**：算完 z 又 `GetValue` 回读，可用向量 Exp/Reciprocal 直接算

### P2：查 Case1/Case2 的 1.16μs 不一致
全榜 6 位用户这两个 case 差 <0.2μs（XiuPang **0.00**），我们差 **1.16μs**。
→ 存在未消除的抖动源，可能是 OJ 计时的关键（F2.1k）。

### P3：确认 Case3/4/5 的 shape
消除最大的元不确定性（见第二节 1）。

---

## 四、本地测量的正确用法（F2.1m）

**硬件一致**（F2.1h，三条判据）：本地 910B4 与 OJ 同规格，测量可外推。
**但口径不同**：本地拆机 = `aclrtSynchronizeStream` + 主机墙钟；OJ 口径未知。

**实测方差**（4 次 `mhc_percall 4 4 1`）：

| 指标 | 范围 | 极差 |
|---|---|---|
| `per_call` | 16.7~18.9 | **±6.5%** ← 抖 |
| `streamed` | 4.346~4.468 | **±1.4%** ← 稳 |

**结论**：
- ✅ 做 **A/B 版本对比**（用 `streamed`，取多次 min）
- ❌ **不可**预测 OJ 绝对分数
- ⚠️ `streamed` 含 ~4.4μs 固定底噪（小 shape 不随数据变）

---

## 五、环境与连接

**详见** `Ascend/ascend-npu-console/docs/session-handoff.md`（158 行）。

速记：
```bash
cd D:/Desktop/OP-Learning/Ascend/ascend-npu-console
python scripts/npu.py inst                      # 提取 instanceId（每次重建都变）
python scripts/npu.py --inst <ID> run "<命令>"   # 跑命令
```

| 项 | 值 |
|---|---|
| NPU | 910B4，`/dev/davinci6`（**不是 0**），20 AI Core |
| CPU | **`nproc=1`（单核）** ← 可能是 launch 开销异常的根因 |
| CANN | `/usr/local/Ascend/cann-8.5.0` |
| 远端工作区 | `/opt/atomgit/mhc/`（**重启即清空，可重建**） |

---

## 六、坑（血泪）

| 坑 | 症状 | 做法 |
|---|---|---|
| `$HOME` 被本地展开 | 远端 `mkdir '/c'` 失败 | **用绝对路径** |
| heredoc 卡死 shell | 命令不返回 | 远端命令**保持单行** |
| `make -j8` | 单核上更慢 | 用 `-j2` |
| `/tmp` 当持久存储 | 重启消失 | 产物拉回本地 |

---

## 七、文件地图

```
C组赛题/【C组中等题】MhcHeadCollapse 算子（mHC四路归一）/
├── README.md        144 行  导航入口
├── FACTS.md         660 行  ★ 稳定事实（F2.1a~F2.1o 共 15+ 条）
├── PITFALLS.md      286 行  经验教训
├── EXPERIMENTS.md   186 行  实验记录
├── 赛题.md                  竞赛原始要求
├── 迭代优化记录.md            142 个版本的完整历史
├── versions/                142 个版本（52 个是纯副本，见 F2.13）
└── scripts/                 check-facts.sh 等

工具包（父仓库）：Ascend/ascend-npu-console/
├── scripts/npu.py           280 行，inst/terms/run/push
└── docs/session-handoff.md  158 行  ★ 环境与恢复
```

---

## 八、下一步（给下个会话）

1. **读** `FACTS.md` 的 F2.1e / F2.1j / F2.1l / F2.1m / F2.1n（五条核心）
2. **确认 Notebook 还活着**（`npu.py inst`），死了就按 `session-handoff.md` 恢复
3. **按 ROI 从 P1 开始**：削标量 → 用 `streamed` 验证（噪声 1.4% 可分辨）
4. **提交前后**跑 `scripts/check-facts.sh`（引号引用完整性自检）

**诚实提醒**：OJ 计时口径**仍未确定**（F2.1k），仅排除了三个假设。
不要基于"本地数字 = OJ 分数"做判断。
