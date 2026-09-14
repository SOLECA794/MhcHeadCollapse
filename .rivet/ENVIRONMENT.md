# Notebook NPU 环境事实（2026-09-14 勘探固化）

> 读者：任何在 /workspace/notebook13/MhcHeadCollapse 工作的 agent/人。
> 目的：环境相关的坑一次勘探、永久固化——**不要再踩第二次**。
> 全部条目带勘探证据，无推测。

## 一、核心事实（一图流）

| 项 | 值 | 证据 |
|---|---|---|
| 机器 | aarch64 (ARM)，Ubuntu 22.04（gcc 11.4.0） | `gcc --version` 实测 |
| NPU | **910B4**，`/dev/davinci2`（另有 `/dev/davinci_manager`） | `npu-smi info` 实测：OK，91.8W，HBM 32GB |
| npu-smi | `/usr/local/bin/npu-smi`，v24.1.0.3 | 实测正常输出（需先设 LD_LIBRARY_PATH，见 §三） |
| CANN | **9.0.0 conda 打包版**，装在 `/opt/conda/Ascend/cann-9.0.0/` | `ls /opt/conda/Ascend/` |
| driver | `/usr/local/Ascend/driver/`（**只有驱动，无 toolkit**） | `ls /usr/local/Ascend/` 仅 `driver/` 一项 |
| ccec 编译器 | `/opt/conda/Ascend/cann-9.0.0/bin/ccec`（clang 15.0.5） | `ccec --version` 实测 |
| cmake | 4.2.1（`/usr/local/bin/cmake`） | `cmake --version` 实测 |
| python3 | 3.11.15（`/usr/bin/python3`） | `python3 --version` 实测 |
| sqlite3 CLI | **未安装**（读记忆库用 `python3 -c "import sqlite3; ..."`） | command not found 实测 |
| ripgrep | **未安装**（grep 工具自动回退慢速模式，输出会提示） | 工具警告实测 |

## 二、头号坑：CANN 不在 /usr/local/Ascend！

这个环境与标准安装（`/usr/local/Ascend/ascend-toolkit/`）不同：

- CANN toolkit 9.0.0 是 **conda 打包版**，在 `/opt/conda/Ascend/cann-9.0.0/`
- `/usr/local/Ascend/` 下**只有 driver**，没有 ascend-toolkit、没有 set_env.sh
- 错误假设：去 `/usr/local/Ascend/` 找 toolkit → 找不到 → 误判"环境不可用"（本会话第一天就踩了）
- 正确做法：一切 CANN 工具/库/头文件从 `/opt/conda/Ascend/cann-9.0.0/` 取

符号链接链（实测）：

```
/opt/conda/Ascend/ascend-toolkit/latest  → /opt/conda/Ascend/cann
/opt/conda/Ascend/cann                   → /opt/conda/Ascend/cann-9.0.0（conda 构建别名）
```

## 三、二号坑：npu-smi 缺 libc_sec.so

裸跑 `npu-smi` 报错：

```
npu-smi: error while loading shared libraries: libc_sec.so: cannot open shared object file
```

根因：shell 会话的 `LD_LIBRARY_PATH` 默认为**空**（PATH 里虽有 CANN bin，但库搜索路径没设）。

**修复（实测 exit=0，npu-smi 正常输出 910B4）——直接抄这一行：**

```bash
export LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
```

⚠ **set_env.sh 在此环境不能替代上面这行**：实测 `source /opt/conda/Ascend/cann-9.0.0/set_env.sh` 后 npu-smi 仍报 libc_sec.so 缺失——脚本会先 remove_env 清掉 cann 相关 LD_LIBRARY_PATH 再按标准安装布局加回，与 conda 打包布局不符。跑 NPU 程序（链接 libascendcl.so 等运行时库）时同样需要这行手工 LD_LIBRARY_PATH。

## 四、环境变量现状（conda 初始化注入了一半）

PATH 里已有 CANN 六项（`/opt/conda/Ascend/cann-9.0.0/bin`、`tools/ccec_compiler/bin`、`tools/profiler/bin` 等），PYTHONPATH 也指向 CANN python 包。**但缺**：

- `LD_LIBRARY_PATH`（空——二号坑根因）
- `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` / `ASCEND_OPP_PATH`（均未设）
- `CMAKE_PREFIX_PATH` 未指向 CANN 的 cmake 配置

**编译算子前的完整初始化（复制即用）：**

```bash
export LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_HOME_PATH=/opt/conda/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME=/opt/conda/Ascend/cann-9.0.0
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
```

（路径均为实测值；若手动设置后仍异常，用 `ls` 核对磁盘真实路径，不要凭记忆改写。）

## 五、代码与测试路径

- 代码目录：`./code/`（op_host/ + op_kernel/ 双 CMake 子项目）——**当前是空壳模板**（35 行骨架，Init/Process 为空）
- 旧工作区版本快照：`.rivet/scratch/v117_stage/`（code/ + tests/ 四件套；v118 与 v117 差异仅 PATH2 gate 向量化 + Group 回退）
- 本地 bench/正确性测试参考：`.rivet/scratch/v117_stage/tests/`（mhc_bench.cpp、mhc_bench_percall.cpp、mhc_correctness.cpp、mhc_isolated.cpp）
- 本地**无** versions/ 与 tests/ 目录（旧 Windows 工作区的 142 个版本目录未带过来）
- 跨会话记忆库 `.rivet/knowledge/memory-index.sqlite` 在此环境**表为空**（旧记录未同步）——项目事实以根目录 md 文档为准（FACTS.md 983 行是单一事实源）

## 六、其他坑（继承自旧环境记忆，仍有效）

- msprof 单次采样是热身曲线切片：NPU 频率 800→1650MHz 热身约 25 次调用（2~3 倍时间差）；对比实验必须 ≥8 次采样取稳态/中位
- CANNJudge 平台时间单位标注 ms 实为 μs
- worker 产出的实验数据可能编造——进 FACTS 的数字必须主会话手跑或核验产物
- 本地 streamed 口径含 ~4.4μs 主机底噪——本地绝对值只做 A/B 相对比较，不可直接对标 OJ 分数（除非先复刻 OJ 计时口径）

## 七、下一步（用户核心目标：对拍测真实 case 形状）

1. 从 `v117_stage` 恢复代码到 `code/`，用本机 CANN 9.0.0 跑通编译 + 正确性测试
2. 本地建 ACL-event 计时 harness（对齐 OJ 端到端口径）
3. shape 扫描矩阵：本地各 (n, nH, outer) 组合的计时曲线
4. 对拍：找哪个 shape 组合能同时拟合 OJ 的逐 case 时间（本队 [4.70, 5.84, 5.10, 4.94, 8.64] 量级，最新值以榜单为准）
5. 辅助验证：`scripts/probe_oj.py sweep outer 3 1,2,4,8,16`（OJ 侧断言探测，补 Case3/4/5 的 outer）
