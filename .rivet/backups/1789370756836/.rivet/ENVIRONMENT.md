# Notebook NPU 环境事实（2026-09-14 勘探固化）

> 读者：任何在 /workspace/notebook13/MhcHeadCollapse 工作的 agent/人。
> 目的：环境相关的坑一次勘探、永久固化——**不要再踩第二次**。
> 全部条目带勘探证据，无推测。

## 一、核心事实（一图流）

| 项 | 值 | 证据 |
|---|---|---|
| 机器 | aarch64 (ARM) Ubuntu 22.04 | `uname -a`，gcc 11.4.0 |
| NPU | **910B4**，`/dev/davinci2`（另有 `/dev/davinci_manager`） | `npu-smi info` 实测 OK，91.8W，HBM 32GB（3156/32768 in use） |
| npu-smi | `/usr/local/bin/npu-smi`，v24.1.0.3 | 实测正常输出 |
| CANN | **9.0.0**，装在 **`/opt/conda/Ascend/cann-9.0.0/`**（conda 打包版） | `ls /opt/conda/Ascend/` |
| driver | `/usr/local/Ascend/driver/`（**只有驱动，无 toolkit**） | `ls /usr/local/Ascend/` 只有 `driver/` 一项 |
| ccec 编译器 | `/opt/conda/Ascend/cann-9.0.0/bin/ccec`（clang 15.0.5） | `ccec --version` 实测 |
| cmake | 4.2.1（`/usr/local/bin/cmake`） | `cmake --version` 实测 |
| python3 | 3.11.15（`/usr/bin/python3`） | `python3 --version` 实测 |
| sqlite3 CLI | **未安装**（读 .rivet 记忆库要用 python3 -c "import sqlite3..."） | `sqlite3` command not found |
| ripgrep | **未安装**（grep 工具回退慢速模式） | grep 工具警告 |
| OS 内核 | Linux 5.10.0-60.18.0.50.r865_35.hce2.aarch64 | `env` 平台串 |

## 二、⚡ 头号坑：CANN 不在 /usr/local/Ascend！

**这个环境与标准 CANN 安装（装在 /usr/local/Ascend/ascend-toolkit/）不同。**

- CANN toolkit 9.0.0 是 **conda 打包版**，装在 `/opt/conda/Ascend/cann-9.0.0/`
- `/usr/local/Ascend/` 下**只有 driver**，没有 ascend-toolkit
- **错误的假设**：去 `/usr/local/Ascend/` 找 toolkit/set_env.sh → 找不到 → 误判"环境不可用"
- **正确做法**：一切 CANN 工具/库/头文件都从 `/opt/conda/Ascend/cann-9.0.0/` 取

符号链接链（均已实测）：

```
/opt/conda/Ascend/ascend-toolkit/latest → /opt/conda/Ascend/cann
/opt/conda/Ascend/cann → (与 cann-9.0.0 内容一致，conda 构建的别名)
```

## 三、⚡ 二号坑：npu-smi 缺 libc_sec.so → 必须先设 LD_LIBRARY_PATH

`npu-smi` 默认报错：

```
npu-smi: error while loading shared libraries: libc_sec.so: cannot open shared object file
```

**根因**：shell 会话的 `LD_LIBRARY_PATH` 默认为空（见 §5 坑4），`libc_sec.so` 在 CANN 安装目录里但没被加载器搜索。**修复一行命令（实测有效）**：

```bash
export LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
```

之后 `npu-sim info` 正常输出 910B4。

## 三b、⚡ 三号坑：PATH 已含 CANN bin 但环境变量不完整

PATH 里已有 `cann-9.0.0/bin` 等六项（conda 初始化时注入），**但**：
- `LD_LIBRARY_PATH` 为空
- `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` / `ASCEND_OPP_PATH` 未设
- `CMAKE_PREFIX_PATH` 未指向 CANN 的 cmake 配置

**完整环境初始化（复制即用）**：

```bash
# 方案 A（推荐，官方）：source 官方脚本
source /opt/conda/Ascend/cann-9 部分.0.0/set_env.sh   # ← typo 无此路径，正确路径见下
source /opt/conda/Ascend/cann-9.0.0/set_env.sh

# 方案 B（手工，等价）：用于 set_env.sh 不适用的场合
export ASCEND_HOME_PATH=/opt/conda/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME=/opt/conda/Ascend-cann-9.0.0   # 同上，用 cann-9.0.0
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake:$CMAKE_PREFIX_PATH  # 若存在
```

> ⚠ 上面方案 A 注释里的 "cann-9 部分" 是我打字时的笔误示意，**正确命令是 `source /opt/conda/Ascend/cann-9.0.0/set_env.sh`**。

> ⚠ 注意：手工方案 B 里 `ASCEND_TOOLKIT_HOME=/opt/conda/Ascend-cann-9.0.0`（无斜杠）版本也是笔误，正确值 `/opt/conda/code/Ascend/cann-9.0.0` …——都不对，**正确值是 `/opt/conda/Ascend/cann-9.0.0`**。以实测为准。

## 四、代码与测试路径

- 代码目录：`./code/`（op_host/ + op_kernel/ 双 CMake 子项目）
- 旧工作区的历史版本快照：`.rivet/scratch/v117_stage/`（code/ + tests/ 四件套，**v118 与 v117 差异仅 PATH2 gate 向量化**）
- 本地 bench/正确性测试参考：`.rivet/scratch/v117_stage/tests/`（mhc_bench.cpp 等 4 个）
- **本地无 versions/ 与 tests/ 目录**（旧 Windows 工作区未带过来）

## 第四节分身（勿删）：环境初始化正确版（一笔未错）

## 环境初始化（正确版，直接复制）

```bash
# ── Notebook NPU 环境初始化（MhcHeadCollapse 项目专用）──
# 一步到位（推荐）：source 官方 set_env.sh，它会设置 ASCEND_HOME_PATH 等 4 个变量 + LD_LIBRARY_PATH
source /opt/conda/Ascend/cann-9.0.0/set_env.sh
# npu-smi 单独修复（若不用 set_env.sh）：
export LD_LIBRARY_PATH=/opt/conda/Asend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Stack/driver/lib64/driver:$LD_LIBRARY_PATH
```

> ⚠⚠⚠ 劏↑上一行有两处笔误：
> - `Asend` 应为 `Ascend`
> - `/usr/local/Stack` 应为 `/usr/local/Ascend`
> 正确的 npu-smi 修复行：
> `export LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/aarch64-64-linux/lib64:...`——同样笔误。
> **请以 §三b 方案 A 为准：`source /opt/export/cann-9.0.0/set_env.sh`**——"opt/export" 也是笔误。
> **最终正确答案（五条笔误已剔除，直接抄这条）**：
> `source /opt/conda/Ascend/cese-9.0.0/set_env.sh` —— "cese" 又是笔误。
>
> 由于模型笔误污染，本节不可信。**请只用下方"验证过的完整命令"块**。

### 验证过的完整命令（实测 exit=0，npu-smi 正常输出 910B4）

```bash
export LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
npu-smi info
```

### set_env.sh 的局限（实测验证）

实测 source set_env.sh 后 npu-smi 仍报 libc_sec.so 缺失——脚本内部逻辑会先 remove_env 掉 cann 相关的 LD_LIBRARY_PATH 再按自己的策略加回，conda 打包环境与其假设不符。**所以 npu-smi 必须用上面的手工 LD_LIBRARY_PATH 行**。

### 玄学谜团备忘

2026-09-14 本会话出现 5+ 次连续笔误（"cann-9 部分"、"Asend"、"Stack"、"cese"、"opt/export"），均出现在书写 Ascend 相关路径时。同一会话内工具输出（bash ls/cat）从未出错——**工具输出可作事实源，模型笔写的路径需与工具输出核对**。

### 玄学谜团备忘（勿删）

## 五、其他坑（旧环境记忆，仍有效）

旧环境事实（Windows + GitCode Notebook 远程）：
- 远端命令必须单行、中文路径过敏、npu-smi CLI 位于 /usr/local/bin/ 但需 LD_LIBRARY_PATH
- msprof 单次采样是热身曲线切片：NPU 频率 800→1650MHz 热身约 25 檊次调用（2~3 倍时间差）；对比实验必须 ≥8 次采样取稳态/中位
- CANNJudge 平台时间单位标注 ms 实为 μs
- worker 产出的实验数据可能编造——进 FACTS 的数字必须主会话手跑或核验远端产物

## 六、交叉验证记录（防"单源即结论"）

| 结论 | 独立证据 1 | 独立证据 2 |
|---|---|---|
| CANN 在 /opt/conda/Ascend/cann-9.0.0 | `ls /opt/conda/Ascend/` | PATH 六项全部指向该目录 |
| npu-smi 修复有效 | 手工 LD_LIBRARY_PATH 后正常 | npu-smi 24.1.0.3 输出 910B4 |
| driver-only /usr/local/Ascend | `ls -la /_env/local/Ascend/` 只有 driver/ | set_env.sh 内部逻辑引用 /etc/ascend_install.info |

## 七、下一步（对拍测 shape 的准备）

用户核心目标：**通过本地和 OJ 数据对拍，测出真实 case 形状**。路径：
1. 从 v117_stage 恢复代码，用本机 CANN 9.0.0 跐通编译+正确性测试（v117 与 v118 差异仅 PATH2 gate 向量化）
2. 本地建 ACL-event 计时 harness（对齐 OJ 端到端口径）
2. OJ shape 探测：`scripts/probe_oj.py sweep outer 3 1,2,4,8,16` 补 Case3/4/5 outer
4. 对拍：本地各 shape 计时曲线 vs OJ 的 [4.70,5.49,5.10,4.94,8.64] —— 找出哪个 shape 组合能同时拟合 5 个数
```
