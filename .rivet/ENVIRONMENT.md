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

## 六、昇腾知识图谱（ascend-kg）已接入

- Skill 位置：`.rivet/skills/ascend-kg/`（SKILL.md + agents + engine + references，来源 https://gitcode.com/agent0/kg-tools）
- API Key：已写入 `~/.bashrc`（`ASCEND_KG_API_KEY`，当前用内置测试 key，共享 5 RPS；正式使用去 https://ascend.wiki/register 免费申请专属 key）
- 调用方式（实测 200 OK）：`curl -s --compressed -X POST https://ascend.wiki/search -H "X-API-Key: $ASCEND_KG_API_KEY" -H "Content-Type: application/json" -d '{"query":"...","top_k":5}'`
- 评分阈值：>0.83 黄金命中（`/source` 取全文）；0.70-0.83 模糊（改写 query）；<0.70 换路径
- 验证记录：2026-09-14 查询 "aclnnGetWorkspaceSize 361001 error custom operator" 命中 5 条，最高 0.94（`.rivet/scratch/kg-verify2.json`）
- 亦可用 skill 工具直接加载：`skill(name="ascend-kg")`

## 七、其他坑（继承自旧环境记忆，仍有效）

- msprof 单次采样是热身曲线切片：NPU 频率 800→1650MHz 热身约 25 次调用（2~3 倍时间差）；对比实验必须 ≥8 次采样取稳态/中位
- CANNJudge 平台时间单位标注 ms 实为 μs
- worker 产出的实验数据可能编造——进 FACTS 的数字必须主会话手跑或核验产物
- 本地 streamed 口径含 ~4.4μs 主机底噪——本地绝对值只做 A/B 相对比较，不可直接对标 OJ 分数（除非先复刻 OJ 计时口径）

## 八、CANN 8.5 双环境（2026-09-14 建成，与 9.0 并存）

**8.5 已装在 `/tmp/cann85/cann-8.5.0/`**（OBS 官方源下载 toolkit run 包 + `--quiet` 安装；OJ 赛题口径就是 8.5，后续优化工作默认在 8.5 上做）。

安装包留存：`/tmp/cann85-pkg/Ascend-cann-toolkit_8.5.0_linux-aarch64.run`（1.1GB，机器重置后可直接重装）。

⚠ **/tmp 是共享盘（20T），notebook 环境重置后 /tmp 可能清空**——重要产物勿只存 /tmp。

### 8.5 环境一键初始化（复制即用）

```bash
export ASCEND_HOME_PATH=/tmp/cann85/cann-8.5.0
export ASCEND_AICPU_PATH=/tmp/cann85/cann-8.5.0
export ASCEND_OPP_PATH=/tmp/cann85/cann-8.5.0/opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/v117_inst85/vendors/custom   # 8.5 编译安装的算子包
export LD_LIBRARY_PATH=/tmp/cann85/cann-8.5.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/tmp/v117_inst85/vendors/custom/op_api/lib:$LD_LIBRARY_PATH
```

### 9.0 环境（conda 版，对照用）

```bash
export ASCEND_HOME_PATH=/opt/conda/Ascend/cann-9.0.0
export ASCEND_AICPU_PATH=/opt/conda/Ascend/cann-9.0.0
export ASCEND_OPP_PATH=/opt/conda/Ascend/cann-9.0.0/opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/v117_inst/vendors/custom
export LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/tmp/v117_inst/vendors/custom/op_api/lib:$LD_LIBRARY_PATH
```

### ★★★ 361001 根因（已解，两版本通用）

`GetWorkspaceSize failed: 361001` 的根因是 **`ASCEND_HOME_PATH` 未设置**——conda 打包环境没有 `/etc/ascend_install.info`，runtime 无法定位 CANN 根，算子元数据加载失败。**只设 LD_LIBRARY_PATH 和 ASCEND_OPP_PATH 不够，必须设 ASCEND_HOME_PATH**（最好连 ASCEND_AICPU_PATH 一起）。证据：debug plog（`~/ascend/log/debug/plog/`）中反复出现 `can not get env [ASCEND_HOME_PATH]`；补上后立即 PASS。

### 8.5 编译验证记录（2026-09-14）

- v117 编译链全通：`build85/` cmake→make→binary 全绿
- 正确性 7/8 PASS：4/16/2、4/16/3、4/16/4、8/512/2、8/512/8 (fp16) + 4/16/2 (fp32) 全过，误差 8.8e-4~9.8e-4（fp16 容差内）与 v116 基线量级一致
- 唯一失败：8/1024/2 fp16 报 507035（stream 同步失败）——**9.0 也同样挂**，非 8.5 特有；NPU 上有 5 个共存 python3 进程，疑似资源冲突（AIV 占用），待独占窗口复测
- 8.5 编译产物：`v117_stage/code/build85/`；测试程序 `tests_bin/mhc_correctness85`（链接时需 `-Wl,-rpath-link` 指向 cann85 lib64 和 driver lib64）

## 九、下一步（用户核心目标：对拍测真实 case 形状）

1. 从 `v117_stage` 恢复代码到 `code/`，用本机 CANN 9.0.0 跑通编译 + 正确性测试
2. 本地建 ACL-event 计时 harness（对齐 OJ 端到端口径）
3. shape 扫描矩阵：本地各 (n, nH, outer) 组合的计时曲线
4. 对拍：找哪个 shape 组合能同时拟合 OJ 的逐 case 时间（本队 [4.70, 5.84, 5.10, 4.94, 8.64] 量级，最新值以榜单为准）
5. 辅助验证：`scripts/probe_oj.py sweep outer 3 1,2,4,8,16`（OJ 侧断言探测，补 Case3/4/5 的 outer）
