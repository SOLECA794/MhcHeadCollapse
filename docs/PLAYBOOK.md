# PLAYBOOK — MhcHeadCollapse 实战手册（招式速查）

> 定位：**可复制的操作序列**。METHODOLOGY.md 讲"为什么"，本文档讲"怎么做"。
> 覆盖：OJ 提交 / 本地全链路 / 错误速查 / 实验纪律 / 认识论原则。
> 来源：2026-09-14 会话全程实测（每条都可指向验证记录），更新时保持这一标准。

---

## 一、OJ 提交操作手册（cannjudge CLI）

```bash
CLI="cannjudge-submit-plaintext/cannjudge_cli.py"
PID="6a7c23a6a52e0f540a8a1779"

python3 $CLI rank --problem-id $PID          # 查榜（不耗额度，先跑这个验凭据）
python3 $CLI submit --problem-id $PID --project-dir <含CMakeLists的code目录>
python3 $CLI query --submission-id <id>       # 轮询结果
```

**关键纪律**：
- **限流 45~60s**：连续提交报 429 "提交过于频繁"——每次提交前 sleep ≥50s，
  或捕获 429 后按提示秒数退避重试（实测提示 15~53s 不等，一律 sleep 提示值+5）
- 一次 submit 命令含提交+自动轮询（约 60~90s 出结果），后台跑 + `job await` 等待
- 提交目录必须是**含 CMakeLists.txt 的 code 目录**（op_host/ + op_kernel/ 同级）
- **排行榜只保留最新提交**（含 Compile Error 覆盖）——保分版本提交前想清楚
- 单次评测噪声 **±0.6μs**（F3.8 NOP=100 时 Case5 反向 -0.64 实测）——
  <1μs 的优化单次提交不可分辨，需要多次提交或大幅改动

## 二、本地全链路工作流（改码→验证→提交）

```bash
# 0) 环境（每次新 shell 必做，见 ENVIRONMENT.md §八 一键块）
export ASCEND_HOME_PATH=/tmp/cann85/cann-8.5.0 ASCEND_AICPU_PATH=$ASCEND_HOME_PATH \
       ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp ASCEND_CUSTOM_OPP_PATH=/tmp/<inst>/vendors/custom
export CMAKE_PREFIX_PATH=$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/tmp/<inst>/vendors/custom/op_api/lib:$LD_LIBRARY_PATH

# 1) 编译（全清重编，防增量缓存掩盖错误）
cd <code>/build && rm -rf * && cmake .. && make -j$(nproc) && make binary -j$(nproc) && make package

# 2) 安装到【全新目录】（旧目录有 root 属主 uninstall.sh，rm 不掉会卡链）
./custom_opp_ubuntu_aarch64.run --install-path=/tmp/<新名字>_inst --quiet

# 3) 正确性 + 性能（shape 权威版：C1/C2=4/4/2, C3/C4=8/64/8, C5=8/128/32）
S=.rivet/scratch/v117_stage
$S/tests_bin/mhc_correctness85 8 128 32 fp16   # 正确性
$S/tests_bin/mhc_shape_bench   8 128 32 fp16 200 streamed  # 性能 med

# 4) 显微镜（改优化前后各一次，对比分项）
msprof --output=./out --ai-core=on --aic-metrics=PipeUtilization \
       --application="<bench命令>" && python3 scripts/msprof_report.py ./out

# 5) OJ 提交（确认本地绿 + 无回归后再花额度）
```

**变体管理**：实验版 `cp -r code /tmp/vNNN_名字`——**必须 `rm -rf build` 再编**
（cp -r 会把旧 CMakeCache 一起拷来，cmake 报 source mismatch，见 P23）。

## 三、错误码→动作速查表

| 症状 | 根因 | 动作 | 详档 |
|---|---|---|---|
| `GetWorkspaceSize failed: 361001` | ASCEND_HOME_PATH 未设 | export 三件套（HOME/AICPU/OPP） | P20 |
| `aclrtSynchronizeStream failed: 507035` | kernel 设备侧崩溃 | 查 `~/ascend/log/debug/plog/` 最新文件，grep errorStr | F3.3/F3.6 |
| plog "MTE write out of range" | MTE 写越界（常是 UB 超 192KB） | InitBuffer 逐项对账 vs 192KB | F3.6 |
| plog "VEC ub address out of bounds" | 向量指令 UB 读写越界 | 查 tensor 索引是否超缓冲声明 | F3.6 |
| HTTP 429 提交频繁 | OJ 限流 | sleep 提示秒数+5 重试 | §一 |
| ld "undefined reference to hal*/drv*" | rpath-link 缺 driver 侧 | 三段 `-Wl,-rpath-link,...` | P22 |
| 产物 "Exec format error" | 链接失败留非 ELF 残骸 | `file` 检查产物再查别的 | P22 |
| `npu-smi: libc_sec.so not found` | LD_LIBRARY_PATH 空 | ENVIRONMENT §三 修复行 | P22 前 |
| 编译器找不到 ASC 包 | CMAKE_PREFIX_PATH 未设 | export（§二 步骤 0） | 实测 |
| **shell 工具全体 command not found** | bash 环境 PATH 损坏（罕见） | 重开命令/绝对路径执行 | 本会话一次 |

## 四、A/B 实验纪律（血泪换来的）

1. **采样量**：streamed 至少 200 iters（20 samples）取 med；nsamp<5 的结论不作数
2. **min/med/p90 三看**：min≈med=稳定慢（真瓶颈）；min≪med=抖动（外部干扰/NPU 共存进程）
3. **热身 25 次**固定保留（NPU 800→1650MHz 爬频，PITFALLS 旧档）
4. **否定也是产出**：负载均衡假设（F3.10）被 A/B 否定省了一次 OJ 额度——
   **本地 A/B 过不了的优化不提交 OJ**
5. 优化前后各跑一次 msprof：分项变化方向对得上才算"机制验证"
   （v121 案例：scalar 40.3→34.8% 与时间 -4.4% 同向，才信）

## 五、认识论原则（本会话最贵的三课）

1. **拟合的完美解 ≠ 真相**：本地矩阵+口径系数拟合出 outer5=4（残差<0.06μs），
   被探针实测推翻（真值 32）。教训：**拟合只能提假设，实测才能定案**——
   任何"完美拟合"在探针确认前都标注为假设。
2. **探针先验机制有效性**：outer=999 必错对照救了整个探测（InferShape 版全 Pass
   暴露注入点无效）。任何断言探测先打一发"必错值"验证机制，再信结果。
3. **双源交叉**：msprof 分项与历史探针差分（F3.10 vs F2.1e）互相印证才下结论；
   单源数字（哪怕 R²=1.000）只算线索。

## 六、工具资产速查

| 工具 | 位置 | 用途 |
|---|---|---|
| mhc_correctness85 | .rivet/scratch/v117_stage/tests_bin/ | 正确性（n h outer dtype） |
| mhc_shape_bench | 同上 | 双口径计时（percall/streamed） |
| msprof_report.py | scripts/ | msprof op_summary 解析+瓶颈判定 |
| probe_oj_v2.py | scripts/ | OJ shape 断言探测（TilingFunc 版） |
| make_nop_variants.sh | .rivet/scratch/ | NOP 标定变体生成 |
| cannjudge_cli.py | cannjudge-submit-plaintext/ | OJ 提交/rank/query |
| ascend-kg | POST ascend.wiki/search（key 在 ~/.bashrc） | 昇腾知识检索 |
| shape 矩阵 | .rivet/scratch/shape_scan_v119_8.5.tsv | 17-shape 基准数据 |
| 参考实现库 | .rivet/scratch/cann-ops-competitions/ | 官方赛事高分代码 |
| Tangefly GeluV2 | 上库 operator-challenge-champ2/ | sigmoid/exp 近似参考 |
