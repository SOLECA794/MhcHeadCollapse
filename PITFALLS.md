# PITFALLS — 已踩过的坑

> **纪律**：每类坑首次出现即写入本文件，并**同步更新脚本/清单**。同坑不得二犯。
> 格式：现象 → 根因 → 正确做法 → 证据。
>
> 本文件只记录**已经踩过的**。猜测性的风险放 `FACTS.md §A`。

---

## 一、编译陷阱（已在实机复现）

### P1. fp16 count 版 Mul/ReduceSum 在 dav_c220 上编译失败

**现象**：`AscendC::Mul<half>` / `AscendC::ReduceSum<half>`（count 版）报错：
```
dav_c220/kernel_operator_vec_binary_impl.h:166 → vmul(dst,src0,src1,...)
dav_c220/kernel_operator_vec_reduce_impl.h:153 → vcadd(dst,src,1,...)
the 1st parameter maybe need a type '__ubuf__ half *'
```

**根因**：编译器内建（vmul/vcadd）对 half 指针绑不上。源码里 `SupportType<T, half, ...>`
**声明**支持 half，但实际内建无法绑定——**声明与实现不一致**，不是调用方式问题。

**正确做法**：不要在 c220 上用 count 版 half 归约。若确需半精度归约，走 DataBlock/PairElem 系列
（但注意 P2）。

**证据**：`迭代优化记录.md:1597`；实机与 OJ 报错行号逐字一致，已交叉验证。

---

### P2. `ReduceDataBlock` / `ReducePairElem` 被架构门控拦住（**不是"非公开 API"**）

**历史误记**：早期记录写"ReduceDataBlock 不是公开 API，`__ASCENDC_INCLUDE_INTERNAL_HEADERS__` 未定义"。

**真相**：
- 它们**在公开头文件里**：`asc-devkit/include/basic_api/kernel_operator_vec_reduce_intf.h:51,71`
  （`kernel_operator_intf.h:54` 链式包含该头，属正常包含路径）
- 真正的拦截是函数上的 `__ASC_USE_RESERVED_UBUF__(3510, "...")` 属性宏，
  白名单只有 `2201` / `3510`：`asc-devkit/impl/utils/sys_macros.h:110-120`

**正确做法**：查 API 可用性时，**看 `__ASC_USE_RESERVED_UBUF__` / `__NPU_ARCH__` 门控值**，
不要用"是否在公开头文件里"判断。910B 对应 `dav_c220`。

**教训**：结论对（确实不可用）但**理由错**，会导致在别的 API 上重蹈误判。

**证据**：`FACTS.md F2.8`

---

### P3. 文件级 pragma 是 DataCopy 出现在 `Process()` 中的前提

**现象**：不带 pragma 时，`Process()` 内的 `DataCopy` 编译失败。

**正确做法**：kernel 文件顶部必须有：
```cpp
#pragma GCC optimize("O3,fast-math,unroll-loops")
```
证据: `versions/v116_vsync_batched/code/op_kernel/mhc_head_collapse.cpp:9`

**其他 pragma 规律**（已验证）：
| 写法 | 结果 |
|---|---|
| 类内 `__pragma__ GCC optimize("O3")` | CE |
| Generic 内循环 `#pragma GCC unroll 16` | CE |
| `#pragma GCC unroll`（无参） | Pass 但无收益 |

**证据**：`迭代优化记录.md:3`（v057/v058 根因节）

---

### P4. `AscendC::Queue<A,B>` 不存在，正确类型是 `TQue<TPosition, depth>`

**来源**：v057 转抄时的臆造 API。

**正确做法**：`AscendC::TQue<AscendC::QuePosition::VECIN, 1>`
证据: `versions/v116_vsync_batched/code/op_kernel/mhc_head_collapse.cpp:577`

**证据**：`迭代优化记录.md:3`

---

### P5. 多路径类必须显式声明 `TPipe pipe_` 成员

**现象**：`Init()` 里用了 `pipe_` 但类内未声明 → CE。

证据: `versions/v116_vsync_batched/code/op_kernel/mhc_head_collapse.cpp:576`（`AscendC::TPipe pipe_;`）

**证据**：`迭代优化记录.md:3`

---

### P6. 版本目录必须完整 copy code tree 再 patch

**现象**：从错误锚点生成版本 → 缺 `op_host`/tiling 文件 → OJ 报"找不到必要的代码文件"。

**正确做法**：完整复制上一版的 `code/` 目录，再在其上改。不要只拷单个 `.cpp`。

**教训**：v056 源码曾因此丢失（只剩 CMakeLists）。

---

### P7. tiling 结构体不可替换为残缺 stub

**现象**：v079 多次 CE，根因是 `tiling.h` 被替换成只剩 `uint32_t length` 的 stub。

**正确做法**：tiling 结构体字段必须与 host 写入端严格一致。
证据: `versions/v116_vsync_batched/code/op_kernel/mhc_head_collapse_tiling.h:1-10`

**证据**：`迭代优化记录.md`（v079 节）

---

## 二、环境陷阱

### P8. 实机验证会改写 target，与 OJ 不等价

`scripts/verify.sh:9-10` 会把目标平台从 `ascend910b` 改成 `ascend910_93`：
```bash
sed -i "s/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/" code/CMakeLists.txt
sed -i "s/.AddConfig(\"ascend910b\")/.AddConfig(\"ascend910_93\")/" code/op_host/mhc_head_collapse.cpp
```

**含义**：实机测得的**绝对时间**不能直接与 OJ 榜对比。

⚠ **更危险的**：若把这个被 sed 改过的目录直接拿去提交 OJ，会带上错误 target。
**提交前必须确认 `op_host` 里是 `.AddConfig("ascend910b")`**（原版见 `op_host:111`）。

**证据**：`scripts/verify.sh:9-10`、`versions/v116_vsync_batched/code/op_host/mhc_head_collapse.cpp:111`

---

### P9. OJ 提交限流 429，需冷却 70-80 秒

**正确做法**：提交前先实机离线排掉 CE。CE 会浪费提交次数且输出 `time=0` 无法用于归因。

**证据**：`MhcHeadCollapse_优化全记录_2026-09-08.md:4.3 节`

---

### P10. 实机是共享环境，勿打扰他人进程

**正确做法**：跑之前先看有没有别人的进程占着芯片；跑完确认无残留。

**证据**：`MhcHeadCollapse_优化全记录_2026-09-08.md:7.2 节`

---

### P11. CE 会让所有 case 的 time=0，探针法失效

**含义**：WA 差分探针必须**能编译通过**才有意义（故意语义错误，但要能跑起来）。

**证据**：`MhcHeadCollapse_优化全记录_2026-09-08.md:4.1 节`

---

## 三、方法学陷阱（★ 本轮侦察发现，代价最大）

### P12. 追加式记录会让勘误失效

**现象**：`迭代优化记录.md:75-79` 的用例表写 `ms` 和错误 shape；
而 `:1211`（单位勘误）与 `:1220`（shape 勘误）写在 **900 行之后**。主表从未回改。

**后果**：读主表的人得到数量级错误 + 路径判断错误，且不会读到勘误。
本轮首轮诊断就差点据此得出错误结论。

**正确做法**：
- 事实集中到 `FACTS.md`（单一源）
- 旧文档在矛盾处**加指针**而非改写（保留证伪过程）
- 推翻结论时**改原条目**，不要追加到文末

---

### P13. 副本目录会伪装成"探索量"

**现象**：142 个版本目录中 **52 个是纯副本**（`FACTS.md F2.13`）。
例如 `v116_vsync_batched` / `v116_batched_reducesum` 与 `v111_path2_group4` **md5 完全相同**。

**后果**：文档声称"v116 实机验证通过"，实际验证的是 v111 的代码——**结论张冠李戴**。

**正确做法**：新建版本后**立即 md5 对比上一版**确认有实质改动；
声明"某版本验证通过"前，先确认该版本的代码确实包含所声称的改动。

---

### P14. 性能归因前先确认"这段代码真的在跑"

**现象**：v110/v111 宣称"批处理改造"并据此得出"同算法重排无效"的结论；
实际因 `block_dim` 公式（`FACTS.md F2.3`），Case5 的 `cnt` 恒为 1，**批处理从未生效**。

**后果**：一条核心结论建立在空转的代码上，且后续"必须换归约原语"的推论也从此出发。

**正确做法**：下"这个优化无效"的结论前，**先验证该优化路径确实被执行**——
加个计数器、或从探针反推。**"代码写了"不等于"代码跑了"。**

---

### P15. floor 模型必须自洽性检查

**现象**：探针给 Case5 floor=3.38μs，但榜首 Case3/4 实测 2.84/2.92μs——
比更大的 case 的 floor 还低。**这在"floor 是常量"的模型下不可能**。

**根因**：P1 探针只清空 `ProcessVectorRow`，而 weight DMA 在 `Process()` 内仍执行，
且搬运量随 block_dim 变化（Case5 是 Case3 的 16 倍）。所谓 floor 里混着**随 shape 变化的成本**。

**正确做法**：声称"X 是固定开销"前，检查它在所有 case 上是否真的恒定。
**跨 case 不自洽的数字，说明模型里混进了别的变量。**

**证据**：`FACTS.md F2.6`

---

### P16. 探针归因是"减法"，只能定界不能定位

**现象**：WA 探针法（清掉一段代码，看时间差）是有效的方法，但只能给出**阶段级**上界，
不能精确定位到指令级。且差值里可能混入其他变量（见 P15）。

**正确做法**：探针给的是**假设**，需再用第二种手段（如 nsys、或对照实验）交叉确认。

---

## 四、本仓库特有陷阱

### P17. 本机无法运行任何 kernel 验证

本机**无 CANN、无 NPU 设备**（`/usr/local/Ascend` 与 `/dev/davinci*` 均不存在）。
所有编译/性能验证必须走远程实机：
```powershell
pwsh -File D:\Desktop\OP-Learning\Ascend\atomgit-devspace-tools\Connect-AtomGitDevEnv.ps1
```

**含义**：不要在本机尝试 `cmake`/`make` 验证 kernel；也不要基于本机无法运行就宣称"未验证是环境问题"——
实机是可达的，只是要显式连接。

---

### P18. 提交工具与凭据（2026-09-11 更正）

⚠ **更正**：此前记录说"凭据不在本仓库、需先确认来源"——**不准确**。

**实际可用的工具是明文版**（`Ascend/cann-learing-hub/skills/cannjudge-submit-plaintext/`）：
- ✅ `.env` **已存在且凭据有效**（实测 `login` 成功，用户 `kkio-`）
- 仅依赖 `requests`（原 RSA 版还需 `pycryptodome`）
- **免登录能力**：`problem` / `rank` / `query` / `download` 四个命令**无需登录**即可用

```bash
cd D:/Desktop/OP-Learning/Ascend/cann-learing-hub/skills/cannjudge-submit-plaintext
python cannjudge_cli.py rank  --problem-id 6a7c23a6a52e0f540a8a1779   # 免登录
python cannjudge_cli.py query --submission-id <sub_id>                 # 免登录
python cannjudge_cli.py login                                          # 用 .env
python cannjudge_cli.py submit --problem-id <id> --project-dir <dir>   # 需登录
```

⚠ **同一个 CLI 的另一处坑**：它把 `time` 字段显示成 `ms`，但后缀是**硬编码**的
（`:225`），API 返回的是裸数字。见 `FACTS.md F2.1`。**不要相信 CLI 输出的单位标签。**

---

### P19b. NPU 实机有两条路，都需先手动准备

**路径 1 — AtomGit Dev Space（`verify.sh` 用的）**
```powershell
pwsh -File D:\Desktop\OP-Learning\Ascendtomgit-devspace-tools\Connect-AtomGitDevEnv.ps1
```
⚠ 2026-09-11 实测：报
`AtomGit plugin did not establish a usable SSH forward. Open AtomGit Dev Space and verify the environment is running.`
→ **需先在 AtomGit Dev Space 里手动启动实例**，SSH forward 才建立。

**路径 2 — GitCode Notebook（`ascend-npu-console` 用的）**
工具包: `Ascend/ascend-npu-console/`
⚠ 它的链路依赖 `mcp__node_repl__js` 宿主运行时（`notebook.js` 内用 `agent.browsers`），
**该运行时在标准工具集中不存在**。替代思路：用浏览器调试工具在页面上下文 `eval` 执行等价 JS。
⚠ Notebook 是**临时环境**（~2 小时倒计时，重启后本地文件全删）——**要保留的先 commit**。
⚠ **铁律：零页面导航**。禁止对 Notebook 页 `goto`/`reload`，整页导航会重载前端并可能重启实例
（2026-09-04 实机踩坑）。

**路径 3 — OJ 在线评测（已验证可用，无需 NPU）**
`cannjudge-submit-plaintext` 的 `.env` 凭据有效，可直接提交。这是当前**最可靠的验证手段**
（见 P9 限流注意）。

### P19. `xingchen-operators` 是独立 git 仓库，且缺 `.gitignore`

- 仓库根：`D:\Desktop\OP-Learning\Ascend\xingchen-operators`（有自己的 `.git`）
- 赛题目录**已被追踪**（2556 个文件）
- **无 `.gitignore`** → `__pycache__/`、`build_*/` 等会混入提交

**正确做法**：提交前看清 `git status`，不要把产物目录带进去。

### P20. 361001 报错的真凶是环境变量缺失（2026-09-14，双 CANN 版本验证）

`GetWorkspaceSize failed: 361001` 与 CANN 版本无关——根因是 **`ASCEND_HOME_PATH` 未导出**。
notebook conda 环境没有 `/etc/ascend_install.info`，runtime 定位不到 CANN 根，算子元数据加载失败。
只设 `LD_LIBRARY_PATH` + `ASCEND_OPP_PATH` **不够**。修复（8.5/9.0 双双实测 PASS）：

```bash
export ASCEND_HOME_PATH=<CANN根>   # 8.5: /tmp/cann85/cann-8.5.0；9.0: /opt/conda/Ascend/cann-9.0.0
export ASCEND_AICPU_PATH=<同上>
export ASCEND_OPP_PATH=<CANN根>/opp
```

诊断路径：`ASCEND_GLOBAL_LOG_LEVEL=0` 重跑，看 `~/ascend/log/debug/plog/*.log` 里的
`can not get env [ASCEND_HOME_PATH]`。

### P21. 8.5 toolkit 安装器交互式 EULA 会假死（2026-09-14）

`--install-path` 安装时不带 `--quiet`，安装器打完 EULA 后等待 stdin，后台跑表现为"exit 0 但无产物"。
**必须加 `--quiet`**（隐含接受 EULA）。官方直链：OBS `ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/`。

### P22. 链接 CANN 库的测试程序需要 rpath-link 三件套（2026-09-14）

g++ 链接 libascendcl/libnnopbase 时报 `undefined reference to hal*/drv*` 符号——不是缺库，
是 ld 找不到 driver 侧依赖链。必须同时给三个 rpath-link：

```bash
-Wl,-rpath-link,<cann>/aarch64-linux/lib64 -Wl,-rpath-link,/usr/local/Ascend/driver/lib64/driver -Wl,-rpath-link,/usr/local/Ascend/driver/lib64/common
```

另注意：链接失败时 g++ 可能留下**非 ELF 残骸文件**（`file` 显示 `data`），chmod +x 后报
Exec format error——报错先 `file` 一下产物再查别的。

### P23. cp -r 拷贝工程会把 build 缓存一起带来（2026-09-14）

`cp -r code /tmp/vNNN` 后直接 cmake 会报
`CMakeCache.txt directory is different` + `source does not match`——旧 cache 指向
原路径（且可能指向另一 CANN 版本的 asc-config）。**拷贝后必须 `rm -rf build` 重建**。
更阴险的变体：build 在别的子目录（build85/）漏删，cmake 走了但链接到旧对象。

### P24. `&&` 长链静默断裂：BUILD_OK 没打但 INSTALL_OK 出现（2026-09-14）

`cmake && make && make package && echo BUILD_OK; ./xx.run --install ... && echo INSTALL_OK`
——当 cmake 失败时 `&&` 链断，但 `;` 后的安装命令照跑（装的是**旧 .run**），打出
INSTALL_OK 造成"成功"假象。教训：**链尾回显（BUILD_OK）没出现=链条断了**，必须
核查；关键步骤间用显式 `echo step=$?` 分隔，或干脆分命令跑。

### P25. .run 安装目录的 scripts/ 是 root 属主，rm -rf 卡权限（2026-09-14）

安装器在 `<inst>/vendors/custom/scripts/uninstall.sh` 写 root 文件；下次
`rm -rf /tmp/xx_inst && ./xx.run --install-path=/tmp/xx_inst` 时 rm 报
Permission denied 且 `&&` 断链。**解法：每次安装换全新目录名**
（v119_inst → v119b_inst → v120_inst），别删旧的；旧目录浪费的 /tmp 空间可忽略
（20T 盘）。若必须清理：`rm -rf` 单独跑并容忍失败（`; true`）。
