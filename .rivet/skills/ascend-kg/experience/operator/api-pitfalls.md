# API / 编译 / 工程避坑清单（声明性）

> **触发**：编译 / 链接 / API 报错，同一报错 >2 次时逐条对照。
> 格式：`症状 → 根因 → 修复`。🔥 = 高频（多任务踩过或处于编译/精度关键路径）。
> 来源：11 任务复盘 + 聚合文档 + 会话现场记录。

## A. 数学 / 向量 API 签名

| 🔥 | 症状 | 根因 | 修复 |
|---|---|---|---|
| 🔥 | 编译报 `Subs` 不存在 | AscendC 无 `Subs` | `Adds(x, -scalar, n)` |
| 🔥 | 编译报 `ReduceType::SUM` 不存在 | 无该枚举 | Level-2 `ReduceSum<T>(dst, src, tmp, count)` |
| | 编译报 `sqrtf` 不匹配 | 不是 C 的 sqrtf | `AscendC::Sqrt<T>` |
| | 设计稿写 `WholeReduceSum` 等意图 API | 设计阶段只写意图 | 实施前必查 KG `ascendc-api-best-practices` 落定精确签名，查一次胜过几轮试编译 |
| 🔥 | `Ands`/`Select`/`Sub`/`Not` 不含 int32 | 向量指令集不支持 int32 | IntSelect mask trick：`~m = Sub(allOnes, m)`；And/Or 走 int16-reinterpret + count×2 |
| | `ReduceMin`/`ReduceMax` 编译过但行为错 | A2 仅支持 float | int32 需先 Cast 到 float |
| | `ShiftRight` 无逐元素张量版 | 910B 仅标量版（仅 950 有张量版） | 用标量 + 循环 / 换算法 |
| | fp32 精度 FAIL（vrec ~2⁻¹⁰ > 阈值 2⁻¹³） | A2 `Reciprocal` 仅 INTRINSIC 精度 | fp32 保留 `Div(1,x)`，仅低精度 dtype 用 Reciprocal（省 UB buffer 是额外红利） |

## B. dtype 与 Cast

| 🔥 | 症状 | 根因 | 修复 |
|---|---|---|---|
| 🔥 | 编译报 `CastIntrinsicsImpl no matching function` | A2 `Cast` float→int8/uint8 无直转重载 | 两步经 half：`float→half[CAST_RINT]→int8[CAST_TRUNC]`；int16/int32 可一步 |
| 🔥 | bf16 输出出现 `[nan,...]` 错乱且难关联 | bf16 标量读取只 `SetValue(0,val)` 写 1 元素，向量 Cast 处理其余未初始化 bf16（含 signaling NaN）污染向量管线 | **向量 Cast 元素必须全部有效初始化**：`Duplicate<bfloat16_t>(s,val,16)` 填满再 Cast |
| | unsigned 高位值右移错 | int32 算术右移填符号位 | `logical = arith & ShiftRight<uint32>(allOnes, s)` 剥符号填充；`ShiftRight<uint32>` dst 显式 ReinterpretCast |
| 🔥 | int 精度 ±1（~20% 元素） | 舍入模式错 | int 用 `CAST_TRUNC`（对齐 numpy.astype/TBE）；half/bf16 用 `CAST_RINT`。KG"推荐"≠内置行为，**以内置实测为准** |
| | `uint→float` 隐式转换禁止 | 编译期约束 | host 侧预算 `invN` 传入 tiling |

## C. 构建 / 工程

| 🔥 | 症状 | 根因 | 修复 |
|---|---|---|---|
| 🔥 | 运行时 561103(ACLNN_ERR_INNER_NULLPTR) 且包内无 kernel .o | **软链**致 `gen_ops_info.cmake` 的 `find`（无 `-L`）不遍历 → 找不到 `OP_ADD` → "op type undefined, skip binary compile" | 源码**真实目录拷贝 `cp -r`**（禁用软链）+ `--make_clean` 清 op_type 缓存 |
| | 561103 **全 dtype** 失败（含原本正常的 float） | `op_api/*.cpp` 的 `IsAiCoreSupport` 运行期关卡硬编码 `DAV_3510`（仅 950），本机 910B3（DAV_2201）被拒 | 放行 `if (npuArch == DAV_3510 || npuArch == DAV_2201)`；dtype 白名单三处（op_api RESULT_DTYPE_SUPPORT_LIST / AICORE_DTYPE_SUPPORT_LIST / op_host def）保持一致 |
| | `--make_clean` 后瞬间返回无产物 | 该参数语义是"仅 clean 不 build" | 本地验证用 `-u --experimental`；改 header 必 `rm -rf build_out` 全量重编 |
| | 增量 build 跑旧 kernel | 不重编译改动的 header | 改 `.h` 后全量重编，不依赖增量 |
| | `--opkernel` 编译 stable 算子致符号冲突 | ops-math 有 **两个同名目录**（`math/` stable + `experimental/math/`） | 必须 `--experimental` 才编译 experimental 版；移开同名 stable 算子 |
| 🔥 | `multiple definition of aclnnXxx/GetWorkspaceSize`（ld error） | 手写 op_api 的 CMakeLists 用 `ACLNNTYPE aclnn` 致 autogen 与手写重复定义 | 用 `ACLNNTYPE aclnn_exclude`（ops-math 隐性约定，读同类算子 CMakeLists 才发现） |
| 🔥 | 报 `kernel entry 'xxx' not implement` / 路径对不上 | 命名不一致：**目录/文件/核入口 = snake_case，op type = CamelCase** | 开工前 `ls experimental/math/` 看既有算子；复制模板连函数名大小写、include 路径都 1:1 |
| | `'BroadcastTo' is not a member of 'l0op'` | aclnn 层用 `l0op::BroadcastTo` 缺头文件 | `#include "conversion/broadcast_to/op_api/broadcast_to.h"` |
| | CPU-debug UT 报 `XXX_SCH_MODE_* not declared` | schMode 宏放 `tiling_key.h`，CPU UT 走 `#if ASCENDC_CPU_DEBUG` 不 include 它 | 宏放 `tiling_data.h` |
| | 手动编译 example 链接失败（绕 3 次） | 链接库不全 / 错 | `-I vendor/op_api/include -I $CANN_INC -I $CANN_INC/aclnnop` + `-lcust_opapi -lnnopbase -lascendcl -lc_sec`（**无 libacl.so**，aclCreateTensor 在 libnnopbase） |
| | `msprof op --output=/tmp/...` 报 "writable by any other users" | 输出目录非世界可写 | 项目内私有目录 + `chmod 700` |
| | 复用开源 aclnn 源码在新架构行为错 | 其路由逻辑假设旧架构（如 `IsRegBase` 仅 A5 true，A2 全走 AICPU 崩溃） | **逐行核对架构相关分支**；nullptr→`AllocTensor(shape={0},FLOAT)` 哨兵 |
| | 宏/API 跨 CANN 版本未声明 | 版本差异 | 以本机头文件 grep 为准，对齐同仓 stable 算子惯例 |
| | 空张量 `GRAPH_FAILED` / 越界 | 只在单层处理空输入 | **三层防御**：aclnn `IsEmpty()` 早退 + tiling `inputNum==0→coreNum=1` + kernel `coreDataNum==0` 守卫 |

## 速查：dtype 支持矩阵（A2 / 910B，以头文件 static_assert 为准）

| API | 限制 |
|---|---|
| `Ands`/`Sub`/`Adds`/`Not` | 不含 int32 |
| `Select` | 数据仅 half/float |
| `ShiftRight` | 仅标量，scalar∈[0,32]；910B 无逐元素张量版 |
| `ReduceMin`/`ReduceMax` | A2 仅 float（int32 需 Cast） |
| `Cast` float→int8/uint8 | A2 不支持直转，两步经 half |
| bf16 标量→float | 不支持静态转换，必须经向量 Cast + 元素全初始化 |
| `Reciprocal` | A2 仅 INTRINSIC（~2⁻¹⁰），fp32 不满足需保留 `Div` |

> KG 对细粒度 dtype 矩阵命中弱（常返 LLVM/attention 噪声）；**头文件 `static_assert(SupportType<...>)` 编译期强制最权威**。
