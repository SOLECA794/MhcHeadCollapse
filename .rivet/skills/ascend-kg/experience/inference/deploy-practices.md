# 推理部署实战经验（昇腾 NPU · vLLM-Ascend）

> 沉淀自 2 个真实推理部署任务：Qwen2.5-7B（`llm-deploy-vllmascend` 会话）与 Qwen3-0.6B（`qwen-deploy` 会话），均在本机 8×910B3（Atlas 800I A2，64GB/卡）上经 vLLM-Ascend 完成。
> 与 KG `vllm-ascend-deploy` / `vllm-ascend-performance-optimization` **互补**：**KG = 标准答案**（该用什么 flag / 版本 / 顺序），**本库 = 现场答案**（这台机器怎么装成 / 怎么救回来 / 怎么清干净 / 怎么选 skill）。

## 0. 核心思想（先抽象，再细节）

**环境就绪 ≠ 推理可用**（用户方法论，两次会话共同验证）：
昇腾推理栈是分层依赖链 `驱动 → CANN → torch_npu → NNAL/ATB → 推理框架`，任一环断裂都会让上层崩在深处。三条铁律：

1. **验证走真实路径**：框架常延迟注册（如 ATB 扩展惰性加载，`torch.npu.is_available()=True` 是假阳性）——`import` / `is_available()` 成功只证当前层，**端到端最小通路（`import vllm_ascend.vllm_ascend_C` + 真实推理请求）才算数**。
2. **同源同版本**：同一组件多实例禁混用（本机两套 CANN、conda 内与系统 NNAL 并存），以 `$ASCEND_*` 环境变量指向的唯一实例为准。
3. **依赖缺失先分两态**：`文件缺失` vs `存在但运行时/版本不兼容`（glibc / torch ABI），同根于依赖链断裂。

> 方法论来源（用户原话）：skill 只存"能指导一类问题的判断"，不存"某一次问题的答案"；实例化细节（版本号 / 路径 / 错误码值 / 单模型坑）不进 skill——无论主文件还是 references。**判断标准：这条经验能否规避一类问题（而非覆盖某一种报错）。**

## 1. 阶段 × Skill 路由表（推理/部署域）

| 阶段 | 加载 Skill（node_id / FAQ） | 说明 |
|---|---|---|
| 部署环境检查 / 编排 | `vllm-ascend-deploy` | 配置必须官方文档验证（**禁止猜测**）；后台启动必须 cron 监控 |
| 推理性能调优 | `vllm-ascend-performance-optimization` | 交互式顾问：FULL_DECODE_ONLY / FlashComm1(EP) / chunked-prefill / multi-stream MLA |
| 故障诊断（启动失败） | `fault_diagnose` | ⚠️ 实测缺口：libatb.so / glibc 两次失败全程本地排查，首查该 skill 可带出故障模式清单 |
| 环境安装（NNAL 缺失） | `cann-nnal-installer` | 比 `find / -name libatb.so` 直接 |
| 多机 / 多卡 | FAQ `N-0457`（Ray+HCCL） | rank_table device_ip 须与 `hccn_tool` RoCE IP 一致 |

> ⚠️ **两个部署 skill 并存**：`vllm-ascend-deploy`（docker 编排 + cron 监控 + 独立 browser 查镜像）vs MindSpeed `vllm-ascend-serving`（脚本层 `serve_start/status/stop/probe_npus.py`）。选型看本机 harness。

## 2. 部署前环境检查清单（可复现）

| 检查项 | 命令 | 本机关键值 / 要点 |
|---|---|---|
| OS / 架构 | `uname -m`、`cat /etc/os-release` | aarch64 / Ubuntu 22.04 |
| NPU 卡 / 型号 / 驱动 | `npu-smi info` | 8×910B3 = **A2**（16×910B3 = A3，Docker 镜像按 `-a2`/`-a3` 后缀匹配）；**记录每卡空闲 HBM 基线**（重启判据用） |
| CANN 归属 | `echo $ASCEND_TOOLKIT_HOME` | 用户自有 `~/Ascend/cann-9.0.0`（release）优先；**禁 source /usr/local** |
| 推理栈版本 | `pip list \| grep -iE 'vllm\|torch\|npu'` | 逐层对齐，别只测 Python 包 |
| C++ 扩展可达 | `import vllm_ascend.vllm_ascend_C` | ⚠️ Python 级 `import vllm` 成功 ≠ C++ 扩展可加载（torch ABI） |
| 模型完整性 | `ls <model_dir>/; du -sh` | config / 权重 / tokenizer 齐全 |

## 3. 踩坑清单（五段式，🔥 高频优先）

### ①🔥🔥 版本钉死不匹配 → `std::bad_alloc`（torch ABI，`--no-deps` 混装必现）
- **触发**：任意 `vllm serve` 启动 / `import vllm_ascend.vllm_ascend_C`。
- **症状**：`Engine core initialization failed. See root cause above`；EngineCore 子进程 <1s 崩，日志 `terminate called after throwing an instance of 'std::bad_alloc'`。ldd 无缺库、torch_npu 设备 alloc/matmul 正常——**迷惑性极强**。
- **根因**：vllm-ascend 预编译 wheel `METADATA` 钉死 `Requires-Dist: torch==2.9.0`，本机 torch 2.7.1 → C++ 扩展按 2.9.0 ABI 编译、运行时加载 2.7.1 库，初始化读垃圾值。
- **修复**：装 wheel 要求的精确 torch/torch_npu 版本；决定性验证 `import vllm_ascend.vllm_ascend_C` OK。
- **教训**：**检查 wheel `Requires-Dist` 版本钉死**；环境检查必须测 C++ 扩展导入，不只是 Python 包。

### ②🔥 CANN 归属错误（用户红线）
- **触发**：部署前置 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- **症状**：生效 root 属主系统 CANN（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0-beta.2`），用户质问"你使用的CANN是哪里的？我要用我自己的"。
- **根因**：误用系统级 CANN；用户自有 `~/Ascend/cann-9.0.0`（release）shell 已配好。
- **修复**：只用 `$ASCEND_TOOLKIT_HOME` 指向的用户 CANN，不碰 `/usr/local`。
- **教训**：昇腾任务先 `echo $ASCEND_TOOLKIT_HOME` 判定归属，**用户自有优先**。

### ③🔥 `libatb.so` 缺失（NNAL 组件，两个会话都踩）
- **触发**：首次 `vllm serve` / 服务走到 worker 设备初始化后。
- **症状**：`OSError: libatb.so: cannot open shared object file`（EngineCore 启动失败，末尾 `ERR99999`）。
- **根因**：手动安装 CANN ≠ 自带 NNAL/ATB；或只 source CANN env 没 source ATB env。
- **修复**：`source ~/Ascend/nnal/atb/set_env.sh`（自动选 cxx_abi_1）；或 `find / -name libatb.so` 定位后把 `cxx_abi_1/lib` 加入 `LD_LIBRARY_PATH`。
- **教训**：裸机环境不止 CANN 一个 env，**"启动序列必须 source 哪些 env"要显式写进 runbook**。

### ④🔥 `torch.npu.is_available()=True` 是假阳性
- **触发**：环境预检用它判定"环境可用"，import `_atb_ops` 也成功。
- **症状**：预检全绿但 `vllm serve` 启动即挂。
- **根因**：ATB 扩展延迟注册——`_atb_ops.py` 顶层 `try/except OSError` 把 libatb 加载失败吞进全局变量，`import` 不报错；只有 `_register_atb_extensions()`（`@lru_cache`）才真正 raise。
- **修复**：验证必须显式 `from torch_npu.op_plugin.atb import _atb_ops; _atb_ops._register_atb_extensions()` → 输出 `ATB register: OK` 才算通过。
- **教训**："环境就绪 ≠ 推理可用"；验证走真实路径，端到端最小通路才算数。

### ⑤🔥 glibc 不兼容（文件存在 ≠ 可用）
- **触发**：改用系统 NNAL 后第二次启动。
- **症状**：`libatb.so` 找到了但进程又退（undefined symbol / GLIBC 版本不足类）。
- **根因**：`readelf --version-info libatb.so` → 系统 NNAL 需 **GLIBC_2.38**，Ubuntu 22.04 只有 **2.35**（`ldd --version`）；conda 内 NNAL 只要求 **GLIBC_2.17**。
- **修复**：`readelf --version-info` 核实后选用兼容实例。
- **教训**：依赖缺失先分两态——"文件缺失" vs "存在但运行时/版本不兼容"；`readelf --version-info` 是标准核对手段。

### ⑥🔥 重启竞争 → `507033 / TsdOpen failed`
- **触发**：`kill $(cat /tmp/vllm.pid)` 后立刻重拉。
- **症状**：`RuntimeError: error code 507033 | TsdOpen failed. devId=0 | open device 0 failed`。
- **根因**：kill 只杀 APIServer 父进程，EngineCore 子进程成孤儿仍持 device 0，新进程抢不到卡；`ps` 无残留、HBM 已回基线时即为**释放竞态**。
- **修复**：等卡 HBM 回到空闲基线再重拉；轻量预检 `python -c "import torch_npu; torch.npu.set_device(0)"` 通过再起。
- **教训**：重启 vLLM 要「先确认设备释放、再拉起」，避免浪费 ~125s 启动周期。

### ⑦ setuptools≥81 移除 pkg_resources → torchair 导入失败
- **触发**：torch_npu 升级后启动。
- **症状**：EngineCore `torch_npu/dynamo/torchair/__init__.py` → `ModuleNotFoundError: No module named 'pkg_resources'`。
- **根因**：setuptools 82.0.1 起移除 `pkg_resources`，torch_npu 的 torchair 子模块仍依赖。
- **修复**：`pip install 'setuptools==75.8.0'`（必须 <81）。
- **教训**：升级 torch_npu 是连锁反应，配套 setuptools 版本也要核对；这类 env 缺件常被前一崩溃掩盖。

### ⑧ fastapi 过新 → HTTP 500
- **触发**：引擎 warmup 成功但 HTTP 层调用。
- **症状**：`/v1/chat/completions` 返回 500，`'_IncludedRouter' object has no attribute 'path'`。
- **根因**：fastapi 0.137.1 改 router API；vllm-ascend 钉 `fastapi<0.124.0`（vllm 只要求 ≥0.115）。
- **修复**：降 `fastapi==0.123.10`（连带 starlette 0.50.0），重启。
- **教训**：pip 警告（依赖钉死）别当耳边风；vllm-ascend 与 vllm 的依赖 pin 交叉区间要精确取交集。

### ⑨ `pkill -f` 自匹配
- **触发**：清理旧进程 `pkill -f "vllm serve"` 后立即重启。
- **症状**：`exit 144`，新进程也被带走、旧日志未被覆盖。
- **根因**：`pkill -f` 匹配到命令自身所在 shell（命令行含 "vllm serve"）。
- **修复**：`pkill -f "[v]llm serve"`（正则括号防自匹配）或 `kill $(cat /tmp/vllm.pid)`。
- **教训**：`pkill -f` 带正则括号防自匹配。

### ⑩ 杂项
- `import vllm_ascend; vllm_ascend.__version__` 报 AttributeError → 用 `pip show vllm_ascend` 查版本。
- `npu-smi info -t pid` 语法非法 → 查占用用 `npu-smi info`（HBM）+ `ps -eo pid,ppid,etime,cmd | grep EngineCore`，或 `npu-smi info -t proc`。
- Qwen3 默认 thinking 会输出 `<think>`，0.6B 建议请求体加 `"chat_template_kwargs":{"enable_thinking":false}`。
- 模型到 A2：BF16 原精度**不加 `--quantization ascend`**；多卡 TP>1 才需 FlashComm1/HCCL/TASK_QUEUE env。

## 4. 服务启停与资源释放

**启动序列**（Qwen3-0.6B / Qwen2.5-7B 实测，~90-125s 就绪）：
```bash
# 自包含 runbook（fresh shell 可执行）：conda activate 必须在 source CANN/NNAL 之前！
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ascend-dev                     # 先激活 Python 环境
source ~/Ascend/cann-9.0.0/set_env.sh         # 再 source CANN
source ~/Ascend/nnal/atb/set_env.sh           # 最后 source NNAL（libatb.so）
export ASCEND_RT_VISIBLE_DEVICES=0            # 固定只用卡0，别占满8卡
nohup vllm serve <model_dir> --served-model-name <name> --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8000 --trust-remote-code > /tmp/vllm.log 2>&1 &
echo $! > /tmp/vllm.pid
```
> ⚠️ **坑：conda activate 会重置 LD_LIBRARY_PATH**。在已激活 conda 的交互 shell 里 `source CANN → source NNAL → vllm serve` 没问题；但写进**自包含脚本**（fresh bash）时，若把 conda activate 放在 source CANN/NNAL 之后，`LD_LIBRARY_PATH` 里 atb 库路径被冲掉 → `libatb.so` 找不到 → EngineCore 崩溃。**顺序必须是 conda activate 在前、CANN/NNAL 在后**（或直接用绝对路径 `~/miniconda3/envs/ascend-dev/bin/vllm`）。判据：`echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -c atb` ≥1 才说明 ATB 在加载路径里。
就绪信号：日志 `Application startup complete` + `curl -s http://localhost:8000/health` 返回 200。

**停止序列**（先确认设备释放，再拉起）：
```bash
kill $(cat /tmp/vllm.pid)          # 只杀父进程，EngineCore 可能残留
kill -9 <EngineCore_pid>           # 残留子进程单独清
pgrep -af "[E]ngineCore"           # 确认清干净
npu-smi info                       # 各卡 HBM 回任务前基线（波动 <10MB）才算释放完成
```

## 5. KG 使用复盘（部署域）

- **用的对**：KG 命中版本兼容矩阵 / 单卡部署样例 / FAQ（libatb.so 线索）→ 直接锚定启动参数，省掉官方文档逐页爬取与参数试错；`vllm-ascend-deploy` skill 给出"禁止猜测 + cron 监控"两条硬约束。
- **该用没用**（两次会话共性，勿重蹈）：① 环境检查全本地 Bash、开局未加载 `vllm-ascend-deploy`（若开局加载，"禁止猜测"会推动先查 wheel METADATA → 提前抓出 torch ABI 坑）；② 故障时惯性滑回通用调试，没查 `fault_diagnose`（多花 3-4 轮试错）；③ 检索一次不理想即弃，没换中英文改写 / `file_type: error_code` 过滤 / 查相邻 FAQ。
- **接力而非互替**：KG 对版本矩阵确认、skill 硬约束、报错消歧增益明确；根因级细节（wheel pin、env 缺件、glibc 版本）靠本地实证（readelf/ldd/wheel METADATA）。

## 数据来源

- 会话 `~/.claude/projects/-home-z60124773-projects-ascend-kg-optimize-llm-deploy-vllmascend/f443e4e2-1866-4e52-bf87-d19a36d40aea.jsonl`（Qwen2.5-7B，956 行）
- 会话 `~/.claude/projects/-home-z60124773-projects-qwen-deploy/ef979bab-edab-419f-a09b-bd4c008b39ad.jsonl`（Qwen3-0.6B 部署）+ `91f3b660-62e0-48c0-adbf-2f7233391e73.jsonl`（服务停止/资源释放）
- 参考：KG `vllm-ascend-deploy` / `vllm-ascend-performance-optimization` / FAQ `N-0801` `N-0813` `N-0457`（标准答案，勿在本库重复）
