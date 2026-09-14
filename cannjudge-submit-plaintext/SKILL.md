# CANNJudge 算子提交 Skill（本地明文版）

帮助用户在本地完成 CANNJudge 算子竞赛的完整流程：配置凭据、登录、下载工程、理解题目、实现泛化算子、提交代码、获取结果。

> 本 Skill 是 `cannjudge-submit` 的本地化改造：**登录改为本地明文凭据**（`.env` / 环境变量），移除了服务器 RSA 密钥对机制，其余平台交互流程保持一致。

## 触发条件

当用户需要：
- 参加 CANNJudge 算子竞赛
- 从 CANNJudge 下载算子工程模板
- 提交算子实现到 CANNJudge
- 查看 CANNJudge 排行榜

## 前置要求

- 已注册 CANNJudge 账号（记录邮箱与密码）
- 已配置 CANN 开发环境与编译工具
- Python 3.8+，依赖 `requests`：`pip install requests`

## 登录机制（本地明文）

### 设计说明

原版面向“服务器 CANNBot 代登”场景，采用 RSA 公私钥加密密码。本版面向**个人本地电脑**直接操作，登录凭据改为明文，来源优先级：

```
命令行参数 --email/--password
        > 环境变量 CANNJUDGE_EMAIL / CANNJUDGE_PASSWORD
        > 本地 .env 配置文件
```

### 首次配置（一次性）

```bash
cd skills/cannjudge-submit-plaintext
python cannjudge_cli.py setup
# 交互式输入邮箱与密码，自动生成 .env（权限 600）
```

或手工复制模板：

```bash
cp .env.example .env
# 编辑 .env 填入 CANNJUDGE_EMAIL / CANNJUDGE_PASSWORD
```

### 安全约定（本地明文版铁律）

1. **密码只写入本地 `.env`（chmod 600）或环境变量，禁止写入代码、日志、对话输出**
2. **运行过程不打印密码**：登录成功仅输出用户 ID 与昵称
3. `.env` 已被 `.gitignore` 排除，提交代码前检查是否误提交
4. 若在共享电脑使用，建议用完删除 `.env` 或改用环境变量注入

## ⚠️ 重要：必须从网站获取题目信息

**禁止使用本地任何算子信息文件，也不要私自决定使用标准定义！**

题目信息（描述、算子原型、工程模板）必须通过 API 从 CANNJudge 获取，原因：
- 题目可能随时更新
- 确保信息准确性与一致性
- 竞赛存在平台限制

获取方式：

```python
# 方式1: 按题目名称（推荐，名称更稳定）
client.get_problem_by_name("depthtospace")

# 方式2: 按题目 ID
client.get_problem("题目ID")
```

## ⚠️ 重要：理解题目与参考算子

### 关键步骤

1. **仔细阅读题目描述**：算子原型（输入/输出/属性）、支持的数据类型、数学公式、参考算子
2. **查阅参考算子官方文档**（最重要）：理解所有行为，特别注意默认值的特殊语义
3. **验证理解**：用参考算子（如 PyTorch）测试边界情况

### 常见陷阱

| 陷阱 | 错误理解 | 正确理解 |
|------|---------|---------|
| `min=0, max=0` | 范围是 [0, 0] | 自动计算数据 min/max |
| 默认值 | 直接使用默认值 | 可能是特殊标志 |
| 参考算子 | 仅参考原型 | 必须完全复现所有行为 |

### 实战案例：Histogram 算子（torch.histc）

- 题目属性：`bins=100, min=0.0, max=0.0`
- 错误实现：认为 min/max 是有效范围，回退到 [0, 1]
- 正确实现：当 `min==0 && max==0` 时设置 `need_compute_range=true`，在 Kernel 中动态计算数据范围

## ⚠️ 重要：算子泛化性设计

**平台不开放测试用例 API，必须设计泛化算子！**

| 要素 | 说明 | 设计要点 |
|------|------|----------|
| Shape 泛化 | 任意维度与大小 | 动态 shape，不硬编码维度 |
| Dtype 泛化 | 多种数据类型 | 模板类 + dtype 分发 |
| 属性泛化 | 各种属性取值 | 边界值、默认值、负 axis |
| 对齐泛化 | 对齐与非对齐 | DataCopyPad 处理尾部数据 |
| 边界泛化 | 边界情况 | 空输入、单元素、极端值、NaN/Inf |

```cpp
// ✅ 动态处理任意维度
int32_t rank = shape.GetDimNum();
uint32_t totalLength = 1;
for (int32_t i = 0; i < rank; i++) { totalLength *= shape.GetDim(i); }

// ✅ 模板支持多 dtype
template<typename T> class KernelOp { ... };
extern "C" __global__ __aicore__ void op_kernel(...) {
    if (dtype == 0) { KernelOp<float> op; op.Process(...); }
    else if (dtype == 1) { KernelOp<half> op; op.Process(...); }
}

// ✅ 处理非对齐尾部
uint32_t tileNum = (totalLength + 255) / 256;
uint32_t lastTileLength = totalLength - (tileNum - 1) * 256;
```

## API 接口

基础 URL：`https://cannjudge.cn`

| 功能 | 方法 | 路径 |
|------|------|------|
| 登录 | POST | `/api/users/login`（Body: `{"email","password"}`） |
| 题目信息（ID） | GET | `/api/problems/{problemId}` |
| 题目信息（名称） | GET | `/api/problems/name/{problemName}`（推荐） |
| 下载工程 | GET | `/api/problems/{problemId}/package?userId={userId}` |
| 提交代码 | POST | `/api/submissions/submit` |
| 查询结果 | GET | `/api/submissions/{submissionId}` |
| 排行榜 | GET | `/api/problems/{problemId}/ranking`（网站同源；回退 `last_submission` / 旧接口） |

提交 Body：

```json
{
  "problemId": "题目ID",
  "userId": "用户ID",
  "tiling_h": "tiling头文件内容",
  "tiling_key_h": "tiling key头文件内容（可空）",
  "host_cpp": "host侧cpp内容",
  "kernel_cpp": "kernel侧cpp内容"
}
```

## 完整流程（命令行）

```bash
# 0. 首次配置凭据（仅 login / submit 需要；problem / rank / query / download 为公开接口）
python cannjudge_cli.py setup

# 1. 登录（验证凭据与网络）
python cannjudge_cli.py login

# 2. 获取题目信息（ID 或名称，无需登录）
python cannjudge_cli.py problem --problem-name depthtospace
python cannjudge_cli.py problem --problem-id 题目ID

# 3. 下载工程模板（无需登录）
python cannjudge_cli.py download --problem-id 题目ID --output ./output

# 4. 实现泛化算子（op_host / op_kernel 下相关文件）

# 5. 提交代码并等待结果（需登录）
python cannjudge_cli.py submit --problem-id 题目ID --project-dir ./output

# 6. 查询历史提交（无需登录）
python cannjudge_cli.py query --submission-id 提交ID

# 7. 查看排行榜（无需登录，与网站 /public/.../ranking 同源）
python cannjudge_cli.py rank --problem-id 题目ID
```

代码方式：

```python
from cannjudge_cli import CANNJudgeClient
client = CANNJudgeClient(email="you@example.com", password="本地密码")
client.login()
problem = client.get_problem_by_name("depthtospace")
project_dir = client.download_package(problem["_id"], "./output")
# ... 实现算子 ...
from cannjudge_cli import read_project_files
files = read_project_files(project_dir)
sid = client.submit(problem["_id"], **files)
print(client.wait_for_result(sid)["status"])
```

## 工程模板结构（兼容两代模板）

```
code/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   ├── {op_name}_def.cpp      # 算子定义（新模板拆分多文件）
│   ├── {op_name}_infershape.cpp
│   └── {op_name}_tiling.cpp   # 新模板按文件名拼接为一个 host_cpp 提交
└── op_kernel/
    ├── CMakeLists.txt
    ├── {op_name}_tiling_data.h   # Tiling 数据结构（新模板；旧模板为 *_tiling.h）
    ├── {op_name}_tiling_key.h    # Tiling Key 定义（可空）
    └── {op_name}.cpp          # Kernel 实现
```

`submit` 命令自动适配两种命名：tiling 数据头文件取 `*_tiling_data.h`（回退 `*_tiling.h`），tiling key 取 `*_tiling_key.h`（回退 `tiling_key_*.h`），Host 多文件按 def/infershape/tiling 顺序拼接。

## 返回结果说明

- 提交状态：`Running` / `Accepted` / `Wrong Answer` / `Compile Error` / `Runtime Error` / `Time Limit Exceeded`
- 用例结果：`testcase_status`、`precision_ratio`（1.0=完全匹配）、`time`（毫秒）

## 注意事项

1. **凭据安全**：明文密码仅存本地 `.env`（600）或环境变量，禁止进日志/仓库/对话
2. **泛化设计**：平台不提供测试用例，必须设计泛化算子
3. **公开接口免登录**：problem / rank / query / download 为公开接口；login / submit 才需凭据
4. **Cookie 管理**：登录后 Session 自动管理 Cookie
5. **轮询间隔**：查询结果建议 3 秒，最长等待默认 120 秒
6. **超时处理**：单请求超时 30 秒，下载包 60 秒
7. **登录限流**：频繁登录会触发平台 429，连续操作请复用 Session 或间隔 10 秒以上
8. **ABI 一致性核验**：提交 ID 只证明服务端接受了请求，不证明评测端使用了本次源码的入口。提交后必须保存四个源码字段的 SHA-256 与评测日志；若日志中的 `aclnn*GetWorkspaceSize` 参数名、数量或顺序仍来自旧模板，应标记为“评测入口/构建配置不一致”，不得立即重复提交同一代码。
9. **榜单交叉验证**：将本提交的逐 case 结果与 `/api/problems/{problemId}/ranking` 对照。榜首仍为部分 WA 时，只能说明存在可运行的部分路径，不能据此断言题面 ABI 已完整生效。

## 相关 Skills

- `ascendc-ops-project`：泛化算子设计、Tiling 设计、Host/Kernel 实现、精度验证标准
- 使用流程：本 Skill 负责平台交互（登录/下载/提交/查询），`ascendc-ops-project` 负责算子实现

## FAQ

### Q: 为什么从 RSA 密文改为明文？

**A**: 原版面向服务器 CANNBot 代登场景，需密钥对保证传输安全；本版面向个人本地电脑直连，`.env` + 权限 600 + 不入库已能满足本地凭据保护需求，流程更简洁。

### Q: float16 精度不达标怎么办？

**A**: float16 存在边界量化问题，需根据参考算子行为调整：对 float16 数据将边界量化到 half 可表示值，float32/int32 使用均匀边界；可在 Host 侧设置 `quantizeEdges` 标志，Kernel 侧按标志选择行为。

### Q: 如何处理默认值特殊语义？

**A**: `min=0,max=0` → 自动计算范围；`axis=-1` → 最后一个维度；`keepdims=False` → 不保持维度。务必查阅参考算子官方文档并用 Python 验证。

### Q: 如何设计泛化算子？

**A**: 参考 `ascendc-ops-project`：Shape/Dtype/属性/对齐/边界五要素泛化，逐项对照检查清单。
