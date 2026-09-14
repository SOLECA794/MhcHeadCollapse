# CANNJudge 算子竞赛实践指南

**——从平台交互到泛化算子实现的全流程方法论文**

> 版本：v1.0（本地明文版）｜适用平台：https://cannjudge.cn｜配套能力包：`cannjudge-submit-plaintext`

---

## 一、引言

CANNJudge 是基于昇腾 CANN 平台的算子在线评测系统，参赛者通过在线提交 Host 侧与 Kernel 侧代码，由平台完成编译与多组测试用例的精度、性能验证。与本地开发不同，该平台具有三个显著特征：**题目信息仅能在线获取**、**不开放测试用例接口**、**提交内容为源码文本而非二进制**。这三个特征决定了参赛者必须以"泛化算子"为设计目标，并建立一套规范、可复用的提交流程。

本文以书面形式系统阐述 CANNJudge 竞赛的完整方法论，并介绍一套经过本地化改造的**可复用能力包**。该能力包将原版"服务器 CANNBot 代登（RSA 密文）"流程改造为"个人电脑直连（明文凭据）"流程，在保持全部平台交互能力的同时，显著降低了部署门槛。

## 二、平台工作流程总览

完整的竞赛周期包含八个阶段，形成闭环：

```
配置凭据 → 登录鉴权 → 获取题目 → 下载模板 → 理解题目 → 实现泛化算子 → 编译提交 → 查询反馈
```

各阶段与平台 API 的对应关系如下：

| 阶段 | 平台 API | 说明 |
|------|----------|------|
| 登录鉴权 | `POST /api/users/login` | 提交邮箱与密码，返回用户信息，会话自动携带 Cookie |
| 获取题目 | `GET /api/problems/name/{name}` | 按名称获取题目（推荐，名称比 ID 更稳定） |
| 下载模板 | `GET /api/problems/{id}/package?userId={id}` | 返回 Zip 工程包，解压即得工程骨架 |
| 提交代码 | `POST /api/submissions/submit` | 提交 tiling_h / tiling_key_h / host_cpp / kernel_cpp 四段文本 |
| 查询结果 | `GET /api/submissions/{id}` | 返回状态与逐用例的精度比例、耗时 |
| 排行榜 | `GET /api/submissions/problem/{id}/latest` | 获取全体参赛者最新提交 |

## 三、本地明文改造：设计动机与方案

### 3.1 原版机制的适用场景

原版 `cannjudge-submit` Skill 面向"服务器 CANNBot 代理登录"场景：用户在个人电脑上用 RSA 公钥加密密码，将密文传给服务器端 Agent，由服务器用私钥解密后调用登录接口。该设计的核心目的是**防止明文密码经对话链路传输**，因此在服务器代登场景下是必要的。

### 3.2 明文改造的依据

当使用场景变为**个人本地电脑直连平台**时，RSA 环节失去了存在意义：密码本就存储在本机，加密后的密文仍由同一台机器解密，并未提升安全性，反而引入了 `pycryptodome` 依赖、密钥管理、密文复用等额外复杂度。

### 3.3 明文方案的安全设计

为在"简洁"与"安全"之间取得平衡，本地明文版采取四项约束：

1. **凭据来源分级**：命令行参数 > 环境变量 > `.env` 文件，覆盖交互式与自动化两种使用方式；
2. **最小暴露**：`.env` 文件写入后自动 `chmod 600`，仅属主可读；
3. **禁入日志**：所有运行路径均不打印密码，登录成功仅输出用户 ID 与昵称；
4. **防误提交**：`.gitignore` 显式排除 `.env`，防止凭据进入代码仓库。

```bash
# 一次性配置
python cannjudge_cli.py setup
# 生成 .env（权限 600），内容格式：
# CANNJUDGE_EMAIL=you@example.com
# CANNJUDGE_PASSWORD=********
```

## 四、泛化算子设计方法论

平台不开放测试用例，参赛者提交后由平台以未知的 shape、dtype、属性组合评测。因此，**泛化性是唯一可靠的得分策略**，须从五个维度逐项落实。

### 4.1 Shape 泛化

不硬编码维度数，以总元素数为处理单位：

```cpp
int32_t rank = shape.GetDimNum();
uint32_t totalLength = 1;
for (int32_t i = 0; i < rank; i++) {
    totalLength *= shape.GetDim(i);
}
```

### 4.2 Dtype 泛化

以模板类承载计算逻辑，入口按 dtype 分发；浮点中间计算建议以 float32 进行以保障精度。

### 4.3 属性泛化

属性默认值往往是"标志"而非"数值"。例如 `torch.histc` 的 `min=0, max=0` 表示自动计算数据范围；`axis=-1` 表示取最后一个维度。**必须查阅参考算子官方文档并用参考实现实测验证**，这是最容易失分也最容易被忽视的环节。

### 4.4 对齐泛化

昇腾 UB 访存存在对齐约束，非对齐尾部数据须使用 `DataCopyPad` 与向上取整的切分策略：

```cpp
uint32_t tileNum = (totalLength + 255) / 256;
uint32_t lastTileLength = totalLength - (tileNum - 1) * 256;
```

### 4.5 边界泛化

覆盖空输入（shape 含 0）、单元素输入、极端值（最大/最小）、特殊值（NaN/Inf）。例如 float16 数据存在边界量化问题，需要在 Host 侧设置 `quantizeEdges` 标志，Kernel 侧按标志决定是否将边界量化到 half 可表示值。

## 五、能力包使用指南

能力包以"Skill + 工具 + 文档"三层结构组织，源码位于 `cannjudge-submit-plaintext/`。

### 5.1 快速上手

```bash
pip install requests
python cannjudge_cli.py setup          # 配置本地凭据
python cannjudge_cli.py login          # 验证登录
python cannjudge_cli.py problem --problem-name depthtospace
python cannjudge_cli.py download --problem-id <ID> --output ./output
# ... 在 ./output/code 中实现算子 ...
python cannjudge_cli.py submit --problem-id <ID> --project-dir ./output
python cannjudge_cli.py query --submission-id <提交ID>
python cannjudge_cli.py rank --problem-id <ID>
```

### 5.2 提交文件映射

`submit` 命令自动从工程目录收集四个提交字段：

| 字段 | 来源文件 | 可空 |
|------|----------|------|
| `tiling_h` | `op_kernel/*_tiling.h` | 否 |
| `tiling_key_h` | `op_kernel/tiling_key_*.h` | 是 |
| `host_cpp` | `op_host/*.cpp` | 否 |
| `kernel_cpp` | `op_kernel/*.cpp`（排除 tiling） | 否 |

### 5.3 结果解读

评测状态机：`Running → Accepted / Wrong Answer / Compile Error / Runtime Error / Time Limit Exceeded`。用例结果字段 `precision_ratio = 1.0` 表示完全匹配；精度达标参考：float16 需 `rtol < 1e-3, atol < 1e-3`，float32 需 `rtol < 1e-4, atol < 1e-4`。

## 六、最佳实践与常见问题

1. **题目信息只信线上**：禁止使用本地旧文件或主观臆断的标准定义，题目随时可能更新；
2. **参考算子文档优先**：实现前先通读官方文档，再以 Python 参考实现实测边界行为；
3. **提交前自查清单**：dtype 全覆盖、维度无硬编码、负 axis 已归一、非对齐尾部已处理、空输入与极端值已覆盖；
4. **轮询节流**：结果查询间隔 3 秒、单请求超时 30 秒，避免对平台造成压力；
5. **凭据卫生**：`.env` 不入库、共享电脑用后即删、密码绝不出现在任何输出中。

**常见问题速查**：

- 部分用例失败 → 泛化覆盖不足，按五要素清单逐项排查；
- float16 精度不达标 → 边界量化与参考实现对齐；
- 模板缺少编译配置（CMakePresets.json / build.sh）→ 从已有 Ascend C 工程复制；
- 需要支持任意维度 → Host 侧按总长度 Tiling，Kernel 侧按总长度循环。

## 七、结语

CANNJudge 竞赛的胜负手不在于"会写某个算子"，而在于建立一套**可复现、可迁移、可验证**的工程方法：以泛化设计应对未知测试，以规范流程压缩往返周期，以安全约定保护凭据。本文所附的本地明文能力包，将平台交互固化为一条命令即可完成的流水线，使参赛者得以将精力集中于算子实现本身。愿读者以此为起点，在算子竞赛中稳步精进。

---

*本文由 CANNJudge 技能体系整理而成，配套能力包位于 `skills/cannjudge-submit-plaintext/`，可通过 `SKILL.md` 或 `README.md` 获取全部细节。*
