# CANNJudge 算子提交能力包（本地明文版）

本地可复用的 CANNJudge 算子竞赛提交流程能力包：凭据明文配置、一键登录、下载工程、提交代码、查询结果、查看排行榜。

## 快速开始

```bash
# 1. 安装依赖
pip install requests

# 2. 配置本地凭据（交互式，自动生成 .env 且权限 600）
python cannjudge_cli.py setup

# 3. 验证登录
python cannjudge_cli.py login

# 4. 下载题目工程（按名称查 ID，再下载；problem/rank/query/download 无需登录）
python cannjudge_cli.py problem --problem-name cube2
python cannjudge_cli.py download --problem-id <题目ID> --output ./output

# 5. 实现算子后提交（需登录）
python cannjudge_cli.py submit --problem-id <题目ID> --project-dir ./output

# 6. 查询结果 / 排行榜（无需登录）
python cannjudge_cli.py query --submission-id <提交ID>
python cannjudge_cli.py rank --problem-id <题目ID>
```

## 凭据优先级

```
命令行 --email/--password  >  环境变量 CANNJUDGE_EMAIL/CANNJUDGE_PASSWORD  >  .env 文件
```

## 文件结构

```
cannjudge-submit-plaintext/
├── SKILL.md              # 技能文档（完整流程 + 泛化算子设计指南）
├── README.md             # 本文件
├── cannjudge_cli.py      # 明文版命令行工具（核心）
├── example.py            # 完整流程示例
├── package.ps1           # 生成不含本地凭据和缓存的复用 ZIP
├── .env.example          # 凭据模板
├── .gitignore            # 排除 .env / 密钥
└── docs/
    └── CANNJudge-算子竞赛实践指南.md   # 正式书面文章（作品输出）
```

## 与原版（cannjudge-submit）差异

| 项目 | 原版（RSA 密文） | 本版（明文） |
|------|----------------|------------|
| 登录凭据 | RSA 密文，服务器私钥解密 | 明文，`.env` / 环境变量 |
| 密钥脚本 | generate_key.py / encrypt_password.py | 移除 |
| 前置依赖 | pycryptodome | 仅 requests |
| 凭据保护 | 私钥留在服务器 | `.env` chmod 600 + 不入库 |
| 适用场景 | 服务器 CANNBot 代登 | 个人本地电脑直连 |
| 排行榜接口 | 旧 `/latest` | `/api/problems/{id}/ranking`（网站同源） |
| 免登录能力 | 全部需登录 | problem/rank/query/download 公开可查 |

## 安全提示

- `.env` 已自动 `chmod 600`，且被 `.gitignore` 排除
- 密码不会出现在终端、日志或对话输出中
- 共享电脑使用后请删除 `.env`

## 生成复用包

在 Windows PowerShell 中执行：

```powershell
pwsh -File .\package.ps1
```

脚本会在上级目录生成 `cannjudge-submit-plaintext.zip`，并强制排除
`.env`、私钥、公钥、Python 缓存和既有压缩包。生成后会再次检查 ZIP，
发现敏感文件时直接失败。
