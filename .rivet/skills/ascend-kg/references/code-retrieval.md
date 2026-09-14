# Code 原文三级取码与本地镜像（SKILL.md §1.2 详细版）

> 本文件是 `SKILL.md` §1.2 的详细版。**当需要 Code 节点源码原文时**读这里（`/source` 对 code 节点返回 404 属设计，非故障）。本地镜像（1.6GB）配置的完整机制在本文件；克隆动作本身的 🔴 CHECKPOINT 在主文件 §1.2，**必须先获用户同意才能执行**。

## 三级取码

`source_file` 形如 `cann-ops/MindSpeed-Ops/ci/build.sh`，即 `{repo}/{rel_path}`；源码真源统一是 `gitcode.com/agent0/ascend-knowledg-graph` 一个仓库，默认分支 `main`。

1. **`/source`** — 仅 document/skill 等带正文节点可读；code 节点会 404（正常）。
2. **`$KG_SOURCE_BASE_DIR/<source_file>`** — 若已配置本地镜像，零网络读取（**反复需要 Code 原文时推荐**）。配置见下方"本地镜像（可选）"。
3. **raw URL 兜底** — `curl -sL "https://raw.gitcode.com/agent0/ascend-knowledg-graph/raw/main/<source_file>"`（⚠️ 是 `raw.gitcode.com` 子域名 + `main` 分支；用 `gitcode.com/.../raw/...` 会返 HTML 页；master 分支报 404）。

## 本地镜像（可选）

**仅当任务反复需要 Code 原文时配置**：

- 仓合集约 **1.6GB**；**🔴 CHECKPOINT · 🛑 STOP：克隆前必须停下、显式征得最终用户同意**（"是否同意下载约 1.6GB 的 ascend-knowledg-graph 代码合集到本地？"）——用户明确确认后才能克隆；未确认不得动手、不得静默跳过。
- 同意后：`git clone --depth=1 https://gitcode.com/agent0/ascend-knowledg-graph.git <dir>`
- 设 `export KG_SOURCE_BASE_DIR=<dir 绝对路径>`，写入对应 shell 配置（`~/.bashrc` / `~/.zshrc` 视 shell 而定）；之后所有 code 取码走路径 2。

⚠️ **该仓库是子集镜像**：收录各 `source_repo` 源码主体，**不含全部文件**（如测试文件常未收录）。raw URL 返 `404 File Not Found` 即未被收录——告知用户"该源文件未被 ascend-knowledg-graph 收录"，**不回退到 `source_repo`**。

⚠️ **不要理会 `source_repo` 字段**——凭它去本地磁盘 / gitcode 搜开源项目 / GitHub 等平台找文件**全是错的，禁止走**。
