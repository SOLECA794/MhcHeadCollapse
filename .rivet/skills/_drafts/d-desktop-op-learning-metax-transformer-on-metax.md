---
name: d-desktop-op-learning-metax-transformer-on-metax
description: d:\Desktop\OP-Learning\Metax\Transformer-on-Metax 把这个路径的仓库文件夹配置成子模块 https://gitlink.org.cn/ccf-ai-infra/Transformer-on-Metax.git指向这个仓库 并且把最新的内容拉下来 — verified by 3 checks
triggers: ['probe-submodule-register', 'Desktop', 'OP-Learning', 'Metax']
---

# d-desktop-op-learning-metax-transformer-on-metax

> 自动从会话 4f88bf50 蒸馏的草稿。审核后用 `/skill approve d-desktop-op-learning-metax-transformer-on-metax` 入库，或 `/skill reject d-desktop-op-learning-metax-transformer-on-metax` 丢弃。

## Steps
1. 阅读 / 搜索：echo "=== cwd ===" && pwd && echo "=== git root ==、echo "=== Metax dir ===" && ls -la Metax/ 2>&1 | h、echo "=== .gitmodules ===" && cat .gitmodules 2>&1
2. 修改：d:\Desktop\OP-Learning\.rivet\scratch\probe-submodule-register.sh
3. 阅读 / 搜索：bash "d:/Desktop/OP-Learning/.rivet/scratch/probe-、echo "=== 待拉取数据量 ===" && git diff --shortstat HEAD、echo "=== 父仓库是否还有别的 dirty 子模块约定 ===" && grep -A3 -
4. 验证：deliver_task
5. 阅读 / 搜索：cd "d:/Desktop/OP-Learning"; echo "=== 提交前索引确认 ===、cd "d:/Desktop/OP-Learning"; echo "=== 提交内容(应只有2文件

## Verified by
- cd "d:/Desktop/OP-Learning" && echo "=== ls-files entry ===" && git ls-files -s -- "Metax/Transformer-on-Metax" 2>&1 && echo "(empty=not tracked as gitlink)" && echo "=== check-ignore ===" && git chec (passed 0)
- cd "d:/Desktop/OP-Learning/Metax/Transformer-on-Metax" && echo "=== 待拉取数据量 ===" && git diff --shortstat HEAD origin/master 2>&1 && echo "=== 大文件 top ===" && git diff --numstat HEAD origin/master 2>&1  (passed 0)
- cd "d:/Desktop/OP-Learning" && echo "=== gitlink 条目 ===" && git ls-files -s -- "Metax/Transformer-on-Metax" && echo "=== .gitmodules 尾部 ===" && tail -6 .gitmodules && echo "=== .git 形态(应仍是目录,未被搬走) === (passed 0)

<!-- skill-draft-key: 63f7be38ba48 -->
<!-- source-session: 4f88bf50 -->
