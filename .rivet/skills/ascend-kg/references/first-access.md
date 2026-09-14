# 首次接入全流程（SKILL.md §0 详细版）

> 本文件是 `SKILL.md` §0 的详细版，被 §0 各小节引用。**首次接入本 skill 时**需完整执行：获取 Key → 预检 → 自检 → 加载 5 份核心指南。需要快速路径时直接在 SKILL.md §0 操作；需要完整 curl / 选项明细时读本文件。

## 0.1 获取 API Key（命中即停）

1. 环境变量 `ASCEND_KG_API_KEY` 存在 → 直接使用
2. **🔴 CHECKPOINT · 无 Key 时必须停下问用户**，三选一：
   - ① 测试 Key（共享，5 RPS）
   - ② 申请专属 Key（免费，推荐）→ https://ascend.wiki/register
   - ③ 我已经有 Key
3. 选②或③拿到有效 Key 后 → 持久化到 `~/.bashrc`：追加 `export ASCEND_KG_API_KEY=<Key>`

**测试 Key**：`kg-test-1489df447809de55e30251bf59bf2d6c`
**配额档位**：standard/测试 Key = 5 RPS；whitelist Key = 100 RPS。超限返 429，处理见主文件 §4。

## 0.2 预检（1 个 curl，秒级）

```bash
curl -sf --max-time 5 -H "X-API-Key: <Key>" https://ascend.wiki/health
```
返回 200 → 可用。非 200 → 将错误告知用户，不要继续尝试。

## 0.2.1 自检（3 条 golden query，30 秒确认环境对）

```bash
# 1) /search 能搜到，响应主键是 id（不是 node_id）
curl -s --max-time 10 -X POST https://ascend.wiki/search -H "Content-Type: application/json" -H "X-API-Key: <Key>" \
  -d '{"query":"AscendC DataCopy","top_k":2}' -o /tmp/g1.json -w "search=%{http_code}\n"
# 2) /source 能读 skill 全文（期望 200）
curl -s --max-time 10 -X POST https://ascend.wiki/source -H "Content-Type: application/json" -H "X-API-Key: <Key>" \
  -d '{"node_id":"ascendc-api-best-practices"}' -o /dev/null -w "source=%{http_code}\n"
# 3) /cypher 统计能用（期望 200）
curl -s --max-time 10 -X POST https://ascend.wiki/cypher -H "Content-Type: application/json" -H "X-API-Key: <Key>" \
  -d '{"template":"stats_node_count","params":{}}' -o /dev/null -w "cypher=%{http_code}\n"
```
任一失败：① search 空/报错 → 查 Key 权限、字段名（是 `id` 不是 `node_id`）；② source 非 200 → skill 名拼错或改走 §1.0.1；③ cypher 400 → 模板名错，看 §1.3。
**3 项全 200（自检通过）→ 进入 0.3 加载 5 份核心检索指南**；任一失败先按上面修复并重跑，自检未过不进入下一步。

## 0.3 加载 5 份核心检索指南（首次必读，务必全部加载）

这 5 份文档是你在知识图谱中检索的完整使用手册——教会你如何构造查询、走决策路径、使用 Cypher 模板、加载 Skill。**首次接入时必须全部读完**，否则你不知道有哪些检索能力可用。

**第 1 步：获取 KG 路由中枢**

```bash
curl -s --compressed -X POST https://ascend.wiki/source \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <Key>" \
  -d '{"node_id": "kg-agent-orientation"}'
```
路由中枢（~9KB）：6 条快速路径决策树、19 个技能类别概览、故障排查表、关键经验（score 阈值、中文查询优势、CONFIGURES 边优先等）。

**第 2 步：获取交互式检索教程**

```bash
curl -s --compressed -X POST https://ascend.wiki/source \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <Key>" \
  -d '{"node_id": "kg-skill-abtest-final"}'
```
6 种检索模式的 curl 示例、决策点、"Done when" 条件和自检清单（~9KB）。

**第 3 步：获取完整 API 参考手册**

```bash
curl -s --compressed -X POST https://ascend.wiki/source \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <Key>" \
  -d '{"node_id": "kg-query-guide-full"}'
```
全部 20 个 Cypher 模板、完整 API 参数、5 轮迭代检索方法论、停止条件定义（~22KB）。

**第 4 步：加载 Skill 分类索引**

```bash
curl -s --compressed -X POST https://ascend.wiki/source \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <Key>" \
  -d '{"node_id": "kg-skill-index-master"}'
```
**605 可执行 Skill + 23 Plugin 的技能库** + 19 个分类目录（~8KB）。这些 Skill 不是文档摘录，是带触发条件、工作流、references/scripts/templates 配套的可执行技能。选择分类 → 获取 `node_id` → `/source` 加载完整 SKILL.md → 按其指令执行。**执行型任务优先走 §1.0 路由表**，不必每次都读本索引。

**第 5 步：学习 Skill 渐进式加载方式**

```bash
curl -s --compressed -X POST https://ascend.wiki/source \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <Key>" \
  -d '{"node_id": "kg-agent-skill-loading-guide"}'
```
从 KG 中动态加载完整 Skill 及 `references/`、`scripts/`、`templates/` 配套文件（多数 Skill 带配套，共 6,734 配套文件），覆盖 4 种入口场景、配套资源 ID 构造规则（~9KB）。加载后**按 Skill 工作流执行**，而非当文档引用。

> 以上 5 份文档加载完成后，你就可以正确使用主文件 §1 中的所有检索方法了。不要跳过任何一份。
