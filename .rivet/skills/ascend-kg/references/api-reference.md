# 完整 API 参考（SKILL.md §1 详细版）

> 本文件是 `SKILL.md` §1.1–§1.4 的完整参数/模板参考。**构造精确请求、核对参数 schema、查 id 形态、查 Cypher 模板时**读这里。主文件 §1 保留每个端点的高频速查。

## /search — 完整参数

参数：`query`(必填)，`top_k`(默认10)，`file_type`(可按 code/document/concept/rationale/code_card/configuration/api/error_code 过滤)，`source_repo`(按仓库名子串过滤，如 `"vllm"` 匹配 `vllm-mindspeed-torch`；鲲鹏用 `"KP-*"` 或 `id STARTS WITH 'kp_'`)，`with_neighbors`(默认 true——搜索引擎与图数据库的本质区别，不发该 flag 也返邻居)。**只接受这 5 个参数**（schema `extra='forbid'`）；`cross_repo_similar` / `path_trace` 是 `/cypher` 模板名、**不是** `/search` 参数（入 body → 422）；`neighbor_limit` 已忽略，邻居数服务端定（上限 12）。 _（参数以 2026-08-10 live 实测为准）_

### 评分阈值

| score | 含义 | 动作 |
|---|---|---|
| **> 0.83** | 黄金命中 | `/source` 取全文 |
| 0.70-0.83 | 模糊 | 换中/英文改写 query 重试；或缩小 `file_type` |
| **< 0.70** | 噪声 | 换检索路径：概念→SEMANTICALLY_IMPLEMENTS→代码；或加载对应 Skill |

### id 体系速查

拿到任何 id 先判形态，再决定怎么取正文：

| id 形态 | 示例 | `/source` 可读？ |
|---|---|---|
| skill 名 | `ascendc-crash-debug` | ✅ 返 SKILL.md |
| skill 资源 | `{skill}_res_{安全路径}` | ✅ 返资源全文 |
| document / 代码卡 | 路径编码长 id（`ascdevkit_docs_...`） | ✅（有 SourceText） |
| 函数级 code 卡 | `..._vfcall` / `..._func` | ❌ → 取 `source_file` 走 §1.2 三级取码（详见 references/code-retrieval.md） |

## /cypher — 模板参考

20 个预注册安全模板，覆盖节点查询、图遍历、跨仓库关系、统计聚合、语义搜索。

**实测可用模板**（2026-08-10）：`node_detail_by_id {id}`（取 `source_file`/`source_repo` 的权威途径）、`stats_node_count {}`、`node_neighbors_extended {id,rel_type,limit}`、`node_calls_chain {id,limit}`。
**易 400**（参数名敏感）：`find_nodes_by_label` / `find_nodes_by_source_file`——参数名非直觉，先查 `kg-query-guide-full` 的 schema 再用。
⚠️ **`/cypher/templates` 端点不存在（404）**；完整 20 模板清单在 guide 节点 `kg-query-guide-full`，不在 API。
总原则：模板名/参数校验严格，不确定时优先用 `/search` + `with_neighbors` 代替。

## /skill/search — 稳定性说明

⚠️ **可能为空**：实测 `/skill/search` 中英文查询常返回 `{"results":[]}`，该端点稳定性较差。若返回空，改走 `/source` 加载 `kg-skill-index-master` 主索引（按 19 个分类浏览），或 `GET /skill/{id}`（实测稳定返 SKILL.md，不含配套资源）。完整 skill 加载流程见主文件 §1.0.1。
