# domain_cache —— KG 动作空间的材料与产物记录

2026-08-27 起语义变更：KG 构建动作空间（plan §7.1）已是 **agent_loop 的循环内必走节点**
（`tools/domain_node.py`，round 0 前自动执行：KG 检索 → 暂停点 LLM 提取 → 与手写
domain 字段级合并，KG 字段覆盖、缺失字段手写继承）。`registry.load_flow` **不再**读
本目录覆盖手写 domain（load 时覆盖会绕过必走节点）。

- `{flow_id}_raw.json` —— domain 必走节点（或独立摸底 `setup_domain.py --flow <id>`）
  检索 KG 落盘的原始材料（`framework` / `kg_status` / `hits` / `sources[{id,score,text}]`）。
- `{flow_id}.json` —— **记录性质**：本次 run 最终动作空间（`{flow_id, domain, meta}`），
  由节点在合并完成后写入，供人查看/跨 run 对比 KG 提取变化；不作为任何 load 时覆盖源。

domain 格式与 `flow.params.domain` 同构：

```json
{ "<param>": { "kind" / "risk" / "locked" / "min" / "max" / "step" / "values" / "flag" / "fixed" / "compose" / "path": … } }
```

字段词表须覆盖引擎实际消费的全部字段（`state.py`：kind/risk/locked/fixed/path/compose/values/flag/min/max/step；
`sampler.py`：flag/values/step/min/max/locked/compose）。节点对 LLM 提取做结构校验
（`domain_node.validate_kg_domain`），不合格条目丢弃并记入 meta.dropped。
