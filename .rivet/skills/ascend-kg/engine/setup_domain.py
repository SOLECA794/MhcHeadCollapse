#!/usr/bin/env python3
"""setup_domain.py —— KG 动作空间材料的独立摸底 CLI（薄包装）。

2026-08-27 起 KG 构建动作空间已是 agent_loop 的循环内必走节点（tools/domain_node.py，
round 0 前自动执行：检索 → 暂停点 LLM 提取 → 与手写 domain 合并）。本脚本仅保留
「不起引擎、单独看某 flow 能检索到什么材料」的摸底用途，逻辑零重复（直接调
domain_node.search_materials）。

用法：
  python engine/setup_domain.py --flow vllm-serve-optimize
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.registry import load_flow  # noqa: E402
from engine.tools import domain_node  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", required=True, help="flow_id（对应 flows/<flow_id>.json）")
    args = ap.parse_args(argv)

    flow = load_flow(args.flow)
    materials = domain_node.search_materials(flow)
    status = materials["kg_status"]
    if status == "unmapped-backend":
        print(f"未映射 backend={flow.backend} 的框架关键词"
              f"（已知: {sorted(domain_node.FRAMEWORK)}），请在 FRAMEWORK 里补充",
              file=sys.stderr)
        return 1
    if status == "unavailable":
        print("KG 不可达（无 Key 或 /health 非 200）——无法检索，请先配置 ASCEND_KG_API_KEY",
              file=sys.stderr)
        return 1

    raw_path = domain_node.dump_raw(flow.flow_id, flow.backend, materials)
    print(f"已落盘原始材料: {raw_path}（hits={len(materials['hits'])} "
          f"sources={len(materials['sources'])}）")
    if not materials["sources"]:
        print("⚠️ 无 score>0.83 的黄金命中，无全文可提取——请人工确认 query 或换英文改写",
              file=sys.stderr)

    print("\n注：正常调优无需手动跑本脚本——agent_loop 的 domain 必走节点会自动检索并"
          "暂停等待 LLM 提取。此处仅为独立摸底。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
