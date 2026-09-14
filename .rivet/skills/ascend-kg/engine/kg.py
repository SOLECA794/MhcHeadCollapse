"""KG 检索与纠偏（平移自 vllm_ascend_opt/machine.py，纯 stdlib + curl）。

KG 不可达时优雅降级（返回 False/[]/原候选），不阻塞闭环。
"""
import json
import os
import re
import subprocess
from pathlib import Path

KG_BASE = "https://ascend.wiki"


def _get_key() -> str:
    key = os.environ.get("ASCEND_KG_API_KEY", "")
    if key:
        return key
    try:
        m = re.search(
            r"ASCEND_KG_API_KEY=(\S+)",
            (Path.home() / ".bashrc").read_text(encoding="utf-8"))
        if m:
            return m.group(1).strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def kg_available() -> bool:
    """探测 KG 是否可达（不阻塞闭环：失败返回 False 用内置规则表）。"""
    key = _get_key()
    if not key:
        return False
    try:
        r = subprocess.run(
            ["curl", "-sf", "--max-time", "5", "-H", f"X-API-Key: {key}",
             f"{KG_BASE}/health"],
            capture_output=True, text=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


def kg_search(query: str, top_k: int = 3) -> list:
    """KG /search 语义检索，失败返回 []。"""
    key = _get_key()
    if not key:
        return []
    try:
        payload = json.dumps({"query": query, "top_k": top_k, "with_neighbors": True})
        r = subprocess.run(
            ["curl", "-s", "--compressed", "--max-time", "10", "-X", "POST",
             f"{KG_BASE}/search", "-H", "Content-Type: application/json",
             "-H", f"X-API-Key: {key}", "-d", payload],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        return data.get("results") or data.get("data") or []
    except Exception:
        return []


def kg_source(node_id: str, max_length: int = 5000) -> str:
    """KG /source 取节点全文（镜像 /source 端点），失败返回 ""。"""
    key = _get_key()
    if not key:
        return ""
    try:
        payload = json.dumps({"node_id": node_id, "max_length": max_length})
        r = subprocess.run(
            ["curl", "-s", "--compressed", "--max-time", "15", "-X", "POST",
             f"{KG_BASE}/source", "-H", "Content-Type: application/json",
             "-H", f"X-API-Key: {key}", "-d", payload],
            capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return ""
        data = json.loads(r.stdout)
        return data.get("text", "") if isinstance(data, dict) else ""
    except Exception:
        return ""


def kg_correct(candidate: dict, notes: list) -> tuple:
    """KG 纠偏：内置规则表优先，KG 在线确认一致则记录。
    返回 (纠正后的候选, 追加的纠偏说明)。

    【遗留兼容（P1-1c/B5，2026-08-27）】已从 recommend 调用链摘除——参数约束
    统一收敛到 flow.params.domain.locked（如 case5 的 block-size {"locked": true,
    "fixed": 128}），KG 在线确认只是装饰性 note。知识约束的新增路径是 domain
    必走节点（r0-domain 的 LLM 提取 + validate_kg_domain），不再走本函数。
    保留签名供考古/独立调用，勿在新链路引用。"""
    corrected = dict(candidate)
    # 内置规则表 —— 已验证的本机/架构约束
    if "block-size" in corrected and corrected["block-size"] != 128:
        notes.append(
            f"KG纠偏: block-size {corrected['block-size']}->128（910B chunked prefill 强制 128）")
        corrected["block-size"] = 128
    # KG 在线确认（可插拔；不可达则用内置表，不阻塞）
    if kg_available():
        res = kg_search("vllm-ascend 910B chunked prefill block size 128")
        hit = [x for x in res if float(x.get("score", 0) or 0) > 0.70]
        if hit:
            # 过滤时就把 score 归一为 float：KG 可能返回字符串 score，
            # 下方 :.2f 对字符串格式化会抛 TypeError，破坏 kg_correct 优雅降级承诺
            for x in hit:
                x["score"] = float(x.get("score", 0) or 0)
            nid = hit[0].get("id") or hit[0].get("node_id")
            sf = hit[0].get("source_file", "")
            notes.append(
                f"KG确认[{nid} score={hit[0].get('score'):.2f}{' ' + sf if sf else ''}]: "
                f"block_size=128 在 910B chunked prefill 下强制")
    return corrected, notes
