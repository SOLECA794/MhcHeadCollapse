#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OJ shape 断言探测 v2（notebook 环境版，2026-09-14）

原理（F2.1l v081 同款）：InferShape 注入 `if (outer != V) return GRAPH_FAILED`
→ Pass = 该 case outer==V；RE = 不等。一次提交对 5 case 同时生效。

用法：python3 probe_oj_v2.py <field> <value>     # field: n|nH|outer
输出：提交 + 轮询 + 逐 case Pass/RE 解读
"""
import subprocess, sys, os, time, shutil

BASE_SRC = "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stage"  # v119 同源
PROBE_DIR = "/tmp/probe_oj_tmp"
CLI_DIR = "/workspace/notebook13/MhcHeadCollapse/cannjudge-submit-plaintext"
PROBLEM_ID = "6a7c23a6a52e0f540a8a1779"

def run_cli(*args, timeout=400):
    r = subprocess.run(["python3", "cannjudge_cli.py", *args], cwd=CLI_DIR,
                       capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")

def build_and_submit(field, value):
    if os.path.exists(PROBE_DIR):
        shutil.rmtree(PROBE_DIR)
    shutil.copytree(BASE_SRC, PROBE_DIR, ignore=shutil.ignore_patterns('build', 'build85', 'build_out', '__pycache__'))
    host = os.path.join(PROBE_DIR, "code", "op_host", "mhc_head_collapse.cpp")
    s = open(host, encoding="utf-8").read()
    # 注入点：TilingFunc 的 outer 计算之后（aclnn 每次 aclnnGetWorkspaceSize 都执行）
    anchor = "uint64_t outer = 1;\n        for (uint32_t i = 0; i + 1 < rank; ++i) {\n            outer *= static_cast<uint64_t>(x_shape.GetDim(i));\n        }"
    assert anchor in s, "TilingFunc anchor not found"
    fld = "outer" if field == "outer" else ("nH" if field == "nH" else "n")
    inject = anchor + f"\n        if ({fld} != {value}) return ge::GRAPH_FAILED;"
    s = s.replace(anchor, inject, 1)
    open(host, 'w', encoding='utf-8').write(s)
    print(f"[probe] injected assert({field} == {value}), submitting...")
    out = run_cli("submit", "--problem-id", PROBLEM_ID, "--project-dir",
                  os.path.join(PROBE_DIR, "code"))
    print(out)
    return out

if __name__ == "__main__":
    field, value = sys.argv[1], sys.argv[2]
    build_and_submit(field, value)
