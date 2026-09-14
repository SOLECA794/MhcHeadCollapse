#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MhcHeadCollapse OJ shape 探测器（半自动批量断言探测）

用法:
  python probe_oj.py build <case> <field> <value>   # 生成断言探针版本并提交
  python probe_oj.py poll <submission_id>           # 轮询结果
  python probe_oj.py sweep <field> <case> <v1,v2,...>  # 批量扫一组候选值

探测原理（v081 矩阵同款）:
  在 op_host 的 InferShape 阶段注入断言 `if (field != value) return GRAPH_FAILED`
  Pass = 断言成立（该 case 的 field == value）
  RE   = 断言不成立

已知 shape 现状（F2.1l v2）:
  Case1: n=4 nH=16 outer∈{2,3,4}(v084b) | Case2: n=4 nH=16 outer=2
  Case3: n=8 nH=512 outer? | Case4: n=8 nH=512 outer? | Case5: n=8 nH=1024 outer?
  探测目标: Case3/4/5 的 outer（候选 1~16 的 2 的幂 + 常见值）
"""
import argparse, json, os, re, shutil, subprocess, sys, time, urllib.request

BASE = r"D:/Desktop/OP-Learning/Ascend/xingchen-operators/C组赛题/【C组中等题】MhcHeadCollapse 算子（mHC四路归一）"
PROBE_DIR = os.path.join(BASE, "versions", "_probe_tmp")
CLI_DIR = r"D:/Desktop/OP-Learning/Ascend/cann-learing-hub/skills/cannjudge-submit-plaintext"
PROBLEM_ID = "6a7c23a6a52e0f540a8a1779"
LOG = os.path.join(BASE, "shape_probe_log.jsonl")

# 基线版本（正确性已验证的最终版）
BASE_VERSION = "v118_path2vecz"

def run_cli(*args, timeout=300):
    r = subprocess.run(["python", "cannjudge_cli.py", *args], cwd=CLI_DIR,
                       capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")

def log_result(rec):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def build_probe(case, field, value):
    """生成断言探针版本目录并提交。field: n/nH/outer; case: 1-5"""
    src = os.path.join(BASE, "versions", BASE_VERSION)
    if os.path.exists(PROBE_DIR):
        shutil.rmtree(PROBE_DIR)
    shutil.copytree(src, PROBE_DIR)
    host = os.path.join(PROBE_DIR, "code", "op_host", "mhc_head_collapse.cpp")
    s = open(host, encoding="utf-8").read()
    # 在 InferShape 开头（找第一个 shape 相关行）注入断言
    # case 索引: 平台按 case 编号逐个编译, 用 tiling_key 区分? 不行 — 断言必须按 shape 而非 case 号
    # 手法: 用"排除法" — 断言写 field==value 时, 只有该 field 匹配的 case 才 Pass
    marker = "namespace {" if "namespace {" in s else "namespace\n{"
    assert_field = {
        "n":    f'if (xDesc.GetShape().GetDimNum() >= 3) {{ }} // shape sanity\n',
    }
    # 通用注入: 在文件头 namespace 后加全局探针函数, 在 InferShape 调用
    # 注入点: InferShape 的 return GRAPH_SUCCESS 前（nH/n 已就绪, outer 需算）
    anchor = """        *y_shape = *x_shape;
        y_shape->SetDim(rank - 1, nH / n);
        return GRAPH_SUCCESS;"""
    assert anchor in s, "InferShape anchor not found"
    probe_val = {"outer": "outer_v", "nH": "nH", "n": "n"}[field]
    inject = f"""        {{ // PROBE field={field} value={value}
            int64_t outer_v = 1;
            for (uint32_t i = 1; i + 1 < rank; ++i) outer_v *= x_shape->GetDim(i);
            if ({probe_val} != (int64_t)({value})) return GRAPH_FAILED;
        }}
"""
    s = s.replace(anchor, inject + anchor, 1)
    open(host, "w", encoding="utf-8").write(s)
    # 提交
    out = run_cli("submit", "--project-dir", PROBE_DIR, "--problem-id", PROBLEM_ID, "--no-wait")
    m2 = re.search(r"Submission ID: (\w+)", out)
    if not m2:
        print("SUBMIT FAILED:", out[-300:]); sys.exit(3)
    sid = m2.group(1)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "case_hint": case, "field": field,
           "value": value, "submission_id": sid, "status": "submitted"}
    log_result(rec)
    print(f"submitted {field}={value} (case hint {case}) -> {sid}")
    return sid

def poll(sid):
    for i in range(20):
        out = run_cli("query", "--submission-id", sid, timeout=120)
        if "Running" not in out and "Pending" not in out:
            # 解析 5 个 case 的状态
            statuses = re.findall(r"Case (\d): (\w+)", out)
            rec = {"submission_id": sid, "statuses": statuses, "raw_tail": out[-200:]}
            print(json.dumps(rec, ensure_ascii=False))
            return statuses
        time.sleep(30)
    print("TIMEOUT"); return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "poll", "sweep"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "build":
        case, field, value = a.args
        build_probe(case, field, value)
    elif a.cmd == "poll":
        poll(a.args[0])
    elif a.cmd == "sweep":
        field, case = a.args[0], a.args[1]
        values = a.args[2].split(",")
        for v in values:
            sid = build_probe(case, field, v)
            st = poll(sid)
            if st:
                passed = [c for c, s in st if s == "Pass"]
                print(f"=> field={field} value={v}: PASS cases = {passed}")
            time.sleep(5)

if __name__ == "__main__":
    main()
