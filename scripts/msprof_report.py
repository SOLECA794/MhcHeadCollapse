#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""msprof 瓶颈报告生成器 — MhcHeadCollapse 观测基础设施 v1

用法:
  python3 msprof_report.py <msprof输出目录>          # 解析最新 PROF_* 目录
  python3 msprof_report.py <op_summary_*.csv路径>     # 直接喂 CSV

采集命令（先跑这个再喂本脚本）:
  $CANN/tools/profiler/bin/msprof --output=./out --ai-core=on \
    --aic-metrics=PipeUtilization --application="<测试程序> <args>"

字段可信性: 2026-09-14 实测（8.5.0, 910B4）——op_summary 含 aiv_vec_ratio/
aiv_scalar_ratio/aiv_mte2_ratio/aiv_mte3_ratio/aiv_icache_miss_rate 等 45+ 列。
"""
import csv
import glob
import sys
import os

# 阈值表（社区经验值，来自建议分析.md，2026-09-14 实测字段有效）
THRESHOLDS = [
    # (字段, 阈值, 判定, 优化动作)
    ("aiv_scalar_ratio", 0.20, "Scalar 受限", "精简 TilingData / 标量展开 / 减少循环控制"),
    ("aiv_mte2_ratio",   0.50, "MTE2 Bound",  "对齐、连续化、蹭 L2、双缓冲"),
    ("aiv_mte3_ratio",   0.30, "MTE3 写受限", "写合并、减少回写量"),
    ("aiv_vec_ratio",    0.60, "计算充分",    "向量已吃满，优化空间在别处"),
    ("aiv_icache_miss_rate", 0.10, "指令缓存缺失", "代码瘦身 / 减少分支"),
]

def find_csv(path):
    if path.endswith(".csv"):
        return [path]
    # 目录模式：找最新 PROF_* 下的 op_summary
    cands = sorted(glob.glob(os.path.join(path, "PROF_*", "mindstudio_profiler_output",
                                           "op_summary_*.csv")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(path, "**", "op_summary_*.csv"), recursive=True))
    return cands

def report(csv_path):
    with open(csv_path) as fp:
        rows = list(csv.DictReader(fp))
    print(f"=== msprof 瓶颈报告: {os.path.basename(csv_path)} ===")
    print(f"算子实例总数: {len(rows)}\n")
    for i, r in enumerate(rows):
        name = r.get("Name") or r.get("Op Name") or f"row{i}"
        task_type = r.get("Task Type", "?")
        aiv_time = r.get("aiv_time(us)", "?")
        # 分项时间列名: aiv_vec_time(us) 等
        print(f"[{i}] {name[:60]} ({task_type})  aiv={aiv_time}us")
        parts = []
        for fld, thr, verdict, action in THRESHOLDS:
            v = r.get(fld, "")
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            mark = ""
            if fld == "aiv_vec_ratio":
                mark = " ✓充分" if fv >= thr else ""
            elif fv >= thr:
                mark = f" ⚠️ {verdict}"
            parts.append((fld.replace("aiv_", "").replace("_ratio", ""), fv, mark, action))
        for nm, fv, mark, _ in parts:
            bar = "█" * int(fv * 40)
            print(f"    {nm:8s} {fv*100:5.1f}% {bar}{mark}")
        hits = [p for p in parts if "⚠️" in p[2]]
        if hits:
            print("  → 瓶颈判定:")
            for nm, fv, mark, action in hits:
                print(f"    - {nm} {fv*100:.1f}% → {action}")
        print()
    # 多实例聚合（同一算子多次调用）
    if len(rows) > 1:
        print("=== 多实例聚合（中位数） ===")
        import statistics
        for fld, thr, verdict, _ in THRESHOLDS:
            vals = [float(r[fld]) for r in rows if r.get(fld) not in ("", None)]
            if vals:
                med = statistics.median(vals)
                print(f"  {fld:28s} 中位 {med*100:5.1f}%  (n={len(vals)})")

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    csvs = find_csv(target)
    if not csvs:
        print(f"未找到 op_summary CSV: {target}")
        sys.exit(1)
    for c in csvs[-1:]:
        report(c)

if __name__ == "__main__":
    main()
