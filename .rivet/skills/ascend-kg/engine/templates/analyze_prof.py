#!/usr/bin/env python3
"""分析 profiling kernel_details.csv：算子热点、AI Core 效率、MFU 估算。
用法: analyze_prof.py <kernel_details.csv> <output.txt> [prefill_tokens] [decode_tokens]
输出重定向到文件，不直接打印到 stdout 避免卡 session。
PEAK 已校正为 910B3 = 294.91 TFLOPS（op-mfu-calculator，非 313）。"""
import csv
import sys
import collections

F = sys.argv[1]
OUT = sys.argv[2]
PRE_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 920
DEC_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 580

# 模型参数（Qwen2.5-7B）
n_layers = 28
hidden = 3584
inter = 18944

def flops_linear(M, N, K): return 2 * M * N * K  # MACs * 2

flops_per_layer = (
    flops_linear(1, 3*hidden, hidden)   # QKV
    + flops_linear(1, hidden, hidden)   # attention out
    + flops_linear(1, 2*inter, hidden)  # gate+up
    + flops_linear(1, hidden, inter)    # down
)
total_flops_per_token = flops_per_layer * n_layers

rows = list(csv.DictReader(open(F)))
out = open(OUT, "w")
def w(s=""): out.write(s + "\n")

# 空数据/缺列防御：不同 profiling 版本列名可能不一致，统一 .get 兜底
def col(row, key, default=""):
    return row.get(key, default)

if not rows:
    w("ERR: kernel_details.csv 为空或无数据，无法分析")
    out.close()
    sys.exit(1)

w(f"估算: 单 token 单层 FLOPs={flops_per_layer/1e9:.3f}G, 全模型/ token={total_flops_per_token/1e9:.3f}G")
w(f"总算子记录: {len(rows)}")

agg = collections.defaultdict(lambda: {"count":0,"dur":0.0,"mac":0.0,"mac_ratio":0.0,"cube_util":[]})
core_cnt = collections.Counter()
for row in rows:
    name = col(row, "Name", "?").split('_Tiling')[0]
    core = col(row, "Accelerator Core", "?")
    core_cnt[core]+=1
    try: dur = float(col(row, "Duration(us)", "0"))
    except (ValueError, TypeError): dur = 0.0
    agg[name]["count"]+=1
    agg[name]["dur"]+=dur
    try: agg[name]["mac"]+= float(col(row, "aic_mac_time(us)", "0") or 0)
    except (ValueError, TypeError): pass
    cbu = col(row, "cube_utilization(%)", "")
    if cbu:
        try: agg[name]["cube_util"].append(float(cbu))
        except (ValueError, TypeError): pass
    macr = col(row, "aic_mac_ratio", "")
    if macr:
        try: agg[name]["mac_ratio"]+= float(macr)
        except (ValueError, TypeError): pass

tot_dur = sum(a["dur"] for a in agg.values())
if tot_dur <= 0:
    w("ERR: 算子总耗时=0（Duration(us) 列缺失或全 0），无有效耗时数据")
    out.close()
    sys.exit(1)
w(f"\n===== 算子类型聚合 (按耗时) 总耗时 {tot_dur/1e6:.3f}s =====")
w(f"{'算子':<30}{'Count':>7}{'Total(ms)':>11}{'占比%':>8}{'AvgCube%':>9}")
for n,a in sorted(agg.items(), key=lambda x:-x[1]["dur"])[:20]:
    avg_cube = sum(a["cube_util"])/len(a["cube_util"]) if a["cube_util"] else 0
    w(f"{n:<30}{a['count']:>7}{a['dur']/1e3:>11.2f}{a['dur']/tot_dur*100:>8.1f}{avg_cube:>9.1f}")

w(f"\n===== Accelerator Core 分布 =====")
for c,n in core_cnt.most_common(): w(f"  {c}: {n} ({n/len(rows)*100:.1f}%)")

# MFU 估算
mac_time_total = sum(a["mac"] for a in agg.values())
PEAK = 294.91e12  # 910B3 单卡 FP16 峰值（校正后）
w(f"\n===== MFU 估算 =====")
w(f"总 aic_mac_time(立方计算时间): {mac_time_total/1e6:.3f}s")
rows_sorted = sorted(rows, key=lambda r: float(col(r, "Start Time(us)", "0")) if str(col(r, "Start Time(us)", "")).strip() else 0)
t_start = float(col(rows_sorted[0], "Start Time(us)", "0"))
t_end = float(col(rows_sorted[-1], "Start Time(us)", "0")) + float(col(rows_sorted[-1], "Duration(us)", "0"))
wall = (t_end - t_start)/1e6
if wall <= 0:
    w("ERR: 采集窗口<=0（Start Time(us)/Duration(us) 列缺失或全 0），无法算 MFU")
else:
    w(f"采集窗口(首算子→末算子): {wall:.3f}s")
    useful = (PRE_TOK + DEC_TOK) * total_flops_per_token
    mfu = useful / (PEAK * wall)
    w(f"有用FLOPs(估算): {useful/1e12:.2f} TFLOPs (prefill={PRE_TOK}, decode={DEC_TOK})")
    w(f"估算 MFU: {mfu*100:.1f}%  (峰值294.91TFLOPS)")

# wait 分析
w(f"\n===== Wait Time Top (等待长=下发/依赖瓶颈) =====")
wait_agg = collections.defaultdict(float)
wait_cnt = collections.Counter()
for row in rows:
    n = col(row, "Name", "?").split('_Tiling')[0]
    try: ww = float(col(row, "Wait Time(us)", "0"))
    except (ValueError, TypeError): ww = 0
    wait_agg[n]+=ww; wait_cnt[n]+=1
for n,ww in sorted(wait_agg.items(), key=lambda x:-x[1])[:10]:
    w(f"  {n}: 总等待 {ww/1e3:.2f}ms, {wait_cnt[n]}次, 均{ww/wait_cnt[n]:.2f}us")

# MatMul 时长分布（prefill vs decode）
w(f"\n===== MatMulV2 Duration 分布 (找 prefill vs decode) =====")
durs = sorted([float(col(r, "Duration(us)", "0")) for r in rows if "MatMulV2" in col(r, "Name", "")])
if durs:
    w(f"  MatMulV2 个数={len(durs)}, min={durs[0]:.1f}us, p50={durs[len(durs)//2]:.1f}us, p90={durs[int(len(durs)*0.9)]:.1f}us, max={durs[-1]:.1f}us")
    big = [d for d in durs if d>200]
    small = [d for d in durs if d<=200]
    w(f"  大 matmul(>200us, prefill类): {len(big)}个, 总耗时{sum(big)/1e3:.2f}ms")
    w(f"  小 matmul(<=200us, decode类): {len(small)}个, 总耗时{sum(small)/1e3:.2f}ms")
else:
    w("  MatMulV2 个数=0（无 MatMulV2 算子记录）")

out.close()
