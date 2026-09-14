import csv, statistics as st

base = r"D:/Desktop/OP-Learning/.rivet/scratch/"

# ---- task_time: temporal structure ----
rows = []
with open(base + "s2_task_time.csv", newline="", encoding="utf-8") as f:
    for r in csv.reader(f):
        if r and r[1] == "MhcHeadCollapse":
            rows.append({
                "tid": int(r[4]),
                "dur": float(r[5]),
                "start": float(r[6].strip()),
                "stop": float(r[7].strip()),
            })

print(f"n={len(rows)}")
durs = [r["dur"] for r in rows]
print(f"duration: min={min(durs):.2f} p10={sorted(durs)[len(durs)//10]:.2f} "
      f"median={st.median(durs):.2f} p90={sorted(durs)[9*len(durs)//10]:.2f} max={max(durs):.2f}")

# gap = next.start - prev.stop (same stream, back-to-back)
gaps = [rows[i+1]["start"] - rows[i]["stop"] for i in range(len(rows)-1)]
print(f"gap stop->start: min={min(gaps):.3f} median={st.median(gaps):.3f} max={max(gaps):.3f}")
big = sum(1 for g in gaps if g > 2)
print(f"gaps > 2us: {big} / {len(gaps)}")

# distribution of durations (coarse histogram)
from collections import Counter
c = Counter(round(d * 2) / 2 for d in durs)  # 0.5us bins
print("duration histogram (0.5us bins):")
for k in sorted(c):
    bar = "#" * min(c[k] // 5 + (1 if c[k] else 0), 60)
    print(f"  {k:5.1f}: {c[k]:4d} {bar}")

# temporal: first 40 durations in order
print("first 40 durations:", [f"{d:.1f}" for d in durs[:40]])
# run-length encoding of fast/slow state (threshold 10us)
state = "".join("F" if d < 10 else "S" for d in durs)
runs = []
for ch in state:
    if runs and runs[-1][0] == ch:
        runs[-1][1] += 1
    else:
        runs.append([ch, 1])
print(f"state runs (F=fast<10us, S=slow>=10us), first 30 runs:")
print(" ".join(f"{c}{n}" for c, n in runs[:30]))
print(f"total runs: {len(runs)}, F-runs: {sum(1 for c,_ in runs if c=='F')}, S-runs: {sum(1 for c,_ in runs if c=='S')}")

# ---- op_summary: per-call mte2/scalar breakdown ----
cols = None
summ = []
with open(base + "s2_op_summary.csv", newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        if r.get("Op Name") != "MhcHeadCollapse":
            continue
        summ.append(r)

print(f"\nop_summary n={len(summ)}")
aiv = [float(r["aiv_time(us)"].strip()) for r in summ]
mte2 = [float(r["aiv_mte2_time(us)"].strip()) for r in summ]
scal = [float(r["aiv_scalar_time(us)"].strip()) for r in summ]
vec = [float(r["aiv_vec_time(us)"].strip()) for r in summ]

def stats(name, xs):
    xs2 = sorted(xs)
    print(f"{name}: n={len(xs)} min={min(xs):.2f} p25={xs2[len(xs2)//4]:.2f} "
          f"median={xs2[len(xs2)//2]:.2f} p75={xs2[3*len(xs2)//4]:.2f} max={max(xs):.2f}")

stats("aiv_time", aiv)
stats("aiv_mte2", mte2)
stats("aiv_scalar", scal)
stats("aiv_vec", vec)

# correlation: mte2 vs aiv total
import math
def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    vx = sum((a-mx)**2 for a in x); vy = sum((b-my)**2 for b in y)
    return cov / math.sqrt(vx*vy) if vx and vy else float("nan")

print(f"\ncorr(mte2, aiv)     = {pearson(mte2, aiv):.3f}")
print(f"corr(scalar, aiv)   = {pearson(scal, aiv):.3f}")
print(f"corr(vec, aiv)      = {pearson(vec, aiv):.3f}")
print(f"corr(scalar, mte2)  = {pearson(scal, mte2):.3f}")

# joint histogram mte2 x scalar ratio buckets
fast = [(m, s, v, a) for m, s, v, a in zip(mte2, scal, vec, aiv) if a < 10]
slow = [(m, s, v, a) for m, s, v, a in zip(mte2, scal, vec, aiv) if a >= 10]
print(f"\nfast bucket (aiv<10): n={len(fast)}")
if fast:
    print(f"  mte2:   min={min(x[0] for x in fast):.2f} median={st.median([x[0] for x in fast]):.2f} max={max(x[0] for x in fast):.2f}")
    print(f"  scalar: min={min(x[1] for x in fast):.2f} median={st.median([x[1] for x in fast]):.2f} max={max(x[1] for x in fast):.2f}")
    print(f"  vec:    min={min(x[2] for x in fast):.2f} median={st.median([x[2] for x in fast]):.2f} max={max(x[2] for x in fast):.2f}")
print(f"slow bucket (aiv>=10): n={len(slow)}")
if slow:
    print(f"  mte2:   min={min(x[0] for x in slow):.2f} median={st.median([x[0] for x in slow]):.2f} max={max(x[0] for x in slow):.2f}")
    print(f"  scalar: min={min(x[1] for x in slow):.2f} median={st.median([x[1] for x in slow]):.2f} max={max(x[1] for x in slow):.2f}")
    print(f"  vec:    min={min(x[2] for x in slow):.2f} median={st.median([x[2] for x in slow]):.2f} max={max(x[2] for x in slow):.2f}")

# mte2 histogram
c2 = Counter(round(m) for m in mte2)
print("\nmte2 histogram (1us bins):")
for k in sorted(c2):
    print(f"  {k:3d}: {c2[k]:4d} {'#'*min(c2[k]//5,60)}")
