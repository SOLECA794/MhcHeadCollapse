#!/usr/bin/env python3
"""
探针: 验证 block_dim 的双重用途如何导致 group-of-4 无法生效

背景(FACTS.md F2.3):
  op_host:33  uint32_t block_dim = outer <= 4U ? 1U : (outer < 20 ? outer : 20);
  op_host:50  context->SetBlockDim(block_dim);          <- 用途1: 并行度(block 数量)
  op_host:44  tiling->block_dim = block_dim;            <- 用途2: 行分配步长
  kernel:346  for (; cnt < kGroupRows && row < outer_; ++cnt, row += block_dim_)
                                                  ^^^^^^^^^^^^^^^^^^^^ 用同一个值

本探针穷举各种改法, 看哪种能让 Case3/4/5 都收满 4 行。
"""

K_GROUP_ROWS = 4
CASES = [
    # name, outer
    ("Case1", 1), ("Case2", 4), ("Case3", 1), ("Case4", 2), ("Case5", 8),
]
K_MAX_CORES = 20


def collect_rows(outer, block_dim, nblocks):
    """模拟 kernel 的行收集: 每个 block 从自己的 block_idx 起, 步长 block_dim, 收满 kGroupRows"""
    result = []
    for b in range(nblocks):
        rows, row, cnt = [], b, 0
        while row < outer and cnt < K_GROUP_ROWS:
            rows.append(row)
            cnt += 1
            row += block_dim
        result.append((b, cnt, rows))
    return result


def show(label, get_bd, get_nb):
    print(f"\n{'='*66}")
    print(f"方案: {label}")
    print(f"{'='*66}")
    print(f"{'case':7}{'outer':>6}{'blocks':>8}{'block_dim':>11}{'各block的cnt':>16}{'收满组?':>10}")
    all_full = True
    for name, outer in CASES:
        bd = get_bd(outer)
        nb = get_nb(outer, bd)
        res = collect_rows(outer, bd, nb)
        cnts = [c for _, c, _ in res]
        covered = sum(cnts)
        full = "✓" if all(c == min(K_GROUP_ROWS, outer) for c in cnts) and covered == outer else "✗"
        if full == "✗":
            all_full = False
        print(f"{name:7}{outer:>6}{nb:>8}{bd:>11}{str(cnts):>16}{full:>10}")
    print(f"→ 全部收满: {'是' if all_full else '否'}")
    return all_full


print("=" * 66)
print("探针: block_dim 双重用途导致 group-of-4 失效 (FACTS.md F2.3)")
print("=" * 66)

# ---- 现状 ----
show(
    "现状: bd = outer<=4 ? 1 : min(outer,20);  blocks = bd",
    lambda o: 1 if o <= 4 else min(o, K_MAX_CORES),
    lambda o, bd: bd,
)

# ---- 天真改法: bd = ceil(outer/kGroupRows)，并行度同时缩小 ----
def naive_bd(o):
    return 1 if o <= 4 else (o + K_GROUP_ROWS - 1) // K_GROUP_ROWS


show(
    "天真改法: bd = ceil(outer/4), blocks = bd  (并行度被牺牲)",
    naive_bd,
    lambda o, bd: bd,
)

# ---- 正确改法: 解耦 —— blocks 保持 outer (并行度不变), 但行按 group 连续切分 ----
# 思路: 每个 block 处理 ceil(outer/nblocks) 个连续行, 组内 stride=1 而非 block_dim
print(f"\n{'='*66}")
print("方案: 解耦 —— blocks 保持不变, 行按「连续块」切分给各 block")
print(f"{'='*66}")
print("  即: 每个 block b 处理行 [b*chunk, min((b+1)*chunk, outer)), 其中 chunk=ceil(outer/nblocks)")
print("      组内 stride 改为 1 (同 block 内连续), 而非 block_dim_")
print()
for label, nb_of in [
    ("blocks = min(outer,20)", lambda o: min(o, K_MAX_CORES)),
    ("blocks = ceil(outer/4)", lambda o: max(1, (o + K_GROUP_ROWS - 1) // K_GROUP_ROWS)),
]:
    print(f"  --- {label} ---")
    print(f"  {'case':7}{'outer':>6}{'blocks':>8}{'chunk':>7}{'各block收行数':>16}{'组内cnt<=4?':>13}")
    ok = True
    for name, outer in CASES:
        nb = nb_of(outer)
        chunk = (outer + nb - 1) // nb
        cnts = []
        for b in range(nb):
            lo = b * chunk
            hi = min(lo + chunk, outer)
            cnts.append(max(0, hi - lo))
        full = all(c <= K_GROUP_ROWS for c in cnts) and sum(cnts) == outer
        if not full:
            ok = False
        print(f"  {name:7}{outer:>6}{nb:>8}{chunk:>7}{str(cnts):>16}{('✓' if full else '✗'):>13}")
    print(f"  → 全部合法: {'是' if ok else '否'}")
    print()

print("=" * 66)
print("结论")
print("=" * 66)
print("""
1. 现状: Case5 (outer=8) 拿到 bd=8 → 每个 block 只收 1 行 → group-of-4 空转
   Case3 (outer=1) 拿到 bd=1 → 只能收 1 行 → 本来就没有可批的
   → 问题不在 Case3, 而在 Case5

2. 天真改法 (bd = ceil(outer/4)) 能让 Case5 收满 4 行, 但:
   - blocks 从 8 降到 2 → 并行度损失 4 倍
   - Case3 (outer=1) 仍是 bd=1 → 无改善
   → 这是"诊断探针"级别, 不是最终方案

3. 正确方向: 解耦并行度与分组粒度。让 blocks 保持 min(outer,20),
   但每个 block 处理「连续的一段行」, 组内 stride=1。
   这样 Case5 用 8 个 block 各处理 1 行 —— 仍然收不满 4 行!

   ⚠ 关键洞察: 若 outer=8 且 blocks=8, 无论怎么切, 每 block 平均只有 1 行。
   要让批处理生效, 必须让 blocks < outer (牺牲并行度换批处理),
   或者接受"批处理对 Case5 无意义"。

4. 因此实验B 的真正问题应当是:
   在 outer=8 这个小规模下, 8 个 block 各跑 1 行 vs 2 个 block 各跑 4 行,
   哪个更快? 这才是需要实机测量的。
   → 批处理不是"没生效所以结论错", 而是"在这个规模下可能本就不划算"。
""")
