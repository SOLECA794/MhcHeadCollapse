import json, requests
BASE = 'https://cannjudge.cn'


def get(sid):
    j = requests.get(f'{BASE}/api/submissions/{sid}', timeout=30).json()
    return (j.get('data') or j).get('result', [])


BEST = [2.12, 2.12, 2.48, 2.48, 4.36]

print('=' * 74)
print('best_time 揭示的结构: floor vs 可压缩部分')
print('=' * 74)
print("""
Case1/2 走 PATH3(TinyH4), best 均为 2.12 -> PATH3 floor = 2.12
Case3/4 走 PATH2(向量),  best 均为 2.48 -> PATH2 floor = 2.48
Case5   走 PATH2, best = 4.36           -> 比 floor 多 1.88

推论:
  PATH2 的 floor(2.48) 比 PATH3(2.12) 高 0.36
  Case5 在同为 PATH2 下多出的 1.88 = 真实可压缩的工作量
""")

print('=' * 74)
print('本队优化空间分解 (假设能达到 best)')
print('=' * 74)
mine = get('6a9fb0d9c76b321ca6e47d5f')
tot_gain = 0.0
for i, x in enumerate(mine):
    t, b = x['time'], x['best_time']
    gain = t - b
    tot_gain += gain
    path = 'PATH3' if i < 2 else 'PATH2'
    print(f'Case{i+1} ({path}): {t:5.2f} -> {b:5.2f}   可省 {gain:5.2f}  ({gain / t * 100:4.1f}%)')
print(f"{'合计':10} {sum(x['time'] for x in mine):5.2f} -> {sum(BEST):5.2f}   可省 {tot_gain:5.2f}")

print()
print('=' * 74)
print('优化重心 (按可省绝对值排序)')
print('=' * 74)
items = [(f"Case{i+1}", x['time'] - x['best_time']) for i, x in enumerate(mine)]
for name, g in sorted(items, key=lambda z: -z[1]):
    print(f'  {name}: 可省 {g:5.2f}  (占总可省 {g / tot_gain * 100:4.1f}%)')

print()
print('  => Case1 差距最大 (2.70x) 且最反常:')
print('     只有 16 个元素 (312 字节), 却要 5.72')
print('     best=2.12 说明它应接近纯 floor -> 本队 Case1 有 3.6 的纯浪费')
print('     Case1/2 都在 PATH3(TinyH4), 说明问题在 PATH3 而非 PATH2')
