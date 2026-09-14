#!/usr/bin/env bash
# 阶段1: 分析 cann-learing-hub 的 pull 障碍（纯查询，不做任何破坏性操作）
set -u
export MSYS_NO_PATHCONV=1
R="/d/Desktop/OP-Learning/Ascend/cann-learing-hub"
cd "$R" || exit 1

echo "=== A. 上游新增的文件 ==="
git diff --name-only --diff-filter=A HEAD origin/master 2>/dev/null | LC_ALL=C sort > /tmp/up_added.txt
echo "上游新增: $(wc -l < /tmp/up_added.txt)"

echo "=== B. 展开未跟踪文件 ==="
: > /tmp/untracked_files.txt
git ls-files --others --exclude-standard 2>/dev/null | while IFS= read -r p; do
  if [ -d "$p" ]; then find "$p" -type f >> /tmp/untracked_files.txt 2>/dev/null
  elif [ -f "$p" ]; then echo "$p" >> /tmp/untracked_files.txt; fi
done
LC_ALL=C sort -u /tmp/untracked_files.txt -o /tmp/untracked_files.txt
echo "未跟踪文件: $(wc -l < /tmp/untracked_files.txt)"

echo "=== C. 交集 = 会被 pull 覆盖的未跟踪文件 ==="
LC_ALL=C comm -12 /tmp/untracked_files.txt /tmp/up_added.txt > /tmp/conflicts.txt
echo "冲突文件数: $(wc -l < /tmp/conflicts.txt)"
head -6 /tmp/conflicts.txt

echo "=== D. 逐一验证内容是否等于上游 ==="
SAME=0; DIFF=0; : > /tmp/conflict_diff.txt
while IFS= read -r f; do
  [ -f "$f" ] || continue
  U=$(git rev-parse "origin/master:$f" 2>/dev/null) || continue
  W=$(git hash-object -- "$f" 2>/dev/null)
  if [ "$U" = "$W" ]; then SAME=$((SAME+1)); else DIFF=$((DIFF+1)); echo "$f" >> /tmp/conflict_diff.txt; fi
done < /tmp/conflicts.txt
echo "内容==上游(无信息量,可移走): $SAME"
echo "内容!=上游(需保留): $DIFF"
[ -s /tmp/conflict_diff.txt ] && { echo "--- 需保留的 ---"; cat /tmp/conflict_diff.txt; }
echo "=== 完成 ==="
