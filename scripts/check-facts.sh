#!/usr/bin/env bash
# check-facts.sh — 校验 FACTS.md / PITFALLS.md 中的 file:line 引用是否仍有效
#
# 存在意义：文档会随代码漂移。FACTS.md 的每条结论都依赖 file:line 证据，
# 若引用的文件被移动/改写，文档就会静默失效（PITFALLS.md P12 的教训）。
# 本脚本把"文档是否还准确"变成一条可执行的检查。
#
# 用法: bash scripts/check-facts.sh
# 退出码: 0 = 全部引用有效; 1 = 有失效引用

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

# 仅在 TTY 下用颜色，重定向/管道时输出纯文本
if [ -t 1 ]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; NC=''
fi
OK=0; BAD=0; WARN=0

# ---------- 1. 校验 FACTS.md 中的行号引用 ----------
check_facts_refs() {
  echo "=== 校验 FACTS.md 的行号引用 ==="
  # 从 FACTS.md 提取形如 path:LINE 或 path:LN-LN 的引用
  grep -oE '`[^`]*\.(cpp|md|h|hpp|py|sh|txt):[0-9]+(-[0-9]+)?`' FACTS.md 2>/dev/null \
    | tr -d '`' | sort -u | while IFS= read -r ref; do
      file="${ref%%:*}"
      lineno="${ref##*:}"
      # 去掉行号范围的后半段
      lineno="${lineno%%-*}"
      # 解析相对路径（FACTS.md 中引用的是相对赛题目录的路径）
      target="$ROOT/$file"
      if [ ! -f "$target" ]; then
        # 尝试 versions/ 前缀
        target="$ROOT/versions/$file"
      fi
      # 外部仓库引用（同级 Ascend/ 下，不在本赛题目录内）
      if [ ! -f "$target" ]; then
        case "$file" in
          asc-devkit/*|catlass/*)
            target="$ROOT/../../../$file"
            ;;
          cannjudge-submit/*|cannjudge-submit-plaintext/*)
            target="$ROOT/../../../cann-learing-hub/skills/$file"
            ;;
        esac
      fi
      if [ ! -f "$target" ]; then
        printf "  ${RED}✗ 文件不存在${NC}: %s\n" "$file"
        continue
      fi
      total=$(wc -l < "$target")
      if [ "$lineno" -gt "$total" ]; then
        printf "  ${RED}✗ 行号超界${NC}: %s (文件仅 %s 行)\n" "$ref" "$total"
      else
        printf "  ${GREEN}✓${NC} %s\n" "$ref"
      fi
    done
}

# ---------- 2. 校验关键文件存在 ----------
check_files() {
  echo ""
  echo "=== 校验关键文件存在 ==="
  for f in FACTS.md PITFALLS.md EXPERIMENTS.md README.md \
           versions/README.md scripts/verify.sh scripts/submit.sh \
           赛题.md 迭代优化记录.md; do
    if [ -f "$ROOT/$f" ]; then
      printf "  ${GREEN}✓${NC} %s\n" "$f"; OK=$((OK+1))
    else
      printf "  ${RED}✗ 缺失${NC} %s\n" "$f"; BAD=$((BAD+1))
    fi
  done
}

# ---------- 3. 校验 FACTS.md 的核心断言仍成立 ----------
check_assertions() {
  echo ""
  echo "=== 校验核心断言（代码层面） ==="

  local V="$ROOT/versions/v116_vsync_batched/code"

  # F2.3: block_dim 公式
  if grep -q "outer <= 4U ? 1U" "$V/op_host/mhc_head_collapse.cpp" 2>/dev/null; then
    printf "  ${GREEN}✓${NC} F2.3 block_dim 公式仍在 op_host:33 附近\n"
  else
    printf "  ${YELLOW}⚠${NC} F2.3 block_dim 公式已变 —— 需复核 FACTS.md F2.3\n"; WARN=$((WARN+1))
  fi

  # F2.3: 行收集步长
  if grep -q "row += block_dim_" "$V/op_kernel/mhc_head_collapse.cpp" 2>/dev/null; then
    printf "  ${GREEN}✓${NC} F2.3 行收集步长仍为 block_dim_\n"
  else
    printf "  ${YELLOW}⚠${NC} F2.3 行收集逻辑已变 —— 需复核\n"; WARN=$((WARN+1))
  fi

  # F2.4: v111/v116 副本关系
  local h111 h116v h116r
  h111=$(md5sum "$ROOT/versions/v111_path2_group4/code/op_kernel/mhc_head_collapse.cpp" 2>/dev/null | cut -c1-8)
  h116v=$(md5sum "$V/op_kernel/mhc_head_collapse.cpp" 2>/dev/null | cut -c1-8)
  h116r=$(md5sum "$ROOT/versions/v116_batched_reduce/code/op_kernel/mhc_head_collapse.cpp" 2>/dev/null | cut -c1-8)
  if [ "$h111" = "$h116v" ]; then
    printf "  ${GREEN}✓${NC} F2.4 v116_vsync_batched 仍与 v111 相同 (%s)\n" "$h111"
  else
    printf "  ${YELLOW}⚠${NC} F2.4 v116_vsync_batched 已不同于 v111 —— 需复核\n"; WARN=$((WARN+1))
  fi
  if [ "$h116r" != "$h111" ]; then
    printf "  ${GREEN}✓${NC} F2.4 v116_batched_reduce 确实有独立内容 (%s)\n" "$h116r"
  else
    printf "  ${YELLOW}⚠${NC} F2.4 v116_batched_reduce 与 v111 相同了\n"; WARN=$((WARN+1))
  fi

  # F2.5: weight DMA 位置
  if grep -q "DataCopy(w_cache, weight_gm_" "$V/op_kernel/mhc_head_collapse.cpp" 2>/dev/null; then
    printf "  ${GREEN}✓${NC} F2.5 weight DMA 仍在 Process() 内\n"
  else
    printf "  ${YELLOW}⚠${NC} F2.5 weight DMA 位置已变 —— 可能已修复真\n"; WARN=$((WARN+1))
  fi

  # P8: verify.sh 改写 target
  if grep -q "ascend910_93" "$ROOT/scripts/verify.sh" 2>/dev/null; then
    printf "  ${GREEN}✓${NC} P8 verify.sh 仍会改写 target（提交前务必检查）\n"
  else
    printf "  ${YELLOW}⚠${NC} P8 verify.sh 已不再改写 target\n"; WARN=$((WARN+1))
  fi

  # op_host 提交目标应为 ascend910b
  if grep -q 'AddConfig("ascend910b")' "$V/op_host/mhc_head_collapse.cpp" 2>/dev/null; then
    printf "  ${GREEN}✓${NC} op_host 提交目标是 ascend910b（正确）\n"
  else
    printf "  ${RED}✗${NC} op_host 提交目标不是 ascend910b —— 可能被 verify.sh 改过！\n"; BAD=$((BAD+1))
  fi
}

# ---------- 4. 统计版本副本 ----------
check_versions() {
  echo ""
  echo "=== 版本目录统计 ==="
  local total uniq dup
  total=$(ls -d "$ROOT"/versions/*/ 2>/dev/null | wc -l)
  uniq=$(for d in "$ROOT"/versions/*/; do
           f="$d/code/op_kernel/mhc_head_collapse.cpp"
           [ -f "$f" ] && md5sum "$f" | cut -c1-8
         done | sort -u | wc -l)
  local withk
  withk=$(for d in "$ROOT"/versions/*/; do
            [ -f "$d/code/op_kernel/mhc_head_collapse.cpp" ] && echo x
          done | wc -l)
  dup=$((withk - uniq))
  printf "  版本目录: %s 个（含 kernel 的 %s 个，唯一内容 %s 组，副本 %s 个）\n" \
    "$total" "$withk" "$uniq" "$dup"
  if [ "$dup" -gt 0 ]; then
    printf "  ${YELLOW}提醒${NC}: 本仓库有 %s 个副本目录，声称某版本有改动前先 md5 对比\n" "$dup"
    WARN=$((WARN+1))
  fi
}

echo "############ check-facts.sh ############"
echo "根目录: $ROOT"
echo ""

check_files
check_assertions
check_versions
check_facts_refs

echo ""
echo "############ 汇总 ############"
printf "  ${GREEN}通过 %s${NC}  ${YELLOW}提醒 %s${NC}  ${RED}失败 %s${NC}\n" "$OK" "$WARN" "$BAD"
if [ "$BAD" -gt 0 ]; then
  echo ""
  echo "→ 有失败项。FACTS.md 可能已与代码脱节，请复核后更新文档。"
  exit 1
fi
if [ "$WARN" -gt 0 ]; then
  echo ""
  echo "→ 有提醒项（通常是 bug 已被修复，或仍有副本）。请判断是否需要更新 FACTS.md。"
fi
exit 0
