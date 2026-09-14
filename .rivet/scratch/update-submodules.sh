#!/usr/bin/env bash
# 一键更新 OP-Learning 下的 git 子模块到远端最新
# 用法: bash .rivet/scratch/update-submodules.sh [子模块相对路径...]
#       不给参数则更新 .gitmodules 中全部子模块。
#
# ⚠ 这是持久工具脚本，不是一次性探针——不要纳入 scratch 的批量清理。
#
# 背景（2026-09-11 踩过的三个 Windows 特有坑）:
#   1. Windows 默认 MAX_PATH 260 → 长路径报 "Filename too long"
#      已修: git config --global core.longpaths true
#   2. 上次 pull 被 (1) 打断后，文件已落盘但 HEAD 指针未跟上
#      → 本地看似"改动/未跟踪"，内容实为上游版本。git 保守拒绝覆盖，pull 反复 abort。
#      本脚本: 先识别"内容 == 上游某历史版本"的伪改动，只对真改动做备份。
#   3. Windows 无法创建符号链接（索引 120000，磁盘成普通文件）→ 恒显示 M/D
#      已修(cannbot-skills): git update-index --assume-unchanged <14个符号链接条目>
set -u
export MSYS_NO_PATHCONV=1
ROOT="/d/Desktop/OP-Learning"
BK="$ROOT/.rivet/scratch/submodule-update-backup"
mkdir -p "$BK"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  mapfile -t TARGETS < <(cd "$ROOT" && git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null | awk '{print $2}')
fi

echo "=== 目标子模块: ${#TARGETS[@]} 个 ==="
for R in "${TARGETS[@]}"; do
  D="$ROOT/$R"
  [ -d "$D/.git" ] || { echo "  [跳过, 非独立仓库] $R"; continue; }
  cd "$D" || continue
  NAME=$(basename "$R")
  echo ""
  echo "########## $NAME ##########"

  timeout 300 git fetch origin >/dev/null 2>&1 || { echo "  fetch 失败, 跳过"; continue; }
  BEHIND=$(git rev-list --count HEAD..origin/master 2>/dev/null || echo "?")
  AHEAD=$(git rev-list --count origin/master..HEAD 2>/dev/null || echo "?")
  echo "  落后: $BEHIND   领先: $AHEAD"
  [ "$BEHIND" = "0" ] && { echo "  已是最新"; continue; }

  echo "--- 识别真实本地改动（排除'内容==上游'的伪改动）---"
  REAL=()
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    LB=$(git hash-object -- "$f" 2>/dev/null) || continue
    HIT=""
    for c in $(git log origin/master --format=%H -- "$f" 2>/dev/null | head -80); do
      [ "$(git rev-parse "$c:$f" 2>/dev/null)" = "$LB" ] && { HIT=1; break; }
    done
    [ -z "$HIT" ] && { REAL+=("$f"); echo "    ★真实本地改动: $f"; }
  done < <(git diff --name-only HEAD 2>/dev/null)

  if [ ${#REAL[@]} -gt 0 ]; then
    echo "--- 备份真实改动 ---"
    git diff HEAD > "$BK/$NAME.local-changes.patch" 2>/dev/null
    echo "    patch: $BK/$NAME.local-changes.patch ($(wc -c < "$BK/$NAME.local-changes.patch") bytes)"
    for f in "${REAL[@]}"; do
      mkdir -p "$BK/$NAME.files/$(dirname "$f")"
      cp -p "$f" "$BK/$NAME.files/$f" 2>/dev/null
    done
    echo "    原文副本: $BK/$NAME.files/"
    echo "--- 暂存这些改动以避免阻塞 pull ---"
    git diff "${REAL[@]}" 2>/dev/null | git apply --cached --reverse 2>/dev/null || true
  fi

  echo "--- pull ---"
  timeout 600 git pull --ff-only > "$BK/$NAME.pull.log" 2>&1
  RC=$?
  echo "  exit=$RC"
  if [ $RC -ne 0 ]; then
    echo "  未完成, 末尾输出:"; grep -v '^	' "$BK/$NAME.pull.log" | tail -8
  fi
  echo "  HEAD: $(git rev-parse --short HEAD)  落后: $(git rev-list --count HEAD..origin/master)"
done

echo ""
echo "=== 汇总 ==="
for R in "${TARGETS[@]}"; do
  cd "$ROOT/$R" 2>/dev/null || continue
  printf "  %-28s HEAD=%s 落后=%s\n" "$(basename "$R")" "$(git rev-parse --short HEAD)" "$(git rev-list --count HEAD..origin/master 2>/dev/null)"
done
