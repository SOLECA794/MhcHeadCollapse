#!/usr/bin/env bash
# ascend-kg 更新自检 —— 由 SKILL.md 顶部指令在「本会话第一次触发 ascend-kg」时调用。
#
# 行为约定（SKILL.md 依赖此契约，更新一律经用户确认后才执行）：
#   无输出 + exit 0 = 无更新/无法检测（离线、仓库缺失等），直接继续当前任务；
#   输出含 ⬆️（不带 --apply 运行）= 检测到远程领先——向用户确认，同意后以 --apply 重跑执行更新；
#   --apply 运行后输出 ✅ = 已拉取并重装；无输出 = 已无需更新；
#   输出含 ⚠️    = 需要人工处理（分叉/本地领先/工作区脏/重装失败），把提示带给用户。
#
# 判定逻辑：git fetch 后仅当「远程严格领先」（本地 HEAD 是上游祖先）才动作——
# 本地领先（未推送 WIP）或分叉都不自动动仓库，防覆盖用户工作。工作区有未提交
# 改动同样跳过（pull --ff-only 也可能与之冲突）。重装走 install.sh --force，
# copy_skill_dir 已内置运行时数据保留（engine/state、domain_cache 记录、flows/case*）。
#
# 会话级去重：CLAUDE_SESSION_ID 可用时按会话 stamp（同会话多次触发只检一次）；
# 拿不到则 1 小时时间窗兜底（stat 非 GNU 时视为过期，宁可多检不漏检）。
#
# 源仓库定位：优先 KG_TOOLS_REPO 环境变量，其次安装时落盘的 .repo_source 戳。
set -u

# --apply：用户确认后的执行阶段（拉取+重装），绕过会话去重；缺省 = 仅检测不动作
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

# Windows/Git Bash 兼容：脚本可能经 `bash 'C:\...\check_update.sh'` 调用——$0 带
# 反斜杠时 dirname 按 POSIX 只认 /，会把整串当文件名返回 "."，SKILL_DIR 错位到
# 工作目录上级。反斜杠统一转正斜杠（C:/... 在 MSYS 下 cd/stat/git -C 均接受）。
self="${BASH_SOURCE[0]:-$0}"
case "$self" in *\\*) self="${self//\\//}" ;; esac
SKILL_DIR="$(cd "$(dirname "$self")/.." && pwd)"
STAMP_BASE="${TMPDIR:-/tmp}/ascend-kg-updchk"

if [[ $APPLY -eq 0 ]]; then
  sid="${CLAUDE_SESSION_ID:-}"
  if [[ -n "$sid" ]]; then
    STAMP="$STAMP_BASE.$sid"
    [[ -e "$STAMP" ]] && exit 0  # 本会话已检过（stamp 存在即去重；touch 失败至多多检一次）
  else
    STAMP="$STAMP_BASE.win"
    mtime="$(stat -c %Y "$STAMP" 2>/dev/null || echo 0)"
    if [[ $(( $(date +%s) - mtime )) -lt 3600 ]]; then
      exit 0
    fi
  fi
  mkdir -p "$(dirname "$STAMP")" 2>/dev/null
  touch "$STAMP" 2>/dev/null
fi

SRC="${KG_TOOLS_REPO:-$(cat "$SKILL_DIR/.repo_source" 2>/dev/null || true)}"
# Windows 安装器写入的戳可能是 CRLF 行尾或反斜杠路径——\r 会让 -d/git -C 全部
# 判失败（路径不存在），先剥 \r 再统一反斜杠为正斜杠
SRC="${SRC%$'\r'}"
case "$SRC" in *\\*) SRC="${SRC//\\//}" ;; esac
[[ -n "$SRC" && -d "$SRC/.git" ]] || exit 0
command -v git >/dev/null 2>&1 || exit 0

# 离线/无权限/超时一律静默跳过——自检绝不阻塞 skill 使用
git_fetch() {
  if command -v timeout >/dev/null 2>&1; then timeout 20 git "$@"; else git "$@"; fi
}
git_fetch -C "$SRC" fetch --quiet 2>/dev/null || exit 0

head="$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" || exit 0
up="$(git -C "$SRC" rev-parse '@{u}' 2>/dev/null)" || exit 0
[[ -n "$head" && -n "$up" && "$head" != "$up" ]] || exit 0

if ! git -C "$SRC" merge-base --is-ancestor "$head" "$up" 2>/dev/null; then
  echo "⚠️ kg-tools 源仓库本地领先或与远程分叉（HEAD ${head:0:7} vs 远程 ${up:0:7}），已跳过自动更新，请手动处理：cd $SRC && git status"
  exit 0
fi
if [[ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ]]; then
  echo "⚠️ kg-tools 源仓库有未提交改动，已跳过自动更新（防覆盖本地工作），处理后再触发：cd $SRC && git status"
  exit 0
fi

if [[ $APPLY -eq 0 ]]; then
  echo "⬆️ kg-tools 有更新（${head:0:7} → ${up:0:7}）。向用户确认是否更新；同意后运行 bash $SKILL_DIR/scripts/check_update.sh --apply 执行（拉取+重装，保留运行时数据），拒绝则继续当前任务。"
  exit 0
fi
echo "⬆️ kg-tools 更新执行中（${head:0:7} → ${up:0:7}）：拉取并重装 ascend-kg skill..."
if ! git -C "$SRC" pull --ff-only --quiet 2>/dev/null; then
  echo "⚠️ git pull 失败，已跳过重装，请手动执行：cd $SRC && git pull"
  exit 0
fi
# 重装旗标按本 skill 实际安装位置自适应（防 Hermes/OpenCode/OpenClaw 安装误装到 Claude）
agent_flag="--claude"
case "$SKILL_DIR" in
  */.hermes/*)          agent_flag="--hermes" ;;
  */.config/opencode/*|*/.opencode/*) agent_flag="--opencode" ;;
  */.openclaw/*)        agent_flag="--openclaw" ;;
esac
# 注：install.sh 的 copy_skill_dir 会 rm -rf 本 skill 目录（含正在运行的本脚本）——
# Linux unlink 语义下已打开的脚本 fd 不受影响，且此调用之后仅剩收尾输出。
if ! bash "$SRC/install.sh" ascend-kg "$agent_flag" --force; then
  echo "⚠️ install.sh 重装失败，请手动执行：bash $SRC/install.sh ascend-kg $agent_flag --force"
  exit 0
fi
echo "✅ ascend-kg 已更新重装（本会话已加载的是旧版 SKILL.md，下次触发生效新内容）"
