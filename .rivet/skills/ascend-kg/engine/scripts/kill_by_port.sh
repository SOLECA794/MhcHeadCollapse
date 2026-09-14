#!/bin/bash
# kill_by_port.sh <port> [grace] [pattern]
# 按端口精确销毁进程（红线工具）。
#   主路径：lsof 定位 LISTEN pid → 若是进程组组长 → kill 整个组（vllm 主进程+EngineCore 同组）
#   回退路径：非组长 → 杀 pid + 递归杀后代
#   兜底：清理 cmdline 匹配 pattern 的孤儿。pattern 由 backend.process_pattern 注入
#         （DESIGN-multi-backend §3.2）；缺省 vllm serve 模式；显式空串 = 禁用兜底。
# 严禁 pkill -f "vllm serve <model>" —— 宽模式会连主服务一起误杀（本会话血泪教训）。
# 端口作用域天然隔离 8000/8001，不依赖模型路径。
set -u
PORT=$1
GRACE=${2:-10}
PATTERN=${3-__VLLM_DEFAULT__}
if [ "$PATTERN" = "__VLLM_DEFAULT__" ]; then
  PATTERN="vllm serve.* --port $PORT"
fi

cleanup_orphans() {
  if [ -n "$PATTERN" ]; then
    # 排除脚本自身（$$）与其直接父 bash（$PPID）：PATTERN 作为命令行参数出现在
    # kill_by_port.sh 自身 cmdline 里，pgrep -f 会自匹配 → kill -9 自杀 → 返回 -9，
    # 引擎 _apply 把 -9 当非零误判 apply 失败（SGLang GLM-4 实测浪费一轮）。
    # 真正的孤儿进程 cmdline 含 pattern 但 pid 必不是脚本自身/其父进程。
    pgrep -f "$PATTERN" 2>/dev/null | grep -v "^$$\$" | grep -v "^$PPID\$" | xargs -r kill -9 2>/dev/null || true
  fi
  # 逃逸的 EngineCore：vllm 把子进程标题改写为 VLLM::EngineCore，cmdline 不含端口/模式串，
  # 上面的组杀与模式兜底都抓不到。判据 = ppid==1（API server 已死，必为孤儿）——有活父进程
  # 的 EngineCore（正常服务，如他人在跑的 vllm 实例）不受影响。
  # 实测案例（case7 scr14-24 连锁失败）：static-kernel 编译超时被 kill 时 EngineCore 卡在
  # 设备调用里躲过组杀成孤儿，占 56GB 显存，后续 11 轮启动全因 free memory 不足失败。
  for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null | grep -v "^$$\$" | grep -v "^$PPID\$"); do
    _ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$_ppid" = "1" ] && kill -9 "$pid" 2>/dev/null || true
  done
}

MAIN=$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1)

if [ -z "$MAIN" ]; then
  echo "kill_by_port: no LISTEN process on port $PORT"
  # 兜底清理可能残留的孤儿（严格限定注入的命令行模式，绝不匹配 orchestrator）
  cleanup_orphans
  sleep 1
  echo "kill_by_port: port $PORT released"
  exit 0
fi

PGID=$(ps -o pgid= -p "$MAIN" 2>/dev/null | tr -d ' ')
SELF=$$

# 排除 kill_by_port.sh 自身（防止脚本被组杀连坐）
[ "$PGID" = "$SELF" ] && PGID="$MAIN"

if [ -n "$PGID" ] && [ "$PGID" = "$MAIN" ]; then
  # 主进程是组组长：整组 TERM
  echo "kill_by_port: TERM group -$PGID (main=$MAIN) on port $PORT"
  kill -TERM -- "-$PGID" 2>/dev/null || kill -TERM "$MAIN" 2>/dev/null || true
  sleep "$GRACE"
  # 组内残留 KILL
  pkill -KILL -g "$PGID" 2>/dev/null || true
else
  # 非组长：杀主进程 + 递归杀后代
  echo "kill_by_port: TERM pid=$MAIN (not group leader, pgid=$PGID) on port $PORT"
  kill -TERM "$MAIN" 2>/dev/null || true
  sleep "$GRACE"
  kill_desc() {
    local pid=$1
    for c in $(pgrep -P "$pid" 2>/dev/null); do kill_desc "$c"; done
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  }
  kill_desc "$MAIN"
fi

# 兜底清理孤儿（仅命中注入模式的进程；逃逸的 EngineCore cmdline 不含模式串，此处抓不到）
cleanup_orphans

sleep 1
if lsof -t -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERR: port $PORT still held after kill" >&2
  exit 1
fi
echo "kill_by_port: port $PORT released"
exit 0
