#!/bin/bash
# wait_port.sh <port> <timeout_seconds> [health_path_or_url]
# 轮询 GET health 路径直到 200 或超时。用于服务启动就绪检测。
# 第 3 参按后端注入（backend.health_url，DESIGN-multi-backend §3.2），缺省 /health；
# 以 http 开头则视为完整 URL（MindIE 健康面挂管理口 127.0.0.2:1026 时用）。
# 用 $SECONDS 计时（墙钟时间），把 curl 耗时也算进去，避免名义 300s 实际最坏 480s。
set -u
PORT=$1
TIMEOUT=${2:-300}
HEALTH_PATH=${3-/health}
case "$HEALTH_PATH" in
  http*) URL="$HEALTH_PATH" ;;
  *)     URL="http://127.0.0.1:${PORT}${HEALTH_PATH}" ;;
esac
SECONDS=0
while ! curl -sf --max-time 3 -o /dev/null "$URL" 2>/dev/null; do
  if [ "$SECONDS" -ge "$TIMEOUT" ]; then
    echo "ERR: ${URL} not healthy after ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep 5
done
echo "OK: ${URL} healthy after ${SECONDS}s"
exit 0
