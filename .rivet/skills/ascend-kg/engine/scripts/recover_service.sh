#!/bin/bash
# recover_service.sh [state_dir]
# 一键兜底：orchestrator 崩溃后人工救回主服务。
# 读 state/run_state.json 的 current_script + port 重启并健康检查。
# current_script 缺失/失效（如 /tmp 易失脚本被清）时，现场重渲染 defaults 基线。
set -u
STATE_DIR=${1:-$(dirname "$0")/../state}
PORT=8000

SCRIPT=""
if [ -f "$STATE_DIR/run_state.json" ]; then
  SCRIPT=$(python3 -c "import json;s=json.load(open('$STATE_DIR/run_state.json'));print(s.get('current_script') or '')" 2>/dev/null || true)
  PORT=$(python3 -c "import json;s=json.load(open('$STATE_DIR/run_state.json'));print(s.get('port') or 8000)" 2>/dev/null || true)
fi
[ -n "$PORT" ] || PORT=8000

if [ -z "$SCRIPT" ] || [ ! -f "$SCRIPT" ]; then
  # 现场重渲染 defaults 基线（机器重启后 /tmp 易失脚本已丢，仍能救回）
  echo "recover_service: current_script 缺失/失效，现场重渲染 defaults 基线"
  ENGINE_DIR=$(cd "$(dirname "$0")/.." && pwd)
  SKILL_DIR=$(cd "$(dirname "$0")/../.." && pwd)
  SCRIPT=$(python3 - "$ENGINE_DIR" "$SKILL_DIR" "$PORT" <<'PY'
import sys, json
from pathlib import Path
engine = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from engine.state import render, params_to_args, params_to_env, TEMPLATE_DIR
flow = json.load(open(engine / "flows" / "vllm-serve-optimize.json"))
port = int(sys.argv[3])
p = flow["params"]
out = Path("/tmp/start_vllm_opt_baseline.sh")
render(TEMPLATE_DIR / "start_vllm_opt.sh.j2", out,
       DEVICE=0, PORT=port, MODEL=flow["paths"]["model_default"],
       VLLM_ARGS=params_to_args(p["defaults"], p["domain"], port),
       VLLM_ENV=params_to_env(p["defaults"], p["domain"]))
print(out)
PY
)
fi

if [ -z "$SCRIPT" ] || [ ! -f "$SCRIPT" ]; then
  echo "recover_service: 无可用启动脚本且现场重渲染失败，请重跑编排引擎建立基线" >&2
  exit 1
fi
echo "recover_service: using $SCRIPT (port $PORT)"

# kill_by_port 返回值不能忽略：杀不干净就拒绝启动，避免 300s 空等掩盖真实故障
if ! bash "$(dirname "$0")/kill_by_port.sh" "$PORT" 15; then
  echo "recover_service: 端口 $PORT 未释放，拒绝启动" >&2
  exit 1
fi

# setsid 让恢复出的服务成为进程组组长，EngineCore 清理才能走组杀而非进程树遍历
setsid bash "$SCRIPT" > "$STATE_DIR/recover_service.log" 2>&1 &
echo "recover_service: launched $SCRIPT pid=$! (setsid)"

if bash "$(dirname "$0")/wait_port.sh" "$PORT" 300; then
  echo "recover_service: OK, port $PORT healthy"
  exit 0
else
  echo "recover_service: FAILED, port $PORT not healthy（见 $STATE_DIR/recover_service.log）" >&2
  exit 1
fi
