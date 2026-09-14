"""确定性执行原语：引擎从 10 阶段流水线瘦身为可独立调用的工具。

工具契约（纯函数，显式接收 flow/cfg/state，返回 dict）：
  capture(flow, cfg, state, round_key) -> {out_dir, validation, signals}
  apply_action(flow, cfg, state, action, round_key) -> {ok, rolled_back, log, current_params}
  verify(flow, cfg, state, round_key) -> {metrics, smoke_ok, accuracy_match}
  rollback_to_last(flow, cfg, state, round_key) -> {ok}
  is_kept(metrics, baseline, flow) -> (kept, rationale)

红线不变：进程管理一律 kill_by_port.sh（端口作用域），严禁宽模式 pkill。
"""
from engine.tools.capture import capture
from engine.tools.apply import apply_action, rollback_to_last
from engine.tools.verify import verify
from engine.tools.decide import is_kept

__all__ = ["capture", "apply_action", "verify", "rollback_to_last", "is_kept"]
