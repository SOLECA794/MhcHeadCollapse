"""vLLM-Ascend 后端：渲染/图捕获校验/兜底签名（平移自 stages/{apply,verify}.py 的硬编码）。"""
from pathlib import Path

from engine.backends.base import Backend
from engine.state import render, params_to_args, params_to_env, TEMPLATE_DIR

# 图捕获校验判据（vllm/v1/worker/gpu_model_runner.py）：
#   成功：logger.info_once("Graph capturing finished in %.0f secs, took %.2f GiB", ...)
#   跳过（cudagraph_mode=NONE）：logger.warning("Skipping CUDA graph capture. To turn on CUDA graph capture, ...")
_GRAPH_OK_LOG = "Graph capturing finished"
_GRAPH_SKIP_LOG = "Skipping CUDA graph capture"


class VllmAscend(Backend):
    name = "vllm-ascend"

    def script_path(self, flow, round_key: str) -> Path:
        return Path(f"/tmp/{flow.flow_id}_start_vllm_opt_{round_key}.sh")

    def render_start_script(self, flow, cfg, params: dict, round_key: str) -> Path:
        script = self.script_path(flow, round_key)
        render(TEMPLATE_DIR / "start_vllm_opt.sh.j2", script,
               DEVICE=cfg.device, PORT=cfg.port, MODEL=cfg.model,
               VLLM_ARGS=params_to_args(params, flow.params.domain, cfg.port),
               VLLM_ENV=params_to_env(params, flow.params.domain))
        return script

    def graph_probe(self, log_path, current_params: dict) -> bool | None:
        """确认图捕获生效。None = 不适用/不可判（enforce-eager、显式 NONE、日志缺失）。

        cudagraph-mode 显式 NONE 是有意关图（A/B 对照）：vllm 会打 "Skipping" 警告，
        实为预期行为而非配置未落地，不判 False。
        """
        if current_params.get("enforce-eager"):
            return None
        if str(current_params.get("cudagraph-mode", "")).upper() == "NONE":
            return None
        p = Path(log_path)
        if not p.exists():
            return None
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if _GRAPH_OK_LOG in text:
            return True
        if _GRAPH_SKIP_LOG in text:
            return False
        return None

    def process_pattern(self, port: int) -> str | None:
        # 与 kill_by_port.sh 原兜底一致：严格限定 vllm serve + 端口，绝不宽模式
        return f"vllm serve.* --port {port}"
