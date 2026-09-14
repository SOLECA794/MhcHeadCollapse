"""SGLang-NPU 后端（DESIGN-multi-backend Phase 2）：CLI flag 载体 + launch_server 启动。

形态事实（KG + 上游源码核实，见记忆 sglang-npu-facts）：
- 官方 NPU 线 v0.5.13+（CANN 9.0.0 + torch_npu 2.10 + py3.11，源码装 pyproject_npu.toml，
  PyPI wheel 无 NPU 后端）。
- 启动 `python3 -m sglang.launch_server`（NPU 上 attention-backend 自动 ascend，此处仍显式带上）。
- GET /health ✓ 默认；/v1/completions + /v1/chat/completions OpenAI 兼容 ✓ 默认 smoke/bench 直接用。
- 图捕获日志判据未核实 → graph_probe 用基类默认 None（不判，不误伤）。
"""
from pathlib import Path

from engine.backends.base import Backend
from engine.state import render, params_to_args, params_to_env, TEMPLATE_DIR


class SGLang(Backend):
    name = "sglang"

    def script_path(self, flow, round_key: str) -> Path:
        return Path(f"/tmp/{flow.flow_id}_start_sglang_opt_{round_key}.sh")

    def render_start_script(self, flow, cfg, params: dict, round_key: str) -> Path:
        script = self.script_path(flow, round_key)
        render(TEMPLATE_DIR / "start_sglang_opt.sh.j2", script,
               DEVICE=cfg.device, PORT=cfg.port, MODEL=cfg.model,
               CONDA_SH=flow.paths.conda_sh, ENV_NAME=flow.paths.sglang_env_name,
               SGLANG_ARGS=params_to_args(
                   params, flow.params.domain, cfg.port,
                   # --trust-remote-code：GLM/GLM-4 家族模型带自定义 tokenizer/config 代码，
                   # 缺失该 flag 时 AutoTokenizer 抛 "custom code must be executed"（GLM-4-9B 实踩）。
                   # 对无自定义代码的模型（Qwen 系）为无害 no-op。
                   base_args=f"--host 0.0.0.0 --port {cfg.port} --attention-backend ascend "
                            f"--trust-remote-code"),
               SGLANG_ENV=params_to_env(params, flow.params.domain))
        return script

    def process_pattern(self, port: int) -> str | None:
        # launch_server 主形态（新 CLI "sglang serve" 不由引擎渲染，不匹配）
        return f"sglang.launch_server.* --port {port}"

    def bench_template(self) -> str:
        # 同 MindIE 口径：非流式首字节=整响应（缓冲），curl time_starttransfer 测到的是
        # prefill+全部输出解码，调度参数不可见。流式首内容 chunk 才是真 TTFT。GLM-4-9B
        # 实测非流式 2481ms vs 流式首内容 <100ms。用通用流式模板 + ttft_probe.py。
        return "concurrency_test_stream.sh.j2"

    def probe_script(self) -> str:
        """bench 模板以 @PROBE@ 引用的单请求流式探针绝对路径（OpenAI 兼容 /v1/chat/completions，
        SGLang 直接可用；输出契约 reqN: ttft=Xs total=Ys out=Ntok 与 parse_benchmark 一致）。"""
        return str(Path(__file__).resolve().parent.parent / "scripts" / "ttft_probe.py")
