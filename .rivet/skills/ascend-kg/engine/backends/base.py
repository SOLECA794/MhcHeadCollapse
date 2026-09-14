"""Backend 接口 + OpenAI 兼容默认实现（DESIGN-multi-backend §3.2）。

新后端继承 Backend 只覆写差异项：请求面非 OpenAI 兼容 → 覆写 smoke_request/bench_template；
health 路径不同 → 覆写 health_path；启动脚本形态不同 → 覆写 render_start_script。

红线不变：本接口不执行任何进程操作，只提供「签名」级信息（health 路径、kill 兜底
pgrep 模式）；进程销毁一律由引擎经 scripts/kill_by_port.sh 端口作用域完成。
"""
from pathlib import Path


class Backend:
    name = "base"

    # ---- 路径约定（/tmp 脚本按 flow 前缀隔离，防多 flow 并跑撞名，DESIGN §3.4）----
    def script_path(self, flow, round_key: str) -> Path:
        raise NotImplementedError

    def log_path(self, flow, round_key: str) -> Path:
        return self.script_path(flow, round_key).with_suffix(".log")

    # ---- 接口六方法 ----
    def render_start_script(self, flow, cfg, params: dict, round_key: str) -> Path:
        """按 params 渲染启动脚本，返回脚本路径（模板与参数渲染由后端自治）。"""
        raise NotImplementedError

    def graph_probe(self, log_path, current_params: dict) -> bool | None:
        """图捕获/图模式生效校验。None = 该后端无图概念或不可判（不惩罚）。"""
        return None

    def health_path(self) -> str:
        """就绪探测的 HTTP 路径（同 host:port 后端覆写此项即可）。"""
        return "/health"

    def health_url(self, flow, port: int) -> str:
        """就绪探测完整 URL（wait_port.sh 第 3 参，http 开头按完整 URL 用）。
        健康面挂不同 host/port 的后端（MindIE 挂管理口）覆写此项。"""
        return f"http://127.0.0.1:{port}{self.health_path()}"

    def process_pattern(self, port: int) -> str | None:
        """kill_by_port.sh 兜底 pgrep -f 模式；None = 该后端禁用兜底。"""
        return None

    def smoke_request(self, port: int, model: str) -> tuple:
        """单次可推理探测请求。返回 (url, body_dict)。默认 OpenAI 兼容。"""
        return (f"http://127.0.0.1:{port}/v1/completions",
                {"model": model, "prompt": "Hello.", "max_tokens": 8, "temperature": 0})

    def bench_template(self) -> str:
        """并发压测模板文件名（templates/ 下；契约：输出行 reqN: ttft=Xs total=Ys）。"""
        return "concurrency_test.sh.j2"

    def probe_script(self) -> str:
        """bench 模板以 @PROBE@ 引用的单请求探针绝对路径。默认空串（模板无 @PROBE@ 则渲染 no-op）。"""
        return ""
