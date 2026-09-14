"""MindIE 3.0.0 后端（DESIGN-multi-backend Phase 3）：config.json 载体 + 容器内 daemon 重启。

形态事实（3.0.0-800I-A2 真机实测，见记忆 mindie-deploy-facts）：
- 无 CLI flag，全部配置在 config.json；顶层 ServerConfig/BackendConfig，调度参数嵌在
  BackendConfig.ScheduleConfig 下；改配置必须重启 daemon（无热更）。
- docker --net=host 官方形态；容器须 --shm-size>=1g（shared_memory.cpp 需 ~656MB，默认 64M 起不来）；
  docker cp 进的 config 须 chown root:root + chmod 640（daemon 安全校验属主与 other 权限）。
- 启动 = 容器内 4+1 set_env（ascend-toolkit / nnal/atb / **/usr/local/Ascend/atb-models**（不在
  mindie/latest 下）/ mindie-llm / mindie-service）+ nohup ./bin/mindieservice_daemon（在
  mindie-service/ 下启动，不在 bin/ 内），回显 "Daemon start success!"。
- API：业务口（ipAddress 默认 127.0.0.1）POST /generate（native）+ /v1/chat/completions
  （OpenAI 兼容，openAiSupport 默认 "vllm"）。
- 健康：/v2/health/{live,ready} 挂**管理口**（managementIpAddress 127.0.0.2:1026），业务口 404。

红线适配：
- process_pattern 返回 None（禁用 kill 兜底）——daemon cmdline 不含端口，按进程名兜底会
  误杀同机其他用户的 MindIE 容器；且 host 侧 lsof 解析不到容器内进程 pid（pid ns 隔离），
  kill_by_port 主路径无害 no-op。停止动作由重启脚本在容器内完成（docker exec 只作用于本
  flow 的容器，作用域比端口更窄）。
"""
import json
from pathlib import Path

from engine.backends.base import Backend
from engine.state import render, params_to_json_config, TEMPLATE_DIR


class MindIE(Backend):
    name = "mindie"

    def script_path(self, flow, round_key: str) -> Path:
        return Path(f"/tmp/{flow.flow_id}_mindie_restart_{round_key}.sh")

    def config_path(self, flow, round_key: str) -> Path:
        """每轮版本化的 config.json（宿主侧生成，docker cp 进容器）。"""
        return Path(f"/tmp/{flow.flow_id}_mindie_cfg_{round_key}.json")

    def base_config(self, flow) -> dict:
        p = Path(flow.paths.mindie_config_base)
        if not p.exists():
            raise FileNotFoundError(
                f"MindIE 基线 config 不存在: {p}。首次使用请从镜像实读生成（不发明字段）：\n"
                f"  docker run --rm --entrypoint cat "
                f"swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0-800I-A2-py311-openeuler24.03-lts "
                f"/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json > {p}\n"
                f"并按本机改：ModelConfig[0] 的 modelWeightPath（容器内路径）/worldSize=1、"
                f"npuDeviceIds（独立卡）、ServerConfig.port=引擎端口、httpsEnabled=false。\n"
                f"容器一次性前置（实测要求）：docker run -d --net=host --runtime=ascend --privileged "
                f"--shm-size=1g --device /dev/davinci<N> --device /dev/davinci_manager "
                f"--device /dev/hisi_hdc --device /dev/devmm_svm "
                f"-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro "
                f"-v /usr/local/dcmi:/usr/local/dcmi:ro "
                f"-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro "
                f"-v <本机模型目录>:/models:ro <mindie镜像> bash -c 'sleep infinity'；"
                f"config 每轮 docker cp 后须 chown root:root + chmod 640（daemon 安全校验）。")
        return json.loads(p.read_text(encoding="utf-8"))

    def render_start_script(self, flow, cfg, params: dict, round_key: str) -> Path:
        cfg_file = self.config_path(flow, round_key)
        cfg_file.write_text(json.dumps(params_to_json_config(self.base_config(flow), params,
                                                            flow.params.domain),
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        script = self.script_path(flow, round_key)
        render(TEMPLATE_DIR / "start_mindie_opt.sh.j2", script,
               CONTAINER=flow.paths.mindie_container, PORT=cfg.port,
               CFG=cfg_file, ROUND=round_key)
        return script

    def health_url(self, flow, port: int) -> str:
        # 实测：/v2/health/{live,ready} 挂管理口（127.0.0.2:managementPort），业务口 404
        return f"http://127.0.0.2:{flow.paths.mindie_management_port}/v2/health/ready"

    def process_pattern(self, port: int) -> str | None:
        return None  # 禁用兜底：daemon cmdline 无端口，按名兜底会误杀他人容器（见模块注释）

    def smoke_request(self, port: int, model: str) -> tuple:
        # openAiSupport 默认 "vllm" → /v1/chat/completions 可用；/v1/completions 未核实故不用
        return (f"http://127.0.0.1:{port}/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "Hello."}],
                 "max_tokens": 8, "temperature": 0})

    def bench_template(self) -> str:
        # 流式首内容 chunk 才是真 TTFT：非流式首字节=整响应（缓冲），2870ms 里 ~99% 是解码。
        # 用流式模板修正指标口径，调度参数改动才在指标上可见（见 scripts/ttft_probe.py）。
        return "concurrency_test_mindie.sh.j2"

    def probe_script(self) -> str:
        """bench 模板以 @PROBE@ 引用的单请求流式探针绝对路径。"""
        return str(Path(__file__).resolve().parent.parent / "scripts" / "ttft_probe.py")

    def warm_until_stable(self, cfg) -> bool:
        """MindIE: 管理口 health-ready 早于模型加载完成（实测业务口拒绝连接 ~90s，之后 NPU
        仍在沉降 ~5 分钟，TTFT 从 ~88ms 缓慢降到稳态 ~44ms）。health=200 直接压测会打在加载
        /沉降窗口（连接失败或 TTFT 假高，轮间不可比）。用流式单请求循环直到首内容 token 落
        真稳态（<60ms 且两轮漂移 <300ms），上限 8 分钟（覆盖加载窗+沉降）。失败返回 False
        不阻断（压测自身兜底，decide 按 10x 基线判 unhealthy）。"""
        import re
        import subprocess
        import time

        probe = self.probe_script()
        prompt = "请写一段200字的关于机器学习基础概念介绍。"
        samples = []
        deadline = time.time() + 480
        while time.time() < deadline:
            try:
                r = subprocess.run(
                    ["python3", probe, str(cfg.port), cfg.model, "8", "30",
                     prompt, "warm"],
                    capture_output=True, text=True, timeout=35)
            except subprocess.TimeoutExpired:
                samples = []
                time.sleep(3)
                continue
            m = re.search(r"ttft=([\d.]+)s", r.stdout)
            if not m or float(m.group(1)) <= 0:
                # 失败 sentinel（连接拒绝/超时）→ 未就绪，不参与稳定性判定
                samples = []
                time.sleep(2)
                continue
            t = float(m.group(1)) * 1000
            samples.append(t)
            if len(samples) > 5:
                samples = samples[-5:]
            # 真稳态 = 连续 5 采样全部 <55ms 且极差 <25ms。加载/沉降窗（88→55ms 滑落）会被
            # <55ms 挡住，残余 CPU 争用（稳态~50ms 不再下滑）算稳定——同轮内可比即可。
            if len(samples) >= 5 and max(samples) < 55 and max(samples) - min(samples) < 25:
                return True
            time.sleep(2)
        return False
