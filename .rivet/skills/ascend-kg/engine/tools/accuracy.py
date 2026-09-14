"""精度护栏（H1）：参考文本困惑度漂移（perplexity drift）对拍。

为什么弃用 greedy 精确匹配：图模式/融合 pass 改变 reduce 顺序 → ~1e-7 浮点差异 → argmax
在近 tie 位置翻转一个 token → 文本逐字符相等直接判 False，把「数值漂移」误判成「语义破坏」
（vLLM Llama-3.1-8B 案例：cudagraph_mode=FULL_DECODE_ONLY 精度 0.88<0.9 被强制回滚，
实测为安全且吞吐 +12%）。

困惑度漂移是连续度量：baseline 用 greedy 生成一批参考文本，后续每轮复用同一批文本，用
/v1/completions 的 echo+logprobs+max_tokens=0 取模型对该文本的逐 token 对数似然 →
ppl=exp(-mean logprob)。reduce 顺序扰动只让 ppl 变 ~1e-7；真正的语义破坏（权重/精度错乱）
才会让模型对正常文本的似然坍塌 → ppl 暴涨。按相对漂移分级：

  drift ≤ ppl_drift_soft                    → accuracy_match=True（无变化）
  ppl_drift_soft < drift ≤ ppl_drift_hard   → accuracy_match=False, cls="A"（数值漂移，放行+惩罚）
  drift > ppl_drift_hard                    → accuracy_match=False, cls="B"（语义破坏，硬否决）

None 语义（不误伤）：flow 未配置 / 无基线 / 采集失败 / 可评分样本不足半数 → None，
调用方（agent_loop.try_action）只对 is False 强制回滚。

配置（flow perf 段）："accuracy": {"prompts": 8, "max_tokens": 64,
  "ppl_drift_soft": 0.05, "ppl_drift_hard": 0.30}
"""
import json
import math
import urllib.request

# 固定 prompt 集（确定性，短/长、中/英混合）；flow accuracy.prompts 取前 N 条
_PROMPTS = [
    "请用一句话介绍机器学习。",
    "1+1 等于几？请直接回答。",
    "写一个 Python 函数计算斐波那契数列第 n 项。",
    "列出三个常见的排序算法并说明各自时间复杂度。",
    "把下面这句话翻译成英文：实践是检验真理的唯一标准。",
    "Explain what a transformer model is in two sentences.",
    "Summarize the key idea of gradient descent in one short paragraph.",
    "请写一段80字左右的文字，介绍NPU在大模型推理中的作用，语言正式、逻辑清晰。",
]


def _completions(port: int, model: str, prompt: str, max_tokens: int,
                 timeout: int = 60) -> str:
    """单条 greedy completion（temperature=0，参考文本生成用）。"""
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
        return (data.get("choices") or [{}])[0].get("text", "")


def _score_perplexity(port: int, model: str, text: str, timeout: int = 60) -> float | None:
    """单条参考文本的困惑度。

    echo+logprobs+max_tokens=0：vllm 该路径返回逐 prompt token 的 token_logprobs
    （vllm/entrypoints/openai/completion/serving.py: max_tokens==0 → token_ids=prompt_token_ids,
    out_logprobs=prompt_logprobs），无生成。ppl = exp(-mean logprob)。
    temperature=1.0 取原始 logits softmax（标准困惑度口径）。失败/空文本/无有效 logprob → None。
    """
    if not text or not text.strip():
        return None
    body = json.dumps({"model": model, "prompt": text, "max_tokens": 0,
                       "temperature": 1.0, "echo": True, "logprobs": 1}).encode()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        lp = ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("token_logprobs")
        vals = [float(v) for v in (lp or []) if v is not None]
        if not vals:
            return None
        return math.exp(-sum(vals) / len(vals))
    except Exception:
        return None


def collect_outputs(flow, cfg) -> list | None:
    """按 flow accuracy 配置生成固定 prompt 集的 greedy 参考文本（baseline 与每轮共用）。

    未配置 / 任一请求失败 → None（调用方按「不判」处理，不误伤）。
    """
    acc = getattr(flow.perf, "accuracy", None)
    if not acc:
        return None
    n = min(int(acc.get("prompts", 8)), len(_PROMPTS))
    max_tokens = int(acc.get("max_tokens", 64))
    outs = []
    try:
        for p in _PROMPTS[:n]:
            outs.append(_completions(cfg.port, cfg.model, p, max_tokens))
    except Exception as e:
        print(f"[accuracy] 参考文本生成失败: {e}", flush=True)
        return None
    return outs


def score_outputs(flow, cfg, texts: list) -> list:
    """对参考文本逐条评分困惑度，返回与 texts 等长列表（失败位 None）。"""
    return [_score_perplexity(cfg.port, cfg.model, t) for t in (texts or [])]


def check_accuracy(flow, cfg, state) -> tuple:
    """返回 (accuracy_match, detail)。None = 不判（未配置/无基线/采集失败/样本不足）。

    按参考文本的困惑度相对漂移分级（见模块 docstring）：A 类数值漂移放行+惩罚，B 类硬否决。
    """
    acc = getattr(flow.perf, "accuracy", None)
    if not acc:
        return None, {"reason": "flow 未配置 perf.accuracy"}
    baseline = (state.get("baseline") or {})
    refs = baseline.get("outputs")
    base_ppl = baseline.get("outputs_ppl")
    if not refs or not base_ppl:
        return None, {"reason": "无基线参考文本/困惑度（旧状态未存 outputs_ppl）"}
    cur_ppl = score_outputs(flow, cfg, refs)
    soft = float(acc.get("ppl_drift_soft", 0.05))
    hard = float(acc.get("ppl_drift_hard", 0.30))
    per = []
    for b, c in zip(base_ppl, cur_ppl):
        per.append(abs(c - b) / b if (b and c) else None)
    valid = [d for d in per if d is not None]
    if len(valid) < max(1, len(refs) // 2):
        return None, {"reason": "可评分样本不足半数，判不判", "per_prompt": per}
    drift = max(valid)
    per_r = [round(d, 4) if d is not None else None for d in per]
    if drift <= soft:
        return True, {"drift": round(drift, 4), "soft": soft, "hard": hard,
                      "cls": None, "per_prompt": per_r}
    cls = "A" if drift <= hard else "B"
    print(f"[accuracy] 困惑度漂移 {drift:.3f} 超 {soft} 阈值, cls={cls}", flush=True)
    return False, {"drift": round(drift, 4), "soft": soft, "hard": hard,
                   "cls": cls, "per_prompt": per_r}
