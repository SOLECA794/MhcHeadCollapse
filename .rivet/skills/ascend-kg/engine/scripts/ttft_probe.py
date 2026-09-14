#!/usr/bin/env python3
"""MindIE 首 token 时延探针（流式 SSE：首内容 chunk = 真 TTFT）+ 输出 token 计数。

背景（2026-08-19 真机实测）：MindIE 对非流式 /v1/chat/completions 缓冲整响应后才发首字节，
curl time_starttransfer 测到的是「prefill + 全部输出解码」而非首 token。GLM-4-9B 稳态
CONC=5 实测：流式首内容 ~49ms，非流式首字节 ~2870ms——引擎把 TTFT 测成了全响应时间，
解码主导下调度参数（动作空间）改动不可见。流式首内容 chunk 才是调度/首 token 口径。

输出 token 数（out=Ntok，engine/stages/verify.parse_benchmark 契约）：
- 流式逐 chunk 的 content delta 计数（每 delta ≈ 一个解码 token），可靠兜底；
- `stream_options.include_usage` 生效时，末 chunk 的 usage.completion_tokens 为权威值，
  存在则覆盖 delta 计数。out 缺失时引擎仅报 TTFT/TTOT，不算吞吐/TPOT/TPS（向后兼容）。

用法: python3 ttft_probe.py <port> <model> <max_tokens> <timeout> <prompt> <label>
输出: "reqN: ttft=Xs total=Ys out=Ntok"（失败打印 ttft=0.000000s → 被 parse 丢弃为假样本）
"""
import http.client
import json
import sys
import time


def probe(port, model, max_tokens, timeout, prompt, label):
    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=int(timeout))
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": int(max_tokens), "temperature": 0,
                       "stream": True,
                       "stream_options": {"include_usage": True}})
    t0 = time.time()
    first_content = None
    out_tokens = 0
    try:
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        while True:
            line = resp.readline()
            if not line:
                break
            txt = line.decode(errors="ignore").strip()
            if not txt.startswith("data:"):
                continue
            payload = txt[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_content is None:
                        first_content = (time.time() - t0) * 1000
                    out_tokens += 1
                # include_usage 末 chunk 的 completion_tokens 是权威计数，覆盖 delta 计数
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    out_tokens = usage["completion_tokens"]
            except Exception:
                continue
    except Exception:
        conn.close()
        print(f"{label}: ttft=0.000000s total=0.000000s")
        return
    total = (time.time() - t0) * 1000
    conn.close()
    fc = first_content if first_content is not None else total
    print(f"{label}: ttft={fc/1000:.6f}s total={total/1000:.6f}s out={out_tokens}tok")


if __name__ == "__main__":
    probe(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
          sys.argv[5], sys.argv[6])
