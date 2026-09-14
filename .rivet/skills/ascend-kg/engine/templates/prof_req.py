#!/usr/bin/env python3
"""并发推理请求（profiling 触发）：5 并发，混合输入长度，覆盖 prefill+decode。
由 orchestrator 渲染 @BASE@/@MODEL@/@INPUT_BASE@/@INPUT_STEP@/@OUTPUT_MAX@。"""
import json
import threading
import urllib.request

BASE = "@BASE@"
MODEL = "@MODEL@"
SENT = "The quick brown fox jumps over the lazy dog. "  # 10 tokens

def send(i):
    plen = @INPUT_BASE@ + i * @INPUT_STEP@
    prompt = SENT * (plen // 10)
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": @OUTPUT_MAX@,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
            n_out = len(resp["choices"][0]["text"].split())
            print(f"[req{i}] plen={plen} out_tokens={n_out} ok")
    except Exception as e:
        print(f"[req{i}] plen={plen} FAIL: {e}")

threads = [threading.Thread(target=send, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("ALL DONE")
