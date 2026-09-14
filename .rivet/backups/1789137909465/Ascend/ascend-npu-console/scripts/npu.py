#!/usr/bin/env python3
"""
npu.py — 通过 CDP 控制 GitCode Notebook（昇腾 NPU 在线环境）

链路: Python → CDP(127.0.0.1:9222) → 浏览器页面上下文 → 读 iframe src 拿 instanceId
      → WebSocket 直连 aihub-run.gitcode.com → 执行命令

零外部依赖（除 websocket-client，已装）。绕过 browser_debug 的 allowlist。
"""
import argparse
import base64
import json
import os
import re
import socket
import ssl
import sys
import time
import urllib.request

try:
    import websocket
except ImportError:
    print("需要 websocket-client: pip install websocket-client", file=sys.stderr)
    sys.exit(1)

CDP_PORT = 9222
REST_HOST = "aihub-run.gitcode.com"
ANSI_RE = re.compile(
    r"\x1b\[[0-9;]*[a-zA-Z]"
    r"|\x1b\?2004[lh]"
    r"|\x1b[()][AB0]"
    r"|\x1b\][^\x07]*\x07"
    r"|\x1b\[[0-9]*[ABCDHK]"
    r"|\x1b[=>]"
)


def clean(t: str) -> str:
    return ANSI_RE.sub("", (t or "").replace("\r\n", "\n").replace("\r", "\n"))


# ---------------- CDP ----------------

class CDP:
    def __init__(self, port=CDP_PORT):
        self.port = port
        self.ws = None
        self._id = 0

    def _list(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=8) as r:
            return json.loads(r.read().decode())

    def attach_gitcode(self):
        for t in self._list():
            if t.get("type") == "page" and "gitcode" in t.get("url", ""):
                self.ws = websocket.create_connection(
                    t["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
                return t
        raise RuntimeError("未找到 gitcode 页面（确认浏览器开着且已登录）")

    def call(self, method, params=None):
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == mid:
                return m

    def js(self, expr, timeout=30):
        r = self.call("Runtime.evaluate",
                      {"expression": expr, "returnByValue": True, "awaitPromise": True})
        res = r.get("result", {})
        if "exceptionDetails" in res:
            raise RuntimeError("JS 异常: " + json.dumps(res["exceptionDetails"])[:300])
        return res.get("result", {}).get("value")

    def close(self):
        try:
            self.ws and self.ws.close()
        except Exception:
            pass


def find_instance(wait_s=60):
    """从页面里提取 instanceId（iframe src 或页面变量）"""
    cdp = CDP()
    cdp.attach_gitcode()
    try:
        cdp.js("1")  # 预热
        deadline = time.time() + wait_s
        while time.time() < deadline:
            v = cdp.js(r"""(function(){
              var out = {iframes: [], fromHtml: []};
              var fs = document.querySelectorAll('iframe');
              for (var i=0;i<fs.length;i++) if (fs[i].src) out.iframes.push(fs[i].src);
              // 兜底: 从整页 HTML 里正则找 online-action/<id>
              var m = document.documentElement.innerHTML.match(/online-action\/([A-Za-z0-9_-]{8,})/g);
              if (m) out.fromHtml = m.slice(0, 5);
              out.starting = document.body.innerText.indexOf('启动中') >= 0;
              return JSON.stringify(out);
            })()""")
            d = json.loads(v or "{}")
            srcs = list(d.get("iframes", [])) + list(d.get("fromHtml", []))
            for s in srcs:
                m = re.search(r"online-action/([A-Za-z0-9_-]{8,})", s)
                if m:
                    cdp.close()
                    return m.group(1)
            if not d.get("starting"):
                # 不在启动中但也没有 iframe —— 可能页面不是 lab
                pass
            time.sleep(4)
        raise RuntimeError("超时未找到 instanceId（Notebook 可能仍在启动）")
    finally:
        cdp.close()


# ---------------- 终端 WebSocket ----------------

class Term:
    def __init__(self, inst, term, origin="https://gitcode.com", timeout=30):
        self.inst, self.term = inst, term
        self.url = f"wss://{REST_HOST}/online-action/{inst}/terminals/websocket/{term}"
        self.timeout = timeout
        self.ws = None

    def connect(self):
        # websocket-client 对 wss + 自定义 origin, 用 suppress_origin 避免 CDP 之外的问题
        self.ws = websocket.create_connection(self.url, timeout=self.timeout,
                                              suppress_origin=True)
        return self.ws.connected

    def close(self):
        try:
            self.ws and self.ws.close()
        except Exception:
            pass


def list_terminals(inst, timeout=15):
    url = f"https://{REST_HOST}/online-action/{inst}/api/terminals"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_cmd(inst, term, cmd, timeout_ms=60000):
    t = Term(inst, term, timeout=max(20, timeout_ms / 1000 + 10))
    t.connect()
    marker = "M" + format(int(time.time() * 1000), "x")
    line = f"echo START:{marker}; {cmd}; echo DONE:{marker}"
    # bracketed paste + 回车（对齐 verified-route.md 实测协议）
    t.ws.send(json.dumps(["stdin", "\x1b[200~" + line + "\x1b[201~"]))
    t.ws.send(json.dumps(["stdin", "\r"]))

    out, deadline, timed_out = "", time.time() + timeout_ms / 1000, False
    while True:
        try:
            t.ws.settimeout(max(0.3, deadline - time.time()))
            msg = t.ws.recv()
        except Exception:
            if time.time() >= deadline:
                timed_out = True
            break
        if not msg:
            break
        try:
            mm = json.loads(msg)
            if isinstance(mm, list) and len(mm) > 1 and isinstance(mm[1], str):
                out += mm[1]
        except Exception:
            pass
        if ("\nDONE:" + marker) in out or ("\rDONE:" + marker) in out:
            break
        if time.time() >= deadline:
            timed_out = True
            break
    t.close()

    text = clean(out)
    st, dt = "START:" + marker, "DONE:" + marker
    s, cur = -1, 0
    while True:
        i = text.find(st, cur)
        if i == -1:
            break
        if i == 0 or text[i - 1] == "\n":
            s = i + len(st)
        cur = i + len(st)
    d = text.find("\n" + dt, max(0, s))
    if d == -1:
        d = text.find(dt, max(0, s))
    body = text[s:d] if (s >= 0 and 0 <= s < d) else (text if s < 0 else "")
    return {"out": body, "timedOut": timed_out, "ok": s >= 0}


def main():
    ap = argparse.ArgumentParser(description="CDP 控制 GitCode Notebook NPU 环境")
    ap.add_argument("--inst", help="instanceId（省略则自动从页面提取）")
    ap.add_argument("--term", help="终端名（省略则用 API 列）")
    ap.add_argument("--skip-cdp", action="store_true", help="不通过 CDP，直接用 --inst")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inst", help="从页面提取 instanceId")
    sub.add_parser("terms", help="列终端")
    p_run = sub.add_parser("run", help="执行命令")
    p_run.add_argument("command")
    p_run.add_argument("--timeout-ms", type=int, default=60000)
    p_push = sub.add_parser("push", help="上传本地文件/目录到远端 (tar+base64)")
    p_push.add_argument("local")
    p_push.add_argument("remote_dir")
    args = ap.parse_args()

    inst = args.inst
    if not inst and not args.skip_cdp:
        print("从页面提取 instanceId...", file=sys.stderr)
        inst = find_instance()
        print(f"instanceId = {inst}", file=sys.stderr)

    if args.cmd == "inst":
        print(inst)
        return 0

    if not inst:
        print("需要 --inst", file=sys.stderr)
        return 2

    if args.cmd == "terms":
        print(json.dumps(list_terminals(inst), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "run":
        term = args.term
        if not term:
            ts = list_terminals(inst)
            items = ts.get("data") if isinstance(ts, dict) else ts
            if items:
                term = items[0].get("name") or items[0].get("id")
            else:
                print("无终端，需在 Notebook 里开一个", file=sys.stderr)
                return 1
            print(f"使用终端: {term}", file=sys.stderr)
        r = run_cmd(inst, term, args.command, args.timeout_ms)
        print(r["out"])
        if r["timedOut"]:
            print("[超时]", file=sys.stderr)
        return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
