#!/usr/bin/env python3
"""
探针: 判定 NPU Notebook 的 WebSocket 能否脱离浏览器直连

背景(ascend-npu-console/docs/verified-route.md):
  - REST 列终端: https.get 无需 cookie (我们已验证: 假 instanceId 返回 502 而非 401)
  - WebSocket 执行命令: 文档说"页面上下文自带会话 cookie, 连接 101 成功"
    -> 但未验证"脱离浏览器是否仍能 101"

本探针只做一件事: 用一个不存在的 instanceId 尝试 WS 握手, 看服务器返回什么。
  若能拿到 101/或明确的"实例不存在" -> 说明鉴权在实例层, 不在 cookie 层
  若返回 401/403 -> 说明确实依赖页面 cookie

⚠ 不做任何真实连接（不碰别人的实例）。
"""
import socket
import ssl
import base64
import os

HOST = "aihub-run.gitcode.com"
FAKE_INST = "probe-nonexistent-instance-0000"
FAKE_TERM = "probe"


def ws_handshake(host, path, extra_headers=None, timeout=10):
    """裸 socket 做 WebSocket 握手, 返回状态行与响应头"""
    key = base64.b64encode(os.urandom(16)).decode()
    headers = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
        f"Origin: https://gitcode.com",
    ]
    if extra_headers:
        headers.extend(extra_headers)
    req = "\r\n".join(headers) + "\r\n\r\n"

    ctx = ssl.create_default_context()
    with socket.create_connection((host, 443), timeout=timeout) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as s:
            s.sendall(req.encode())
            data = b""
            s.settimeout(timeout)
            try:
                while b"\r\n\r\n" not in data and len(data) < 4096:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    data += chunk
            except socket.timeout:
                pass
    return data.decode("utf-8", "replace")


print("=" * 72)
print("探针: WebSocket 握手鉴权层判定")
print("=" * 72)

path = f"/online-action/{FAKE_INST}/terminals/websocket/{FAKE_TERM}"
print(f"目标: wss://{HOST}{path}")
print()

try:
    resp = ws_handshake(HOST, path)
    lines = resp.split("\r\n")
    print("响应首行:", lines[0] if lines else "(空)")
    print()
    for ln in lines[:12]:
        if ln.strip():
            print("   ", ln)
    print()
    # 判定
    status = lines[0] if lines else ""
    print("=" * 72)
    print("判定")
    print("=" * 72)
    if "101" in status:
        print("  -> 握手成功(101)。说明鉴权在实例层, 不在 cookie 层")
        print("     脱浏览器直连【可行】")
    elif "401" in status or "403" in status:
        print("  -> 认证拒绝。说明确实依赖页面 cookie")
        print("     脱浏览器直连【不可行】, 必须走浏览器上下文")
    elif "502" in status or "404" in status:
        print("  -> 实例不存在类错误(502/404)。这是好消息:")
        print("     服务器在校验【实例存在性】之前没有要求 cookie")
        print("     => 若 instanceId 有效, 很可能可直连。需真实 instanceId 复验")
    else:
        print(f"  -> 未预期状态: {status}")
except Exception as e:
    print(f"握手异常: {type(e).__name__}: {e}")

print()
print("=" * 72)
print("REST 端点对照(已验证)")
print("=" * 72)
print("  假 instanceId -> HTTP 502 Bad Gateway (非 401/403)")
print("  => REST 不校验 cookie, 只校验实例存在性 ✓ 已证实")
