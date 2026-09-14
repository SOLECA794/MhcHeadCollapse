#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MhcHeadCollapse 专用 SSH 远程执行器（跳板机直连 NPU 服务器）

用法:
  python mhc_ssh.py run "<命令>"          # 远程执行
  python mhc_ssh.py push <本地路径> <远端路径>   # 上传文件/目录(sftp)
  python mhc_ssh.py pull <远端路径> <本地路径>   # 下载

链路: 本地 -> 跳板(113.47.8.48:2234, user jt_xxx) -> NPU(199.98.55.200, root)
替代原 notebook 链路(CDP->WS->base64), 标准 ssh/sftp, /workspace 高IO 300GB。
"""
import os, sys, stat
import paramiko

JUMP_HOST, JUMP_PORT = "113.47.8.48", 2234
JUMP_USER = "jt_847D314397BDC0DD5D2D6742:B3F9034877D09406F640100B9009231F0A08BF24FA8065A805B1B4F04A3CD9F39940D2ABA56C49B789C34CBE6E3181B92F"
JUMP_PASS = "Sj9eim1bS0uf00BN"
NPU_HOST, NPU_USER, NPU_PASS = "199.98.55.200", "root", "Sj9eim1bS0uf00BN"

def connect():
    # 跳板通道
    jsock = None
    jump = paramiko.SSHClient()
    jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        jump.connect(JUMP_HOST, port=JUMP_PORT, username=JUMP_USER, password=JUMP_PASS, timeout=30)
        jchan = jump.get_transport().open_channel("direct-tcpip", (NPU_HOST, 22), ("127.0.0.1", 0))
    except Exception as e:
        # 跳板可能不需要密码或用户名格式不同 — 报详细信息
        print(f"JUMP FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    # NPU 主机经由跳板通道
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(NPU_HOST, username=NPU_USER, password=NPU_PASS, sock=jchan, timeout=30)
    return jump, cli

def run(cli, cmd, timeout=600):
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    rc = out.channel.recv_exit_status()
    return rc, o, e

def push(cli, local, remote):
    sftp = cli.open_sftp()
    if os.path.isdir(local):
        # 递归上传目录
        def ensure_dir(r):
            try: sftp.stat(r)
            except FileNotFoundError:
                ensure_dir(os.path.dirname(r.rstrip('/')))
                sftp.mkdir(r)
        ensure_dir(remote)
        count = 0
        for root, dirs, files in os.walk(local):
            rel = os.path.relpath(root, local).replace("\\", "/")
            rdir = remote if rel == "." else f"{remote}/{rel}"
            ensure_dir(rdir)
            for f in files:
                sftp.put(os.path.join(root, f), f"{rdir}/{f}")
                count += 1
        return f"pushed {count} files -> {remote}"
    else:
        rdir = os.path.dirname(remote)
        if rdir:
            try: sftp.stat(rdir)
            except FileNotFoundError: sftp.mkdir(rdir)
        sftp.put(local, remote)
        return f"pushed {local} -> {remote}"

def pull(cli, remote, local):
    sftp = cli.open_sftp()
    sftp.get(remote, local)
    return f"pulled {remote} -> {local}"

if __name__ == "__main__":
    jump, cli = connect()
    try:
        cmd = sys.argv[1]
        if cmd == "run":
            rc, o, e = run(cli, sys.argv[2])
            print(o)
            if e.strip(): print("[STDERR]", e[-500:], file=sys.stderr)
            sys.exit(rc)
        elif cmd == "push":
            print(push(cli, sys.argv[2], sys.argv[3]))
        elif cmd == "pull":
            print(pull(cli, sys.argv[2], sys.argv[3]))
        else:
            print(__doc__)
    finally:
        cli.close(); jump.close()
