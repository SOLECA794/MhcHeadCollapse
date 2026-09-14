#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANNJudge 算子提交辅助工具（本地明文版）

在本地直接使用 邮箱 + 明文密码 完成 CANNJudge 算子竞赛的完整流程：
登录、获取题目、下载工程模板、提交代码、查询结果、查看排行榜。

凭据来源优先级：命令行参数 > 当前 shell 环境变量 > .env 配置文件。

安全约定：
- 明文密码只保存在本地 .env（建议权限 600）或环境变量中；
- 运行过程中绝不把密码打印到终端 / 日志 / 提交到仓库；
- 登录成功仅输出用户 ID 与昵称。
"""

import argparse
import os
import sys
import time
import zipfile
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.stderr.write("缺少依赖 requests，请先执行: pip install requests\n")
    sys.exit(1)

DEFAULT_ENV_FILE = ".env"
BASE_URL = "https://cannjudge.cn"


def load_env_file(path: str) -> dict:
    """解析 .env 文件为字典（不依赖 python-dotenv）。"""
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def write_env_template(path: str, email: str = "", password: str = "") -> str:
    """写入 .env 凭据文件（带安全提示注释）。"""
    content = (
        "# CANNJudge 本地凭据（请勿提交到代码仓库）\n"
        "# 生成方式: python cannjudge_cli.py setup\n"
        f"CANNJUDGE_EMAIL={email}\n"
        f"CANNJUDGE_PASSWORD={password}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows 平台无 chmod 语义
    return path


class CANNJudgeClient:
    """CANNJudge API 客户端（明文登录）。"""

    def __init__(self, email: str = None, password: str = None):
        self.session = requests.Session()
        self.email = email
        self._password = password  # 仅内存持有，禁止输出
        self.user_id = None
        self.user_info = None

    def login(self, email: str = None, password: str = None) -> dict:
        """登录 CANNJudge，成功后保存用户信息与 Cookie。"""
        email = email or self.email
        password = password or self._password
        if not email or not password:
            raise ValueError(
                "缺少登录凭据：请通过 --email/--password、环境变量或 .env 文件提供"
            )
        resp = self.session.post(
            f"{BASE_URL}/api/users/login",
            json={"email": email, "password": password},
            timeout=30,
        )
        resp.raise_for_status()
        self.user_info = resp.json()
        self.user_id = self.user_info["_id"]
        return self.user_info

    def get_problem(self, problem_id: str) -> dict:
        """按题目 ID 获取题目信息。"""
        resp = self.session.get(
            f"{BASE_URL}/api/problems/{problem_id}", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def get_problem_by_name(self, problem_name: str) -> dict:
        """按题目名称获取题目信息（推荐，名称更稳定）。"""
        resp = self.session.get(
            f"{BASE_URL}/api/problems/name/{problem_name}", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def download_package(self, problem_id: str, output_dir: str = None) -> str:
        """下载并解压工程模板，返回解压目录（公开接口，无需登录）。"""
        params = {"userId": self.user_id} if self.user_id else None
        resp = self.session.get(
            f"{BASE_URL}/api/problems/{problem_id}/package",
            params=params,
            timeout=60,
        )
        resp.raise_for_status()

        output_dir = output_dir or tempfile.mkdtemp(prefix="cannjudge_")
        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, "project.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)

        extract_dir = os.path.join(output_dir, "project")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    def submit(self, problem_id: str, tiling_h: str, tiling_key_h: str,
               host_cpp: str, kernel_cpp: str, files: list = None) -> str:
        """提交代码，返回 submissionId。"""
        if not self.user_id:
            raise RuntimeError("请先登录")
        payload = {
            "problemId": problem_id,
            "userId": self.user_id,
            "tiling_h": tiling_h,
            "tiling_key_h": tiling_key_h,
            "host_cpp": host_cpp,
            "kernel_cpp": kernel_cpp,
        }
        if files:
            payload["files"] = files
        resp = self.session.post(
            f"{BASE_URL}/api/submissions/submit", json=payload, timeout=30
        )
        if not resp.ok:
            try:
                detail = resp.json()
                code = detail.get("code")
                message = detail.get("msg") or detail.get("message")
                if code or message:
                    raise RuntimeError(
                        f"提交被平台拒绝 (HTTP {resp.status_code}"
                        f"{f', {code}' if code else ''}): {message or '未知错误'}"
                    )
            except ValueError:
                pass
        resp.raise_for_status()
        return resp.json()["data"]["submissionId"]

    def get_submission(self, submission_id: str) -> dict:
        """查询提交结果。"""
        resp = self.session.get(
            f"{BASE_URL}/api/submissions/{submission_id}", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_result(self, submission_id: str, max_wait: int = 120,
                        interval: int = 3) -> dict:
        """轮询等待评测完成。"""
        terminal = ["Accepted", "Wrong Answer", "Compile Error",
                    "Runtime Error", "Time Limit Exceeded", "Fail"]
        for _ in range(max_wait // interval):
            time.sleep(interval)
            result = self.get_submission(submission_id)
            if result.get("status") in terminal:
                return result
        return self.get_submission(submission_id)

    def get_rankings(self, problem_id: str, limit: int = 20) -> list:
        """获取题目排行榜（与网站 /public/.../{problem}/ranking 一致）。

        优先级：
        1. GET /api/problems/{problemId}/ranking  ← 网站实际使用的接口，含 rank/昵称/分数/用例明细
        2. 题目详情的 last_submission 字段
        3. 旧的 GET /api/submissions/problem/{problemId}/latest
        """
        try:
            resp = self.session.get(
                f"{BASE_URL}/api/problems/{problem_id}/ranking", timeout=30
            )
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("rows") or payload.get("list") or []
            if rows:
                return rows
        except Exception:
            pass
        info = self.get_problem(problem_id)
        rankings = info.get("last_submission") or []
        if rankings:
            return rankings
        resp = self.session.get(
            f"{BASE_URL}/api/submissions/problem/{problem_id}/latest", timeout=30
        )
        resp.raise_for_status()
        return resp.json()


def print_submission_result(result: dict):
    """格式化打印提交结果（不含任何敏感信息）。"""
    status = result.get("status", "Unknown")
    print(f"\n{'=' * 60}")
    print(f"状态: {status}")
    print(f"{'=' * 60}")
    for idx, tc in enumerate(result.get("result", [])):
        tc_status = tc.get("testcase_status", "Unknown")
        precision = tc.get("precision_ratio", "N/A")
        time_ms = tc.get("time", "N/A")
        print(f"  Case {idx + 1}: {tc_status} | precision: {precision} | time: {time_ms}ms")


def read_project_files(project_dir: str) -> dict:
    """从工程目录收集提交所需的四个字段内容。

    兼容两代模板命名：
    - 新模板: op_kernel/{op}_tiling_data.h + {op}_tiling_key.h，
      op_host 拆分为 {op}_def.cpp / {op}_infershape.cpp / {op}_tiling.cpp（按序拼接）
    - 旧模板: op_kernel/{op}_tiling.h + tiling_key_{op}.h，
      op_host/{op}.cpp 单文件
    """
    code_dir = Path(project_dir)
    if (code_dir / "code").exists():
        code_dir = code_dir / "code"

    kernel_dir = code_dir / "op_kernel"
    host_dir = code_dir / "op_host"

    # Tiling 数据结构定义：新模板 *_tiling_data.h，旧模板 *_tiling.h
    tiling_h_files = sorted(kernel_dir.glob("*_tiling_data.h"))
    if not tiling_h_files:
        tiling_h_files = sorted(kernel_dir.glob("*_tiling.h"))
    # Tiling Key 定义：新模板 *_tiling_key.h，旧模板 tiling_key_*.h
    tiling_key_h_files = sorted(kernel_dir.glob("*_tiling_key.h"))
    if not tiling_key_h_files:
        tiling_key_h_files = sorted(kernel_dir.glob("tiling_key_*.h"))
    # Kernel 实现：op_kernel/*.cpp，排除 tiling 相关
    kernel_cpp_files = [
        p for p in sorted(kernel_dir.glob("*.cpp")) if "tiling" not in p.name
    ]
    # 提交 API 没有独立的 Kernel 辅助头文件字段。模板将主体实现放在
    # {op}.h 时，必须在提交前内联该头文件，否则评测端无法解析 include。
    kernel_helper_headers = [
        p for p in sorted(kernel_dir.glob("*.h"))
        if p not in tiling_h_files and p not in tiling_key_h_files
    ]
    # Host 实现：单文件直接取；多文件（def/infershape/tiling）按名拼接
    host_cpp_files = sorted(host_dir.glob("*.cpp"))

    if not tiling_h_files or not host_cpp_files or not kernel_cpp_files:
        raise FileNotFoundError(
            "找不到必要的代码文件，请确认工程目录包含 op_host/*.cpp、"
            "op_kernel/*_tiling_data.h（或 *_tiling.h）、op_kernel/*.cpp"
        )

    editable_files = []
    for directory in (host_dir, kernel_dir):
        for path in sorted(directory.glob("*")):
            if path.is_file() and path.suffix in {".h", ".cpp", ".cc", ".cxx"}:
                editable_files.append({
                    "path": str(path.relative_to(code_dir)).replace("\\", "/"),
                    "content": path.read_text(encoding="utf-8"),
                })

    return {
        "tiling_h": tiling_h_files[0].read_text(encoding="utf-8"),
        "tiling_key_h": tiling_key_h_files[0].read_text(encoding="utf-8")
        if tiling_key_h_files else "",
        "host_cpp": host_cpp_files[0].read_text(encoding="utf-8"),
        "kernel_cpp": kernel_cpp_files[0].read_text(encoding="utf-8"),
        "files": editable_files,
    }


def build_client(args, env: dict) -> CANNJudgeClient:
    """按 命令行 > 环境变量 > .env 的优先级构造客户端。"""
    email = args.email or env.get("CANNJUDGE_EMAIL") or os.environ.get("CANNJUDGE_EMAIL", "")
    password = args.password or env.get("CANNJUDGE_PASSWORD") or os.environ.get("CANNJUDGE_PASSWORD", "")
    return CANNJudgeClient(email=email, password=password)


def main():
    parser = argparse.ArgumentParser(
        description="CANNJudge 算子提交辅助工具（本地明文版）"
    )
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE,
                        help=f".env 凭据文件路径（默认: {DEFAULT_ENV_FILE}）")

    def add_auth(sub):
        sub.add_argument("--email", help="登录邮箱（优先级高于环境变量）")
        sub.add_argument("--password", help="登录密码（优先级高于环境变量，注意 shell 历史）")
        return sub
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    subparsers.add_parser("setup", help="交互式生成 .env 凭据文件")
    add_auth(subparsers.add_parser("login", help="登录并保存会话"))
    add_auth(subparsers.add_parser("problem", help="按 ID 或名称获取题目信息"))
    add_auth(subparsers.add_parser("download", help="下载并解压工程模板"))
    add_auth(subparsers.add_parser("submit", help="提交代码并等待结果"))
    add_auth(subparsers.add_parser("query", help="查询提交结果"))
    add_auth(subparsers.add_parser("rank", help="查看排行榜"))

    # 公共子命令参数
    for name in ("problem", "download", "submit", "rank"):
        subparsers._name_parser_map[name].add_argument(
            "--problem-id", help="题目 ID"
        )
    subparsers._name_parser_map["problem"].add_argument(
        "--problem-name", help="题目名称（与 --problem-id 二选一）"
    )
    subparsers._name_parser_map["download"].add_argument(
        "--output", help="输出目录"
    )
    subparsers._name_parser_map["submit"].add_argument(
        "--project-dir", required=True, help="工程目录"
    )
    subparsers._name_parser_map["submit"].add_argument(
        "--no-wait", action="store_true", help="提交后不等待结果"
    )
    subparsers._name_parser_map["query"].add_argument(
        "--submission-id", required=True, help="提交 ID"
    )

    args = parser.parse_args()

    if args.command == "setup":
        email = input("CANNJudge 登录邮箱: ").strip()
        import getpass
        password = getpass.getpass("CANNJudge 登录密码（输入不显示）: ").strip()
        path = write_env_template(args.env_file, email, password)
        print(f"凭据已保存到 {path}（权限已设为 600，请勿提交到仓库）")
        return

    env = load_env_file(args.env_file)
    client = build_client(args, env)

    try:
        if args.command == "login":
            client.login()
            print(f"登录成功！用户ID: {client.user_id} | 昵称: {client.user_info.get('nickname', 'N/A')}")

        elif args.command == "problem":
            if args.problem_name:
                info = client.get_problem_by_name(args.problem_name)
                print(f"题目名称: {info.get('title') or info.get('name')}")
                print(f"题目 ID: {info.get('_id')}")
                print(f"描述: {info.get('desc', 'N/A')[:500]}")
            else:
                info = client.get_problem(args.problem_id)
                print(f"题目名称: {info.get('title') or info.get('name')}")
                print(f"描述: {info.get('desc', 'N/A')[:500]}")

        elif args.command == "download":
            extract_dir = client.download_package(args.problem_id, args.output)
            print(f"工程模板已解压到: {extract_dir}")

        elif args.command == "submit":
            files = read_project_files(args.project_dir)
            client.login()
            submission_id = client.submit(args.problem_id, **files)
            print(f"提交成功！Submission ID: {submission_id}")
            if not args.no_wait:
                print("等待评测结果...")
                print_submission_result(client.wait_for_result(submission_id))

        elif args.command == "query":
            print_submission_result(client.get_submission(args.submission_id))

        elif args.command == "rank":
            rankings = client.get_rankings(args.problem_id)
            print(f"\n题目 {args.problem_id} 排行榜:")
            print("=" * 60)
            sorted_rankings = sorted(
                rankings,
                key=lambda x: x.get("rank") or 0 if x.get("rank") else (x.get("score", 0) or 0),
                reverse=False,
            )
            for item in sorted_rankings[:20]:
                user = item.get("user") or {}
                nickname = user.get("nickname") or item.get("user_id", "unknown")
                email = user.get("email") or ""
                score = item.get("score", "N/A")
                status = item.get("status", "N/A")
                sid = item.get("submission_id", "N/A")
                rank = item.get("rank", "-")
                print(f"{str(rank):>4}. {nickname:20s} | score: {str(score):>8} | "
                      f"{status:8s} | sub: {sid} | {email}")

        else:
            parser.print_help()
    except requests.exceptions.HTTPError as e:
        sys.stderr.write(f"请求失败: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"错误: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
