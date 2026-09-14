#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANNJudge 算子提交完整流程示例（本地明文版）

运行前：
1. 复制 .env.example 为 .env 并填写邮箱/密码，或运行:
     python cannjudge_cli.py setup
2. pip install requests
"""

import os
from pathlib import Path

from cannjudge_cli import CANNJudgeClient, load_env_file, read_project_files

BASE_DIR = Path(__file__).resolve().parent


def main():
    # 1. 读取本地凭据（也可用环境变量 CANNJUDGE_EMAIL / CANNJUDGE_PASSWORD）
    env = load_env_file(str(BASE_DIR / ".env"))
    email = env.get("CANNJUDGE_EMAIL") or os.environ.get("CANNJUDGE_EMAIL")
    password = env.get("CANNJUDGE_PASSWORD") or os.environ.get("CANNJUDGE_PASSWORD")
    if not email or not password:
        raise SystemExit("请先配置 .env（python cannjudge_cli.py setup）")

    client = CANNJudgeClient(email=email, password=password)

    # 2. 登录（明文，凭据仅在内存中使用）
    user_info = client.login()
    print(f"登录成功！用户ID: {client.user_id} | 昵称: {user_info.get('nickname', 'N/A')}")

    # 3. 获取题目信息（按名称更稳定；也可按 ID: client.get_problem("题目ID")）
    problem = client.get_problem_by_name("depthtospace")
    problem_id = problem["_id"]
    print(f"题目: {problem.get('title') or problem.get('name')} (ID: {problem_id})")

    # 4. 下载工程模板
    project_dir = client.download_package(problem_id, output_dir="./output")
    print(f"工程模板已解压到: {project_dir}")

    # 5. 在此实现泛化算子（Tiling / Host / Kernel），然后收集文件内容
    files = read_project_files(project_dir)

    # 6. 提交代码
    submission_id = client.submit(problem_id, **files)
    print(f"提交成功！Submission ID: {submission_id}")

    # 7. 轮询等待评测结果
    result = client.wait_for_result(submission_id)
    print(f"最终状态: {result.get('status')}")

    # 8. 查看排行榜
    rankings = client.get_rankings(problem_id)
    print(f"当前排行榜前 5 名:")
    for i, item in enumerate(rankings[:5]):
        print(f"  {i + 1}. {item.get('user_id', 'unknown')} | "
              f"{item.get('status', 'unknown')} | score: {item.get('score', 'N/A')}")


if __name__ == "__main__":
    main()
