"""checkpoint 暂停点：LLM 步骤回传主会话 agent 的约定文件读写。

引擎在需要 LLM 的 stage（如 diagnose）写 agent_task.json 并置 ctx["pending"]，
主会话 agent 读任务、加载 skill_refs 里的 SKILL.md、执行推理、写回 agent_result.json，
引擎 --resume 续跑。引擎本身保持纯确定性、零 LLM 依赖。

纯 stdlib。文件约定（均位于 engine/state/ 下）：
  agent_task.json   — 引擎写：{task_id, stage, round_key, goal, context, skill_refs,
                        expected_output_schema, status:"pending"}
  agent_result.json — agent 写：{task_id, result, status:"done"}

task_id 由 round_key + stage 组合，保证 resume 时同一暂停点幂等（同 task_id 重写安全）。
"""
import json
from pathlib import Path

from engine.state import STATE_DIR

TASK_FILE = STATE_DIR / "agent_task.json"
RESULT_FILE = STATE_DIR / "agent_result.json"


def _task_id(ctx) -> str:
    return f"{ctx.get('round_key', 'r?')}-{ctx.get('stage', '?')}"


def emit_task(ctx, goal: str, skill_refs: list, schema: dict,
              context: dict | None = None) -> dict:
    """写 agent_task.json 并置 ctx["pending"]=True / ctx["pending_task_id"]。返回 task 记录。"""
    task = {
        "task_id": _task_id(ctx),
        "stage": ctx.get("stage"),
        "round_key": ctx.get("round_key"),
        "goal": goal,
        "context": context or {},
        "skill_refs": skill_refs,
        "expected_output_schema": schema,
        "status": "pending",
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TASK_FILE.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    # 清掉旧 result，避免 stale 结果被误读
    if RESULT_FILE.exists():
        RESULT_FILE.unlink()
    ctx["pending"] = True
    ctx["pending_task_id"] = task["task_id"]
    return task


def consume_result(ctx) -> dict | None:
    """读 agent_result.json；不存在 / 未 done / task_id 不匹配 均返回 None。

    task_id 校验防陈旧结果：上一轮 agent 回传的 result 残留时，若其 task_id
    非当前暂停点（round_key-stage），视为无结果（下一轮重新 emit 并覆盖）。
    """
    if not RESULT_FILE.exists():
        return None
    try:
        data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("status") != "done":
        return None
    if data.get("task_id") != _task_id(ctx):
        return None
    return data.get("result")
