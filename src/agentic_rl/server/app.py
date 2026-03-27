"""FastAPI application for the Code Review OpenEnv environment.

Uses the official openenv create_fastapi_app() which auto-generates:
  /ws, /reset, /step, /state, /health, /web, /docs, /schema

Custom endpoints added for hackathon compliance:
  /tasks   - List all tasks with action schema
  /grader  - Grade agent output against ground truth
  /baseline - Run heuristic baseline on a task

Run locally:
    uvicorn src.agentic_rl.server.app:app --reload --port 8000
"""

import sys
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel

from .environment import CodeReviewEnvironment
from ..models import ReviewAction, ReviewObservation
from ..tasks import get_task, list_all_tasks, TASKS

# Add project root to path so graders module is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from graders.example_graders import KeywordMatchGrader

app = create_fastapi_app(
    env=CodeReviewEnvironment,
    action_cls=ReviewAction,
    observation_cls=ReviewObservation,
)


# ---------------------------------------------------------------------------
# Custom endpoints required by hackathon pre-submission checklist
# ---------------------------------------------------------------------------


@app.get("/tasks")
def endpoint_list_tasks():
    """Return list of all tasks and the action schema (fields required for a step)."""
    return {
        "tasks": list_all_tasks(),
        "action_schema": ReviewAction.model_json_schema(),
    }


class GraderRequest(BaseModel):
    task_id: str
    issues_found: List[Dict[str, Any]] = []


@app.post("/grader")
def endpoint_grade(req: GraderRequest):
    """Grade agent-found issues against ground truth for a task."""
    try:
        task = get_task(req.task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {req.task_id}")

    grader = KeywordMatchGrader()
    result = grader.grade(
        task_id=req.task_id,
        ground_truth=task["ground_truth"],
        agent_issues=req.issues_found,
    )
    return asdict(result)


class BaselineRequest(BaseModel):
    task_id: Optional[str] = None


@app.post("/baseline")
def endpoint_baseline(req: BaselineRequest):
    """Run a simple heuristic baseline agent against task(s) and return scores.

    If task_id is provided, runs on that single task.
    If task_id is None, runs on all tasks (required by judges).
    """
    env = CodeReviewEnvironment()
    grader = KeywordMatchGrader()

    task_ids = [req.task_id] if req.task_id else list(TASKS.keys())
    results = []

    for tid in task_ids:
        try:
            task = get_task(tid)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Unknown task_id: {tid}")

        # Reset environment for this task
        obs = env.reset(task_id=tid)

        # Heuristic baseline: submit a generic review that mentions common issues
        heuristic_action = ReviewAction(
            issues_found=[
                {
                    "line": "1",
                    "severity": "major",
                    "category": "bug",
                    "description": "Potential error handling issue detected in the code.",
                    "suggestion": "Add proper validation and error handling.",
                }
            ],
            overall_assessment="request_changes",
            confidence=0.5,
        )

        # Step environment
        step_obs = env.step(heuristic_action)

        # Also grade with the standalone grader
        grade_result = grader.grade(
            task_id=tid,
            ground_truth=task["ground_truth"],
            agent_issues=heuristic_action.issues_found,
        )

        results.append({
            "task_id": tid,
            "difficulty": task["difficulty"],
            "env_reward": step_obs.reward,
            "grader_score": grade_result.score,
            "grader_passed": grade_result.passed,
            "grader_feedback": grade_result.feedback,
        })

    return {
        "results": results,
        "total_tasks": len(results),
        "avg_grader_score": sum(r["grader_score"] for r in results) / len(results) if results else 0.0,
    }
