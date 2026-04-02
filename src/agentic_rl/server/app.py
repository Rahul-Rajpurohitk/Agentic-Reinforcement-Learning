"""FastAPI application for the Fish Farm OpenEnv environment.

Uses the official openenv create_fastapi_app() which auto-generates:
  /ws, /reset, /step, /state, /health, /web, /docs, /schema

Custom endpoints added for hackathon compliance:
  /tasks   - List all tasks with action schema
  /grader  - Grade a completed episode
  /baseline - Run heuristic baseline on task(s)

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

from .environment import FishFarmEnvironment
from ..models import FarmAction, FarmObservation
from ..tasks import get_task, list_all_tasks, TASKS

# Add project root to path so graders module is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from graders.farm_graders import FarmGrader  # noqa: E402

app = create_fastapi_app(
    env=FishFarmEnvironment,
    action_cls=FarmAction,
    observation_cls=FarmObservation,
)


# ---------------------------------------------------------------------------
# Custom endpoints required by hackathon
# ---------------------------------------------------------------------------


@app.get("/tasks")
def endpoint_list_tasks():
    """Return list of all 12 tasks and the action schema."""
    return {
        "tasks": list_all_tasks(),
        "action_schema": FarmAction.model_json_schema(),
    }


class GraderRequest(BaseModel):
    task_id: str
    final_state: Dict[str, Any] = {}
    episode_history: List[Dict[str, Any]] = []


@app.post("/grader")
def endpoint_grade(req: GraderRequest):
    """Grade a completed episode using task-specific grader."""
    try:
        task = get_task(req.task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {req.task_id}")

    grader = FarmGrader()
    result = grader.grade(
        task_id=req.task_id,
        final_state=req.final_state,
        episode_history=req.episode_history,
        task_config=task,
    )
    return asdict(result)


class BaselineRequest(BaseModel):
    task_id: Optional[str] = None


@app.post("/baseline")
def endpoint_baseline(req: BaselineRequest):
    """Run a constant-action baseline agent on task(s) and return scores.

    This heuristic uses moderate feeding (0.4), moderate aeration (0.6),
    no heating, light water exchange — a reasonable but unoptimized strategy.
    """
    from ..engine.simulator import FishFarmSimulator

    grader = FarmGrader()
    task_ids = [req.task_id] if req.task_id else list(TASKS.keys())
    results = []

    for tid in task_ids:
        try:
            task = get_task(tid)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Unknown task_id: {tid}")

        # Run simulation with constant baseline action
        sim = FishFarmSimulator(seed=42)
        ic = task["initial_conditions"]
        sim.reset(
            initial_weight=ic["weight_g"],
            initial_population=ic["population"],
            initial_temp=ic["temp"],
            initial_DO=ic["DO"],
            initial_TAN=ic["TAN"],
            initial_pH=ic["pH"],
            day_of_year=ic["day_of_year"],
            base_air_temp=ic.get("base_air_temp", 30.0),
            seed=42,
            scheduled_events=task["events"][:] if task["events"] else None,
        )

        history = []
        max_hours = min(task["episode_hours"], 720)  # cap at 30 days for baseline speed

        for _ in range(max_hours):
            state = sim.step(
                feeding_rate=0.4,
                aeration_rate=0.6,
                heater_setting=0.0,
                water_exchange_rate=0.02,
                harvest=False,
                treatment="none",
            )
            history.append(state)
            if state["done"]:
                break

        grade_result = grader.grade(tid, state, history, task)

        results.append({
            "task_id": tid,
            "difficulty": task["difficulty"],
            "grader_score": grade_result.score,
            "grader_passed": grade_result.passed,
            "grader_feedback": grade_result.feedback,
            "hours_simulated": len(history),
        })

    return {
        "results": results,
        "total_tasks": len(results),
        "avg_grader_score": sum(r["grader_score"] for r in results) / len(results) if results else 0.0,
    }
