"""FastAPI application for the Code Review OpenEnv environment.

Run locally:
    uvicorn src.agentic_rl.server.app:app --reload --port 8000

Endpoints:
    POST /reset   - Start new episode (accepts {"task_id": "easy_001"})
    POST /step    - Submit review action
    GET  /state   - Get internal state (for grading)
    GET  /health  - Health check
    GET  /tasks   - List all available tasks
    GET  /docs    - Interactive API docs (Swagger)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .environment import CodeReviewEnvironment
from ..models import ReviewAction, ReviewObservation, ReviewState
from ..tasks import list_all_tasks

app = FastAPI(
    title="Code Review Environment",
    description=(
        "An OpenEnv-compatible RL environment where AI agents review code "
        "for bugs, logic errors, and security vulnerabilities. "
        "Tasks range from easy (obvious bugs) to hard (subtle security issues)."
    ),
    version="0.1.0",
)

env = CodeReviewEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy_001"


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "code_review_env"}


@app.get("/tasks")
async def tasks():
    """List all available tasks with difficulty levels."""
    return {"tasks": list_all_tasks()}


@app.post("/reset", response_model=ReviewObservation)
async def reset(request: ResetRequest = None):
    """Start a new code review episode."""
    task_id = request.task_id if request else "easy_001"
    return env.reset(task_id=task_id)


@app.post("/step", response_model=ReviewObservation)
async def step(action: ReviewAction):
    """Submit a code review and get graded feedback."""
    return env.step(action)


@app.get("/state", response_model=ReviewState)
async def get_state():
    """Get the current internal state (ground truth for grading)."""
    return env.state
