"""FastAPI application wiring for the OpenEnv environment.

Run locally:
    uvicorn src.agentic_rl.server.app:app --reload --port 8000

This auto-generates endpoints: /reset, /step, /state, /health, /docs
"""

from fastapi import FastAPI
from pydantic import BaseModel

from .environment import AgenticRLEnvironment
from ..models import AgentAction, EnvObservation, EnvState

app = FastAPI(
    title="Agentic RL Environment",
    description="OpenEnv-compatible RL environment for the Meta PyTorch Hackathon",
    version="0.1.0",
)

# Global environment instance
env = AgenticRLEnvironment()


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    response: str
    metadata: dict | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset", response_model=EnvObservation)
async def reset(request: ResetRequest | None = None):
    task_id = request.task_id if request else None
    return env.reset(task_id=task_id)


@app.post("/step", response_model=EnvObservation)
async def step(request: StepRequest):
    action = AgentAction(response=request.response, metadata=request.metadata)
    return env.step(action)


@app.get("/state", response_model=EnvState)
async def get_state():
    return env.state
