"""FastAPI application for the Code Review OpenEnv environment.

Uses the official openenv create_fastapi_app() which auto-generates:
  /ws, /reset, /step, /state, /health, /web, /docs, /schema

Run locally:
    uvicorn src.agentic_rl.server.app:app --reload --port 8000
"""

from openenv.core.env_server import create_fastapi_app

from .environment import CodeReviewEnvironment
from ..models import ReviewAction, ReviewObservation

app = create_fastapi_app(
    env=CodeReviewEnvironment,
    action_cls=ReviewAction,
    observation_cls=ReviewObservation,
)
