"""Fish Farm Environment Client.

WebSocket-based client for interacting with the Fish Farm OpenEnv server.
Provides async and sync interfaces for reset/step/state operations.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FarmAction, FarmObservation


class FishFarmEnv(EnvClient[FarmAction, FarmObservation, State]):
    """
    Client for the Fish Farm OpenEnv Environment.

    Maintains a persistent WebSocket connection for efficient multi-step
    interactions with a Nile Tilapia RAS aquaculture simulation.

    Example (async):
        >>> async with FishFarmEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id="feeding_basics")
        ...     while not result.done:
        ...         action = FarmAction(feeding_rate=0.4, aeration_rate=0.6)
        ...         result = await env.step(action)

    Example (sync):
        >>> env = FishFarmEnv(base_url="http://localhost:8000").sync()
        >>> with env:
        ...     result = env.reset(task_id="feeding_basics")
        ...     result = env.step(FarmAction(feeding_rate=0.5))

    Example (from HuggingFace):
        >>> env = await FishFarmEnv.from_env("rahul24raj/fish-farm-env")
    """

    def _step_payload(self, action: FarmAction) -> Dict[str, Any]:
        """Convert FarmAction to JSON payload for step message."""
        return {
            "feeding_rate": action.feeding_rate,
            "aeration_rate": action.aeration_rate,
            "heater_setting": action.heater_setting,
            "water_exchange_rate": action.water_exchange_rate,
            "harvest_decision": action.harvest_decision,
            "treatment": action.treatment,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FarmObservation]:
        """Parse server response into StepResult[FarmObservation]."""
        obs_data = payload.get("observation", {})
        observation = FarmObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
