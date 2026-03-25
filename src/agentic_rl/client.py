"""Official OpenEnv client for the Code Review environment.

Uses openenv.core.env_client.EnvClient with the 3 required abstract methods:
  _step_payload(action) -> dict
  _parse_result(payload) -> StepResult
  _parse_state(payload) -> State
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import ReviewAction, ReviewObservation, ReviewState


class CodeReviewEnv(EnvClient[ReviewAction, ReviewObservation, ReviewState]):
    """WebSocket client for the Code Review environment.

    Usage (sync):
        with CodeReviewEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_id="easy_001")
            print(result.observation.code_snippet)
            result = env.step(ReviewAction(
                issues_found=[...],
                overall_assessment="request_changes",
            ))
            print(result.reward, result.observation.feedback)

    Usage (async):
        async with CodeReviewEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id="easy_001")
            result = await env.step(ReviewAction(...))
    """

    def _step_payload(self, action: ReviewAction) -> Dict[str, Any]:
        """Serialize ReviewAction to JSON for the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ReviewObservation]:
        """Parse server response into StepResult[ReviewObservation]."""
        obs_data = payload.get("observation", payload)
        observation = ReviewObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            metadata=obs_data.get("metadata", {}),
            task_id=obs_data.get("task_id", ""),
            task_difficulty=obs_data.get("task_difficulty", ""),
            code_snippet=obs_data.get("code_snippet", ""),
            language=obs_data.get("language", "python"),
            context=obs_data.get("context", ""),
            feedback=obs_data.get("feedback", ""),
        )
        return StepResult(
            observation=observation,
            reward=obs_data.get("reward"),
            done=obs_data.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ReviewState:
        """Parse server state response into ReviewState."""
        return ReviewState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            ground_truth_issues=payload.get("ground_truth_issues", []),
            agent_found_issues=payload.get("agent_found_issues", []),
            max_steps=payload.get("max_steps", 3),
            is_complete=payload.get("is_complete", False),
            final_score=payload.get("final_score", 0.0),
        )
