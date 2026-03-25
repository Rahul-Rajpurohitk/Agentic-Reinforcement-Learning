"""OpenEnv client for the Code Review environment.

Usage:
    from src.agentic_rl.client import CodeReviewClient

    client = CodeReviewClient(base_url="http://localhost:8000")
    obs = client.reset(task_id="easy_001")
    print(obs.code_snippet)

    obs = client.step(issues_found=[...], overall_assessment="request_changes")
    print(obs.reward, obs.feedback)
"""

from typing import Dict, List, Optional

import httpx

from .models import ReviewAction, ReviewObservation, ReviewState


class CodeReviewClient:
    """HTTP client for the Code Review environment server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str = "easy_001") -> ReviewObservation:
        """Start a new code review episode."""
        resp = self._client.post(
            f"{self.base_url}/reset", json={"task_id": task_id}
        )
        resp.raise_for_status()
        return ReviewObservation(**resp.json())

    def step(
        self,
        issues_found: List[Dict[str, str]],
        overall_assessment: str = "request_changes",
        confidence: float = 1.0,
    ) -> ReviewObservation:
        """Submit a code review."""
        action = ReviewAction(
            issues_found=issues_found,
            overall_assessment=overall_assessment,
            confidence=confidence,
        )
        resp = self._client.post(
            f"{self.base_url}/step", json=action.model_dump()
        )
        resp.raise_for_status()
        return ReviewObservation(**resp.json())

    def get_state(self) -> ReviewState:
        """Get internal state (for grading/debugging)."""
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return ReviewState(**resp.json())

    def list_tasks(self) -> list:
        """List all available tasks."""
        resp = self._client.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()["tasks"]

    def health(self) -> bool:
        """Check if the server is running."""
        try:
            resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
