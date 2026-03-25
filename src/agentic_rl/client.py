"""OpenEnv Client for interacting with the environment server.

Usage:
    client = AgenticRLClient(base_url="http://localhost:8000")
    obs = client.reset()
    obs = client.step("my answer")
"""

from typing import Optional

import httpx

from .models import AgentAction, EnvObservation, EnvState


class AgenticRLClient:
    """HTTP client for the Agentic RL environment server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def reset(self, task_id: Optional[str] = None) -> EnvObservation:
        """Start a new episode."""
        payload = {"task_id": task_id} if task_id else {}
        resp = self._client.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return EnvObservation(**resp.json())

    def step(self, response: str, metadata: Optional[dict] = None) -> EnvObservation:
        """Send an action and get the next observation."""
        payload = {"response": response}
        if metadata:
            payload["metadata"] = metadata
        resp = self._client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return EnvObservation(**resp.json())

    def get_state(self) -> EnvState:
        """Get the current internal state (for debugging/grading)."""
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return EnvState(**resp.json())

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
