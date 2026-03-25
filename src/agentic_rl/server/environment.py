"""OpenEnv Environment - Core game/task logic.

Implements reset() and step() following the OpenEnv 3-component pattern.
Customize the logic here for your specific problem statement.

This template provides a generic task-solving environment where an agent
receives a prompt and must produce a correct response within a turn limit.
"""

import random
from typing import Any, Dict, List, Tuple

from ..models import AgentAction, EnvObservation, EnvState


# ---------------------------------------------------------------------------
# Sample tasks — replace with your problem statement's tasks
# ---------------------------------------------------------------------------
SAMPLE_TASKS: List[Dict[str, str]] = [
    {
        "id": "task_001",
        "prompt": "What is the capital of France?",
        "target": "paris",
    },
    {
        "id": "task_002",
        "prompt": "Solve: 2 + 2 = ?",
        "target": "4",
    },
    {
        "id": "task_003",
        "prompt": "Complete the sequence: 1, 1, 2, 3, 5, ?",
        "target": "8",
    },
]


class AgenticRLEnvironment:
    """OpenEnv-compatible RL environment.

    Follows the 3-component pattern:
    - reset()  -> EnvObservation  (start a new episode)
    - step()   -> EnvObservation  (process an action)
    - state    -> EnvState        (internal state for grading)

    Set SUPPORTS_CONCURRENT_SESSIONS = True if your environment is stateless
    enough to handle multiple simultaneous clients.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, tasks: List[Dict[str, str]] | None = None):
        self._tasks = tasks or SAMPLE_TASKS
        self._state = EnvState()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, task_id: str | None = None) -> EnvObservation:
        """Start a new episode. Optionally specify a task_id."""
        if task_id:
            task = next((t for t in self._tasks if t["id"] == task_id), None)
        else:
            task = random.choice(self._tasks)

        if task is None:
            task = random.choice(self._tasks)

        self._state = EnvState(
            task_id=task["id"],
            target=task["target"],
            history=[],
            current_turn=0,
            max_turns=10,
            is_complete=False,
            final_score=0.0,
        )

        return EnvObservation(
            prompt=task["prompt"],
            feedback="New episode started. Provide your answer.",
            score=0.0,
            done=False,
            turn=0,
            max_turns=self._state.max_turns,
        )

    def step(self, action: AgentAction) -> EnvObservation:
        """Process an agent action and return the next observation."""
        if self._state.is_complete:
            return self._get_terminal_observation()

        self._state.current_turn += 1
        self._state.history.append(action.response)

        # --- Evaluate the action ---
        score, feedback, done = self._evaluate(action.response)

        if done or self._state.current_turn >= self._state.max_turns:
            self._state.is_complete = True
            self._state.final_score = score

        return EnvObservation(
            prompt=f"Task: {self._state.task_id}",
            feedback=feedback,
            score=score,
            done=self._state.is_complete,
            turn=self._state.current_turn,
            max_turns=self._state.max_turns,
        )

    @property
    def state(self) -> EnvState:
        """Return the current internal state (for grading, not for the agent)."""
        return self._state

    # ------------------------------------------------------------------
    # Evaluation logic — customize for your problem statement
    # ------------------------------------------------------------------
    def _evaluate(self, response: str) -> Tuple[float, str, bool]:
        """Evaluate the agent's response against the target.

        Returns:
            (score, feedback_message, is_done)
        """
        normalized = response.strip().lower()
        target = self._state.target.strip().lower()

        if normalized == target:
            return 1.0, "Correct! Well done.", True

        # Partial credit: check if target is contained in response
        if target in normalized:
            return 0.7, "Partially correct — the answer is in your response but not exact.", False

        remaining = self._state.max_turns - self._state.current_turn
        return 0.0, f"Incorrect. {remaining} turns remaining. Try again.", False

    def _get_terminal_observation(self) -> EnvObservation:
        """Return observation for an already-completed episode."""
        return EnvObservation(
            prompt="Episode complete.",
            feedback=f"Final score: {self._state.final_score}",
            score=self._state.final_score,
            done=True,
            turn=self._state.current_turn,
            max_turns=self._state.max_turns,
        )
