"""Base reward function interface for the Fish Farm environment.

Reward functions provide shaping signals for GRPO training.
They score agent actions against the current farm state.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseReward(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        prev_state: Dict[str, Any],
        **kwargs,
    ) -> float:
        """Compute reward for a farm management action.

        Args:
            state: Current farm state after action.
            action: The action that was taken.
            prev_state: Farm state before the action.

        Returns:
            Float reward (can be negative for penalties).
        """
        ...

    def __call__(self, **kwargs) -> float:
        return self.compute(**kwargs)
