"""Base reward function interface for the Code Review environment.

Reward functions score agent completions during GRPO training.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseReward(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(
        self,
        issues_found: List[Dict[str, str]],
        ground_truth: List[Dict[str, str]],
        **kwargs,
    ) -> float:
        """Compute reward for a code review.

        Args:
            issues_found: Issues reported by the agent.
            ground_truth: Actual issues in the code.

        Returns:
            Float reward in [0.0, 1.0].
        """
        ...

    def __call__(self, **kwargs) -> float:
        return self.compute(**kwargs)
