"""Base reward function interface for OpenEnv environments.

Reward functions are used by the GRPO trainer to score agent completions.
Each reward function receives the prompt, the agent's completion, and
optionally the ground truth, and returns a scalar reward.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseReward(ABC):
    """Abstract base class for reward functions.

    Subclass this and implement compute() for your problem statement.
    """

    @abstractmethod
    def compute(
        self,
        prompt: str,
        completion: str,
        target: Optional[str] = None,
        **kwargs,
    ) -> float:
        """Compute reward for a single (prompt, completion) pair.

        Args:
            prompt: The task prompt given to the agent.
            completion: The agent's generated response.
            target: The ground-truth answer (if available).
            **kwargs: Additional context (turn number, history, etc.)

        Returns:
            A float reward value. Convention: 0.0 = worst, 1.0 = best.
        """
        ...

    def __call__(self, prompt: str, completion: str, **kwargs) -> float:
        return self.compute(prompt, completion, **kwargs)
