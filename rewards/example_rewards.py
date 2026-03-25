"""Example reward function implementations.

These serve as templates — adapt them for your specific problem statement.
"""

from typing import List, Optional

from .base_reward import BaseReward


class ExactMatchReward(BaseReward):
    """Binary reward: 1.0 if completion matches target exactly, else 0.0."""

    def __init__(self, case_sensitive: bool = False, strip: bool = True):
        self.case_sensitive = case_sensitive
        self.strip = strip

    def compute(
        self,
        prompt: str,
        completion: str,
        target: Optional[str] = None,
        **kwargs,
    ) -> float:
        if target is None:
            return 0.0

        a = completion.strip() if self.strip else completion
        b = target.strip() if self.strip else target

        if not self.case_sensitive:
            a, b = a.lower(), b.lower()

        return 1.0 if a == b else 0.0


class PartialMatchReward(BaseReward):
    """Shaped reward based on token/character overlap with target."""

    def compute(
        self,
        prompt: str,
        completion: str,
        target: Optional[str] = None,
        **kwargs,
    ) -> float:
        if target is None:
            return 0.0

        comp_tokens = set(completion.lower().split())
        target_tokens = set(target.lower().split())

        if not target_tokens:
            return 0.0

        overlap = comp_tokens & target_tokens
        precision = len(overlap) / len(comp_tokens) if comp_tokens else 0.0
        recall = len(overlap) / len(target_tokens)

        if precision + recall == 0:
            return 0.0

        # F1 score as reward
        return 2 * (precision * recall) / (precision + recall)


class MultiObjectiveReward(BaseReward):
    """Combine multiple reward functions with weights.

    Usage:
        reward = MultiObjectiveReward(
            rewards=[ExactMatchReward(), PartialMatchReward()],
            weights=[0.7, 0.3],
        )
    """

    def __init__(self, rewards: List[BaseReward], weights: Optional[List[float]] = None):
        self.rewards = rewards
        self.weights = weights or [1.0 / len(rewards)] * len(rewards)
        assert len(self.rewards) == len(self.weights)

    def compute(
        self,
        prompt: str,
        completion: str,
        target: Optional[str] = None,
        **kwargs,
    ) -> float:
        total = sum(
            w * r.compute(prompt, completion, target, **kwargs)
            for w, r in zip(self.weights, self.rewards)
        )
        return total
