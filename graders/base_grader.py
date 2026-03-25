"""Base grader interface for OpenEnv environments.

Graders evaluate complete trajectories (episode histories) and produce
structured evaluation results. The hackathon uses both programmatic checks
and LLM-based scoring for evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GradeResult:
    """Structured grading output."""

    score: float  # 0.0 to 1.0
    passed: bool  # Did the agent pass the task?
    feedback: str  # Human-readable feedback
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown


class BaseGrader(ABC):
    """Abstract base class for graders.

    Graders receive the full trajectory and produce a GradeResult.
    """

    @abstractmethod
    def grade(
        self,
        task_id: str,
        target: str,
        history: List[str],
        final_score: float,
        **kwargs,
    ) -> GradeResult:
        """Grade a complete episode trajectory.

        Args:
            task_id: Identifier for the task.
            target: The ground-truth target/answer.
            history: List of agent actions taken during the episode.
            final_score: The environment's final score.
            **kwargs: Additional context.

        Returns:
            GradeResult with score, pass/fail, and feedback.
        """
        ...

    def __call__(self, **kwargs) -> GradeResult:
        return self.grade(**kwargs)
