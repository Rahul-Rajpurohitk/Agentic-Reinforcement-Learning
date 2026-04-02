"""Base grader interface for the Fish Farm environment.

Graders produce deterministic scores in [0.0, 1.0] for each task.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GradeResult:
    """Structured grading output."""

    score: float
    passed: bool
    feedback: str
    details: Dict[str, Any] = field(default_factory=dict)


class BaseGrader(ABC):
    """Abstract base class for graders."""

    @abstractmethod
    def grade(
        self,
        task_id: str,
        final_state: Dict[str, Any],
        episode_history: List[Dict[str, Any]],
        task_config: Dict[str, Any],
        **kwargs,
    ) -> GradeResult:
        """Grade a completed episode.

        Args:
            task_id: Identifier for the task.
            final_state: Final simulation state dict.
            episode_history: List of state dicts from each hour.
            task_config: Task configuration (weights, targets, etc).

        Returns:
            GradeResult with score in [0.0, 1.0].
        """
        ...
