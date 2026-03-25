"""Base grader interface for the Code Review environment.

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
        ground_truth: List[Dict[str, str]],
        agent_issues: List[Dict[str, str]],
        **kwargs,
    ) -> GradeResult:
        """Grade a completed code review episode.

        Returns GradeResult with score in [0.0, 1.0].
        """
        ...
