"""Example grader implementations.

Adapt these for your specific problem statement.
"""

from typing import Any, Dict, List

from .base_grader import BaseGrader, GradeResult


class ExactMatchGrader(BaseGrader):
    """Grade based on whether the agent produced the exact target answer."""

    def grade(
        self,
        task_id: str,
        target: str,
        history: List[str],
        final_score: float,
        **kwargs,
    ) -> GradeResult:
        if not history:
            return GradeResult(
                score=0.0,
                passed=False,
                feedback="No actions taken.",
                details={"task_id": task_id, "attempts": 0},
            )

        best_match = max(
            history,
            key=lambda h: 1.0 if h.strip().lower() == target.strip().lower() else 0.0,
        )
        passed = best_match.strip().lower() == target.strip().lower()

        return GradeResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            feedback="Correct answer found!" if passed else f"Expected '{target}', not found.",
            details={
                "task_id": task_id,
                "attempts": len(history),
                "best_answer": best_match,
                "target": target,
            },
        )


class RubricGrader(BaseGrader):
    """Grade based on multiple rubric criteria with weights.

    Each criterion is a callable: (history, target, **kwargs) -> float [0..1]
    """

    def __init__(self, criteria: Dict[str, Any]):
        """
        Args:
            criteria: Dict of {name: {"fn": callable, "weight": float, "description": str}}
        """
        self.criteria = criteria

    def grade(
        self,
        task_id: str,
        target: str,
        history: List[str],
        final_score: float,
        **kwargs,
    ) -> GradeResult:
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for name, criterion in self.criteria.items():
            fn = criterion["fn"]
            weight = criterion.get("weight", 1.0)
            score = fn(history, target, **kwargs)
            scores[name] = {"score": score, "weight": weight}
            weighted_sum += score * weight
            total_weight += weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        passed = overall >= 0.5

        feedback_lines = [f"Overall: {overall:.2f}"]
        for name, data in scores.items():
            desc = self.criteria[name].get("description", name)
            feedback_lines.append(f"  {desc}: {data['score']:.2f} (weight: {data['weight']})")

        return GradeResult(
            score=overall,
            passed=passed,
            feedback="\n".join(feedback_lines),
            details={"task_id": task_id, "criteria_scores": scores},
        )
