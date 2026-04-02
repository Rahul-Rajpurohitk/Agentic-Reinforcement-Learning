"""Example graders for the Fish Farm environment.

These demonstrate the grading interface with simple scoring strategies.
The production graders are in farm_graders.py (12 task-specific graders).
"""

from typing import Any, Dict, List

from .base_grader import BaseGrader, GradeResult


class SurvivalGrader(BaseGrader):
    """Grade based purely on fish survival rate.

    Simple grader: score = survival_rate. Useful as a baseline
    to compare against more nuanced task-specific graders.
    """

    def grade(
        self,
        task_id: str,
        final_state: Dict[str, Any],
        episode_history: List[Dict[str, Any]],
        **kwargs,
    ) -> GradeResult:
        survival = final_state.get("fish", {}).get("survival_rate", 0.0)
        return GradeResult(
            score=round(survival, 3),
            passed=survival >= 0.8,
            feedback=f"Survival: {survival:.1%}",
        )


class WaterQualityGrader(BaseGrader):
    """Grade based on average water quality throughout the episode.

    Score = time-weighted average of water_quality_score (0-1).
    Strict: penalizes any hour where DO < 3 or UIA > 0.1.
    """

    def grade(
        self,
        task_id: str,
        final_state: Dict[str, Any],
        episode_history: List[Dict[str, Any]],
        **kwargs,
    ) -> GradeResult:
        if not episode_history:
            return GradeResult(score=0.0, passed=False, feedback="No history")

        total_wq = sum(h["water"]["water_quality_score"] for h in episode_history)
        avg_wq = total_wq / len(episode_history)

        violations = sum(
            1
            for h in episode_history
            if h["water"]["DO"] < 3.0 or h["water"]["UIA"] > 0.1
        )
        penalty = min(0.3, violations * 0.01)
        score = max(0.0, avg_wq - penalty)

        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.6,
            feedback=f"Avg WQ: {avg_wq:.3f}, Violations: {violations}",
        )


class ProfitGrader(BaseGrader):
    """Grade based on economic performance.

    Score = normalized profit against a $3,000 benchmark.
    Partial credit for any positive profit.
    """

    def grade(
        self,
        task_id: str,
        final_state: Dict[str, Any],
        episode_history: List[Dict[str, Any]],
        **kwargs,
    ) -> GradeResult:
        profit = final_state.get("economics", {}).get("current_profit", 0)
        benchmark = 3000.0

        if profit <= 0:
            score = 0.0
        else:
            score = min(1.0, profit / benchmark)

        return GradeResult(
            score=round(score, 3),
            passed=profit > 0,
            feedback=f"Profit: ${profit:.0f} (benchmark: ${benchmark:.0f})",
        )
