"""Deterministic graders for the Code Review environment.

Each grader produces reproducible scores between 0.0 and 1.0.
Graders have clear, deterministic success/failure criteria.
"""

from typing import Dict, List

from .base_grader import BaseGrader, GradeResult


class KeywordMatchGrader(BaseGrader):
    """Grade based on keyword overlap between found issues and ground truth.

    This is the primary grader — deterministic and reproducible.
    Score = weighted average of per-issue keyword match scores.
    """

    def __init__(self, match_threshold: float = 0.3):
        self.match_threshold = match_threshold

    def grade(
        self,
        task_id: str,
        ground_truth: List[Dict[str, str]],
        agent_issues: List[Dict[str, str]],
        **kwargs,
    ) -> GradeResult:
        if not ground_truth:
            return GradeResult(score=1.0, passed=True, feedback="No issues expected.")

        issue_scores = []
        matched_count = 0

        for truth in ground_truth:
            keywords = truth.get("keywords", [])
            best_match = 0.0

            for found in agent_issues:
                text = (
                    f"{found.get('description', '')} {found.get('suggestion', '')}"
                ).lower()
                if keywords:
                    match_count = sum(1 for kw in keywords if kw.lower() in text)
                    match_ratio = match_count / len(keywords)
                    best_match = max(best_match, match_ratio)

            issue_scores.append(best_match)
            if best_match >= self.match_threshold:
                matched_count += 1

        avg_score = sum(issue_scores) / len(issue_scores)

        # Penalize false positives gently
        false_positives = max(0, len(agent_issues) - matched_count)
        penalty = min(0.15, false_positives * 0.03)
        final_score = max(0.0, min(1.0, avg_score - penalty))

        passed = final_score >= 0.5

        feedback = (
            f"Matched {matched_count}/{len(ground_truth)} issues. "
            f"Score: {final_score:.3f}."
        )
        if false_positives > 0:
            feedback += f" {false_positives} false positive(s)."

        return GradeResult(
            score=final_score,
            passed=passed,
            feedback=feedback,
            details={
                "task_id": task_id,
                "matched": matched_count,
                "total_expected": len(ground_truth),
                "false_positives": false_positives,
                "per_issue_scores": issue_scores,
            },
        )


class StrictGrader(BaseGrader):
    """Strict grader: requires ALL issues found to pass.

    Score = 1.0 if all issues matched, 0.0 otherwise.
    Used for validating that the hard tasks truly challenge models.
    """

    def grade(
        self,
        task_id: str,
        ground_truth: List[Dict[str, str]],
        agent_issues: List[Dict[str, str]],
        **kwargs,
    ) -> GradeResult:
        if not ground_truth:
            return GradeResult(score=1.0, passed=True, feedback="No issues expected.")

        all_found = True
        for truth in ground_truth:
            keywords = truth.get("keywords", [])
            found = False
            for issue in agent_issues:
                text = (
                    f"{issue.get('description', '')} {issue.get('suggestion', '')}"
                ).lower()
                if keywords:
                    matches = sum(1 for kw in keywords if kw.lower() in text)
                    if matches / len(keywords) >= 0.3:
                        found = True
                        break
            if not found:
                all_found = False
                break

        score = 1.0 if all_found else 0.0
        return GradeResult(
            score=score,
            passed=all_found,
            feedback="All issues found!" if all_found else "Not all issues identified.",
            details={"task_id": task_id, "all_found": all_found},
        )
