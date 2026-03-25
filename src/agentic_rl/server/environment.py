"""Code Review Environment — core game logic.

A real-world OpenEnv environment where an AI agent reviews code for bugs,
logic errors, and security vulnerabilities.

Extends openenv.core.env_server.Environment with the official interface:
  reset(seed, episode_id, **kwargs) -> Observation
  step(action, timeout_s, **kwargs) -> Observation
  state -> State
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

from ..models import ReviewAction, ReviewObservation, ReviewState
from ..tasks import get_task


def _match_issue(found: Dict[str, str], truth: Dict[str, str]) -> float:
    """Score how well a found issue matches a ground-truth issue (0.0-1.0)."""
    score = 0.0
    description_lower = found.get("description", "").lower()
    suggestion_lower = found.get("suggestion", "").lower()
    combined_text = f"{description_lower} {suggestion_lower}"

    # Keyword matches (primary signal)
    keywords = truth.get("keywords", [])
    if keywords:
        matches = sum(1 for kw in keywords if kw.lower() in combined_text)
        score += 0.7 * (matches / len(keywords))

    # Line proximity
    try:
        found_line = int(found.get("line", "0"))
        truth_line = int(truth.get("line", "0"))
        distance = abs(found_line - truth_line)
        score += 0.2 * max(0, 1.0 - distance / 10.0)
    except (ValueError, TypeError):
        pass

    # Severity match
    if found.get("severity", "").lower() == truth.get("severity", "").lower():
        score += 0.1

    return min(score, 1.0)


def _grade_review(
    found_issues: List[Dict[str, str]],
    ground_truth: List[Dict[str, str]],
) -> Tuple[float, str, Dict[str, Any]]:
    """Grade a code review. Returns (score, feedback, details)."""
    if not ground_truth:
        return 1.0, "No issues expected.", {"matched": 0, "total": 0}

    matched_truths = set()
    issue_scores = []

    for truth_idx, truth in enumerate(ground_truth):
        best_score = 0.0
        for found in found_issues:
            s = _match_issue(found, truth)
            if s > best_score:
                best_score = s
        if best_score >= 0.3:
            matched_truths.add(truth_idx)
        issue_scores.append(best_score)

    recall = len(matched_truths) / len(ground_truth)
    false_positives = max(0, len(found_issues) - len(matched_truths))
    noise_penalty = min(0.15, false_positives * 0.03)
    avg_quality = sum(issue_scores) / len(issue_scores) if issue_scores else 0.0
    raw_score = avg_quality * 0.6 + recall * 0.4
    final_score = max(0.0, min(1.0, raw_score - noise_penalty))

    found_count = len(matched_truths)
    total_count = len(ground_truth)
    feedback_parts = [f"Found {found_count}/{total_count} issues."]
    if found_count == total_count:
        feedback_parts.append("Excellent — all issues identified!")
    elif found_count > 0:
        feedback_parts.append(f"Missed {total_count - found_count} issue(s).")
    else:
        feedback_parts.append("No correct issues identified.")
    if false_positives > 0:
        feedback_parts.append(f"{false_positives} false positive(s).")

    details = {
        "matched": found_count,
        "total": total_count,
        "false_positives": false_positives,
        "per_issue_scores": issue_scores,
        "recall": recall,
    }
    return final_score, " ".join(feedback_parts), details


class CodeReviewEnvironment(Environment[ReviewAction, ReviewObservation, ReviewState]):
    """OpenEnv Code Review environment.

    AI agents review code snippets for bugs, logic errors, and security
    vulnerabilities. Tasks range from easy (obvious bugs) to hard (subtle
    security issues that challenge frontier models).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = ReviewState()
        self._current_task: Dict[str, Any] = {}

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy_001",
        **kwargs: Any,
    ) -> ReviewObservation:
        """Start a new code review episode."""
        task = get_task(task_id)
        eid = episode_id or str(uuid.uuid4())

        self._current_task = task
        self._state = ReviewState(
            episode_id=eid,
            step_count=0,
            task_id=task_id,
            ground_truth_issues=task["ground_truth"],
            agent_found_issues=[],
            max_steps=3,
            is_complete=False,
            final_score=0.0,
        )

        return ReviewObservation(
            done=False,
            reward=None,
            metadata={"episode_id": eid, "max_steps": 3},
            task_id=task_id,
            task_difficulty=task["difficulty"],
            code_snippet=task["code"],
            language=task["language"],
            context=task["context"],
            feedback=(
                "Review this code. Identify all bugs, logic errors, and "
                "security vulnerabilities. Report each issue with line number, "
                "severity, category, description, and fix suggestion."
            ),
        )

    def step(
        self,
        action: ReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ReviewObservation:
        """Process the agent's code review and return graded observation."""
        if self._state.is_complete:
            return self._terminal_observation()

        self._state.step_count += 1
        self._state.agent_found_issues = action.issues_found

        score, feedback, details = _grade_review(
            action.issues_found,
            self._state.ground_truth_issues,
        )

        done = score >= 0.85 or self._state.step_count >= self._state.max_steps

        if done:
            self._state.is_complete = True
            self._state.final_score = score

        return ReviewObservation(
            done=done,
            reward=score,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "grading_details": details,
                "overall_assessment": action.overall_assessment,
            },
            task_id=self._state.task_id,
            task_difficulty=self._current_task.get("difficulty", "unknown"),
            code_snippet=self._current_task.get("code", ""),
            language=self._current_task.get("language", "python"),
            context=self._current_task.get("context", ""),
            feedback=feedback,
        )

    @property
    def state(self) -> ReviewState:
        """Return current internal state."""
        return self._state

    def _terminal_observation(self) -> ReviewObservation:
        """Return observation for a completed episode."""
        return ReviewObservation(
            done=True,
            reward=self._state.final_score,
            metadata={"final": True},
            task_id=self._state.task_id,
            task_difficulty=self._current_task.get("difficulty", "unknown"),
            code_snippet="",
            language=self._current_task.get("language", "python"),
            context="Episode already complete.",
            feedback=f"Episode ended. Final score: {self._state.final_score:.2f}",
        )
