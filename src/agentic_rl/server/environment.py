"""Code Review Environment — core game logic.

A real-world OpenEnv environment where an AI agent reviews code for bugs,
logic errors, and security vulnerabilities. The agent receives code snippets
and must identify issues with their severity, category, and suggested fixes.

Implements the OpenEnv interface: reset() / step() / state
"""

import uuid
from typing import Any, Dict, List, Tuple

from ..models import ReviewAction, ReviewObservation, ReviewState
from ..tasks import TASKS, get_task, list_all_tasks


def _match_issue(found: Dict[str, str], truth: Dict[str, str]) -> float:
    """Score how well a found issue matches a ground-truth issue.

    Returns 0.0 - 1.0 based on keyword overlap and line proximity.
    """
    score = 0.0
    description_lower = found.get("description", "").lower()
    suggestion_lower = found.get("suggestion", "").lower()
    combined_text = f"{description_lower} {suggestion_lower}"

    # Check keyword matches (primary signal)
    keywords = truth.get("keywords", [])
    if keywords:
        matches = sum(1 for kw in keywords if kw.lower() in combined_text)
        keyword_score = matches / len(keywords)
        score += 0.7 * keyword_score

    # Check line number proximity
    try:
        found_line = int(found.get("line", "0"))
        truth_line = int(truth.get("line", "0"))
        distance = abs(found_line - truth_line)
        line_score = max(0, 1.0 - distance / 10.0)
        score += 0.2 * line_score
    except (ValueError, TypeError):
        pass

    # Check severity match
    if found.get("severity", "").lower() == truth.get("severity", "").lower():
        score += 0.1

    return min(score, 1.0)


def _grade_review(
    found_issues: List[Dict[str, str]],
    ground_truth: List[Dict[str, str]],
) -> Tuple[float, str, Dict[str, Any]]:
    """Grade a code review against ground truth.

    Returns (score, feedback, details) where score is 0.0-1.0.
    """
    if not ground_truth:
        return 1.0, "No issues expected.", {"matched": 0, "total": 0}

    # Match found issues to ground truth using best-match greedy assignment
    matched_truths = set()
    issue_scores = []

    for truth_idx, truth in enumerate(ground_truth):
        best_score = 0.0
        for found in found_issues:
            s = _match_issue(found, truth)
            if s > best_score:
                best_score = s
        if best_score >= 0.3:  # Threshold for considering it a match
            matched_truths.add(truth_idx)
        issue_scores.append(best_score)

    # Calculate recall (how many ground truth issues were found)
    recall = len(matched_truths) / len(ground_truth)

    # Penalize false positives (noise) but gently
    false_positives = max(0, len(found_issues) - len(matched_truths))
    noise_penalty = min(0.15, false_positives * 0.03)

    # Weighted score: average quality of matches * recall
    avg_quality = sum(issue_scores) / len(issue_scores) if issue_scores else 0.0
    raw_score = avg_quality * 0.6 + recall * 0.4
    final_score = max(0.0, min(1.0, raw_score - noise_penalty))

    # Generate feedback
    found_count = len(matched_truths)
    total_count = len(ground_truth)
    feedback_parts = [f"Found {found_count}/{total_count} issues."]

    if found_count == total_count:
        feedback_parts.append("Excellent — all issues identified!")
    elif found_count > 0:
        missed = total_count - found_count
        feedback_parts.append(f"Missed {missed} issue(s). Look more carefully.")
    else:
        feedback_parts.append("No correct issues identified. Try again.")

    if false_positives > 0:
        feedback_parts.append(f"{false_positives} false positive(s) reported.")

    details = {
        "matched": found_count,
        "total": total_count,
        "false_positives": false_positives,
        "per_issue_scores": issue_scores,
        "recall": recall,
    }

    return final_score, " ".join(feedback_parts), details


class CodeReviewEnvironment:
    """OpenEnv-compatible Code Review environment.

    An AI agent reviews code snippets for bugs, logic errors, and security
    vulnerabilities. Tasks range from easy (obvious bugs) to hard (subtle
    security vulnerabilities that challenge frontier models).

    The environment follows the OpenEnv 3-component pattern:
    - reset(task_id)  -> ReviewObservation
    - step(action)    -> ReviewObservation (with reward and done)
    - state           -> ReviewState (ground truth for grading)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = ReviewState()
        self._current_task: Dict[str, Any] = {}

    def reset(self, task_id: str = "easy_001") -> ReviewObservation:
        """Start a new code review episode for the given task."""
        task = get_task(task_id)
        episode_id = str(uuid.uuid4())

        self._current_task = task
        self._state = ReviewState(
            episode_id=episode_id,
            task_id=task_id,
            step_count=0,
            ground_truth_issues=task["ground_truth"],
            agent_found_issues=[],
            max_steps=3,
            is_complete=False,
            final_score=0.0,
        )

        return ReviewObservation(
            task_id=task_id,
            task_difficulty=task["difficulty"],
            code_snippet=task["code"],
            language=task["language"],
            context=task["context"],
            feedback="Review this code. Identify all bugs, logic errors, and "
            "security vulnerabilities. Report each issue with its line number, "
            "severity, category, description, and fix suggestion.",
            reward=0.0,
            done=False,
            info={"episode_id": episode_id, "max_steps": 3},
        )

    def step(self, action: ReviewAction) -> ReviewObservation:
        """Process the agent's code review and return graded observation."""
        if self._state.is_complete:
            return self._terminal_observation()

        self._state.step_count += 1
        self._state.agent_found_issues = action.issues_found

        # Grade the review
        score, feedback, details = _grade_review(
            action.issues_found,
            self._state.ground_truth_issues,
        )

        # Determine if episode is done
        done = (
            score >= 0.85  # High-quality review
            or self._state.step_count >= self._state.max_steps
        )

        if done:
            self._state.is_complete = True
            self._state.final_score = score

        # Partial progress reward: reward improvement over steps
        reward = score

        return ReviewObservation(
            task_id=self._state.task_id,
            task_difficulty=self._current_task.get("difficulty", "unknown"),
            code_snippet=self._current_task.get("code", ""),
            language=self._current_task.get("language", "python"),
            context=self._current_task.get("context", ""),
            feedback=feedback,
            reward=reward,
            done=done,
            info={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "grading_details": details,
                "overall_assessment": action.overall_assessment,
            },
        )

    @property
    def state(self) -> ReviewState:
        """Return current internal state (for grading)."""
        return self._state

    def _terminal_observation(self) -> ReviewObservation:
        """Return observation for a completed episode."""
        return ReviewObservation(
            task_id=self._state.task_id,
            task_difficulty=self._current_task.get("difficulty", "unknown"),
            code_snippet="",
            language=self._current_task.get("language", "python"),
            context="Episode already complete.",
            feedback=f"Episode ended. Final score: {self._state.final_score:.2f}",
            reward=self._state.final_score,
            done=True,
            info={"final": True},
        )
