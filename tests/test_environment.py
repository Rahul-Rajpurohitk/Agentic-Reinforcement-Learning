"""Tests for the Code Review environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_rl.models import ReviewAction, ReviewObservation, ReviewState
from agentic_rl.server.environment import CodeReviewEnvironment


class TestReset:
    def setup_method(self):
        self.env = CodeReviewEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="easy_001")
        assert isinstance(obs, ReviewObservation)
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.code_snippet != ""
        assert obs.task_id == "easy_001"

    def test_reset_sets_difficulty(self):
        obs = self.env.reset(task_id="easy_001")
        assert obs.task_difficulty == "easy"

        obs = self.env.reset(task_id="medium_001")
        assert obs.task_difficulty == "medium"

        obs = self.env.reset(task_id="hard_001")
        assert obs.task_difficulty == "hard"

    def test_reset_produces_clean_state(self):
        # First episode
        self.env.reset(task_id="easy_001")
        self.env.step(ReviewAction(
            issues_found=[{"line": "1", "severity": "minor", "category": "style",
                           "description": "test", "suggestion": "test"}],
            overall_assessment="comment",
        ))
        # Reset should clear state
        self.env.reset(task_id="easy_002")
        assert self.env.state.step_count == 0
        assert self.env.state.agent_found_issues == []
        assert self.env.state.is_complete is False

    def test_reset_invalid_task_raises(self):
        try:
            self.env.reset(task_id="nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestStep:
    def setup_method(self):
        self.env = CodeReviewEnvironment()

    def test_correct_review_scores_high(self):
        """Easy task: find ZeroDivisionError in calculate_average."""
        self.env.reset(task_id="easy_001")
        action = ReviewAction(
            issues_found=[{
                "line": "6",
                "severity": "critical",
                "category": "bug",
                "description": "ZeroDivisionError when empty list is passed, len is zero",
                "suggestion": "Add check for empty list before division",
            }],
            overall_assessment="request_changes",
        )
        obs = self.env.step(action)
        assert obs.reward > 0.5
        assert obs.done is True  # High score ends episode

    def test_wrong_review_scores_low(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(
            issues_found=[{
                "line": "1",
                "severity": "minor",
                "category": "style",
                "description": "Function name should be more descriptive",
                "suggestion": "Rename function",
            }],
            overall_assessment="comment",
        )
        obs = self.env.step(action)
        assert obs.reward < 0.3

    def test_empty_review_scores_zero(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(
            issues_found=[],
            overall_assessment="approve",
        )
        obs = self.env.step(action)
        assert obs.reward == 0.0

    def test_step_counter_increments(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(
            issues_found=[],
            overall_assessment="comment",
        )
        self.env.step(action)
        assert self.env.state.step_count == 1
        self.env.step(action)
        assert self.env.state.step_count == 2

    def test_max_steps_ends_episode(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(issues_found=[], overall_assessment="comment")
        for _ in range(3):
            obs = self.env.step(action)
        assert obs.done is True

    def test_completed_episode_returns_terminal(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(issues_found=[], overall_assessment="comment")
        for _ in range(3):
            self.env.step(action)
        obs = self.env.step(action)  # After completion
        assert obs.done is True


class TestDifficultyProgression:
    """Verify tasks have meaningful difficulty progression."""

    def setup_method(self):
        self.env = CodeReviewEnvironment()

    def test_easy_has_one_obvious_issue(self):
        self.env.reset(task_id="easy_001")
        assert len(self.env.state.ground_truth_issues) == 1

    def test_medium_has_subtle_logic_bugs(self):
        self.env.reset(task_id="medium_001")
        gt = self.env.state.ground_truth_issues
        assert len(gt) >= 1
        categories = [i.get("category") for i in gt]
        assert any(c in ["bug", "logic"] for c in categories)

    def test_hard_has_multiple_security_issues(self):
        self.env.reset(task_id="hard_001")
        gt = self.env.state.ground_truth_issues
        assert len(gt) >= 2
        categories = [i.get("category") for i in gt]
        assert "security" in categories


class TestRewardSignal:
    """Verify reward function provides partial progress signals."""

    def setup_method(self):
        self.env = CodeReviewEnvironment()

    def test_partial_match_gives_partial_reward(self):
        """Finding some but not all issues should give partial credit."""
        self.env.reset(task_id="hard_001")  # Has 2 issues
        action = ReviewAction(
            issues_found=[{
                "line": "22",
                "severity": "critical",
                "category": "security",
                "description": "Mass assignment - arbitrary field names allow admin escalation",
                "suggestion": "Whitelist allowed fields",
            }],
            overall_assessment="request_changes",
        )
        obs = self.env.step(action)
        # Should get partial credit (found 1 of 2)
        assert 0.1 < obs.reward < 0.9

    def test_reward_in_valid_range(self):
        """All rewards should be in [0.0, 1.0]."""
        for task_id in ["easy_001", "medium_001", "hard_001"]:
            self.env.reset(task_id=task_id)
            action = ReviewAction(
                issues_found=[{
                    "line": "1", "severity": "minor", "category": "style",
                    "description": "random issue", "suggestion": "fix it",
                }],
                overall_assessment="comment",
            )
            obs = self.env.step(action)
            assert 0.0 <= obs.reward <= 1.0
