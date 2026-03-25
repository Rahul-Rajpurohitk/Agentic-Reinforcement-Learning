"""Tests for the Code Review environment using official openenv-core types."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_rl.models import ReviewAction, ReviewObservation, ReviewState
from agentic_rl.server.environment import CodeReviewEnvironment
from openenv.core.env_server import Action, Observation, State, Environment


class TestSpecCompliance:
    """Verify the environment follows the official OpenEnv spec."""

    def test_models_inherit_from_openenv(self):
        assert issubclass(ReviewAction, Action)
        assert issubclass(ReviewObservation, Observation)
        assert issubclass(ReviewState, State)

    def test_environment_inherits_from_openenv(self):
        assert issubclass(CodeReviewEnvironment, Environment)

    def test_observation_has_done_and_reward(self):
        obs = ReviewObservation(task_id="test")
        assert hasattr(obs, "done")
        assert hasattr(obs, "reward")
        assert hasattr(obs, "metadata")

    def test_state_has_episode_id_and_step_count(self):
        state = ReviewState()
        assert hasattr(state, "episode_id")
        assert hasattr(state, "step_count")

    def test_action_has_metadata(self):
        action = ReviewAction(issues_found=[], overall_assessment="comment")
        assert hasattr(action, "metadata")

    def test_supports_concurrent_sessions(self):
        assert CodeReviewEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True


class TestReset:
    def setup_method(self):
        self.env = CodeReviewEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="easy_001")
        assert isinstance(obs, ReviewObservation)
        assert isinstance(obs, Observation)  # Official base class
        assert obs.done is False
        assert obs.reward is None  # Initial obs has no reward per spec
        assert obs.code_snippet != ""
        assert obs.task_id == "easy_001"

    def test_reset_sets_difficulty(self):
        for tid, expected in [("easy_001", "easy"), ("medium_001", "medium"), ("hard_001", "hard")]:
            obs = self.env.reset(task_id=tid)
            assert obs.task_difficulty == expected

    def test_reset_produces_clean_state(self):
        self.env.reset(task_id="easy_001")
        self.env.step(ReviewAction(
            issues_found=[{"line": "1", "severity": "minor", "category": "style",
                           "description": "test", "suggestion": "test"}],
            overall_assessment="comment",
        ))
        self.env.reset(task_id="easy_002")
        assert self.env.state.step_count == 0
        assert self.env.state.agent_found_issues == []
        assert self.env.state.is_complete is False

    def test_reset_accepts_seed_and_episode_id(self):
        """Official spec: reset(seed, episode_id, **kwargs)."""
        obs = self.env.reset(seed=42, episode_id="test-ep-123", task_id="easy_001")
        assert self.env.state.episode_id == "test-ep-123"

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
        self.env.reset(task_id="easy_001")
        action = ReviewAction(
            issues_found=[{
                "line": "6",
                "severity": "critical",
                "category": "bug",
                "description": "ZeroDivisionError when empty list, len is zero",
                "suggestion": "Check for empty list before division",
            }],
            overall_assessment="request_changes",
        )
        obs = self.env.step(action)
        assert obs.reward > 0.5
        assert obs.done is True

    def test_wrong_review_scores_low(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(
            issues_found=[{
                "line": "1", "severity": "minor", "category": "style",
                "description": "Bad name", "suggestion": "Rename",
            }],
            overall_assessment="comment",
        )
        obs = self.env.step(action)
        assert obs.reward < 0.3

    def test_empty_review_scores_zero(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(issues_found=[], overall_assessment="approve")
        obs = self.env.step(action)
        assert obs.reward == 0.0

    def test_step_counter_increments(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(issues_found=[], overall_assessment="comment")
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

    def test_step_accepts_timeout_s(self):
        """Official spec: step(action, timeout_s, **kwargs)."""
        self.env.reset(task_id="easy_001")
        action = ReviewAction(issues_found=[], overall_assessment="comment")
        obs = self.env.step(action, timeout_s=5.0)
        assert isinstance(obs, ReviewObservation)

    def test_completed_episode_returns_terminal(self):
        self.env.reset(task_id="easy_001")
        action = ReviewAction(issues_found=[], overall_assessment="comment")
        for _ in range(3):
            self.env.step(action)
        obs = self.env.step(action)
        assert obs.done is True


class TestDifficultyProgression:
    def setup_method(self):
        self.env = CodeReviewEnvironment()

    def test_easy_has_one_obvious_issue(self):
        self.env.reset(task_id="easy_001")
        assert len(self.env.state.ground_truth_issues) == 1

    def test_medium_has_subtle_logic_bugs(self):
        self.env.reset(task_id="medium_001")
        gt = self.env.state.ground_truth_issues
        assert len(gt) >= 1
        assert any(i.get("category") in ["bug", "logic"] for i in gt)

    def test_hard_has_multiple_security_issues(self):
        self.env.reset(task_id="hard_001")
        gt = self.env.state.ground_truth_issues
        assert len(gt) >= 2
        assert "security" in [i.get("category") for i in gt]


class TestRewardSignal:
    def setup_method(self):
        self.env = CodeReviewEnvironment()

    def test_partial_match_gives_partial_reward(self):
        self.env.reset(task_id="hard_001")
        action = ReviewAction(
            issues_found=[{
                "line": "22", "severity": "critical", "category": "security",
                "description": "Mass assignment - arbitrary field names allow admin escalation",
                "suggestion": "Whitelist allowed fields",
            }],
            overall_assessment="request_changes",
        )
        obs = self.env.step(action)
        assert 0.1 < obs.reward < 0.9

    def test_reward_in_valid_range(self):
        for task_id in ["easy_001", "medium_001", "hard_001"]:
            self.env.reset(task_id=task_id)
            action = ReviewAction(
                issues_found=[{
                    "line": "1", "severity": "minor", "category": "style",
                    "description": "random", "suggestion": "fix",
                }],
                overall_assessment="comment",
            )
            obs = self.env.step(action)
            assert 0.0 <= obs.reward <= 1.0


class TestSerializability:
    """Verify all models serialize to/from JSON (required for WebSocket)."""

    def test_action_round_trip(self):
        action = ReviewAction(
            issues_found=[{"line": "1", "severity": "critical", "category": "bug",
                           "description": "test", "suggestion": "fix"}],
            overall_assessment="request_changes",
        )
        data = action.model_dump()
        restored = ReviewAction(**data)
        assert restored.issues_found == action.issues_found

    def test_observation_round_trip(self):
        env = CodeReviewEnvironment()
        obs = env.reset(task_id="easy_001")
        data = obs.model_dump()
        restored = ReviewObservation(**data)
        assert restored.task_id == obs.task_id
        assert restored.done == obs.done

    def test_state_round_trip(self):
        env = CodeReviewEnvironment()
        env.reset(task_id="easy_001")
        data = env.state.model_dump()
        restored = ReviewState(**data)
        assert restored.task_id == env.state.task_id
