"""Tests for the OpenEnv environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_rl.models import AgentAction, EnvObservation, EnvState
from agentic_rl.server.environment import AgenticRLEnvironment


class TestEnvironment:
    def setup_method(self):
        self.env = AgenticRLEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert isinstance(obs, EnvObservation)
        assert obs.done is False
        assert obs.turn == 0
        assert obs.prompt != ""

    def test_reset_with_task_id(self):
        obs = self.env.reset(task_id="task_001")
        assert self.env.state.task_id == "task_001"

    def test_step_correct_answer(self):
        self.env.reset(task_id="task_002")  # "Solve: 2 + 2 = ?", target: "4"
        action = AgentAction(response="4")
        obs = self.env.step(action)
        assert obs.score == 1.0
        assert obs.done is True

    def test_step_wrong_answer(self):
        self.env.reset(task_id="task_002")
        action = AgentAction(response="5")
        obs = self.env.step(action)
        assert obs.score == 0.0
        assert obs.done is False

    def test_step_partial_match(self):
        self.env.reset(task_id="task_001")  # target: "paris"
        action = AgentAction(response="The answer is paris, the capital")
        obs = self.env.step(action)
        assert obs.score == 0.7
        assert obs.done is False

    def test_turn_counter_increments(self):
        self.env.reset()
        action = AgentAction(response="wrong")
        obs1 = self.env.step(action)
        assert obs1.turn == 1
        obs2 = self.env.step(action)
        assert obs2.turn == 2

    def test_max_turns_ends_episode(self):
        self.env.reset()
        action = AgentAction(response="wrong")
        for _ in range(10):
            obs = self.env.step(action)
        assert obs.done is True

    def test_state_tracks_history(self):
        self.env.reset()
        self.env.step(AgentAction(response="first"))
        self.env.step(AgentAction(response="second"))
        assert self.env.state.history == ["first", "second"]

    def test_completed_episode_returns_terminal(self):
        self.env.reset(task_id="task_002")
        self.env.step(AgentAction(response="4"))  # Correct, episode ends
        obs = self.env.step(AgentAction(response="anything"))  # After completion
        assert obs.done is True
