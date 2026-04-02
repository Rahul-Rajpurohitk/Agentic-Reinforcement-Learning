"""Tests for Pydantic models — requires openenv-core installed."""
import pytest

try:
    from agentic_rl.models import FarmAction, FarmObservation, FarmState
    HAS_OPENENV = True
except ImportError:
    HAS_OPENENV = False


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestFarmAction:
    def test_default_action_valid(self):
        action = FarmAction()
        assert 0.0 <= action.feeding_rate <= 1.0
        assert 0.0 <= action.aeration_rate <= 1.0

    def test_action_schema_has_descriptions(self):
        schema = FarmAction.model_json_schema()
        assert "feeding_rate" in schema["properties"]
        assert "description" in schema["properties"]["feeding_rate"]


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestFarmObservation:
    def test_observation_has_required_fields(self):
        obs = FarmObservation(
            done=False, reward=0.5,
            avg_fish_weight=50.0, population=10000,
            temperature=28.0, dissolved_oxygen=7.0,
            ph=7.5, ammonia=0.1, nitrite=0.05,
            day_in_cycle=1, time_of_day=8,
        )
        assert obs.done is False
        assert obs.reward == 0.5

    def test_observation_includes_feedback(self):
        obs = FarmObservation(
            done=False, reward=0.0,
            avg_fish_weight=50.0, population=10000,
            temperature=28.0, dissolved_oxygen=7.0,
            ph=7.5, ammonia=0.1, nitrite=0.05,
            day_in_cycle=1, time_of_day=8,
            feedback="Fish are feeding eagerly."
        )
        assert "eagerly" in obs.feedback


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestFarmState:
    def test_state_has_episode_id(self):
        state = FarmState(episode_id="test-123")
        assert state.episode_id == "test-123"
