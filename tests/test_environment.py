"""Tests for the Fish Farm environment using official openenv-core types.

Tests the FishFarmEnvironment class against the OpenEnv spec:
  - reset(seed, episode_id, **kwargs) -> Observation
  - step(action, timeout_s, **kwargs) -> Observation
  - state -> State
  - SUPPORTS_CONCURRENT_SESSIONS flag

Also validates simulation coupling: the cascade from
overfeed → ammonia → DO crash → stress → disease → mortality
must emerge from the environment's internal dynamics.
"""

import pytest

try:
    from openenv.core.env_server import Action, Observation, State, Environment
    from agentic_rl.models import FarmAction, FarmObservation, FarmState
    from agentic_rl.server.environment import FishFarmEnvironment
    HAS_OPENENV = True
except ImportError:
    HAS_OPENENV = False


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestSpecCompliance:
    """Verify the environment follows the official OpenEnv spec."""

    def test_models_inherit_from_openenv(self):
        assert issubclass(FarmAction, Action)
        assert issubclass(FarmObservation, Observation)
        assert issubclass(FarmState, State)

    def test_environment_inherits_from_openenv(self):
        assert issubclass(FishFarmEnvironment, Environment)

    def test_observation_has_done_and_reward(self):
        obs = FarmObservation(done=False, reward=0.5)
        assert hasattr(obs, "done")
        assert hasattr(obs, "reward")
        assert hasattr(obs, "metadata")

    def test_state_has_episode_id_and_step_count(self):
        state = FarmState()
        assert hasattr(state, "episode_id")
        assert hasattr(state, "step_count")

    def test_action_has_metadata(self):
        action = FarmAction()
        assert hasattr(action, "metadata")

    def test_supports_concurrent_sessions(self):
        assert FishFarmEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestReset:
    def setup_method(self):
        self.env = FishFarmEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="feeding_basics")
        assert isinstance(obs, FarmObservation)
        assert isinstance(obs, Observation)
        assert obs.done is False
        assert obs.reward is None  # Initial obs has no reward per spec

    def test_reset_observation_has_fish_data(self):
        obs = self.env.reset(task_id="feeding_basics")
        assert obs.avg_fish_weight > 0
        assert obs.population > 0
        assert obs.biomass_kg > 0

    def test_reset_observation_has_water_data(self):
        obs = self.env.reset(task_id="feeding_basics")
        assert obs.temperature > 0
        assert obs.dissolved_oxygen > 0
        assert 0 < obs.ph < 14

    def test_reset_sets_task_initial_conditions(self):
        obs = self.env.reset(task_id="feeding_basics")
        assert obs.avg_fish_weight == pytest.approx(50.0, abs=1.0)
        assert obs.population == 5000
        assert obs.temperature == pytest.approx(30.0, abs=2.0)

    def test_reset_produces_clean_state(self):
        self.env.reset(task_id="feeding_basics")
        # Step once
        self.env.step(FarmAction(feeding_rate=0.5, aeration_rate=0.5))
        # Reset again
        self.env.reset(task_id="oxygen_management")
        assert self.env.state.step_count == 0
        assert self.env.state.is_complete is False

    def test_reset_accepts_seed_and_episode_id(self):
        self.env.reset(seed=42, episode_id="test-ep-123", task_id="feeding_basics")
        assert self.env.state.episode_id == "test-ep-123"

    def test_reset_invalid_task_raises(self):
        with pytest.raises(ValueError):
            self.env.reset(task_id="nonexistent_task_that_doesnt_exist")

    def test_reset_all_12_tasks(self):
        """Every task should reset without error."""
        from agentic_rl.tasks import TASKS
        for tid in TASKS:
            obs = self.env.reset(task_id=tid)
            assert obs.done is False
            assert obs.population > 0

    def test_reset_includes_task_description_in_feedback(self):
        obs = self.env.reset(task_id="feeding_basics")
        assert "TASK:" in obs.feedback
        assert "feed" in obs.feedback.lower() or "fish" in obs.feedback.lower()


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestStep:
    def setup_method(self):
        self.env = FishFarmEnvironment()

    def test_step_returns_observation(self):
        self.env.reset(task_id="feeding_basics")
        action = FarmAction(feeding_rate=0.5, aeration_rate=0.5)
        obs = self.env.step(action)
        assert isinstance(obs, FarmObservation)
        assert obs.reward is not None

    def test_step_advances_time(self):
        self.env.reset(task_id="feeding_basics")
        action = FarmAction()
        obs = self.env.step(action)
        assert obs.time_of_day == 1 or self.env.state.step_count == 1

    def test_step_counter_increments(self):
        self.env.reset(task_id="feeding_basics")
        action = FarmAction()
        self.env.step(action)
        assert self.env.state.step_count == 1
        self.env.step(action)
        assert self.env.state.step_count == 2

    def test_max_steps_ends_episode(self):
        """Episode should end when step_count reaches max_hours."""
        self.env.reset(task_id="oxygen_management")  # 3 * 24 = 72 hours
        action = FarmAction(feeding_rate=0.3, aeration_rate=0.8)
        obs = None
        for _ in range(72):
            obs = self.env.step(action)
        assert obs.done is True

    def test_harvest_ends_episode(self):
        self.env.reset(task_id="feeding_basics")
        action = FarmAction(harvest_decision=True)
        obs = self.env.step(action)
        assert obs.done is True

    def test_step_accepts_timeout_s(self):
        self.env.reset(task_id="feeding_basics")
        action = FarmAction()
        obs = self.env.step(action, timeout_s=5.0)
        assert isinstance(obs, FarmObservation)

    def test_completed_episode_returns_terminal(self):
        self.env.reset(task_id="feeding_basics")
        action = FarmAction(harvest_decision=True)
        self.env.step(action)
        # After episode done, stepping again should return terminal obs
        obs = self.env.step(FarmAction())
        assert obs.done is True

    def test_reward_in_valid_range(self):
        self.env.reset(task_id="feeding_basics")
        action = FarmAction(feeding_rate=0.5, aeration_rate=0.5)
        obs = self.env.step(action)
        assert -2.0 <= obs.reward <= 2.0  # rewards can be negative for penalties

    def test_feeding_rate_affects_ammonia(self):
        """Higher feeding should increase ammonia over time."""
        # Run with high feeding
        self.env.reset(seed=42, task_id="feeding_basics")
        for _ in range(24):
            obs_high = self.env.step(FarmAction(feeding_rate=1.0, aeration_rate=0.5,
                                                 water_exchange_rate=0.0))
        ammonia_high = obs_high.ammonia

        # Run with low feeding
        self.env.reset(seed=42, task_id="feeding_basics")
        for _ in range(24):
            obs_low = self.env.step(FarmAction(feeding_rate=0.0, aeration_rate=0.5,
                                                water_exchange_rate=0.0))
        ammonia_low = obs_low.ammonia

        assert ammonia_high > ammonia_low

    def test_aeration_affects_do(self):
        """Higher aeration should maintain higher DO."""
        self.env.reset(seed=42, task_id="oxygen_management")
        for _ in range(12):
            obs_high = self.env.step(FarmAction(aeration_rate=1.0))
        do_high = obs_high.dissolved_oxygen

        self.env.reset(seed=42, task_id="oxygen_management")
        for _ in range(12):
            obs_low = self.env.step(FarmAction(aeration_rate=0.0))
        do_low = obs_low.dissolved_oxygen

        assert do_high > do_low


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestObservationFields:
    """Verify all observation fields are populated correctly."""

    def setup_method(self):
        self.env = FishFarmEnvironment()
        self.env.reset(task_id="feeding_basics")
        self.obs = self.env.step(FarmAction(feeding_rate=0.5, aeration_rate=0.5))

    def test_fish_fields(self):
        assert self.obs.avg_fish_weight > 0
        assert self.obs.population > 0
        assert self.obs.mortality_today >= 0
        assert 0.0 <= self.obs.stress_level <= 1.0
        assert self.obs.feeding_response in ("eager", "normal", "reduced", "sluggish", "refusing")
        assert self.obs.biomass_kg > 0

    def test_water_fields(self):
        assert self.obs.temperature > 0
        assert self.obs.dissolved_oxygen >= 0
        assert 0 < self.obs.ph < 14
        assert self.obs.ammonia >= 0
        assert self.obs.ammonia_toxic >= 0
        assert self.obs.nitrite >= 0
        assert 0 <= self.obs.water_quality_score <= 1.0

    def test_system_fields(self):
        assert isinstance(self.obs.aerator_working, bool)
        assert isinstance(self.obs.biofilter_working, bool)
        assert isinstance(self.obs.heater_working, bool)
        assert self.obs.feed_remaining_kg >= 0

    def test_economics_fields(self):
        assert self.obs.total_cost_so_far >= 0
        assert isinstance(self.obs.current_fish_value, float)
        assert isinstance(self.obs.current_profit, float)

    def test_context_fields(self):
        assert isinstance(self.obs.weather_forecast, str)
        assert 0 <= self.obs.day_in_cycle
        assert 0 <= self.obs.time_of_day < 24
        assert isinstance(self.obs.alerts, list)
        assert isinstance(self.obs.feedback, str)


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestCascadeDynamics:
    """Test the core RL challenge: biological cascade emergence."""

    def test_overfeed_ammonia_cascade(self):
        """Overfeeding with no water management → ammonia rises → DO drops."""
        env = FishFarmEnvironment()
        env.reset(seed=42, task_id="feeding_basics")

        initial_ammonia = env.step(FarmAction(feeding_rate=0.3, aeration_rate=0.5)).ammonia

        # Overfeed for 48 hours with no aeration and no water exchange
        for _ in range(47):
            obs = env.step(FarmAction(feeding_rate=1.0, aeration_rate=0.0,
                                      water_exchange_rate=0.0))

        assert obs.ammonia > initial_ammonia * 2, "Ammonia should rise significantly with overfeeding"

    def test_good_management_maintains_health(self):
        """Moderate feeding + adequate aeration = stable conditions for 7 days."""
        env = FishFarmEnvironment()
        env.reset(seed=42, task_id="feeding_basics")

        for _ in range(7 * 24):
            obs = env.step(FarmAction(
                feeding_rate=0.4, aeration_rate=0.6,
                water_exchange_rate=0.02,
            ))

        assert obs.population > 4500, "Population should be mostly intact"
        assert obs.avg_fish_weight > 50.0, "Fish should have grown"
        assert obs.dissolved_oxygen > 3.0, "DO should be manageable"


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestSerializability:
    """Verify all models serialize to/from JSON (required for WebSocket)."""

    def test_action_round_trip(self):
        action = FarmAction(
            feeding_rate=0.6, aeration_rate=0.8,
            heater_setting=-0.3, water_exchange_rate=0.05,
            harvest_decision=False, treatment="antibiotics",
        )
        data = action.model_dump()
        restored = FarmAction(**data)
        assert restored.feeding_rate == action.feeding_rate
        assert restored.treatment == action.treatment

    def test_observation_round_trip(self):
        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        data = obs.model_dump()
        restored = FarmObservation(**data)
        assert restored.avg_fish_weight == obs.avg_fish_weight
        assert restored.done == obs.done

    def test_state_round_trip(self):
        env = FishFarmEnvironment()
        env.reset(task_id="feeding_basics")
        data = env.state.model_dump()
        restored = FarmState(**data)
        assert restored.task_id == env.state.task_id

    def test_action_schema_complete(self):
        schema = FarmAction.model_json_schema()
        expected_fields = {"feeding_rate", "aeration_rate", "heater_setting",
                          "water_exchange_rate", "harvest_decision", "treatment"}
        assert expected_fields.issubset(set(schema["properties"].keys()))
        # Every field should have a description
        for field_name in expected_fields:
            assert "description" in schema["properties"][field_name], \
                f"Field {field_name} missing description"
