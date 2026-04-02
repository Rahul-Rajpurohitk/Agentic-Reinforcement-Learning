"""Integration tests for the full simulator."""
import pytest
from agentic_rl.engine.simulator import FishFarmSimulator


class TestSimulatorBasics:
    def test_reset_creates_valid_state(self):
        sim = FishFarmSimulator(seed=42)
        state = sim.reset()
        assert state["fish"]["weight_g"] > 0
        assert state["fish"]["population"] > 0
        assert state["water"]["DO"] > 0
        assert state["water"]["temperature"] > 0

    def test_step_advances_time(self):
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        state = sim.step(feeding_rate=0.5, aeration_rate=0.5,
                        heater_setting=0.0, water_exchange_rate=0.01,
                        harvest=False, treatment="none")
        assert state["time"]["hour"] == 1

    def test_24_hours_equals_one_day(self):
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        for _ in range(24):
            state = sim.step(0.5, 0.5, 0.0, 0.01, False, "none")
        assert state["time"]["day"] == 1

    def test_overfeeding_causes_ammonia_rise(self):
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        initial_tan = sim.water.TAN
        for _ in range(48):  # 2 days of overfeeding
            sim.step(1.0, 0.3, 0.0, 0.0, False, "none")  # max feed, low aeration, no exchange
        assert sim.water.TAN > initial_tan

    def test_no_aeration_causes_do_drop(self):
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        for _ in range(12):  # 12 hours nighttime without aeration
            sim.step(0.0, 0.0, 0.0, 0.0, False, "none")
        assert sim.water.DO < 7.0  # should drop from initial

    def test_fish_grow_over_time(self):
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        initial_weight = sim.fish.weight_g
        for _ in range(24 * 7):  # 1 week
            sim.step(0.5, 0.5, 0.0, 0.02, False, "none")
        assert sim.fish.weight_g > initial_weight

    def test_harvest_ends_episode(self):
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        state = sim.step(0.5, 0.5, 0.0, 0.01, True, "none")  # harvest=True
        assert state["harvested"] is True

    def test_mass_mortality_is_catastrophe(self):
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        # Force lethal conditions
        sim.water.DO = 0.5
        sim.water.TAN = 5.0
        sim.water.temperature = 40.0
        state = sim.step(0.0, 0.0, 0.0, 0.0, False, "none")
        assert state["fish"]["mortality_today"] > 0

    def test_cascade_overfeed_to_mortality(self):
        """The signature RL challenge: overfeed -> ammonia -> DO crash -> deaths."""
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        # Heavy overfeeding for 3 days with no aeration or exchange
        for _ in range(72):
            sim.step(1.0, 0.0, 0.0, 0.0, False, "none")
        # Should see elevated ammonia and reduced survival
        assert sim.water.TAN > 1.0 or sim.fish.population < 10000

    def test_state_includes_enhanced_economics(self):
        """State dict should include new economics fields from engine enhancement."""
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        state = sim.step(0.5, 0.5, 0.0, 0.02, False, "none")
        econ = state["economics"]
        assert "feed_price_per_kg" in econ
        assert "marginal_cost_per_hour" in econ
        assert "roi_pct" in econ
        assert econ["feed_price_per_kg"] > 0

    def test_stochastic_feed_price_varies(self):
        """Feed price should vary stochastically over time."""
        sim = FishFarmSimulator(seed=42)
        sim.reset()
        prices = []
        for _ in range(48):
            state = sim.step(0.5, 0.5, 0.0, 0.02, False, "none")
            prices.append(state["economics"]["feed_price_per_kg"])
        # Price should not be perfectly constant (OU process adds noise)
        assert len(set(prices)) > 1

    def test_seasonal_price_varies_by_day(self):
        """Market price multiplier should reflect seasonal demand."""
        from agentic_rl.engine.economics import EconomicsEngine
        econ = EconomicsEngine()
        econ.reset()

        econ.apply_seasonal_price(day_of_year=360)  # Christmas → premium
        xmas_price = econ.market_price_multiplier

        econ.market_price_multiplier = 1.0  # reset
        econ.apply_seasonal_price(day_of_year=180)  # mid-year → dip
        midyear_price = econ.market_price_multiplier

        assert xmas_price > midyear_price

    def test_vaccination_treatment_option(self):
        """Vaccination should move susceptible fish to recovered."""
        from agentic_rl.engine.disease import DiseaseEngine
        de = DiseaseEngine()
        de.reset(population=10000)
        initial_susceptible = de.susceptible

        de.apply_treatment("vaccination")
        assert de.recovered > 0
        assert de.susceptible < initial_susceptible
        # 80% should be vaccinated
        assert de.recovered >= int(initial_susceptible * 0.79)

    def test_temperature_affects_disease_virulence(self):
        """Disease should progress differently at different temperatures."""
        from agentic_rl.engine.disease import DiseaseEngine
        de_warm = DiseaseEngine()
        de_warm.reset(population=10000)
        de_warm.trigger_outbreak(50)

        de_cold = DiseaseEngine()
        de_cold.reset(population=10000)
        de_cold.trigger_outbreak(50)

        # Run for 5 days
        for _ in range(120):
            de_warm.step(1.0, de_warm.susceptible + de_warm.exposed +
                        de_warm.infected + de_warm.recovered, temperature=30.0)
            de_cold.step(1.0, de_cold.susceptible + de_cold.exposed +
                        de_cold.infected + de_cold.recovered, temperature=15.0)

        # Warm conditions should produce more disease deaths
        assert de_warm.total_disease_deaths >= de_cold.total_disease_deaths


class TestObservationCompleteness:
    """Verify the FarmObservation includes all enhanced fields."""

    def test_observation_has_fish_growth_fields(self):
        """Observation should include FCR, SGR, growth rate, stocking density."""
        from agentic_rl.server.environment import FishFarmEnvironment
        from agentic_rl.models import FarmAction
        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        assert hasattr(obs, "fcr")
        assert hasattr(obs, "sgr")
        assert hasattr(obs, "growth_rate_g_day")
        assert hasattr(obs, "stocking_density")

    def test_observation_has_economics_fields(self):
        """Observation should include stochastic feed price, ROI, marginal cost."""
        from agentic_rl.server.environment import FishFarmEnvironment
        from agentic_rl.models import FarmAction
        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        step_obs = env.step(FarmAction(feeding_rate=0.5, aeration_rate=0.5))
        assert hasattr(step_obs, "feed_price_per_kg")
        assert hasattr(step_obs, "market_price_multiplier")
        assert hasattr(step_obs, "marginal_cost_per_hour")
        assert hasattr(step_obs, "roi_pct")
        assert step_obs.feed_price_per_kg > 0

    def test_observation_has_weather_fields(self):
        """Observation should include daytime, storm, humidity."""
        from agentic_rl.server.environment import FishFarmEnvironment
        from agentic_rl.models import FarmAction
        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        assert hasattr(obs, "is_daytime")
        assert hasattr(obs, "storm_active")
        assert hasattr(obs, "humidity")

    def test_observation_has_disease_signal(self):
        """Observation should have disease_suspected (behavioral indicator)."""
        from agentic_rl.server.environment import FishFarmEnvironment
        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        assert hasattr(obs, "disease_suspected")
        # No disease initially
        assert obs.disease_suspected is False

    def test_observation_has_survival_fields(self):
        """Observation should include cumulative mortality and survival rate."""
        from agentic_rl.server.environment import FishFarmEnvironment
        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        assert hasattr(obs, "cumulative_mortality")
        assert hasattr(obs, "survival_rate")
        assert obs.survival_rate == 1.0

    def test_observation_has_nitrate_and_algae(self):
        """Observation should include NO3 and algae bloom status."""
        from agentic_rl.server.environment import FishFarmEnvironment
        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        assert hasattr(obs, "nitrate")
        assert hasattr(obs, "algae_bloom")


class TestHeuristicAgent:
    """Test the rule-based heuristic fallback agent."""

    def test_heuristic_reduces_feed_on_low_do(self):
        from inference import heuristic_action
        obs = {"dissolved_oxygen": 2.0, "ammonia_toxic": 0.01,
               "temperature": 28.0, "stress_level": 0.3,
               "feeding_response": "sluggish", "avg_fish_weight": 100.0,
               "population": 5000, "feed_remaining_kg": 200.0,
               "biofilter_working": True, "aerator_working": True,
               "disease_suspected": False, "is_daytime": True,
               "market_price_multiplier": 1.0}
        action = heuristic_action(obs, "feeding_basics", 10, 168)
        assert action["feeding_rate"] <= 0.2
        assert action["aeration_rate"] == 1.0  # emergency DO

    def test_heuristic_treats_disease(self):
        from inference import heuristic_action
        obs = {"dissolved_oxygen": 6.0, "ammonia_toxic": 0.01,
               "temperature": 28.0, "stress_level": 0.5,
               "feeding_response": "sluggish", "avg_fish_weight": 200.0,
               "population": 5000, "feed_remaining_kg": 200.0,
               "biofilter_working": True, "aerator_working": True,
               "disease_suspected": True, "mortality_today": 15,
               "is_daytime": True, "market_price_multiplier": 1.0}
        action = heuristic_action(obs, "disease_outbreak", 50, 240)
        assert action["treatment"] == "antibiotics"

    def test_heuristic_harvests_at_market_weight(self):
        from inference import heuristic_action
        obs = {"dissolved_oxygen": 7.0, "ammonia_toxic": 0.01,
               "temperature": 28.0, "stress_level": 0.1,
               "feeding_response": "eager", "avg_fish_weight": 550.0,
               "population": 5000, "feed_remaining_kg": 200.0,
               "biofilter_working": True, "aerator_working": True,
               "disease_suspected": False, "is_daytime": True,
               "market_price_multiplier": 1.15, "mortality_today": 0}
        action = heuristic_action(obs, "full_growout", 1400, 1440)
        assert action["harvest_decision"] is True

    def test_heuristic_heats_cold_water(self):
        from inference import heuristic_action
        obs = {"dissolved_oxygen": 7.0, "ammonia_toxic": 0.01,
               "temperature": 22.0, "stress_level": 0.2,
               "feeding_response": "normal", "avg_fish_weight": 100.0,
               "population": 5000, "feed_remaining_kg": 200.0,
               "biofilter_working": True, "aerator_working": True,
               "disease_suspected": False, "is_daytime": True,
               "market_price_multiplier": 1.0, "mortality_today": 0}
        action = heuristic_action(obs, "temperature_stress", 10, 120)
        assert action["heater_setting"] > 0

    def test_heuristic_increases_exchange_for_high_ammonia(self):
        from inference import heuristic_action
        obs = {"dissolved_oxygen": 6.0, "ammonia_toxic": 0.15, "ammonia": 2.5,
               "temperature": 28.0, "stress_level": 0.3,
               "feeding_response": "sluggish", "avg_fish_weight": 150.0,
               "population": 5000, "feed_remaining_kg": 200.0,
               "biofilter_working": True, "aerator_working": True,
               "disease_suspected": False, "is_daytime": True,
               "market_price_multiplier": 1.0, "mortality_today": 0}
        action = heuristic_action(obs, "ammonia_crisis", 10, 72)
        assert action["water_exchange_rate"] >= 0.05
