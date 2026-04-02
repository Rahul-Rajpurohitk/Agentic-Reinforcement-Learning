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
