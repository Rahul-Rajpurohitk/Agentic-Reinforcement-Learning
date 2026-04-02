"""Tests for the economic model engine."""
from agentic_rl.engine.economics import EconomicsEngine
from agentic_rl.constants import ECONOMICS, DISEASE


class TestCostTracking:
    def setup_method(self):
        self.econ = EconomicsEngine()
        self.econ.reset(initial_population=10000)

    def test_fingerling_cost_on_reset(self):
        assert self.econ.total_fingerling_cost == 10000 * ECONOMICS.fingerling_cost

    def test_feed_cost_accumulates(self):
        self.econ.record_hourly_costs(feed_kg=5.0, aeration_rate=0.5,
                                       heater_setting=0.0, water_exchange_rate=0.02,
                                       tank_volume_m3=100.0)
        assert self.econ.total_feed_cost > 0
        assert self.econ.total_feed_kg == 5.0

    def test_energy_cost_from_aeration(self):
        self.econ.record_hourly_costs(feed_kg=0.0, aeration_rate=1.0,
                                       heater_setting=0.0, water_exchange_rate=0.0,
                                       tank_volume_m3=100.0)
        expected_kwh = ECONOMICS.aeration_power_kw * 1.0
        expected_cost = expected_kwh * ECONOMICS.electricity_cost_per_kwh
        assert abs(self.econ.total_energy_cost - expected_cost) < 0.01

    def test_energy_cost_from_heater(self):
        self.econ.record_hourly_costs(feed_kg=0.0, aeration_rate=0.0,
                                       heater_setting=1.0, water_exchange_rate=0.0,
                                       tank_volume_m3=100.0)
        assert self.econ.total_energy_kwh == ECONOMICS.heater_power_kw

    def test_water_cost(self):
        self.econ.record_hourly_costs(feed_kg=0.0, aeration_rate=0.0,
                                       heater_setting=0.0, water_exchange_rate=0.10,
                                       tank_volume_m3=100.0)
        expected = 10.0 * ECONOMICS.water_cost_per_m3
        assert abs(self.econ.total_water_cost - expected) < 0.01

    def test_fixed_cost_prorated_hourly(self):
        self.econ.record_hourly_costs(feed_kg=0.0, aeration_rate=0.0,
                                       heater_setting=0.0, water_exchange_rate=0.0,
                                       tank_volume_m3=100.0)
        expected = ECONOMICS.fixed_cost_per_day / 24.0
        assert abs(self.econ.total_fixed_cost - expected) < 0.01

    def test_total_cost_includes_all_components(self):
        self.econ.record_hourly_costs(5.0, 0.5, 0.5, 0.05, 100.0)
        self.econ.record_treatment("antibiotics")
        self.econ.record_mortality(10, 100.0)
        total = self.econ.total_cost
        assert total > 0
        # Should include all components
        assert total >= (self.econ.total_feed_cost + self.econ.total_energy_cost
                        + self.econ.total_fingerling_cost)

    def test_hours_operated_increments(self):
        for _ in range(5):
            self.econ.record_hourly_costs(1.0, 0.5, 0.0, 0.02, 100.0)
        assert self.econ.hours_operated == 5


class TestTreatmentCosts:
    def setup_method(self):
        self.econ = EconomicsEngine()
        self.econ.reset()

    def test_antibiotics_cost(self):
        self.econ.record_treatment("antibiotics")
        expected = DISEASE.treatment_cost_per_day / 24.0
        assert abs(self.econ.total_treatment_cost - expected) < 0.01

    def test_salt_cost(self):
        self.econ.record_treatment("salt")
        expected = 10.0 / 24.0
        assert abs(self.econ.total_treatment_cost - expected) < 0.01

    def test_probiotics_cost(self):
        self.econ.record_treatment("probiotics")
        expected = 30.0 / 24.0
        assert abs(self.econ.total_treatment_cost - expected) < 0.01

    def test_vaccination_cost(self):
        self.econ.record_treatment("vaccination")
        assert self.econ.total_treatment_cost == 100.0

    def test_unknown_treatment_no_cost(self):
        self.econ.record_treatment("unknown")
        assert self.econ.total_treatment_cost == 0.0


class TestFishValuation:
    def setup_method(self):
        self.econ = EconomicsEngine()
        self.econ.reset()

    def test_weight_premium_underweight(self):
        value = self.econ.calculate_fish_value(biomass_kg=100.0, avg_weight_g=100.0)
        base = 100.0 * ECONOMICS.market_price_per_kg * 0.8
        assert abs(value - base) < 1.0

    def test_weight_premium_market_weight(self):
        value_small = self.econ.calculate_fish_value(100.0, avg_weight_g=150.0)
        value_large = self.econ.calculate_fish_value(100.0, avg_weight_g=500.0)
        assert value_large > value_small

    def test_weight_premium_peak(self):
        value = self.econ.calculate_fish_value(100.0, avg_weight_g=700.0)
        base = 100.0 * ECONOMICS.market_price_per_kg * 1.2
        assert abs(value - base) < 1.0

    def test_harvest_revenue_positive(self):
        rev = self.econ.calculate_harvest_revenue(500.0)
        assert rev > 0


class TestProfitAndROI:
    def setup_method(self):
        self.econ = EconomicsEngine()
        self.econ.reset(initial_population=10000)

    def test_profit_can_be_positive(self):
        # Minimal costs, big fish
        self.econ.record_hourly_costs(1.0, 0.3, 0.0, 0.01, 100.0)
        profit = self.econ.profit(biomass_kg=500.0, avg_weight_g=400.0)
        assert profit > 0

    def test_profit_can_be_negative(self):
        # Lots of costs, small fish
        for _ in range(240):  # 10 days of operation
            self.econ.record_hourly_costs(10.0, 1.0, 1.0, 0.1, 100.0)
        profit = self.econ.profit(biomass_kg=10.0, avg_weight_g=50.0)
        assert profit < 0

    def test_roi_calculation(self):
        self.econ.record_hourly_costs(1.0, 0.3, 0.0, 0.01, 100.0)
        roi = self.econ.roi(biomass_kg=500.0, avg_weight_g=400.0)
        assert isinstance(roi, float)

    def test_roi_zero_when_no_cost(self):
        econ = EconomicsEngine()
        econ.reset()
        roi = econ.roi(biomass_kg=100.0)
        assert roi == 0.0


class TestStochasticFeedPrice:
    def test_feed_price_initializes_to_base(self):
        econ = EconomicsEngine()
        econ.reset()
        assert econ.feed_price_current == ECONOMICS.feed_price_per_kg

    def test_feed_price_changes_with_random(self):
        econ = EconomicsEngine()
        econ.reset()
        # With a positive shock, price should increase
        econ.record_hourly_costs(1.0, 0.0, 0.0, 0.0, 100.0, rng_value=3.0)
        assert econ.feed_price_current > ECONOMICS.feed_price_per_kg * 0.99

    def test_feed_price_bounded(self):
        econ = EconomicsEngine()
        econ.reset()
        # Apply extreme shocks
        for _ in range(100):
            econ.record_hourly_costs(0.0, 0.0, 0.0, 0.0, 100.0, rng_value=5.0)
        # Should be bounded to ±40% of mean
        assert econ.feed_price_current <= ECONOMICS.feed_price_per_kg * 1.41
        assert econ.feed_price_current >= ECONOMICS.feed_price_per_kg * 0.59

    def test_feed_price_mean_reverts(self):
        econ = EconomicsEngine()
        econ.reset()
        # Push price up
        econ.feed_price_current = ECONOMICS.feed_price_per_kg * 1.3
        # Run with zero shocks (mean reversion only)
        for _ in range(1000):
            econ.record_hourly_costs(0.0, 0.0, 0.0, 0.0, 100.0, rng_value=0.0)
        # Should revert toward mean (within 20% of original deviation)
        initial_deviation = ECONOMICS.feed_price_per_kg * 0.3
        current_deviation = abs(econ.feed_price_current - ECONOMICS.feed_price_per_kg)
        assert current_deviation < initial_deviation


class TestSeasonalPricing:
    def test_christmas_premium(self):
        econ = EconomicsEngine()
        econ.reset()
        econ.apply_seasonal_price(day_of_year=360)
        assert econ.market_price_multiplier == 1.15

    def test_lent_premium(self):
        econ = EconomicsEngine()
        econ.reset()
        econ.apply_seasonal_price(day_of_year=90)
        assert econ.market_price_multiplier == 1.10

    def test_midyear_dip(self):
        econ = EconomicsEngine()
        econ.reset()
        econ.apply_seasonal_price(day_of_year=180)
        assert econ.market_price_multiplier == 0.95

    def test_normal_season(self):
        econ = EconomicsEngine()
        econ.reset()
        econ.apply_seasonal_price(day_of_year=150)
        assert econ.market_price_multiplier == 1.0

    def test_event_override_prevents_seasonal(self):
        """If a price event already set multiplier != 1.0, seasonal shouldn't override."""
        econ = EconomicsEngine()
        econ.reset()
        econ.set_market_price(0.5)  # price crash event
        econ.apply_seasonal_price(day_of_year=360)
        assert econ.market_price_multiplier == 0.5  # event price preserved


class TestMarginalCost:
    def test_marginal_cost_tracked(self):
        econ = EconomicsEngine()
        econ.reset()
        econ.record_hourly_costs(5.0, 0.5, 0.5, 0.05, 100.0)
        assert econ.marginal_cost_per_hour > 0

    def test_marginal_cost_increases_with_usage(self):
        econ = EconomicsEngine()
        econ.reset()
        econ.record_hourly_costs(1.0, 0.1, 0.0, 0.0, 100.0)
        low_cost = econ.marginal_cost_per_hour

        econ.record_hourly_costs(10.0, 1.0, 1.0, 0.1, 100.0)
        high_cost = econ.marginal_cost_per_hour
        assert high_cost > low_cost


class TestCostBreakdown:
    def test_breakdown_sums_to_total(self):
        econ = EconomicsEngine()
        econ.reset(initial_population=5000)
        for _ in range(24):
            econ.record_hourly_costs(2.0, 0.5, 0.3, 0.02, 100.0)
        econ.record_treatment("antibiotics")
        econ.record_mortality(5, 100.0)

        breakdown = econ.cost_breakdown()
        component_sum = sum(v["amount"] for v in breakdown.values() if isinstance(v, dict))
        assert abs(component_sum - breakdown["total"]) < 0.1

    def test_breakdown_has_all_categories(self):
        econ = EconomicsEngine()
        econ.reset()
        breakdown = econ.cost_breakdown()
        for key in ["feed", "energy", "water", "fixed", "treatment", "fingerlings", "disposal"]:
            assert key in breakdown
