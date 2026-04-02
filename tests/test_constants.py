"""Tests for constants module — utility functions and parameter sanity."""
import pytest
import math
from agentic_rl.constants import (
    TILAPIA, WATER, DISEASE, ECONOMICS, SYSTEM,
    uia_fraction, do_saturation, photoperiod_hours,
    temperature_factor, do_factor, uia_factor,
)


class TestTilapiaParams:
    def test_temperature_range_valid(self):
        assert TILAPIA.T_min < TILAPIA.T_opt < TILAPIA.T_max
        assert TILAPIA.T_lethal_low < TILAPIA.T_min
        assert TILAPIA.T_max < TILAPIA.T_lethal_high

    def test_growth_exponents_valid(self):
        assert 0 < TILAPIA.m < 1  # anabolism exponent
        assert 0 < TILAPIA.n < 1  # catabolism exponent
        assert TILAPIA.m < TILAPIA.n  # catabolism scales faster (biological constraint)

    def test_fcr_target_realistic(self):
        assert 1.0 < TILAPIA.fcr_target < 3.0

    def test_do_thresholds_ordered(self):
        assert TILAPIA.DO_lethal < TILAPIA.DO_stress < TILAPIA.DO_optimal


class TestWaterParams:
    def test_uia_thresholds_ordered(self):
        assert WATER.UIA_safe < WATER.UIA_crit < WATER.UIA_lethal

    def test_ph_range_valid(self):
        assert WATER.pH_lethal_low < WATER.pH_min < WATER.pH_max < WATER.pH_lethal_high

    def test_no2_thresholds_ordered(self):
        assert WATER.NO2_safe < WATER.NO2_stress < WATER.NO2_lethal


class TestUIAFraction:
    def test_uia_at_neutral_ph(self):
        f = uia_fraction(7.0, 25.0)
        assert 0.0 < f < 0.05  # small fraction at neutral pH

    def test_uia_increases_with_ph(self):
        f_low = uia_fraction(7.0, 25.0)
        f_high = uia_fraction(9.0, 25.0)
        assert f_high > f_low * 10  # exponential increase

    def test_uia_increases_with_temperature(self):
        f_cold = uia_fraction(8.0, 15.0)
        f_warm = uia_fraction(8.0, 35.0)
        assert f_warm > f_cold

    def test_uia_bounded(self):
        f = uia_fraction(14.0, 40.0)  # extreme
        assert 0 < f <= 1.0


class TestDOSaturation:
    def test_decreases_with_temperature(self):
        assert do_saturation(15.0) > do_saturation(30.0)

    def test_realistic_values(self):
        assert 7.0 < do_saturation(25.0) < 9.0
        assert 6.5 < do_saturation(30.0) < 8.0

    def test_always_positive(self):
        assert do_saturation(40.0) > 0


class TestPhotoperiodHours:
    def test_tropical_roughly_12h(self):
        """At 10° latitude, daylight is roughly 12 hours year-round."""
        for day in [1, 90, 180, 270]:
            p = photoperiod_hours(day, 10.0)
            assert 11.0 < p < 13.0

    def test_higher_latitude_more_variation(self):
        """At 45° latitude, summer should have more daylight than winter."""
        summer = photoperiod_hours(172, 45.0)  # June solstice
        winter = photoperiod_hours(355, 45.0)  # December solstice
        assert summer > winter + 4.0

    def test_equator_constant_12h(self):
        p = photoperiod_hours(90, 0.0)
        assert abs(p - 12.0) < 0.1


class TestTemperatureFactor:
    def test_optimal_near_one(self):
        tau = temperature_factor(TILAPIA.T_opt)
        assert tau > 0.95

    def test_zero_at_extremes(self):
        assert temperature_factor(TILAPIA.T_min) == 0.0
        assert temperature_factor(TILAPIA.T_max) == 0.0

    def test_below_min_is_zero(self):
        assert temperature_factor(10.0) == 0.0

    def test_above_max_is_zero(self):
        assert temperature_factor(45.0) == 0.0

    def test_symmetric_decay(self):
        """Temperature response should be roughly symmetric around T_opt."""
        t_below = temperature_factor(TILAPIA.T_opt - 3)
        t_above = temperature_factor(TILAPIA.T_opt + 3)
        # Not exactly symmetric (different T_min/T_max), but both should be high
        assert t_below > 0.5
        assert t_above > 0.5


class TestDOFactor:
    def test_above_critical_is_one(self):
        assert do_factor(8.0) == 1.0

    def test_below_min_is_zero(self):
        assert do_factor(2.0) == 0.0

    def test_linear_interpolation(self):
        mid = (WATER.DO_crit + WATER.DO_min) / 2
        sigma = do_factor(mid)
        assert 0.4 < sigma < 0.6


class TestUIAFactorFunction:
    def test_below_critical_is_one(self):
        assert uia_factor(0.01) == 1.0

    def test_above_lethal_is_zero(self):
        assert uia_factor(1.0) == 0.0

    def test_intermediate_value(self):
        mid_uia = (WATER.UIA_crit + WATER.UIA_lethal) / 2
        v = uia_factor(mid_uia)
        assert 0.3 < v < 0.7


class TestEconomicsParams:
    def test_feed_cheaper_than_fish(self):
        assert ECONOMICS.feed_price_per_kg < ECONOMICS.market_price_per_kg

    def test_positive_costs(self):
        assert ECONOMICS.fixed_cost_per_day > 0
        assert ECONOMICS.electricity_cost_per_kwh > 0
        assert ECONOMICS.fingerling_cost > 0


class TestSystemParams:
    def test_tank_dimensions(self):
        assert SYSTEM.tank_volume_m3 > 0
        assert SYSTEM.tank_depth_m > 0
        # Surface area should be reasonable
        surface = SYSTEM.tank_volume_m3 / SYSTEM.tank_depth_m
        assert 50 < surface < 200  # 50-200 m² is reasonable for 100m³ tank

    def test_sub_steps_reasonable(self):
        assert SYSTEM.sub_steps >= 4  # at least 4 sub-steps for stability
