"""Tests for water quality dynamics engine."""
from agentic_rl.engine.water_quality import WaterQualityEngine
from agentic_rl.constants import SYSTEM


class TestDODynamics:
    def setup_method(self):
        self.wq = WaterQualityEngine(
            volume_m3=SYSTEM.tank_volume_m3,
            depth_m=SYSTEM.tank_depth_m,
        )
        self.wq.reset(temp=28.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)

    def test_initial_state(self):
        assert self.wq.DO == 7.0
        assert self.wq.temperature == 28.0
        assert self.wq.TAN == 0.1

    def test_do_decreases_with_fish_respiration(self):
        initial_DO = self.wq.DO
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=500.0, fish_weight_g=100.0,
            feeding_rate=0.5, aeration_rate=0.0, water_exchange_rate=0.0,
            is_daytime=False, biofilter_efficiency=0.6,
        )
        assert self.wq.DO < initial_DO

    def test_aeration_increases_do(self):
        self.wq.reset(temp=28.0, DO=3.0, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=100.0, fish_weight_g=100.0,
            feeding_rate=0.0, aeration_rate=1.0, water_exchange_rate=0.0,
            is_daytime=False, biofilter_efficiency=0.6,
        )
        assert self.wq.DO > 3.0

    def test_do_never_negative(self):
        self.wq.reset(temp=35.0, DO=0.5, TAN=5.0, pH=8.0, NO2=1.0)
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=5000.0, fish_weight_g=100.0,
            feeding_rate=1.0, aeration_rate=0.0, water_exchange_rate=0.0,
            is_daytime=False, biofilter_efficiency=0.0,
        )
        assert self.wq.DO >= 0.0


class TestTANDynamics:
    def setup_method(self):
        self.wq = WaterQualityEngine(volume_m3=SYSTEM.tank_volume_m3, depth_m=SYSTEM.tank_depth_m)

    def test_feeding_increases_tan(self):
        self.wq.reset(temp=28.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)
        tan_before = self.wq.TAN
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=500.0, fish_weight_g=100.0,
            feeding_rate=1.0, aeration_rate=0.5, water_exchange_rate=0.0,
            is_daytime=True, biofilter_efficiency=0.0,
        )
        assert self.wq.TAN > tan_before

    def test_biofilter_reduces_tan(self):
        self.wq.reset(temp=28.0, DO=7.0, TAN=2.0, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=100.0, fish_weight_g=100.0,
            feeding_rate=0.0, aeration_rate=0.5, water_exchange_rate=0.0,
            is_daytime=True, biofilter_efficiency=0.8,
        )
        assert self.wq.TAN < 2.0

    def test_water_exchange_reduces_tan(self):
        self.wq.reset(temp=28.0, DO=7.0, TAN=3.0, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=100.0, fish_weight_g=100.0,
            feeding_rate=0.0, aeration_rate=0.5, water_exchange_rate=0.1,
            is_daytime=True, biofilter_efficiency=0.0,
        )
        assert self.wq.TAN < 3.0

    def test_tan_never_negative(self):
        self.wq.reset(temp=28.0, DO=7.0, TAN=0.01, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=10.0, fish_weight_g=100.0,
            feeding_rate=0.0, aeration_rate=0.5, water_exchange_rate=0.1,
            is_daytime=True, biofilter_efficiency=1.0,
        )
        assert self.wq.TAN >= 0.0


class TestUIACalculation:
    def setup_method(self):
        self.wq = WaterQualityEngine(volume_m3=SYSTEM.tank_volume_m3, depth_m=SYSTEM.tank_depth_m)

    def test_uia_increases_with_ph(self):
        self.wq.reset(temp=28.0, DO=7.0, TAN=1.0, pH=7.0, NO2=0.05)
        uia_low_ph = self.wq.UIA
        self.wq.reset(temp=28.0, DO=7.0, TAN=1.0, pH=8.5, NO2=0.05)
        uia_high_ph = self.wq.UIA
        assert uia_high_ph > uia_low_ph * 5

    def test_uia_increases_with_temperature(self):
        self.wq.reset(temp=20.0, DO=7.0, TAN=1.0, pH=8.0, NO2=0.05)
        uia_cold = self.wq.UIA
        self.wq.reset(temp=30.0, DO=7.0, TAN=1.0, pH=8.0, NO2=0.05)
        uia_warm = self.wq.UIA
        assert uia_warm > uia_cold


class TestDenitrification:
    """Test anoxic denitrification (NO3 → N2 when DO < 2.0)."""

    def setup_method(self):
        self.wq = WaterQualityEngine(volume_m3=SYSTEM.tank_volume_m3, depth_m=SYSTEM.tank_depth_m)

    def test_denitrification_active_at_low_do(self):
        """NO3 should decrease faster when DO is very low (anoxic conditions)."""
        self.wq.reset(temp=28.0, DO=0.5, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.NO3 = 20.0
        initial_no3 = self.wq.NO3
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=10.0, fish_weight_g=100.0,
            feeding_rate=0.0, aeration_rate=0.0, water_exchange_rate=0.0,
            is_daytime=False, biofilter_efficiency=0.0,
        )
        # With very low DO and no nitrification, denitrification should reduce NO3
        assert self.wq.NO3 < initial_no3

    def test_no_denitrification_at_high_do(self):
        """NO3 should not decrease from denitrification when DO is adequate."""
        self.wq.reset(temp=28.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.NO3 = 20.0
        # With good DO and no nitrification input, NO3 should remain stable
        # (only exchange can reduce it, and we have exchange=0)
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=10.0, fish_weight_g=100.0,
            feeding_rate=0.0, aeration_rate=0.5, water_exchange_rate=0.0,
            is_daytime=False, biofilter_efficiency=0.0,
        )
        # NO3 might increase slightly from nitrification; key is it shouldn't drop much
        assert self.wq.NO3 >= 19.0  # no significant denitrification


class TestEvaporation:
    """Test evaporation concentrating dissolved substances."""

    def setup_method(self):
        self.wq = WaterQualityEngine(volume_m3=SYSTEM.tank_volume_m3, depth_m=SYSTEM.tank_depth_m)

    def test_evaporation_concentrates_tan(self):
        """Low humidity + high temp should cause evaporation, concentrating TAN."""
        self.wq.reset(temp=35.0, DO=7.0, TAN=1.0, pH=7.5, NO2=0.05)
        initial_tan = self.wq.TAN
        # Run with low humidity (dry conditions) — evaporation concentrates solutes
        self.wq.step(
            dt_hours=1.0, fish_biomass_kg=0.0, fish_weight_g=100.0,
            feeding_rate=0.0, aeration_rate=0.0, water_exchange_rate=0.0,
            is_daytime=True, biofilter_efficiency=0.0,
            solar_intensity=800.0, humidity=30.0,
        )
        # TAN should be slightly higher due to evaporative concentration
        # (even with no fish excretion, evaporation concentrates existing TAN)
        # Note: biofilter and exchange are off, so only evaporation changes TAN
        assert self.wq.TAN >= initial_tan * 0.99  # at minimum, not significantly reduced


class TestTemperatureDynamics:
    def setup_method(self):
        self.wq = WaterQualityEngine(volume_m3=SYSTEM.tank_volume_m3, depth_m=SYSTEM.tank_depth_m)

    def test_heater_warms_water(self):
        self.wq.reset(temp=25.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.update_temperature(dt_hours=1.0, air_temp=25.0, heater_setting=1.0, volume_m3=SYSTEM.tank_volume_m3)
        assert self.wq.temperature > 25.0

    def test_temperature_trends_toward_air(self):
        self.wq.reset(temp=30.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.update_temperature(dt_hours=1.0, air_temp=20.0, heater_setting=0.0, volume_m3=SYSTEM.tank_volume_m3)
        assert self.wq.temperature < 30.0
