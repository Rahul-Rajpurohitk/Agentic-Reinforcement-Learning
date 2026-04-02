import pytest
from agentic_rl.engine.fish_biology import FishBiologyEngine
from agentic_rl.constants import TILAPIA, WATER


class TestGrowthModel:
    def setup_method(self):
        self.bio = FishBiologyEngine()
        self.bio.reset(weight_g=50.0, population=10000, day_of_year=1)

    def test_fish_grow_with_optimal_conditions(self):
        initial_weight = self.bio.weight_g
        self.bio.grow(dt_hours=24.0, feeding_rate=0.5, temperature=TILAPIA.T_opt,
                      DO=7.0, UIA=0.01, photoperiod_h=12.0)
        assert self.bio.weight_g > initial_weight

    def test_no_growth_without_feeding(self):
        initial_weight = self.bio.weight_g
        self.bio.grow(dt_hours=24.0, feeding_rate=0.0, temperature=TILAPIA.T_opt,
                      DO=7.0, UIA=0.01, photoperiod_h=12.0)
        assert self.bio.weight_g < initial_weight

    def test_growth_faster_at_optimal_temp(self):
        bio_opt = FishBiologyEngine()
        bio_opt.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_opt.grow(24.0, 0.5, TILAPIA.T_opt, 7.0, 0.01, 12.0)
        bio_cold = FishBiologyEngine()
        bio_cold.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_cold.grow(24.0, 0.5, 22.0, 7.0, 0.01, 12.0)
        assert bio_opt.weight_g > bio_cold.weight_g

    def test_growth_reduced_by_low_do(self):
        bio_good = FishBiologyEngine()
        bio_good.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_good.grow(24.0, 0.5, 30.0, 7.0, 0.01, 12.0)
        bio_low_do = FishBiologyEngine()
        bio_low_do.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_low_do.grow(24.0, 0.5, 30.0, 3.5, 0.01, 12.0)
        assert bio_good.weight_g > bio_low_do.weight_g

    def test_growth_reduced_by_high_ammonia(self):
        bio_clean = FishBiologyEngine()
        bio_clean.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_clean.grow(24.0, 0.5, 30.0, 7.0, 0.01, 12.0)
        bio_toxic = FishBiologyEngine()
        bio_toxic.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_toxic.grow(24.0, 0.5, 30.0, 7.0, 0.3, 12.0)
        assert bio_clean.weight_g > bio_toxic.weight_g

    def test_weight_stays_positive(self):
        self.bio.grow(240.0, 0.0, 20.0, 2.0, 0.5, 12.0)
        assert self.bio.weight_g > 0


class TestMortality:
    def setup_method(self):
        self.bio = FishBiologyEngine()
        self.bio.reset(weight_g=100.0, population=10000, day_of_year=1)

    def test_no_mortality_optimal_conditions(self):
        deaths = self.bio.apply_mortality(
            dt_hours=24.0, DO=7.0, UIA=0.01, temperature=30.0, stocking_density=50.0
        )
        assert deaths <= 10

    def test_high_mortality_under_stress(self):
        deaths = self.bio.apply_mortality(
            dt_hours=24.0, DO=1.0, UIA=0.5, temperature=38.0, stocking_density=100.0
        )
        assert deaths > 50

    def test_population_never_negative(self):
        self.bio.apply_mortality(24.0, 0.5, 1.0, 40.0, 200.0)
        assert self.bio.population >= 0


class TestStressLevel:
    def setup_method(self):
        self.bio = FishBiologyEngine()
        self.bio.reset(weight_g=100.0, population=10000, day_of_year=1)

    def test_zero_stress_optimal(self):
        stress = self.bio.calculate_stress(
            DO=7.0, UIA=0.01, temperature=30.0, stocking_density=50.0
        )
        assert stress < 0.15

    def test_high_stress_bad_conditions(self):
        stress = self.bio.calculate_stress(
            DO=2.0, UIA=0.3, temperature=36.0, stocking_density=100.0
        )
        assert stress > 0.5


class TestRespirationModel:
    """Test the dual respiration model (tilapia polynomial + allometric fallback)."""

    def setup_method(self):
        self.bio = FishBiologyEngine()

    def test_tilapia_model_used_in_valid_range(self):
        """Tilapia polynomial should be used for 20-200g fish at 24-32°C."""
        self.bio.reset(weight_g=100.0, population=10000, day_of_year=1)
        rate = self.bio.respiration_rate(28.0)
        # Polynomial: 2014.45 + 2.75*100 - 165.2*28 + 0.007*10000 + 3.93*784 - 0.21*100*28
        # = 2014.45 + 275 - 4625.6 + 70 + 3081.12 - 588 = 226.97
        assert 100 < rate < 500  # reasonable range for tilapia

    def test_allometric_model_used_outside_range(self):
        """General model should be used for fish outside 20-200g or 24-32°C."""
        self.bio.reset(weight_g=300.0, population=10000, day_of_year=1)
        rate = self.bio.respiration_rate(28.0)
        assert rate > 50  # should still return a valid positive rate

    def test_respiration_increases_with_temperature(self):
        self.bio.reset(weight_g=100.0, population=10000, day_of_year=1)
        rate_cool = self.bio.respiration_rate(25.0)
        rate_warm = self.bio.respiration_rate(31.0)
        assert rate_warm > rate_cool

    def test_respiration_always_positive(self):
        self.bio.reset(weight_g=5.0, population=10000, day_of_year=1)
        rate = self.bio.respiration_rate(20.0)
        assert rate >= 50.0


class TestSizeFeedingFactor:
    """Test that size-dependent feeding rate is properly applied."""

    def test_fingerlings_grow_faster_per_bw(self):
        """Fingerlings (<10g) should eat a higher % of body weight than adults."""
        small = FishBiologyEngine()
        small.reset(weight_g=5.0, population=10000, day_of_year=1)
        big = FishBiologyEngine()
        big.reset(weight_g=300.0, population=10000, day_of_year=1)

        assert small._size_feeding_factor(5.0) > big._size_feeding_factor(300.0)

    def test_size_factor_affects_growth(self):
        """Large fish should grow proportionally less per feeding_rate=1.0."""
        small = FishBiologyEngine()
        small.reset(weight_g=8.0, population=10000, day_of_year=1)
        big = FishBiologyEngine()
        big.reset(weight_g=250.0, population=10000, day_of_year=1)

        dw_small = small.grow(24.0, 1.0, 30.0, 7.0, 0.01, 12.0)
        dw_big = big.grow(24.0, 1.0, 30.0, 7.0, 0.01, 12.0)

        # Smaller fish with higher feeding factor should grow more relative to body weight
        sgr_small = dw_small / 8.0
        sgr_big = dw_big / 250.0
        assert sgr_small > sgr_big


class TestFeedingResponse:
    def setup_method(self):
        self.bio = FishBiologyEngine()
        self.bio.reset(weight_g=100.0, population=10000, day_of_year=1)

    def test_eager_feeding_optimal(self):
        response = self.bio.feeding_response(
            temperature=30.0, DO=7.0, UIA=0.01, stress=0.1
        )
        assert response in ("eager", "normal")

    def test_refusing_feed_high_stress(self):
        response = self.bio.feeding_response(
            temperature=38.0, DO=2.0, UIA=0.4, stress=0.8
        )
        assert response == "refusing"
