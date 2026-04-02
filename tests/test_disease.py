"""Tests for the SEIR disease model engine."""
import pytest
from agentic_rl.engine.disease import DiseaseEngine
from agentic_rl.constants import DISEASE


class TestSEIRDynamics:
    def setup_method(self):
        self.de = DiseaseEngine()
        self.de.reset(population=10000)

    def test_initial_state_all_susceptible(self):
        assert self.de.susceptible == 10000
        assert self.de.infected == 0
        assert self.de.exposed == 0
        assert self.de.recovered == 0
        assert self.de.is_active is False

    def test_trigger_outbreak_moves_to_infected(self):
        self.de.trigger_outbreak(initial_infected=50)
        assert self.de.infected == 50
        assert self.de.susceptible == 9950
        assert self.de.is_active is True

    def test_outbreak_spreads_over_time(self):
        self.de.trigger_outbreak(initial_infected=50)
        for _ in range(24 * 5):  # 5 days
            self.de.step(1.0, 10000, stress_level=0.0, temperature=28.0)
        # Should have some exposed/infected spread
        assert self.de.exposed > 0 or self.de.recovered > 0

    def test_disease_causes_deaths(self):
        self.de.trigger_outbreak(initial_infected=100)
        total_deaths = 0
        for _ in range(24 * 10):  # 10 days
            deaths = self.de.step(1.0, max(1, self.de.susceptible + self.de.exposed +
                                           self.de.infected + self.de.recovered),
                                  stress_level=0.0, temperature=28.0)
            total_deaths += deaths
        assert total_deaths > 0

    def test_disease_resolves_eventually(self):
        """Without reinfection, disease should eventually clear."""
        self.de.trigger_outbreak(initial_infected=10)
        for _ in range(24 * 60):  # 60 days
            self.de.step(1.0, max(1, self.de.susceptible + self.de.exposed +
                                  self.de.infected + self.de.recovered),
                         stress_level=0.0, temperature=28.0)
        # Either resolved or very few infected (immunity waning can sustain low-level)
        assert self.de.infected <= 20

    def test_r0_calculated(self):
        """Basic reproduction number should be positive."""
        assert self.de.R0 > 0

    def test_r0_above_one_means_epidemic(self):
        """With default params, R0 > 1 means disease can spread."""
        assert self.de.R0 > 1.0


class TestTreatmentEffects:
    def setup_method(self):
        self.de = DiseaseEngine()
        self.de.reset(population=10000)
        self.de.trigger_outbreak(initial_infected=100)

    def test_antibiotics_boost_recovery(self):
        self.de.apply_treatment("antibiotics")
        assert self.de.treatment_active is True
        assert self.de.treatment_type == "antibiotics"
        mult = self.de._treatment_gamma_multiplier()
        assert mult == DISEASE.treatment_recovery_boost  # 2.0

    def test_salt_treatment(self):
        self.de.apply_treatment("salt")
        mult = self.de._treatment_gamma_multiplier()
        assert mult == 1.3

    def test_probiotics_treatment(self):
        self.de.apply_treatment("probiotics")
        mult = self.de._treatment_gamma_multiplier()
        assert mult == 1.5

    def test_antibiotics_harm_biofilter(self):
        self.de.apply_treatment("antibiotics")
        impact = self.de.get_biofilter_impact()
        assert impact < 1.0  # reduces efficiency

    def test_probiotics_help_biofilter(self):
        self.de.apply_treatment("probiotics")
        impact = self.de.get_biofilter_impact()
        assert impact > 1.0  # enhances efficiency

    def test_treatment_expires(self):
        self.de.apply_treatment("antibiotics")
        # Run for longer than treatment duration
        for _ in range(24 * (DISEASE.treatment_duration_days + 1)):
            self.de.step(1.0, 10000, stress_level=0.0, temperature=28.0)
        assert self.de.treatment_active is False

    def test_vaccination_moves_susceptible_to_recovered(self):
        de = DiseaseEngine()
        de.reset(population=10000)
        initial_s = de.susceptible
        de.apply_treatment("vaccination")
        assert de.recovered >= int(initial_s * 0.79)
        assert de.susceptible < initial_s

    def test_vaccination_prevents_future_outbreak(self):
        """Vaccinated population should have smaller outbreaks."""
        # Unvaccinated
        de_unvac = DiseaseEngine()
        de_unvac.reset(population=10000)
        de_unvac.trigger_outbreak(50)
        for _ in range(24 * 20):
            de_unvac.step(1.0, 10000, stress_level=0.2, temperature=28.0)

        # Vaccinated before outbreak
        de_vac = DiseaseEngine()
        de_vac.reset(population=10000)
        de_vac.apply_treatment("vaccination")  # 80% immune
        de_vac.trigger_outbreak(50)
        for _ in range(24 * 20):
            de_vac.step(1.0, 10000, stress_level=0.2, temperature=28.0)

        assert de_vac.total_disease_deaths <= de_unvac.total_disease_deaths


class TestStressTrigger:
    def setup_method(self):
        self.de = DiseaseEngine()
        self.de.reset(population=10000)

    def test_no_trigger_under_optimal_conditions(self):
        triggered = self.de.check_stress_trigger(
            stress_level=0.0, DO=7.0, UIA=0.01, temperature=28.0,
            stocking_density=50.0, rng_value=0.9,
        )
        assert triggered is False

    def test_high_stress_increases_trigger_prob(self):
        """With high stress and low rng, should trigger outbreak."""
        triggered = self.de.check_stress_trigger(
            stress_level=0.9, DO=1.5, UIA=0.3, temperature=38.0,
            stocking_density=100.0, rng_value=0.001,
        )
        assert triggered is True
        assert self.de.is_active is True


class TestImmunityWaning:
    def test_recovered_lose_immunity_over_time(self):
        de = DiseaseEngine()
        de.reset(population=10000)
        # Manually set recovered state
        de.recovered = 5000
        de.susceptible = 5000
        de.is_active = False

        initial_recovered = de.recovered
        # Run for 60 days without active disease
        for _ in range(24 * 60):
            de.step(1.0, 10000, stress_level=0.0, temperature=28.0)

        # Some recovered should have lost immunity
        assert de.recovered < initial_recovered
        assert de.susceptible > 5000


class TestTemperatureVirulence:
    def test_cold_reduces_virulence(self):
        """Pathogens should be less virulent in cold water."""
        # Run two identical outbreaks at different temperatures
        de_warm = DiseaseEngine()
        de_warm.reset(population=10000)
        de_warm.trigger_outbreak(50)

        de_cold = DiseaseEngine()
        de_cold.reset(population=10000)
        de_cold.trigger_outbreak(50)

        for _ in range(24 * 10):
            pop_warm = de_warm.susceptible + de_warm.exposed + de_warm.infected + de_warm.recovered
            pop_cold = de_cold.susceptible + de_cold.exposed + de_cold.infected + de_cold.recovered
            de_warm.step(1.0, max(1, pop_warm), stress_level=0.0, temperature=30.0)
            de_cold.step(1.0, max(1, pop_cold), stress_level=0.0, temperature=15.0)

        # Warm water should have more disease progression
        assert de_warm.total_disease_deaths >= de_cold.total_disease_deaths


class TestPopulationSync:
    def test_sync_adjusts_compartments(self):
        de = DiseaseEngine()
        de.reset(population=10000)
        de.trigger_outbreak(100)
        # Simulate external mortality reducing population
        de.sync_population(8000)
        total = de.susceptible + de.exposed + de.infected + de.recovered
        assert total == 8000
