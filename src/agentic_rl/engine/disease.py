"""SEIR disease model for fish populations.

Source: KB-03 Sec 4.1-4.2, KB-01 Sec 7

Implements the standard SEIR compartmental epidemic model:
    dS/dt = -β × S × I / N  + ω × R           (new infections + immunity waning)
    dE/dt =  β × S × I / N  - σ × E            (exposed → infected)
    dI/dt =  σ × E  - γ × I  - α × I           (progression, recovery, death)
    dR/dt =  γ × I  - ω × R                    (recovery - immunity loss)

Enhancements over base model:
- Differentiated treatment types (antibiotics, salt, probiotics)
- Immunity waning (recovered fish lose immunity over weeks)
- Density-dependent transmission coefficient β
- Stress-modulated disease progression
- Multiple disease phases with severity tracking
- Treatment side effects (antibiotics harm biofilter)
"""

import math
from ..constants import DISEASE


class DiseaseEngine:
    """SEIR compartmental disease model for aquaculture.

    The key biological insight: fish under environmental stress (low DO, high UIA,
    temperature extremes) become immunocompromised, making disease outbreaks
    far more likely and severe. This creates the cascade:
    poor water quality → stress → disease → mortality → reduced biomass →
    changed water quality dynamics.
    """

    def __init__(self):
        self.susceptible: int = 0
        self.exposed: int = 0
        self.infected: int = 0
        self.recovered: int = 0
        self.is_active: bool = False
        self.treatment_active: bool = False
        self.treatment_type: str = "none"
        self.treatment_days_remaining: float = 0.0
        self.total_disease_deaths: int = 0
        self.outbreak_count: int = 0
        self.disease_severity: float = 0.0  # 0.0 = none, 1.0 = catastrophic
        self.days_since_outbreak: float = 0.0

        # Immunity waning rate: recovered → susceptible over ~30 days
        self.immunity_waning_rate: float = 1.0 / 30.0  # omega (per day)

    def reset(self, population: int):
        """Reset disease state for new episode."""
        self.susceptible = population
        self.exposed = 0
        self.infected = 0
        self.recovered = 0
        self.is_active = False
        self.treatment_active = False
        self.treatment_type = "none"
        self.treatment_days_remaining = 0.0
        self.total_disease_deaths = 0
        self.outbreak_count = 0
        self.disease_severity = 0.0
        self.days_since_outbreak = 0.0

    def trigger_outbreak(self, initial_infected: int = 5):
        """Initiate a disease outbreak.

        Moves a small number of susceptible fish directly to infected status,
        simulating pathogen introduction (e.g., from feed, water intake, birds).
        """
        initial_infected = min(initial_infected, self.susceptible)
        if initial_infected <= 0:
            return
        self.susceptible -= initial_infected
        self.infected = initial_infected
        self.is_active = True
        self.outbreak_count += 1
        self.days_since_outbreak = 0.0

    def apply_treatment(self, treatment_type: str = "antibiotics"):
        """Start a treatment course.

        Treatment types and their effects:
        - "antibiotics": Boosts recovery rate γ by 2×, but harms biofilter bacteria
          (reduces nitrification efficiency). Most effective against bacterial infections.
        - "salt": NaCl bath (2-3 ppt). Reduces nitrite toxicity by competing for
          gill chloride cells. Mild disease treatment, no biofilter harm.
        - "probiotics": Competitive exclusion of pathogens. Slower acting but
          enhances biofilter and has no negative side effects. Boosts γ by 1.3×.
        - "vaccination": Preventive measure. Moves susceptible fish to recovered
          (immune) directly. Expensive but prevents future outbreaks. Only works
          when disease is NOT active (prophylactic use). KB-03 Sec 4.2 shows
          80% immunization reduces mortality from 42,900 to 900.
        """
        if treatment_type == "vaccination":
            # Vaccination: move 80% of susceptible directly to recovered
            # Only effective as prophylaxis or early intervention
            vaccinated = int(self.susceptible * 0.8)
            self.susceptible -= vaccinated
            self.recovered += vaccinated
            self.treatment_active = True
            self.treatment_type = treatment_type
            self.treatment_days_remaining = 1  # one-time application
            return

        self.treatment_active = True
        self.treatment_type = treatment_type
        self.treatment_days_remaining = DISEASE.treatment_duration_days

    def step(self, dt_hours: float, population: int, stress_level: float = 0.0,
             temperature: float = 28.0) -> int:
        """Advance disease model by one time step.

        SEIR differential equations discretized with Euler method.
        Stress level and temperature modulate disease severity.

        Temperature effects on pathogen virulence (KB-03 Sec 4.2):
        - Bacterial pathogens grow faster in warm water (25-35°C optimal)
        - Below 20°C: pathogens less virulent, slower progression
        - Above 35°C: some pathogens thrive, fish immune system compromised

        Args:
            dt_hours: Time step (hours).
            population: Current total fish population.
            stress_level: Fish stress index [0.0, 1.0] from FishBiologyEngine.
            temperature: Water temperature (°C) for pathogen virulence.

        Returns:
            Number of disease-induced deaths this step.
        """
        if not self.is_active and self.recovered == 0:
            return 0

        dt_days = dt_hours / 24.0
        N = max(1, self.susceptible + self.exposed + self.infected + self.recovered)

        # Base SEIR parameters
        beta = DISEASE.beta
        sigma = DISEASE.sigma
        gamma = DISEASE.gamma
        alpha = DISEASE.alpha
        omega = self.immunity_waning_rate

        # Temperature-dependent pathogen virulence
        # Most aquaculture bacterial pathogens (Aeromonas, Streptococcus) are
        # mesophilic: optimal growth 25-35°C, reduced below 20°C
        if temperature < 20.0:
            # Cold slows pathogen replication
            temp_virulence = 0.5 + 0.5 * (temperature - 10.0) / 10.0
            temp_virulence = max(0.3, temp_virulence)
        elif temperature <= 35.0:
            # Optimal range: virulence peaks around 30°C
            temp_virulence = 0.8 + 0.2 * math.exp(-0.1 * (temperature - 30.0) ** 2)
        else:
            # Very warm: pathogen AND host both stressed
            temp_virulence = 1.2
        beta *= temp_virulence
        sigma *= temp_virulence

        # Stress modulates disease parameters
        # Higher stress → faster transmission and progression, slower recovery
        if stress_level > 0.3:
            stress_mod = 1.0 + (stress_level - 0.3) * 2.0
            beta *= stress_mod           # faster transmission when stressed
            sigma *= (1.0 + stress_level * 0.5)  # faster E→I progression
            alpha *= stress_mod          # higher disease mortality when stressed
            gamma *= max(0.5, 1.0 - stress_level * 0.3)  # slower recovery

        # Treatment effects on recovery rate
        if self.treatment_active:
            gamma *= self._treatment_gamma_multiplier()

        # Density-dependent transmission (KB-03 Sec 4.2)
        # Higher density → more contact → higher effective β
        density = N / max(1.0, SYSTEM_VOLUME_PLACEHOLDER())
        density_factor = min(2.0, max(0.5, density / 50.0))
        beta *= density_factor

        # ---- SEIR transitions (integer arithmetic) ----
        # S → E: new exposures
        if self.infected > 0 and self.susceptible > 0:
            infection_prob = beta * self.infected / N * dt_days
            new_exposed = int(self.susceptible * min(1.0, infection_prob))
        else:
            new_exposed = 0

        # E → I: incubation complete
        new_infected = int(sigma * self.exposed * dt_days)

        # I → R: recovery
        new_recovered = int(gamma * self.infected * dt_days)

        # I → dead: disease mortality
        disease_deaths = int(alpha * self.infected * dt_days)

        # R → S: immunity waning (recovered lose immunity over ~30 days)
        immunity_lost = int(omega * self.recovered * dt_days) if self.recovered > 0 else 0

        # Clamp transitions to available compartment sizes
        new_exposed = min(new_exposed, self.susceptible)
        new_infected = min(new_infected, self.exposed)
        disease_deaths = min(disease_deaths, self.infected)
        new_recovered = min(new_recovered, self.infected - disease_deaths)
        immunity_lost = min(immunity_lost, self.recovered)

        # Apply transitions
        self.susceptible += -new_exposed + immunity_lost
        self.exposed += new_exposed - new_infected
        self.infected += new_infected - new_recovered - disease_deaths
        self.recovered += new_recovered - immunity_lost

        # Enforce non-negative
        self.susceptible = max(0, self.susceptible)
        self.exposed = max(0, self.exposed)
        self.infected = max(0, self.infected)
        self.recovered = max(0, self.recovered)

        self.total_disease_deaths += disease_deaths

        # Update disease severity (infected fraction of population)
        if N > 0:
            infected_fraction = (self.infected + self.exposed) / N
            self.disease_severity = min(1.0, infected_fraction * 5.0)
        else:
            self.disease_severity = 0.0

        # Track outbreak duration
        if self.is_active:
            self.days_since_outbreak += dt_days

        # Treatment countdown
        if self.treatment_active:
            self.treatment_days_remaining -= dt_days
            if self.treatment_days_remaining <= 0:
                self.treatment_active = False
                self.treatment_type = "none"

        # Check if outbreak resolved
        if self.infected == 0 and self.exposed == 0:
            self.is_active = False

        return disease_deaths

    def _treatment_gamma_multiplier(self) -> float:
        """Recovery rate multiplier based on treatment type.

        Returns a multiplier for the base recovery rate γ.
        """
        if self.treatment_type == "antibiotics":
            # Most effective: 2× recovery speed
            # But has side effect: harms biofilter (handled in simulator)
            return DISEASE.treatment_recovery_boost  # 2.0
        elif self.treatment_type == "salt":
            # Mild treatment: 1.3× recovery
            # Also reduces nitrite toxicity (handled in water quality)
            return 1.3
        elif self.treatment_type == "probiotics":
            # Slow acting: 1.5× recovery
            # Also helps biofilter (positive side effect)
            return 1.5
        else:
            return 1.0

    def get_biofilter_impact(self) -> float:
        """Return biofilter efficiency modifier from treatment.

        Antibiotics kill nitrifying bacteria, reducing biofilter effectiveness.
        Probiotics slightly enhance biofilter.

        Returns:
            Multiplier for biofilter efficiency (0.0 to 1.2).
        """
        if not self.treatment_active:
            return 1.0
        if self.treatment_type == "antibiotics":
            return 0.5  # 50% biofilter efficiency during antibiotic treatment
        elif self.treatment_type == "probiotics":
            return 1.1  # slight enhancement
        return 1.0

    def check_stress_trigger(self, stress_level, DO, UIA, temperature,
                             stocking_density, rng_value):
        """Check if environmental stress triggers a disease outbreak.

        Multiple stressors multiplicatively increase outbreak probability.
        This is the key coupling: poor water quality → disease risk.

        Args:
            stress_level: Composite stress index [0.0, 1.0].
            DO: Dissolved oxygen (mg/L).
            UIA: Unionized ammonia (mg/L).
            temperature: Water temperature (°C).
            stocking_density: fish/m³.
            rng_value: Random number [0, 1) for stochastic trigger.

        Returns:
            True if outbreak was triggered, False otherwise.
        """
        if self.is_active:
            return False

        # Base outbreak probability per hour
        prob = DISEASE.outbreak_prob_per_hour  # 0.0005 → ~1.2%/day

        # Stressor multipliers (compound multiplicatively)
        if DO < DISEASE.stress_DO_threshold:
            # Low DO immunocompromises fish
            do_severity = (DISEASE.stress_DO_threshold - DO) / DISEASE.stress_DO_threshold
            prob *= (1.0 + 4.0 * do_severity)
        if UIA > DISEASE.stress_ammonia_threshold:
            # Ammonia damages gill epithelium, opening pathogen entry
            uia_severity = min(3.0, (UIA - DISEASE.stress_ammonia_threshold) / DISEASE.stress_ammonia_threshold)
            prob *= (1.0 + 3.0 * uia_severity)
        if abs(temperature - 30.0) > DISEASE.stress_temp_deviation:
            # Temperature extremes suppress immune function
            prob *= 2.0
        if stocking_density > DISEASE.stress_density_threshold:
            # High density increases pathogen contact rate
            density_excess = (stocking_density - DISEASE.stress_density_threshold) / DISEASE.stress_density_threshold
            prob *= (1.0 + 2.0 * density_excess)
        if stress_level > 0.4:
            # General stress is immunosuppressive
            prob *= (1.0 + stress_level * 4.0)

        if rng_value < prob:
            # Initial infected count: 0.1% of susceptible, minimum 1
            initial = max(1, int(self.susceptible * 0.001))
            self.trigger_outbreak(initial_infected=initial)
            return True
        return False

    def sync_population(self, total_population: int):
        """Synchronize SEIR compartments with actual fish population.

        Called after non-disease mortality events to keep compartment
        counts consistent with the actual population.
        """
        current = self.susceptible + self.exposed + self.infected + self.recovered
        if current <= 0:
            self.susceptible = total_population
            return
        if current > total_population and current > 0:
            ratio = total_population / current
            self.susceptible = int(self.susceptible * ratio)
            self.exposed = int(self.exposed * ratio)
            self.infected = int(self.infected * ratio)
            # Remainder goes to recovered to ensure exact count
            self.recovered = total_population - self.susceptible - self.exposed - self.infected
            self.recovered = max(0, self.recovered)
        elif current < total_population:
            # More fish than tracked (shouldn't happen, but be safe)
            self.susceptible += total_population - current

    @property
    def R0(self) -> float:
        """Basic reproduction number R₀ = β / (γ + α + μ).

        R₀ > 1: epidemic will spread
        R₀ < 1: disease will die out
        """
        denominator = DISEASE.gamma + DISEASE.alpha + DISEASE.mu
        if denominator > 0:
            return DISEASE.beta / denominator
        return 0.0


def SYSTEM_VOLUME_PLACEHOLDER():
    """Get system tank volume. Avoids circular import."""
    try:
        from ..constants import SYSTEM
        return SYSTEM.tank_volume_m3
    except ImportError:
        return 100.0
