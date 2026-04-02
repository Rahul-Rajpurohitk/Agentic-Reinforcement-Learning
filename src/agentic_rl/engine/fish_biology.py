"""Fish biology engine — bioenergetic growth, feeding, stress, mortality.

Core equation (KB-03 Sec 1.1, from KAUST Q-learning paper / FAO Annex 3):
    dW/dt = [Ψ(f,T,DO) × v(UIA) × W^m]  -  [k(T) × W^n]

Enhancements over base model:
- Size-dependent feeding rate (fingerlings eat 5% BW/day, adults 2%)
- Density-dependent growth reduction at high stocking
- Weight variance tracking (CV of fish weights in population)
- 24-hour accumulated mortality (not per-call overwrite)
- Specific Dynamic Action (SDA) metabolic cost of digestion
- Condition factor tracking (Fulton's K)

Source: FAO Fish Growth Model (Annex 3), arxiv:2306.09915, KB-01 Sec 1-9
"""

import math
from ..constants import (
    TILAPIA, WATER, SYSTEM, temperature_factor, do_factor, uia_factor
)


class FishBiologyEngine:
    """Manages fish population biology: growth, feeding, stress, mortality.

    The bioenergetic model captures the fundamental trade-off:
    higher feeding → faster growth BUT more ammonia excretion → water quality decline.
    """

    def __init__(self):
        self.weight_g: float = TILAPIA.w_initial
        self.population: int = TILAPIA.N_initial
        self.initial_population: int = TILAPIA.N_initial
        self.total_feed_consumed_kg: float = 0.0  # population-total (set by simulator)
        self.initial_biomass_kg: float = 0.0  # for FCR calculation
        self.mortality_today: int = 0
        self.cumulative_mortality: int = 0
        self.stress_level: float = 0.0
        self.day_of_year: int = 1
        self.growth_rate: float = 0.0

        # Enhanced tracking
        self._mortality_24h_buffer: list = []  # (hour, deaths) tuples
        self._current_hour: int = 0
        self.weight_cv: float = 0.10   # coefficient of variation of fish weights
        self.condition_factor: float = 1.0  # Fulton's K (relative body condition)
        self.sgr: float = 0.0         # specific growth rate %/day
        self._weight_history: list = []  # last 24 weights for SGR calc

    def reset(self, weight_g: float, population: int, day_of_year: int):
        """Reset fish biology to initial conditions."""
        self.weight_g = weight_g
        self.population = population
        self.initial_population = population
        self.total_feed_consumed_kg = 0.0
        self.initial_biomass_kg = weight_g * population / 1000.0
        self.mortality_today = 0
        self.cumulative_mortality = 0
        self.stress_level = 0.0
        self.day_of_year = day_of_year
        self.growth_rate = 0.0
        self._mortality_24h_buffer = []
        self._current_hour = 0
        self.weight_cv = 0.10
        self.condition_factor = 1.0
        self.sgr = 0.0
        self._weight_history = [weight_g]

    def grow(self, dt_hours, feeding_rate, temperature, DO, UIA, photoperiod_h):
        """Bioenergetic growth step.

        dW/dt = [h × π × f × b × (1-a) × τ(T) × σ(DO) × v(UIA)] × W^m
              - [k_min × exp(s × (T - T_min))] × W^n

        Includes:
        - Size-dependent max feeding rate (fingerlings eat more % BW)
        - Density-dependent growth depression at high stocking
        - Specific Dynamic Action (SDA) metabolic cost
        - Weight CV tracking (population size uniformity)

        Args:
            dt_hours: Time step in hours.
            feeding_rate: Normalized feeding rate 0..1.
            temperature: Water temperature (°C).
            DO: Dissolved oxygen (mg/L).
            UIA: Unionized ammonia (mg/L).
            photoperiod_h: Hours of daylight.

        Returns:
            dw — weight change (g) over this time step.
        """
        dt_days = dt_hours / 24.0
        w = max(self.weight_g, 0.1)
        f = max(0.0, min(1.0, feeding_rate))

        # Environmental response factors (KB-03 Sec 1.2)
        tau = temperature_factor(temperature)
        sigma = do_factor(DO)
        v = uia_factor(UIA)
        pi = photoperiod_h / 12.0

        # Size-dependent actual feeding rate
        # Fingerlings (<10g): up to 5% BW/day
        # Juveniles (10-100g): 3-4% BW/day
        # Adults (>100g): 2-3% BW/day
        size_feeding_multiplier = self._size_feeding_factor(w)

        # Effective feeding rate (stress reduces appetite, size adjusts max intake)
        stress_appetite = max(0.2, 1.0 - self.stress_level * 0.8)
        f_effective = f * stress_appetite * size_feeding_multiplier

        # Anabolism: H × W^m
        H = (TILAPIA.h * pi * f_effective * TILAPIA.b * (1.0 - TILAPIA.a)
             * tau * sigma * v)

        # Catabolism: k(T) × W^n
        if temperature > TILAPIA.T_min:
            k = TILAPIA.k_min * math.exp(TILAPIA.s * (temperature - TILAPIA.T_min))
        else:
            k = TILAPIA.k_min

        anabolism = H * (w ** TILAPIA.m)
        catabolism = k * (w ** TILAPIA.n)

        # SDA — Specific Dynamic Action: metabolic cost of digestion
        # ~15% of consumed energy goes to processing food (KB-01 Sec 1.6)
        sda_cost = 0.15 * anabolism * f_effective

        # Density-dependent growth depression
        # At high stocking (>60 fish/m³), growth reduces from competition/stress
        density = self.population / SYSTEM.tank_volume_m3
        density_growth_factor = self._density_growth_factor(density)

        dw_dt = (anabolism - catabolism - sda_cost) * density_growth_factor

        dw = dw_dt * dt_days
        self.weight_g = max(0.1, self.weight_g + dw)
        self.growth_rate = dw_dt

        # Update SGR (Specific Growth Rate) — over 24h window
        self._weight_history.append(self.weight_g)
        if len(self._weight_history) > 25:
            self._weight_history = self._weight_history[-25:]
        if len(self._weight_history) >= 24:
            w_start = self._weight_history[-24]
            if w_start > 0.1:
                self.sgr = (math.log(self.weight_g) - math.log(w_start)) * 100

        # Update weight CV (size variation increases with stress, decreases with good conditions)
        if stress_appetite < 0.5:
            self.weight_cv = min(0.30, self.weight_cv + 0.001 * dt_days)
        else:
            self.weight_cv = max(0.05, self.weight_cv - 0.0005 * dt_days)

        # Update condition factor (Fulton's K = W / L³ × 100)
        # Using W = a_wl × L^b_wl → L = (W/a_wl)^(1/b_wl)
        if self.weight_g > 0.1:
            L = (self.weight_g / TILAPIA.a_wl) ** (1.0 / TILAPIA.b_wl)
            self.condition_factor = (self.weight_g / (L ** 3)) * 100.0

        # Advance hour counter
        self._current_hour += 1

        return dw

    def apply_mortality(self, dt_hours, DO, UIA, temperature, stocking_density):
        """Apply mortality for a time step.

        Mortality sources:
        1. Natural background mortality (size-dependent)
        2. Stress-amplified mortality (exponential above threshold)
        3. Acute lethal events (extreme DO / UIA / temperature)

        Uses a 24-hour rolling buffer to track daily mortality accurately.

        Args:
            dt_hours: Time step in hours.
            DO: Dissolved oxygen (mg/L).
            UIA: Unionized ammonia (mg/L).
            temperature: Water temperature (°C).
            stocking_density: fish/m³.

        Returns:
            Number of fish that died this step.
        """
        dt_days = dt_hours / 24.0
        stress = self.calculate_stress(DO, UIA, temperature, stocking_density)
        self.stress_level = stress

        # Size-dependent natural mortality (KB-03 Sec 5.1)
        # Smaller fish have higher natural mortality
        # M_nat ~ a_M × L^(-b_M), simplified to weight-based
        if self.weight_g < 10:
            base_rate = TILAPIA.base_mortality * 2.0  # fingerlings: 2× mortality
        elif self.weight_g < 50:
            base_rate = TILAPIA.base_mortality * 1.3
        else:
            base_rate = TILAPIA.base_mortality

        # Stress amplification (quadratic above 0.3 threshold)
        if stress > 0.3:
            stress_multiplier = 1.0 + 50.0 * ((stress - 0.3) ** 2)
        else:
            stress_multiplier = 1.0

        # Acute lethal events — immediate population fraction killed
        acute_mortality = 0.0
        if DO < WATER.DO_lethal:
            # Below 1.0 mg/L: 5% per day, severity scales with how far below
            severity = max(0, (WATER.DO_lethal - DO) / WATER.DO_lethal)
            acute_mortality += (0.05 + 0.15 * severity) * dt_days
        if UIA > WATER.UIA_lethal:
            # Above 0.6 mg/L: 10% per day base
            severity = min(2.0, (UIA - WATER.UIA_lethal) / WATER.UIA_lethal)
            acute_mortality += (0.10 + 0.10 * severity) * dt_days
        if temperature > TILAPIA.T_max:
            # Above 39.7°C: rapid death
            severity = min(2.0, (temperature - TILAPIA.T_max) / 3.0)
            acute_mortality += (0.15 + 0.20 * severity) * dt_days
        elif temperature < TILAPIA.T_lethal_low:
            # Below 11°C: cold shock
            severity = min(2.0, (TILAPIA.T_lethal_low - temperature) / 5.0)
            acute_mortality += (0.10 + 0.15 * severity) * dt_days

        total_rate = base_rate * stress_multiplier * dt_days + acute_mortality
        total_rate = min(total_rate, 0.5)  # cap at 50% per step

        deaths = int(self.population * total_rate)
        deaths = min(deaths, self.population)
        self.population = max(0, self.population - deaths)

        # Rolling 24-hour mortality buffer
        self._mortality_24h_buffer.append((self._current_hour, deaths))
        # Prune entries older than 24 hours
        cutoff = self._current_hour - 24
        self._mortality_24h_buffer = [
            (h, d) for h, d in self._mortality_24h_buffer if h > cutoff
        ]
        self.mortality_today = sum(d for _, d in self._mortality_24h_buffer)

        self.cumulative_mortality += deaths
        return deaths

    def calculate_stress(self, DO, UIA, temperature, stocking_density):
        """Composite stress index in [0.0, 1.0].

        Weighted sum of four stressors (KB-03 Sec 15.6):
          DO (35%), UIA (30%), temperature (20%), density (15%).

        Each stressor uses species-specific thresholds with smooth transitions.
        """
        # --- DO stress: inversely proportional to DO availability ---
        if DO >= WATER.DO_optimal:
            do_stress = 0.0
        elif DO >= WATER.DO_min:
            do_stress = 1.0 - (DO - WATER.DO_min) / (WATER.DO_optimal - WATER.DO_min)
        elif DO >= WATER.DO_lethal:
            do_stress = 1.0
        else:
            do_stress = 1.0  # already in lethal zone

        # --- UIA stress: proportional to toxicity ---
        if UIA <= WATER.UIA_safe:
            uia_stress = 0.0
        elif UIA <= WATER.UIA_crit:
            uia_stress = (UIA - WATER.UIA_safe) / (WATER.UIA_crit - WATER.UIA_safe) * 0.5
        elif UIA <= WATER.UIA_lethal:
            uia_stress = 0.5 + 0.5 * (UIA - WATER.UIA_crit) / (WATER.UIA_lethal - WATER.UIA_crit)
        else:
            uia_stress = 1.0

        # --- Temperature stress: deviation from optimal ---
        temp_dev = abs(temperature - TILAPIA.T_opt)
        if temp_dev <= 3.0:
            temp_stress = 0.0
        elif temp_dev <= 10.0:
            temp_stress = (temp_dev - 3.0) / 7.0
        else:
            temp_stress = 1.0

        # --- Density stress: above 50 fish/m³ starts causing problems ---
        if stocking_density <= 50:
            density_stress = 0.0
        elif stocking_density <= 80:
            density_stress = (stocking_density - 50) / 30.0 * 0.5
        elif stocking_density <= 120:
            density_stress = 0.5 + (stocking_density - 80) / 40.0 * 0.5
        else:
            density_stress = 1.0

        composite = (0.35 * do_stress + 0.30 * uia_stress
                     + 0.20 * temp_stress + 0.15 * density_stress)
        return min(1.0, composite)

    def feeding_response(self, temperature, DO, UIA, stress):
        """Qualitative feeding behaviour label.

        Returns one of: "eager", "normal", "reduced", "sluggish", "refusing".
        Based on stress level, DO, temperature, and UIA thresholds.
        """
        if self.population == 0:
            return "refusing"
        if stress > 0.7 or DO < WATER.DO_lethal or temperature > TILAPIA.T_max:
            return "refusing"
        elif stress > 0.5 or DO < WATER.DO_min or UIA > WATER.UIA_crit:
            return "sluggish"
        elif stress > 0.3 or DO < WATER.DO_crit:
            return "reduced"
        elif stress < 0.1 and temperature_factor(temperature) > 0.8:
            return "eager"
        else:
            return "normal"

    def _size_feeding_factor(self, weight_g: float) -> float:
        """Size-dependent feeding rate adjustment.

        Fingerlings eat a higher percentage of body weight per day.
        KB-02 Sec 3 / KB-03 Sec 3.2:
          <10g: 5% BW/day (max_feeding_pct is already 5%)
          10-50g: 4% BW/day → multiplier 0.8
          50-200g: 3% BW/day → multiplier 0.6
          >200g: 2% BW/day → multiplier 0.4
        """
        if weight_g < 10:
            return 1.0
        elif weight_g < 50:
            return 0.8
        elif weight_g < 200:
            return 0.6
        else:
            return 0.4

    def _density_growth_factor(self, density: float) -> float:
        """Density-dependent growth reduction.

        g(N) = g_max × (1 - N/K_pop) from KB-03 Sec 5.1
        At moderate density (<60/m³): no effect
        At high density (>80/m³): growth reduced up to 30%
        """
        if density <= 60:
            return 1.0
        elif density <= 120:
            return 1.0 - 0.3 * (density - 60) / 60.0
        else:
            return 0.7

    def respiration_rate(self, temperature: float) -> float:
        """Fish respiration rate (mg O2/kg/h) using the best available model.

        Uses the tilapia-specific polynomial model (R²=0.99, KB-03 Sec 2.1)
        when within its valid range (20-200g, 24-32°C), falling back to the
        general allometric model otherwise.

        FR_tilapia = 2014.45 + 2.75W - 165.2T + 0.007W² + 3.93T² - 0.21WT
        FR_general = 10^(0.40 + 0.016T - 0.0006T² - 0.016·ln(W)) × 1000

        Args:
            temperature: Water temperature (°C).

        Returns:
            Respiration rate in mg O2/kg fish/h.
        """
        w = max(self.weight_g, 1.0)
        T = temperature

        # Tilapia-specific polynomial (high accuracy within valid range)
        if 20.0 <= w <= 200.0 and 24.0 <= T <= 32.0:
            FR = (2014.45 + 2.75 * w - 165.2 * T
                  + 0.007 * w ** 2 + 3.93 * T ** 2 - 0.21 * w * T)
            return max(50.0, FR)

        # General allometric model (broader range)
        X = 0.40 + 0.016 * T - 0.0006 * T ** 2 - 0.016 * math.log(w)
        return max(50.0, (10 ** X) * 1000)

    @property
    def biomass_kg(self):
        """Total live biomass (kg)."""
        return self.population * self.weight_g / 1000.0

    def record_feed(self, feed_kg: float):
        """Record feed consumed by the population (called by simulator).

        This is the single source of truth for feed consumption tracking.
        """
        self.total_feed_consumed_kg += feed_kg

    @property
    def fcr(self):
        """Feed conversion ratio (kg feed / kg biomass gain).

        FCR = Total feed consumed / Net biomass increase
        Good values: 1.2-1.8 for tilapia (KB-01 Sec 1.6).

        Uses population-level biomass change (current biomass vs initial),
        which accounts for mortality reducing the denominator.
        """
        net_biomass_gain = self.biomass_kg - self.initial_biomass_kg
        if net_biomass_gain > 0 and self.total_feed_consumed_kg > 0:
            return self.total_feed_consumed_kg / net_biomass_gain
        return 0.0

    @property
    def survival_rate(self):
        """Fraction of original population still alive."""
        if self.initial_population > 0:
            return self.population / self.initial_population
        return 0.0
