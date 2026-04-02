"""Biological, physical, and economic constants for the Fish Farm simulation.

ALL values sourced from the knowledge base:
- 01-BIOLOGY-AND-SCIENCE.md (species data, water quality thresholds)
- 02-REAL-WORLD-OPERATIONS.md (operational parameters, economics)
- 03-MATHEMATICAL-MODELS.md (equation parameters, MDP spec)

References inline as KB-XX-Section.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TilapiaParams:
    """Nile Tilapia (Oreochromis niloticus) biological parameters.
    Source: FAO Annex 3 Fish Growth Model + KB-03 Section 15.6
    """
    # Growth model (bioenergetic: dW/dt = H*W^m - k*W^n)
    h: float = 0.4768       # consumption coefficient (KB-03 Sec 1.1)
    m: float = 0.6277       # anabolism exponent (KB-03 Sec 1.1)
    n: float = 0.8373       # catabolism exponent (KB-03 Sec 1.1)
    b: float = 0.7108       # assimilation efficiency (KB-03 Sec 1.1)
    a: float = 0.0559       # feeding catabolism fraction (KB-03 Sec 1.1)
    k_min: float = 0.0104   # catabolism base rate g^(1-n)/d (KB-03 Sec 1.1)
    s: float = 0.0288       # catabolism temp sensitivity 1/C (KB-03 Sec 1.1)

    # Temperature (KB-01 Sec 6.2, KB-03 Sec 1.1)
    T_min: float = 18.7     # minimum survivable temp (C)
    T_opt: float = 32.4     # optimal growth temp (C)
    T_max: float = 39.7     # maximum survivable temp (C)
    T_lethal_low: float = 11.0   # lethal low (KB-01 Sec 16)
    T_lethal_high: float = 42.0  # lethal high (KB-01 Sec 16)

    # Stocking / lifecycle
    w_initial: float = 5.0       # initial fingerling weight (g)
    w_market: float = 500.0      # target market weight (g)
    N_initial: int = 10000       # default stocking count
    base_mortality: float = 0.0005  # natural daily mortality rate (KB-03 Sec 15.6)

    # Feeding
    fcr_target: float = 1.6      # target FCR (KB-02 Sec 3)
    max_feeding_pct: float = 5.0 # max % body weight/day for fingerlings (KB-01 Sec 5.2)
    protein_fraction: float = 0.40  # feed protein content (KB-02 Sec 3)
    n_wasted_fraction: float = 0.50 # fraction of consumed N not retained (KB-03 Sec 2.2)

    # DO requirements (KB-01 Sec 2.2)
    DO_optimal: float = 5.0    # mg/L, growth reduction begins below
    DO_stress: float = 3.0     # mg/L, significant stress
    DO_lethal: float = 1.0     # mg/L, fish die

    # Allometric: W = a_wl * L^b_wl
    a_wl: float = 0.0282      # weight-length coefficient
    b_wl: float = 3.0         # isometric growth


@dataclass(frozen=True)
class WaterParams:
    """Water quality parameters and thresholds.
    Source: KB-01 Sec 2-4, KB-03 Sec 2, KB-01 Sec 16
    """
    # Dissolved Oxygen thresholds (mg/L) — tilapia-specific
    DO_optimal: float = 5.0       # above this = no growth reduction
    DO_crit: float = 5.0          # growth reduction begins (KB-03 Sec 1.2)
    DO_min: float = 3.0           # growth ceases (KB-03 Sec 1.2)
    DO_lethal: float = 1.0        # fish die (KB-01 Sec 2.2)
    DO_saturation_30C: float = 7.54  # mg/L at 30C (KB-01 Sec 2.1)

    # Unionized Ammonia thresholds (mg/L) — KB-01 Sec 3.3, KB-03 Sec 1.2
    UIA_safe: float = 0.02        # safe zone
    UIA_crit: float = 0.05        # chronic stress begins
    UIA_lethal: float = 0.6       # lethal (KB-03 Sec 15.6)

    # TAN (mg/L)
    TAN_max: float = 5.0          # maximum before emergency (KB-03 Sec 15.6)

    # Nitrite (mg/L) — KB-01 Sec 3.4
    NO2_safe: float = 0.1
    NO2_stress: float = 0.5
    NO2_lethal: float = 5.0

    # pH — KB-01 Sec 4.1
    pH_min: float = 6.5
    pH_max: float = 8.5
    pH_lethal_low: float = 4.0
    pH_lethal_high: float = 11.0
    pH_default: float = 7.5

    # Nitrification — KB-01 Sec 3.1
    O2_per_TAN: float = 4.57      # g O2 per g TAN oxidized (KB-01 Sec 3.1)
    alkalinity_per_TAN: float = 7.14  # g CaCO3 per g TAN oxidized

    # Reaeration coefficient (1/h) — KB-03 Sec 2.1
    K_a_base: float = 0.04        # base reaeration at moderate wind

    # Biofilter removal efficiency (fraction) — KB-03 Sec 2.2
    biofilter_efficiency: float = 0.6


@dataclass(frozen=True)
class DiseaseParams:
    """SEIR disease model parameters.
    Source: KB-03 Sec 4, KB-01 Sec 7
    """
    # SEIR rates (per day) — KB-03 Sec 4.1-4.2
    beta: float = 0.4          # transmission coefficient
    sigma: float = 0.2         # 1/latent_period (5 day latent)
    gamma: float = 0.1         # recovery rate (10 day infectious)
    alpha: float = 0.05        # disease-induced mortality
    mu: float = 0.0005         # natural mortality (background)

    # Environmental triggers — KB-01 Sec 7
    stress_DO_threshold: float = 3.5     # DO below this increases disease risk
    stress_ammonia_threshold: float = 0.04  # UIA above this increases disease risk
    stress_temp_deviation: float = 5.0   # degrees from optimal that increases risk
    stress_density_threshold: float = 80.0  # fish/m3 that increases disease risk

    # Disease probability per hour when stressed
    outbreak_prob_per_hour: float = 0.0005  # ~1.2% chance per day under stress

    # Treatment effectiveness
    treatment_recovery_boost: float = 2.0  # multiplier on gamma during treatment
    treatment_cost_per_day: float = 50.0   # $/day
    treatment_duration_days: int = 5


@dataclass(frozen=True)
class EconomicsParams:
    """Economic parameters.
    Source: KB-02 Sec 7, KB-03 Sec 6
    """
    feed_price_per_kg: float = 0.50       # $/kg feed (KB-03 Sec 15.6)
    market_price_per_kg: float = 3.00     # $/kg fish (KB-03 Sec 15.6)
    fixed_cost_per_day: float = 10.0      # $/day operating (KB-03 Sec 15.6)
    harvest_cost_per_kg: float = 0.30     # $/kg harvested
    fingerling_cost: float = 0.05         # $/fingerling
    electricity_cost_per_kwh: float = 0.12
    aeration_power_kw: float = 2.0        # aerator power consumption
    heater_power_kw: float = 5.0
    water_cost_per_m3: float = 0.50


@dataclass(frozen=True)
class SystemParams:
    """Physical system parameters.
    Source: KB-02 Sec 14, KB-03 Sec 15.6
    """
    dt_hours: float = 1.0            # agent decision interval (hours)
    sub_steps: int = 10              # water quality sub-steps per hour (6 min each)
    tank_volume_m3: float = 100.0    # RAS tank volume
    tank_depth_m: float = 1.5        # tank depth
    initial_stocking_density: float = 50.0  # fish/m3

    # Aeration
    # Real RAS: 2kW aerator × 1.8 SAE = 3.6 kg O2/h ÷ 100m³ tank = 36 mg/L/h
    # We cap at 20 to represent a well-sized but not over-engineered system
    max_aeration_rate: float = 20.0  # mg O2/L/h at full power (KB-02 Sec 4)
    aerator_SAE: float = 1.8         # kg O2/kWh standard aeration efficiency

    # Water exchange
    max_exchange_rate: float = 0.10  # fraction of volume per hour
    incoming_water_DO: float = 7.0   # mg/L (fresh water supply)
    incoming_water_temp: float = 28.0
    incoming_water_TAN: float = 0.0

    # Biofilter
    biofilter_volume_m3: float = 5.0
    biofilter_VTR: float = 350.0     # g TAN/m3/d moving-bed reactor (KB-03 Sec 2.2)

    # Latitude for photoperiod (tropical)
    latitude: float = 10.0          # degrees N (tropical fish farm)


# ---- Singleton instances ----
TILAPIA = TilapiaParams()
WATER = WaterParams()
DISEASE = DiseaseParams()
ECONOMICS = EconomicsParams()
SYSTEM = SystemParams()


# ---- Utility functions ----

def uia_fraction(pH: float, temp_c: float) -> float:
    """Calculate fraction of TAN that is toxic unionized ammonia (NH3).

    Source: KB-01 Sec 3.2
    Fraction NH3 = 1 / (1 + 10^(pKa - pH))
    pKa = 0.09018 + 2729.92 / T_kelvin
    """
    T_kelvin = temp_c + 273.15
    pKa = 0.09018 + 2729.92 / T_kelvin
    fraction = 1.0 / (1.0 + 10.0 ** (pKa - pH))
    return fraction


def do_saturation(temp_c: float) -> float:
    """DO saturation concentration at given temperature (freshwater, sea level).

    Source: KB-03 Sec 2.1
    DO_sat(T) = 468 / (31.6 + T)  [approximate, mg/L]
    """
    return 468.0 / (31.6 + temp_c)


def photoperiod_hours(day_of_year: int, latitude: float = 10.0) -> float:
    """Calculate daylight hours from day of year and latitude.

    Source: KB-03 Sec 15.3 (photoperiod scalar pi = P_h / 12)
    """
    declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)

    cos_hour_angle = -math.tan(lat_rad) * math.tan(dec_rad)
    cos_hour_angle = max(-1.0, min(1.0, cos_hour_angle))
    hour_angle = math.degrees(math.acos(cos_hour_angle))
    return 2.0 * hour_angle / 15.0


def temperature_factor(T: float, T_min: float = TILAPIA.T_min,
                       T_opt: float = TILAPIA.T_opt,
                       T_max: float = TILAPIA.T_max) -> float:
    """Bell-shaped temperature response function tau(T).

    Source: KB-03 Sec 1.2
    tau(T) = exp{-4.6 * ((T_opt - T) / (T_opt - T_min))^4}  if T < T_opt
    tau(T) = exp{-4.6 * ((T - T_opt) / (T_max - T_opt))^4}  if T >= T_opt
    """
    if T <= T_min or T >= T_max:
        return 0.0
    if T < T_opt:
        x = (T_opt - T) / (T_opt - T_min)
    else:
        x = (T - T_opt) / (T_max - T_opt)
    return math.exp(-4.6 * x ** 4)


def do_factor(DO: float, DO_crit: float = WATER.DO_crit,
              DO_min: float = WATER.DO_min) -> float:
    """Piecewise linear DO response function sigma(DO).

    Source: KB-03 Sec 1.2
    """
    if DO >= DO_crit:
        return 1.0
    elif DO >= DO_min:
        return (DO - DO_min) / (DO_crit - DO_min)
    else:
        return 0.0


def uia_factor(UIA: float, UIA_crit: float = WATER.UIA_crit,
               UIA_max: float = WATER.UIA_lethal) -> float:
    """Piecewise linear UIA response function v(UIA).

    Source: KB-03 Sec 1.2
    """
    if UIA <= UIA_crit:
        return 1.0
    elif UIA <= UIA_max:
        return (UIA_max - UIA) / (UIA_max - UIA_crit)
    else:
        return 0.0
