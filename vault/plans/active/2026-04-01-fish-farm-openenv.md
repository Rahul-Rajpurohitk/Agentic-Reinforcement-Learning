# AquaRL: Smart Fish Farm OpenEnv Environment — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the world's first OpenEnv/Gymnasium-compatible aquaculture farming RL environment — a biologically accurate simulation of a Recirculating Aquaculture System (RAS) for Nile Tilapia, with 12 tasks spanning feeding, water quality, disease management, storm response, and full grow-out economics.

**Architecture:** Modular simulation engine (physics → biology → economics) wrapped in OpenEnv's `Environment[Action, Observation, State]` interface. The sim engine is pure Python with NumPy, no external ML dependencies. Each task configures the engine with different initial conditions, event schedules, and grading criteria. An LLM-powered inference agent reads natural-language observations and outputs structured actions via the OpenAI client.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, OpenEnv-core, NumPy, OpenAI client, Docker, HuggingFace Spaces

---

## File Structure

```
Agentic-Reinforcement-Learning/
├── openenv.yaml                          # OpenEnv manifest (MODIFY)
├── Dockerfile                            # Container spec (MODIFY)
├── requirements.txt                      # Dependencies (MODIFY)
├── pyproject.toml                        # Package config (MODIFY)
├── inference.py                          # LLM agent script (CREATE — was baseline_inference.py)
├── README.md                             # Full documentation (REWRITE)
│
├── src/
│   └── agentic_rl/
│       ├── __init__.py                   # Package init (KEEP)
│       ├── models.py                     # Pydantic: FarmAction, FarmObservation, FarmState (REWRITE)
│       ├── tasks.py                      # 12 task scenario definitions + graders (REWRITE)
│       ├── constants.py                  # All biological/physical constants (CREATE)
│       │
│       ├── engine/                       # Simulation engine package (CREATE)
│       │   ├── __init__.py
│       │   ├── water_quality.py          # DO, TAN, UIA, pH, nitrite, temperature dynamics
│       │   ├── fish_biology.py           # Growth (bioenergetic), feeding, stress, mortality
│       │   ├── disease.py                # SEIR disease model with environmental triggers
│       │   ├── economics.py              # Feed cost, fish value, operating cost, market price
│       │   ├── weather.py                # Diel cycle, seasonal variation, storm events
│       │   ├── events.py                 # Event scheduler: disease, storms, equipment failure, algae bloom
│       │   └── simulator.py             # FishFarmSimulator — orchestrates all subsystems
│       │
│       └── server/
│           ├── __init__.py               # (KEEP)
│           ├── environment.py            # FishFarmEnvironment class (REWRITE)
│           └── app.py                    # FastAPI with /tasks, /grader, /baseline (REWRITE)
│
├── graders/
│   ├── __init__.py                       # (KEEP)
│   ├── base_grader.py                    # BaseGrader + GradeResult (KEEP)
│   └── farm_graders.py                   # Task-specific graders (CREATE — replaces example_graders.py)
│
├── tests/
│   ├── test_constants.py                 # Validate all constants have correct units/ranges (CREATE)
│   ├── test_water_quality.py             # Water quality dynamics tests (CREATE)
│   ├── test_fish_biology.py              # Growth model, feeding, mortality tests (CREATE)
│   ├── test_disease.py                   # SEIR model tests (CREATE)
│   ├── test_economics.py                 # Economic model tests (CREATE)
│   ├── test_simulator.py                 # Integration: full sim step tests (CREATE)
│   ├── test_models.py                    # Pydantic model validation tests (CREATE)
│   ├── test_tasks.py                     # Task loading + grader tests (CREATE)
│   ├── test_environment.py               # Environment reset/step/state tests (CREATE)
│   └── test_server.py                    # FastAPI endpoint tests (CREATE)
│
├── docs/
│   └── knowledge-base/                   # Research foundation (4,427 lines — KEEP)
│
└── training/                             # Optional GRPO training (KEEP as-is)
```

### Key Design Decisions

1. **Separate engine/ package** — Simulation logic is 100% independent of OpenEnv. Can be tested, debugged, and visualized standalone. Environment class is a thin wrapper.
2. **constants.py** — Single source of truth for ALL biological parameters. Every equation references these constants. Makes the research foundation auditable.
3. **1-hour timestep** — Balances biological dynamics (DO changes rapidly) with decision frequency (agent makes 24 decisions/day). Water quality sub-stepped internally at 6-minute intervals (10 sub-steps per hour).
4. **Task configs, not task classes** — Each task is a dict with initial conditions, event schedule, episode length, and grader name. No class inheritance needed.
5. **Deterministic with optional stochasticity** — Seed-controlled RNG for reproducible baseline scores. Weather noise, growth variation, and disease triggers all use seeded random.

---

## Chunk 1: Constants, Water Quality Engine, and Fish Biology

### Task 1: Create constants.py — The Biological Truth

All numerical values sourced from our knowledge base documents (01-BIOLOGY-AND-SCIENCE.md, 03-MATHEMATICAL-MODELS.md).

**Files:**
- Create: `src/agentic_rl/constants.py`
- Test: `tests/test_constants.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_constants.py
"""Validate that all biological constants are within published ranges."""
import pytest
from agentic_rl.constants import TILAPIA, WATER, DISEASE, ECONOMICS, SYSTEM


class TestTilapiaConstants:
    def test_growth_exponents(self):
        """Anabolism exponent m and catabolism exponent n from FAO calibration."""
        assert 0.6 < TILAPIA.m < 0.7, f"m={TILAPIA.m} outside FAO range [0.63-0.67]"
        assert 0.8 < TILAPIA.n < 0.9, f"n={TILAPIA.n} outside FAO range [0.80-0.84]"

    def test_temperature_range(self):
        """Tilapia temp range from species biology."""
        assert TILAPIA.T_min < TILAPIA.T_opt < TILAPIA.T_max
        assert 18 < TILAPIA.T_min < 20  # FAO: 18.7
        assert 31 < TILAPIA.T_opt < 34  # FAO: 32.4
        assert 38 < TILAPIA.T_max < 41  # FAO: 39.7

    def test_assimilation_efficiency(self):
        """b (assimilation) from FAO Table."""
        assert 0.65 < TILAPIA.b < 0.75  # FAO: 0.7108

    def test_consumption_coefficient(self):
        assert 0.4 < TILAPIA.h < 0.55  # FAO: 0.4768

    def test_catabolism_base(self):
        assert 0.008 < TILAPIA.k_min < 0.015  # FAO: 0.0104

    def test_initial_weight(self):
        assert 1 <= TILAPIA.w_initial <= 10  # fingerling 5g

    def test_fcr_range(self):
        assert 1.2 <= TILAPIA.fcr_target <= 2.0


class TestWaterConstants:
    def test_do_thresholds_ordered(self):
        """DO thresholds must be: lethal < stress < critical < optimal."""
        assert WATER.DO_lethal < WATER.DO_min < WATER.DO_crit < WATER.DO_optimal

    def test_uia_thresholds_ordered(self):
        assert WATER.UIA_safe < WATER.UIA_crit < WATER.UIA_lethal

    def test_ph_range(self):
        assert 6.0 <= WATER.pH_min <= 7.0
        assert 8.0 <= WATER.pH_max <= 9.5

    def test_tan_fraction_increases_with_ph(self):
        """Higher pH = more toxic NH3 fraction. Verify lookup is monotonic."""
        from agentic_rl.constants import uia_fraction
        f7 = uia_fraction(7.0, 25.0)
        f8 = uia_fraction(8.0, 25.0)
        f9 = uia_fraction(9.0, 25.0)
        assert f7 < f8 < f9


class TestDiseaseConstants:
    def test_seir_rates(self):
        assert 0 < DISEASE.beta <= 1.0
        assert 0 < DISEASE.sigma <= 1.0
        assert 0 < DISEASE.gamma <= 1.0
        assert 0 < DISEASE.alpha <= 0.5

    def test_r0_above_one(self):
        """R0 > 1 means disease can spread (needed for interesting dynamics)."""
        r0 = DISEASE.beta / (DISEASE.gamma + DISEASE.alpha + DISEASE.mu)
        assert r0 > 1.0, f"R0={r0} must be > 1 for disease to spread"


class TestEconomicsConstants:
    def test_feed_price_positive(self):
        assert ECONOMICS.feed_price_per_kg > 0

    def test_market_price_above_cost(self):
        """Fish must sell for more than feed cost per kg gained (FCR adjusted)."""
        cost_per_kg_fish = ECONOMICS.feed_price_per_kg * TILAPIA.fcr_target
        assert ECONOMICS.market_price_per_kg > cost_per_kg_fish


class TestSystemConstants:
    def test_timestep(self):
        assert SYSTEM.dt_hours == 1.0
        assert SYSTEM.sub_steps == 10

    def test_tank_volume(self):
        assert SYSTEM.tank_volume_m3 > 0

    def test_stocking_density(self):
        assert 10 <= SYSTEM.initial_stocking_density <= 100  # fish/m3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/rahulrajpurohit/IdeaProjects/Agentic-Reinforcement-Learning && python -m pytest tests/test_constants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_rl.constants'`

- [ ] **Step 3: Write constants.py**

```python
# src/agentic_rl/constants.py
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
    max_aeration_rate: float = 10.0  # mg O2/L/h at full power (KB-02 Sec 4)
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
    # Simplified formula
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/rahulrajpurohit/IdeaProjects/Agentic-Reinforcement-Learning && python -m pytest tests/test_constants.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rl/constants.py tests/test_constants.py
git commit -m "feat: add biological constants and utility functions for tilapia RAS simulation"
```

---

### Task 2: Create water_quality.py — DO, TAN, UIA, pH, Temperature Dynamics

The water quality engine is the heartbeat of the simulation. Every 6 minutes (10 sub-steps per 1-hour agent decision), it updates:
- Dissolved oxygen (DO) via mass balance
- Total ammonia nitrogen (TAN) via excretion + nitrification
- Unionized ammonia (UIA) via pH/temperature-dependent equilibrium
- pH drift from nitrification + CO2
- Temperature from weather + heater control

**Files:**
- Create: `src/agentic_rl/engine/__init__.py`
- Create: `src/agentic_rl/engine/water_quality.py`
- Test: `tests/test_water_quality.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_water_quality.py
"""Tests for water quality dynamics engine."""
import pytest
import math
from agentic_rl.engine.water_quality import WaterQualityEngine
from agentic_rl.constants import WATER, TILAPIA, SYSTEM, do_saturation


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
        """Fish consume oxygen — DO must decrease."""
        initial_DO = self.wq.DO
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=500.0,  # 500 kg fish in tank
            fish_weight_g=100.0,
            feeding_rate=0.5,
            aeration_rate=0.0,     # no aeration
            water_exchange_rate=0.0,
            is_daytime=False,      # no photosynthesis
            biofilter_efficiency=0.6,
        )
        assert self.wq.DO < initial_DO, "DO should decrease with fish respiration"

    def test_aeration_increases_do(self):
        """Mechanical aeration should raise DO."""
        self.wq.reset(temp=28.0, DO=3.0, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=100.0,
            fish_weight_g=100.0,
            feeding_rate=0.0,
            aeration_rate=1.0,     # full aeration
            water_exchange_rate=0.0,
            is_daytime=False,
            biofilter_efficiency=0.6,
        )
        assert self.wq.DO > 3.0, "Full aeration should increase DO from 3.0"

    def test_do_cannot_exceed_saturation_significantly(self):
        """DO shouldn't far exceed saturation (supersaturation capped)."""
        self.wq.reset(temp=28.0, DO=do_saturation(28.0), TAN=0.0, pH=7.5, NO2=0.0)
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=10.0,
            fish_weight_g=100.0,
            feeding_rate=0.0,
            aeration_rate=1.0,
            water_exchange_rate=0.0,
            is_daytime=True,
            biofilter_efficiency=0.6,
        )
        assert self.wq.DO <= do_saturation(28.0) * 1.5

    def test_do_never_negative(self):
        """DO is physically bounded at 0."""
        self.wq.reset(temp=35.0, DO=0.5, TAN=5.0, pH=8.0, NO2=1.0)
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=5000.0,  # massive biomass
            fish_weight_g=100.0,
            feeding_rate=1.0,
            aeration_rate=0.0,
            water_exchange_rate=0.0,
            is_daytime=False,
            biofilter_efficiency=0.0,
        )
        assert self.wq.DO >= 0.0


class TestTANDynamics:
    def setup_method(self):
        self.wq = WaterQualityEngine(
            volume_m3=SYSTEM.tank_volume_m3,
            depth_m=SYSTEM.tank_depth_m,
        )

    def test_feeding_increases_tan(self):
        """More feeding = more ammonia excretion."""
        self.wq.reset(temp=28.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)
        tan_before = self.wq.TAN
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=500.0,
            fish_weight_g=100.0,
            feeding_rate=1.0,     # maximum feeding
            aeration_rate=0.5,
            water_exchange_rate=0.0,
            is_daytime=True,
            biofilter_efficiency=0.0,  # no biofilter
        )
        assert self.wq.TAN > tan_before, "Heavy feeding with no biofilter should raise TAN"

    def test_biofilter_reduces_tan(self):
        """Active biofilter should reduce TAN via nitrification."""
        self.wq.reset(temp=28.0, DO=7.0, TAN=2.0, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=100.0,
            fish_weight_g=100.0,
            feeding_rate=0.0,
            aeration_rate=0.5,
            water_exchange_rate=0.0,
            is_daytime=True,
            biofilter_efficiency=0.8,
        )
        assert self.wq.TAN < 2.0, "Biofilter should reduce TAN"

    def test_water_exchange_reduces_tan(self):
        """Fresh water exchange dilutes TAN."""
        self.wq.reset(temp=28.0, DO=7.0, TAN=3.0, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=100.0,
            fish_weight_g=100.0,
            feeding_rate=0.0,
            aeration_rate=0.5,
            water_exchange_rate=0.1,  # 10% exchange
            is_daytime=True,
            biofilter_efficiency=0.0,
        )
        assert self.wq.TAN < 3.0

    def test_tan_never_negative(self):
        self.wq.reset(temp=28.0, DO=7.0, TAN=0.01, pH=7.5, NO2=0.05)
        self.wq.step(
            dt_hours=1.0,
            fish_biomass_kg=10.0,
            fish_weight_g=100.0,
            feeding_rate=0.0,
            aeration_rate=0.5,
            water_exchange_rate=0.1,
            is_daytime=True,
            biofilter_efficiency=1.0,
        )
        assert self.wq.TAN >= 0.0


class TestUIACalculation:
    def setup_method(self):
        self.wq = WaterQualityEngine(
            volume_m3=SYSTEM.tank_volume_m3,
            depth_m=SYSTEM.tank_depth_m,
        )

    def test_uia_increases_with_ph(self):
        """Higher pH = more toxic NH3."""
        self.wq.reset(temp=28.0, DO=7.0, TAN=1.0, pH=7.0, NO2=0.05)
        uia_low_ph = self.wq.UIA

        self.wq.reset(temp=28.0, DO=7.0, TAN=1.0, pH=8.5, NO2=0.05)
        uia_high_ph = self.wq.UIA

        assert uia_high_ph > uia_low_ph * 5, "pH 8.5 should have >5x more UIA than pH 7.0"

    def test_uia_increases_with_temperature(self):
        """Higher temp = more toxic NH3."""
        self.wq.reset(temp=20.0, DO=7.0, TAN=1.0, pH=8.0, NO2=0.05)
        uia_cold = self.wq.UIA

        self.wq.reset(temp=30.0, DO=7.0, TAN=1.0, pH=8.0, NO2=0.05)
        uia_warm = self.wq.UIA

        assert uia_warm > uia_cold


class TestTemperatureDynamics:
    def setup_method(self):
        self.wq = WaterQualityEngine(
            volume_m3=SYSTEM.tank_volume_m3,
            depth_m=SYSTEM.tank_depth_m,
        )

    def test_heater_warms_water(self):
        self.wq.reset(temp=25.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.update_temperature(
            dt_hours=1.0,
            air_temp=25.0,
            heater_setting=1.0,    # full heat
            volume_m3=SYSTEM.tank_volume_m3,
        )
        assert self.wq.temperature > 25.0

    def test_temperature_trends_toward_air(self):
        """Without heater, water temp drifts toward air temp."""
        self.wq.reset(temp=30.0, DO=7.0, TAN=0.1, pH=7.5, NO2=0.05)
        self.wq.update_temperature(
            dt_hours=1.0,
            air_temp=20.0,
            heater_setting=0.0,
            volume_m3=SYSTEM.tank_volume_m3,
        )
        assert self.wq.temperature < 30.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/rahulrajpurohit/IdeaProjects/Agentic-Reinforcement-Learning && python -m pytest tests/test_water_quality.py -v`
Expected: FAIL

- [ ] **Step 3: Create engine package and water_quality.py**

```python
# src/agentic_rl/engine/__init__.py
"""Fish Farm Simulation Engine.

Modular subsystems:
- water_quality: DO, TAN, UIA, pH, temperature dynamics
- fish_biology: Growth (bioenergetic), feeding response, stress, mortality
- disease: SEIR epidemic model with environmental triggers
- economics: Feed cost, fish value, operating expenses
- weather: Diel cycle, seasonal variation, storm events
- events: Event scheduler (disease, storms, equipment failures, algae blooms)
- simulator: Orchestrator combining all subsystems
"""
```

```python
# src/agentic_rl/engine/water_quality.py
"""Water quality dynamics engine.

Implements the mass balance equations for:
- Dissolved Oxygen (DO): fish respiration, photosynthesis, aeration, reaeration, nitrification
- Total Ammonia Nitrogen (TAN): fish excretion, biofilter nitrification, water exchange
- Unionized Ammonia (UIA): pH/temperature-dependent equilibrium
- pH: drift from nitrification acid production
- Nitrite (NO2): intermediate nitrification product
- Temperature: air exchange + heater control

All equations sourced from KB-03 Sections 2.1-2.2.
Internal sub-stepping at 6-minute intervals for numerical stability.
"""

import math
from ..constants import WATER, SYSTEM, TILAPIA, uia_fraction, do_saturation


class WaterQualityEngine:
    """Manages water quality state and dynamics for a single RAS tank."""

    def __init__(self, volume_m3: float, depth_m: float):
        self.volume_m3 = volume_m3
        self.depth_m = depth_m

        # State variables
        self.DO: float = 7.0           # mg/L
        self.temperature: float = 28.0 # C
        self.TAN: float = 0.1          # mg/L
        self.UIA: float = 0.0          # mg/L (derived)
        self.pH: float = 7.5
        self.NO2: float = 0.05         # mg/L
        self.NO3: float = 5.0          # mg/L
        self.alkalinity: float = 150.0 # mg CaCO3/L

    def reset(self, temp: float, DO: float, TAN: float, pH: float, NO2: float):
        """Reset water quality to specified initial conditions."""
        self.temperature = temp
        self.DO = DO
        self.TAN = TAN
        self.pH = pH
        self.NO2 = NO2
        self.NO3 = 5.0
        self.alkalinity = 150.0
        self._update_uia()

    def _update_uia(self):
        """Recalculate UIA from current TAN, pH, and temperature."""
        self.UIA = self.TAN * uia_fraction(self.pH, self.temperature)

    def step(
        self,
        dt_hours: float,
        fish_biomass_kg: float,
        fish_weight_g: float,
        feeding_rate: float,
        aeration_rate: float,
        water_exchange_rate: float,
        is_daytime: bool,
        biofilter_efficiency: float,
    ):
        """Advance water quality by dt_hours using sub-stepping.

        Args:
            dt_hours: Time step in hours (typically 1.0)
            fish_biomass_kg: Total fish mass in tank (kg)
            fish_weight_g: Average individual fish weight (g)
            feeding_rate: 0.0-1.0 fraction of max feeding
            aeration_rate: 0.0-1.0 fraction of max aeration
            water_exchange_rate: 0.0-0.1 fraction of volume exchanged per hour
            is_daytime: Whether photosynthesis is active
            biofilter_efficiency: 0.0-1.0 biofilter removal efficiency
        """
        n_sub = SYSTEM.sub_steps
        sub_dt = dt_hours / n_sub  # hours per sub-step

        for _ in range(n_sub):
            self._sub_step(
                sub_dt, fish_biomass_kg, fish_weight_g, feeding_rate,
                aeration_rate, water_exchange_rate, is_daytime,
                biofilter_efficiency,
            )

    def _sub_step(
        self, dt_h: float, biomass_kg: float, fish_w_g: float,
        feed_rate: float, aeration: float, exchange: float,
        daytime: bool, biofilter_eff: float,
    ):
        """Single sub-step of water quality dynamics."""

        # ---- 1. DO dynamics (KB-03 Sec 2.1) ----

        # Fish respiration: FR = 10^X * 1000 (mg O2/kg/h)
        # X = 0.40 + 0.016*T - 0.0006*T^2 - 0.016*ln(W)
        T = self.temperature
        w = max(fish_w_g, 1.0)
        X = 0.40 + 0.016 * T - 0.0006 * T**2 - 0.016 * math.log(w)
        FR = (10**X) * 1000  # mg O2/kg fish/h
        DO_fish = FR * biomass_kg / (self.volume_m3 * 1000)  # mg/L/h (1000 L/m3)

        # Nitrification oxygen demand: 4.57 g O2 per g TAN nitrified
        nitrif_rate = self._nitrification_rate(biofilter_eff)
        TAN_nitrified_rate = nitrif_rate * self.TAN  # mg/L/h
        DO_nitrif = WATER.O2_per_TAN * TAN_nitrified_rate  # mg/L/h

        # Photosynthesis (daytime only, simplified)
        if daytime:
            # Simple model: photosynthesis proportional to light
            DO_photo = 0.5  # mg/L/h base daytime production (simplified)
        else:
            DO_photo = 0.0

        # Water column respiration (microbial, ~0.1 mg/L/h)
        DO_water = 0.1 * (1.047 ** (T - 20))  # Q10-adjusted

        # Reaeration: K_a * (DO_sat - DO)
        DO_sat = do_saturation(T)
        DO_reaer = WATER.K_a_base * (DO_sat - self.DO)  # mg/L/h (can be negative if supersaturated)

        # Mechanical aeration
        DO_mech = aeration * SYSTEM.max_aeration_rate  # mg/L/h

        # Water exchange DO contribution
        DO_exchange = exchange * (SYSTEM.incoming_water_DO - self.DO)  # mg/L/h

        # Net DO change
        dDO = (DO_photo + DO_reaer + DO_mech + DO_exchange
               - DO_fish - DO_nitrif - DO_water) * dt_h

        self.DO = max(0.0, self.DO + dDO)
        # Cap at ~1.5x saturation (supersaturation limit)
        self.DO = min(self.DO, DO_sat * 1.5)

        # ---- 2. TAN dynamics (KB-03 Sec 2.2) ----

        # TAN excretion from feeding
        # TAN_produced = Feed_rate * Protein% * 0.16 * N_wasted% * 1.2
        # Feed amount in kg/h
        feed_amount_kg_h = (feed_rate * TILAPIA.max_feeding_pct / 100.0
                            * biomass_kg / 24.0)  # daily rate / 24
        TAN_excretion = (feed_amount_kg_h * TILAPIA.protein_fraction
                         * 0.16 * TILAPIA.n_wasted_fraction * 1.2)  # kg TAN/h
        TAN_excretion_mg_L = TAN_excretion * 1e6 / (self.volume_m3 * 1000)  # mg/L/h

        # Nitrification removal
        TAN_nitrif = nitrif_rate * self.TAN  # mg/L/h

        # Water exchange dilution
        TAN_exchange = exchange * (SYSTEM.incoming_water_TAN - self.TAN)  # mg/L/h

        dTAN = (TAN_excretion_mg_L - TAN_nitrif + TAN_exchange) * dt_h
        self.TAN = max(0.0, self.TAN + dTAN)

        # ---- 3. Nitrite dynamics ----
        # Nitrite produced by first-stage nitrification, consumed by second-stage
        # Simplified: NO2 rises if biofilter is stressed
        dNO2 = (TAN_nitrified_rate * 0.3 - nitrif_rate * self.NO2 * 0.8) * dt_h
        self.NO2 = max(0.0, self.NO2 + dNO2)

        # ---- 4. pH dynamics ----
        # Nitrification produces acid (lowers pH)
        # Each g TAN nitrified consumes 7.14 g alkalinity
        alk_consumed = WATER.alkalinity_per_TAN * TAN_nitrified_rate * dt_h * self.volume_m3 * 1000 / 1e6
        self.alkalinity = max(50.0, self.alkalinity - alk_consumed * 1000 / (self.volume_m3 * 1000))

        # pH drifts based on alkalinity buffer
        if self.alkalinity < 80:
            self.pH = max(6.0, self.pH - 0.01 * dt_h)
        elif self.alkalinity > 200:
            self.pH = min(9.0, self.pH + 0.005 * dt_h)

        # ---- 5. Update derived values ----
        self._update_uia()

    def _nitrification_rate(self, biofilter_eff: float) -> float:
        """Calculate nitrification rate coefficient (1/h).

        Source: KB-03 Sec 2.1
        K_NR = 0.11 * 1.08^(T-20), adjusted for biofilter efficiency
        """
        K_NR = 0.11 * (1.08 ** (self.temperature - 20)) / 24.0  # convert per day to per hour
        return K_NR * biofilter_eff

    def update_temperature(
        self,
        dt_hours: float,
        air_temp: float,
        heater_setting: float,
        volume_m3: float,
    ):
        """Update water temperature from air exchange and heater.

        Simplified thermal model:
        - Water slowly equilibrates with air temperature
        - Heater adds/removes energy

        Args:
            heater_setting: -1.0 (max cool) to 1.0 (max heat), 0 = off
        """
        # Thermal equilibration (slow, ~1% per hour for RAS)
        equilibration_rate = 0.01  # fraction per hour
        dT_air = equilibration_rate * (air_temp - self.temperature) * dt_hours

        # Heater effect (C/hour at full power)
        # P = m*c*dT => dT = P*dt / (m*c)
        # At 5kW heater, 100m3 water: dT = 5000 * 3600 / (100000 * 4186) = 0.043 C/h
        heater_dT_max = (ECONOMICS.heater_power_kw * 3600) / (volume_m3 * 1000 * 4186) * 1000
        dT_heater = heater_setting * heater_dT_max * dt_hours

        self.temperature += dT_air + dT_heater
        self.temperature = max(10.0, min(42.0, self.temperature))

    def get_water_quality_score(self) -> float:
        """Calculate composite water quality score (0.0-1.0).

        Scores each parameter against optimal range and returns weighted average.
        """
        # DO score
        if self.DO >= WATER.DO_optimal:
            do_score = 1.0
        elif self.DO >= WATER.DO_min:
            do_score = (self.DO - WATER.DO_min) / (WATER.DO_optimal - WATER.DO_min)
        else:
            do_score = 0.0

        # UIA score
        if self.UIA <= WATER.UIA_safe:
            uia_score = 1.0
        elif self.UIA <= WATER.UIA_crit:
            uia_score = 1.0 - (self.UIA - WATER.UIA_safe) / (WATER.UIA_crit - WATER.UIA_safe)
        else:
            uia_score = max(0.0, 1.0 - (self.UIA - WATER.UIA_crit) / (WATER.UIA_lethal - WATER.UIA_crit))

        # pH score
        if WATER.pH_min <= self.pH <= WATER.pH_max:
            ph_score = 1.0
        else:
            deviation = max(WATER.pH_min - self.pH, self.pH - WATER.pH_max, 0)
            ph_score = max(0.0, 1.0 - deviation / 2.0)

        # Temperature score (using temperature_factor)
        from ..constants import temperature_factor
        temp_score = temperature_factor(self.temperature)

        # Weighted composite
        return 0.35 * do_score + 0.25 * uia_score + 0.20 * temp_score + 0.20 * ph_score
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/rahulrajpurohit/IdeaProjects/Agentic-Reinforcement-Learning && python -m pytest tests/test_water_quality.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rl/engine/ tests/test_water_quality.py
git commit -m "feat: add water quality dynamics engine (DO, TAN, UIA, pH, temperature)"
```

---

### Task 3: Create fish_biology.py — Growth, Feeding, Stress, Mortality

The bioenergetic growth model from KAUST/FAO research, adapted for hourly timesteps.

**Files:**
- Create: `src/agentic_rl/engine/fish_biology.py`
- Test: `tests/test_fish_biology.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_fish_biology.py
"""Tests for fish biology engine — growth, feeding, stress, mortality."""
import pytest
from agentic_rl.engine.fish_biology import FishBiologyEngine
from agentic_rl.constants import TILAPIA


class TestGrowthModel:
    def setup_method(self):
        self.bio = FishBiologyEngine()
        self.bio.reset(weight_g=50.0, population=10000, day_of_year=1)

    def test_fish_grow_with_optimal_conditions(self):
        """Fish should gain weight under optimal conditions."""
        initial_weight = self.bio.weight_g
        self.bio.grow(
            dt_hours=24.0,  # 1 day
            feeding_rate=0.5,
            temperature=TILAPIA.T_opt,
            DO=7.0,
            UIA=0.01,
            photoperiod_h=12.0,
        )
        assert self.bio.weight_g > initial_weight

    def test_no_growth_without_feeding(self):
        """Weight should decrease (catabolism) with zero feeding."""
        initial_weight = self.bio.weight_g
        self.bio.grow(
            dt_hours=24.0,
            feeding_rate=0.0,
            temperature=TILAPIA.T_opt,
            DO=7.0,
            UIA=0.01,
            photoperiod_h=12.0,
        )
        assert self.bio.weight_g < initial_weight

    def test_growth_faster_at_optimal_temp(self):
        """Growth is maximized near T_opt."""
        bio_opt = FishBiologyEngine()
        bio_opt.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_opt.grow(24.0, 0.5, TILAPIA.T_opt, 7.0, 0.01, 12.0)

        bio_cold = FishBiologyEngine()
        bio_cold.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_cold.grow(24.0, 0.5, 22.0, 7.0, 0.01, 12.0)  # cold

        assert bio_opt.weight_g > bio_cold.weight_g

    def test_growth_reduced_by_low_do(self):
        """Low DO suppresses growth via sigma(DO) factor."""
        bio_good = FishBiologyEngine()
        bio_good.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_good.grow(24.0, 0.5, 30.0, 7.0, 0.01, 12.0)

        bio_low_do = FishBiologyEngine()
        bio_low_do.reset(weight_g=50.0, population=10000, day_of_year=1)
        bio_low_do.grow(24.0, 0.5, 30.0, 3.5, 0.01, 12.0)

        assert bio_good.weight_g > bio_low_do.weight_g

    def test_growth_reduced_by_high_ammonia(self):
        """High UIA suppresses growth via v(UIA) factor."""
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
        """Zero or near-zero mortality under optimal conditions."""
        deaths = self.bio.apply_mortality(
            dt_hours=24.0,
            DO=7.0,
            UIA=0.01,
            temperature=30.0,
            stocking_density=50.0,
        )
        assert deaths <= 10  # < 0.1% per day

    def test_high_mortality_under_stress(self):
        """Significant mortality under lethal conditions."""
        deaths = self.bio.apply_mortality(
            dt_hours=24.0,
            DO=1.0,       # lethal
            UIA=0.5,      # near-lethal
            temperature=38.0,
            stocking_density=100.0,
        )
        assert deaths > 50  # noticeable die-off

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
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Write fish_biology.py**

```python
# src/agentic_rl/engine/fish_biology.py
"""Fish biology engine — bioenergetic growth, feeding, stress, mortality.

Core equation (KB-03 Sec 1.1, from KAUST Q-learning paper):
    dW/dt = [Psi(f,T,DO) * v(UIA) * W^m]  -  [k(T) * W^n]

Where:
    Psi(f,T,DO) = h * rho * f * b * (1-a) * tau(T) * sigma(DO)
    k(T) = k_min * exp(s * (T - T_min))
    m = 0.6277 (anabolism exponent)
    n = 0.8373 (catabolism exponent)

All parameters for Nile Tilapia from FAO calibration data.
"""

import math
from ..constants import (
    TILAPIA, WATER, temperature_factor, do_factor, uia_factor
)


class FishBiologyEngine:
    """Manages fish population biology — growth, stress, mortality."""

    def __init__(self):
        self.weight_g: float = TILAPIA.w_initial
        self.population: int = TILAPIA.N_initial
        self.total_feed_consumed_kg: float = 0.0
        self.total_weight_gained_g: float = 0.0
        self.mortality_today: int = 0
        self.cumulative_mortality: int = 0
        self.stress_level: float = 0.0
        self.day_of_year: int = 1
        self.growth_rate: float = 0.0  # g/day (last computed)

    def reset(self, weight_g: float, population: int, day_of_year: int):
        self.weight_g = weight_g
        self.population = population
        self.total_feed_consumed_kg = 0.0
        self.total_weight_gained_g = 0.0
        self.mortality_today = 0
        self.cumulative_mortality = 0
        self.stress_level = 0.0
        self.day_of_year = day_of_year
        self.growth_rate = 0.0

    def grow(
        self,
        dt_hours: float,
        feeding_rate: float,
        temperature: float,
        DO: float,
        UIA: float,
        photoperiod_h: float,
    ) -> float:
        """Apply bioenergetic growth model.

        Returns: weight gain in grams (can be negative if starving)
        """
        dt_days = dt_hours / 24.0
        w = max(self.weight_g, 0.1)
        f = max(0.0, min(1.0, feeding_rate))

        # Environmental response factors
        tau = temperature_factor(temperature)
        sigma = do_factor(DO)
        v = uia_factor(UIA)

        # Photoperiod scalar (KB-03 Sec 15.3)
        pi = photoperiod_h / 12.0

        # Anabolism: H = h * pi * f * b * (1-a) * tau * sigma * v
        H = (TILAPIA.h * pi * f * TILAPIA.b * (1.0 - TILAPIA.a)
             * tau * sigma * v)

        # Catabolism: k(T) = k_min * exp(s * (T - T_min))
        if temperature > TILAPIA.T_min:
            k = TILAPIA.k_min * math.exp(TILAPIA.s * (temperature - TILAPIA.T_min))
        else:
            k = TILAPIA.k_min

        # Growth: dW/dt = H * W^m - k * W^n
        anabolism = H * (w ** TILAPIA.m)
        catabolism = k * (w ** TILAPIA.n)
        dw_dt = anabolism - catabolism  # g/day

        # Apply over dt
        dw = dw_dt * dt_days
        old_weight = self.weight_g
        self.weight_g = max(0.1, self.weight_g + dw)
        self.growth_rate = dw_dt

        # Track feed consumed
        biomass_kg = self.population * self.weight_g / 1000
        feed_today = (f * TILAPIA.max_feeding_pct / 100.0
                      * biomass_kg * dt_days)
        self.total_feed_consumed_kg += feed_today
        self.total_weight_gained_g += max(0, self.weight_g - old_weight)

        return dw

    def apply_mortality(
        self,
        dt_hours: float,
        DO: float,
        UIA: float,
        temperature: float,
        stocking_density: float,
    ) -> int:
        """Apply mortality based on environmental stress.

        Returns: number of fish that died
        """
        dt_days = dt_hours / 24.0
        stress = self.calculate_stress(DO, UIA, temperature, stocking_density)
        self.stress_level = stress

        # Base mortality (KB-03 Sec 15.6)
        base_rate = TILAPIA.base_mortality

        # Stress-induced mortality multiplier
        # Logistic increase with stress
        if stress > 0.3:
            stress_multiplier = 1.0 + 50.0 * ((stress - 0.3) ** 2)
        else:
            stress_multiplier = 1.0

        # Acute lethal conditions
        acute_mortality = 0.0
        if DO < WATER.DO_lethal:
            acute_mortality += 0.05 * dt_days  # 5% per day at lethal DO
        if UIA > WATER.UIA_lethal:
            acute_mortality += 0.10 * dt_days  # 10% per day at lethal UIA
        if temperature > TILAPIA.T_max or temperature < TILAPIA.T_lethal_low:
            acute_mortality += 0.15 * dt_days  # 15% per day at lethal temp

        # Total mortality rate
        total_rate = base_rate * stress_multiplier * dt_days + acute_mortality
        total_rate = min(total_rate, 0.5)  # cap at 50% per step

        deaths = int(self.population * total_rate)
        deaths = min(deaths, self.population)
        self.population = max(0, self.population - deaths)
        self.mortality_today = deaths
        self.cumulative_mortality += deaths

        return deaths

    def calculate_stress(
        self,
        DO: float,
        UIA: float,
        temperature: float,
        stocking_density: float,
    ) -> float:
        """Calculate fish stress level (0.0 = no stress, 1.0 = maximum).

        Multi-factor stress from KB-01 Sec 8 (behavioral indicators).
        """
        # DO stress
        if DO >= WATER.DO_optimal:
            do_stress = 0.0
        elif DO >= WATER.DO_min:
            do_stress = 1.0 - (DO - WATER.DO_min) / (WATER.DO_optimal - WATER.DO_min)
        else:
            do_stress = 1.0

        # Ammonia stress
        if UIA <= WATER.UIA_safe:
            uia_stress = 0.0
        elif UIA <= WATER.UIA_crit:
            uia_stress = (UIA - WATER.UIA_safe) / (WATER.UIA_crit - WATER.UIA_safe) * 0.5
        else:
            uia_stress = min(1.0, 0.5 + 0.5 * (UIA - WATER.UIA_crit) / (WATER.UIA_lethal - WATER.UIA_crit))

        # Temperature stress
        temp_dev = abs(temperature - TILAPIA.T_opt)
        if temp_dev <= 3.0:
            temp_stress = 0.0
        else:
            temp_stress = min(1.0, (temp_dev - 3.0) / 7.0)

        # Density stress (KB-01 Sec 9)
        if stocking_density <= 50:
            density_stress = 0.0
        elif stocking_density <= 80:
            density_stress = (stocking_density - 50) / 80.0
        else:
            density_stress = min(1.0, (stocking_density - 50) / 50.0)

        # Weighted composite (DO is most important)
        return min(1.0, 0.35 * do_stress + 0.30 * uia_stress
                   + 0.20 * temp_stress + 0.15 * density_stress)

    def feeding_response(
        self,
        temperature: float,
        DO: float,
        UIA: float,
        stress: float,
    ) -> str:
        """Classify fish feeding behavior.

        Source: KB-01 Sec 8 (behavioral indicators)
        Returns: 'eager', 'normal', 'sluggish', or 'refusing'
        """
        if stress > 0.7 or DO < WATER.DO_min or temperature > TILAPIA.T_max - 2:
            return "refusing"
        elif stress > 0.4 or DO < WATER.DO_crit:
            return "sluggish"
        elif stress < 0.15 and temperature_factor(temperature) > 0.8:
            return "eager"
        else:
            return "normal"

    @property
    def biomass_kg(self) -> float:
        return self.population * self.weight_g / 1000.0

    @property
    def fcr(self) -> float:
        """Current Feed Conversion Ratio."""
        if self.total_weight_gained_g > 0:
            return self.total_feed_consumed_kg / (self.total_weight_gained_g / 1000.0)
        return 0.0

    @property
    def stocking_density(self) -> float:
        """Current stocking density (fish/m3)."""
        # This uses a default volume; caller should provide actual
        return self.population / 100.0  # default 100m3

    @property
    def survival_rate(self) -> float:
        """Fraction of initial population still alive."""
        if TILAPIA.N_initial > 0:
            return self.population / TILAPIA.N_initial
        return 0.0
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/rahulrajpurohit/IdeaProjects/Agentic-Reinforcement-Learning && python -m pytest tests/test_fish_biology.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rl/engine/fish_biology.py tests/test_fish_biology.py
git commit -m "feat: add bioenergetic fish growth model with stress and mortality"
```

---

## Chunk 2: Disease, Economics, Weather, Events, and Simulator

### Task 4: Create disease.py — SEIR Epidemic Model

**Files:**
- Create: `src/agentic_rl/engine/disease.py`
- Test: `tests/test_disease.py`

- [ ] **Step 1: Write test file**

```python
# tests/test_disease.py
"""Tests for SEIR disease model."""
import pytest
from agentic_rl.engine.disease import DiseaseEngine
from agentic_rl.constants import DISEASE


class TestSEIRModel:
    def setup_method(self):
        self.disease = DiseaseEngine()

    def test_no_disease_initially(self):
        self.disease.reset(population=10000)
        assert self.disease.infected == 0
        assert self.disease.exposed == 0
        assert not self.disease.is_active

    def test_disease_spreads_after_trigger(self):
        self.disease.reset(population=10000)
        self.disease.trigger_outbreak(initial_infected=10)
        assert self.disease.infected == 10
        assert self.disease.is_active

        # Step forward
        deaths = self.disease.step(dt_hours=24.0, population=10000)
        assert self.disease.exposed > 0 or self.disease.infected > 10

    def test_treatment_speeds_recovery(self):
        self.disease.reset(population=10000)
        self.disease.trigger_outbreak(initial_infected=100)

        # Without treatment
        d1 = DiseaseEngine()
        d1.reset(population=10000)
        d1.trigger_outbreak(initial_infected=100)
        d1.step(dt_hours=48.0, population=10000)

        # With treatment
        d2 = DiseaseEngine()
        d2.reset(population=10000)
        d2.trigger_outbreak(initial_infected=100)
        d2.apply_treatment()
        d2.step(dt_hours=48.0, population=10000)

        assert d2.recovered >= d1.recovered  # treatment increases recovery

    def test_disease_dies_out_eventually(self):
        self.disease.reset(population=10000)
        self.disease.trigger_outbreak(initial_infected=5)
        total_deaths = 0
        for _ in range(365 * 24):  # 1 year of hourly steps
            deaths = self.disease.step(dt_hours=1.0, population=self.disease.susceptible + self.disease.exposed + self.disease.infected + self.disease.recovered)
            total_deaths += deaths
            if self.disease.infected == 0 and self.disease.exposed == 0:
                break
        assert self.disease.infected == 0, "Disease should eventually resolve"

    def test_stress_triggers_outbreak(self):
        self.disease.reset(population=10000)
        # Simulate high stress for many hours
        triggered = False
        for _ in range(24 * 30):  # 30 days
            self.disease.check_stress_trigger(
                stress_level=0.8,
                DO=2.0,
                UIA=0.3,
                temperature=36.0,
                stocking_density=100.0,
                rng_value=0.0001,  # force trigger for test
            )
            if self.disease.is_active:
                triggered = True
                break
        assert triggered
```

- [ ] **Step 2: Run to verify fail**
- [ ] **Step 3: Write disease.py**

```python
# src/agentic_rl/engine/disease.py
"""SEIR disease model for fish populations.

Source: KB-03 Sec 4.1-4.2

dS/dt = -beta * S * I / N
dE/dt =  beta * S * I / N  -  sigma * E
dI/dt =  sigma * E  -  gamma * I  -  alpha * I
dR/dt =  gamma * I

Environmental triggers increase outbreak probability when:
- DO < 3.5 mg/L
- UIA > 0.04 mg/L
- Temperature > T_opt + 5C or < T_opt - 10C
- Stocking density > 80 fish/m3
"""

from ..constants import DISEASE


class DiseaseEngine:
    """SEIR compartmental disease model."""

    def __init__(self):
        self.susceptible: int = 0
        self.exposed: int = 0
        self.infected: int = 0
        self.recovered: int = 0
        self.is_active: bool = False
        self.treatment_active: bool = False
        self.treatment_days_remaining: int = 0
        self.total_disease_deaths: int = 0

    def reset(self, population: int):
        self.susceptible = population
        self.exposed = 0
        self.infected = 0
        self.recovered = 0
        self.is_active = False
        self.treatment_active = False
        self.treatment_days_remaining = 0
        self.total_disease_deaths = 0

    def trigger_outbreak(self, initial_infected: int = 5):
        """Start a disease outbreak."""
        initial_infected = min(initial_infected, self.susceptible)
        self.susceptible -= initial_infected
        self.infected = initial_infected
        self.is_active = True

    def apply_treatment(self):
        """Apply antibiotic/treatment."""
        self.treatment_active = True
        self.treatment_days_remaining = DISEASE.treatment_duration_days

    def step(self, dt_hours: float, population: int) -> int:
        """Advance SEIR model by dt_hours. Returns disease-induced deaths."""
        if not self.is_active:
            return 0

        dt_days = dt_hours / 24.0
        N = max(1, self.susceptible + self.exposed + self.infected + self.recovered)

        # Recovery rate (boosted by treatment)
        gamma = DISEASE.gamma
        if self.treatment_active:
            gamma *= DISEASE.treatment_recovery_boost

        # SEIR transitions (KB-03 Sec 4.1)
        new_exposed = int(DISEASE.beta * self.susceptible * self.infected / N * dt_days)
        new_infected = int(DISEASE.sigma * self.exposed * dt_days)
        new_recovered = int(gamma * self.infected * dt_days)
        disease_deaths = int(DISEASE.alpha * self.infected * dt_days)

        # Clamp to available populations
        new_exposed = min(new_exposed, self.susceptible)
        new_infected = min(new_infected, self.exposed)
        new_recovered = min(new_recovered, self.infected - disease_deaths)
        disease_deaths = min(disease_deaths, self.infected)

        # Update compartments
        self.susceptible -= new_exposed
        self.exposed += new_exposed - new_infected
        self.infected += new_infected - new_recovered - disease_deaths
        self.recovered += new_recovered

        # Ensure non-negative
        self.susceptible = max(0, self.susceptible)
        self.exposed = max(0, self.exposed)
        self.infected = max(0, self.infected)
        self.recovered = max(0, self.recovered)

        self.total_disease_deaths += disease_deaths

        # Treatment countdown
        if self.treatment_active:
            self.treatment_days_remaining -= dt_days
            if self.treatment_days_remaining <= 0:
                self.treatment_active = False

        # Check if disease is resolved
        if self.infected == 0 and self.exposed == 0:
            self.is_active = False

        return disease_deaths

    def check_stress_trigger(
        self,
        stress_level: float,
        DO: float,
        UIA: float,
        temperature: float,
        stocking_density: float,
        rng_value: float,
    ) -> bool:
        """Check if environmental stress triggers a disease outbreak.

        Args:
            rng_value: random float 0-1 from seeded RNG
        Returns: True if outbreak triggered
        """
        if self.is_active:
            return False  # already have disease

        # Calculate outbreak probability based on stress factors
        prob = DISEASE.outbreak_prob_per_hour

        if DO < DISEASE.stress_DO_threshold:
            prob *= 3.0
        if UIA > DISEASE.stress_ammonia_threshold:
            prob *= 2.5
        if abs(temperature - 30.0) > DISEASE.stress_temp_deviation:
            prob *= 2.0
        if stocking_density > DISEASE.stress_density_threshold:
            prob *= 2.0
        if stress_level > 0.5:
            prob *= (1.0 + stress_level * 3.0)

        if rng_value < prob:
            self.trigger_outbreak(initial_infected=max(1, int(self.susceptible * 0.001)))
            return True
        return False

    def sync_population(self, total_population: int):
        """Sync SEIR compartments with actual population (after external mortality)."""
        current = self.susceptible + self.exposed + self.infected + self.recovered
        if current > total_population and current > 0:
            ratio = total_population / current
            self.susceptible = int(self.susceptible * ratio)
            self.exposed = int(self.exposed * ratio)
            self.infected = int(self.infected * ratio)
            self.recovered = total_population - self.susceptible - self.exposed - self.infected
            self.recovered = max(0, self.recovered)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

---

### Task 5: Create economics.py — Feed Cost, Fish Value, P&L

**Files:**
- Create: `src/agentic_rl/engine/economics.py`
- Test: `tests/test_economics.py`

- [ ] **Step 1: Write test**

```python
# tests/test_economics.py
import pytest
from agentic_rl.engine.economics import EconomicsEngine
from agentic_rl.constants import ECONOMICS


class TestEconomics:
    def setup_method(self):
        self.econ = EconomicsEngine()
        self.econ.reset()

    def test_feed_cost_accumulates(self):
        self.econ.record_feed(feed_kg=10.0)
        assert self.econ.total_feed_cost == 10.0 * ECONOMICS.feed_price_per_kg

    def test_harvest_revenue(self):
        rev = self.econ.calculate_harvest_revenue(biomass_kg=500.0)
        assert rev == 500.0 * ECONOMICS.market_price_per_kg

    def test_profit_calculation(self):
        self.econ.record_feed(feed_kg=100.0)
        self.econ.record_operating_day(aeration_hours=24.0, heater_hours=0.0, water_exchanged_m3=5.0)
        harvest_rev = self.econ.calculate_harvest_revenue(biomass_kg=500.0)
        profit = harvest_rev - self.econ.total_cost
        assert isinstance(profit, float)

    def test_daily_cost_positive(self):
        self.econ.record_operating_day(aeration_hours=24.0, heater_hours=12.0, water_exchanged_m3=10.0)
        assert self.econ.total_operating_cost > 0
```

- [ ] **Step 2-3: Write economics.py**

```python
# src/agentic_rl/engine/economics.py
"""Economic model — feed cost, energy, fish value, profit/loss.

Source: KB-02 Sec 7, KB-03 Sec 6
Feed = 50-70% of operating costs. FCR optimization is the #1 economic lever.
"""

from ..constants import ECONOMICS


class EconomicsEngine:
    def __init__(self):
        self.total_feed_cost: float = 0.0
        self.total_operating_cost: float = 0.0
        self.total_treatment_cost: float = 0.0
        self.total_feed_kg: float = 0.0
        self.days_operated: int = 0

    def reset(self):
        self.total_feed_cost = 0.0
        self.total_operating_cost = 0.0
        self.total_treatment_cost = 0.0
        self.total_feed_kg = 0.0
        self.days_operated = 0

    def record_feed(self, feed_kg: float):
        self.total_feed_kg += feed_kg
        self.total_feed_cost += feed_kg * ECONOMICS.feed_price_per_kg

    def record_operating_day(
        self,
        aeration_hours: float = 0.0,
        heater_hours: float = 0.0,
        water_exchanged_m3: float = 0.0,
    ):
        energy_cost = (
            aeration_hours * ECONOMICS.aeration_power_kw * ECONOMICS.electricity_cost_per_kwh
            + heater_hours * ECONOMICS.heater_power_kw * ECONOMICS.electricity_cost_per_kwh
        )
        water_cost = water_exchanged_m3 * ECONOMICS.water_cost_per_m3
        self.total_operating_cost += ECONOMICS.fixed_cost_per_day + energy_cost + water_cost
        self.days_operated += 1

    def record_treatment(self, days: int = 1):
        self.total_treatment_cost += days * ECONOMICS.feed_price_per_kg * 100  # simplified

    def calculate_harvest_revenue(self, biomass_kg: float) -> float:
        return biomass_kg * ECONOMICS.market_price_per_kg

    @property
    def total_cost(self) -> float:
        return self.total_feed_cost + self.total_operating_cost + self.total_treatment_cost

    def profit(self, biomass_kg: float) -> float:
        revenue = self.calculate_harvest_revenue(biomass_kg)
        harvest_cost = biomass_kg * ECONOMICS.harvest_cost_per_kg
        return revenue - self.total_cost - harvest_cost
```

- [ ] **Step 4-5: Run tests, commit**

---

### Task 6: Create weather.py — Diel Cycle, Seasons, Storms

**Files:**
- Create: `src/agentic_rl/engine/weather.py`

- [ ] **Step 1-5: Write and test weather engine**

```python
# src/agentic_rl/engine/weather.py
"""Weather and environmental cycle engine.

Models:
- Diel (daily) temperature cycle: sinusoidal, peaking at 14:00
- Seasonal temperature variation
- Storm events: temperature drops, DO crashes, equipment risk
- Light/dark cycle for photosynthesis

Source: KB-01 Sec 2.6 (diel DO cycle), KB-02 Sec 12 (weather impact)
"""

import math
from ..constants import SYSTEM, photoperiod_hours


class WeatherEngine:
    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.base_air_temp: float = 30.0  # tropical baseline
        self.storm_active: bool = False
        self.storm_hours_remaining: int = 0
        self.storm_severity: float = 0.0  # 0-1

    def reset(self, seed: int = 42, base_temp: float = 30.0):
        import random
        self.rng = random.Random(seed)
        self.base_air_temp = base_temp
        self.storm_active = False
        self.storm_hours_remaining = 0
        self.storm_severity = 0.0

    def get_conditions(self, day: int, hour: int) -> dict:
        """Get current weather conditions.

        Returns dict with: air_temp, is_daytime, solar_intensity, wind_speed, cloud_cover
        """
        # Diel temperature cycle (KB-01 Sec 2.6: min at dawn ~6AM, max at ~2PM)
        hour_angle = (hour - 14) * (2 * math.pi / 24)
        diel_variation = 3.0 * math.cos(hour_angle)  # +/- 3C

        # Seasonal variation (tropical: small, +/- 2C)
        seasonal = 2.0 * math.sin(2 * math.pi * (day - 80) / 365)

        # Storm effect
        storm_temp_drop = 0.0
        if self.storm_active:
            storm_temp_drop = self.storm_severity * 8.0  # up to 8C drop

        air_temp = self.base_air_temp + diel_variation + seasonal - storm_temp_drop

        # Daylight (KB-03 Sec 15.3)
        p_hours = photoperiod_hours(day, SYSTEM.latitude)
        sunrise = 12.0 - p_hours / 2
        sunset = 12.0 + p_hours / 2
        is_daytime = sunrise <= hour < sunset

        # Solar intensity (bell curve peaking at noon)
        if is_daytime:
            solar_peak = 800.0  # W/m2 tropical peak
            solar_angle = (hour - 12) * (math.pi / p_hours)
            solar_intensity = solar_peak * max(0, math.cos(solar_angle))
            if self.storm_active:
                solar_intensity *= (1.0 - self.storm_severity * 0.8)
        else:
            solar_intensity = 0.0

        # Wind
        wind_base = 2.0 + 1.0 * math.sin(2 * math.pi * hour / 24)
        if self.storm_active:
            wind_base += self.storm_severity * 15.0

        return {
            "air_temp": air_temp,
            "is_daytime": is_daytime,
            "solar_intensity": solar_intensity,
            "wind_speed": wind_base,
            "cloud_cover": self.storm_severity if self.storm_active else 0.1,
            "storm_active": self.storm_active,
            "photoperiod_hours": p_hours,
        }

    def step(self, hour: int):
        """Advance weather state by 1 hour."""
        if self.storm_active:
            self.storm_hours_remaining -= 1
            if self.storm_hours_remaining <= 0:
                self.storm_active = False
                self.storm_severity = 0.0

    def trigger_storm(self, severity: float = 0.5, duration_hours: int = 48):
        self.storm_active = True
        self.storm_severity = min(1.0, severity)
        self.storm_hours_remaining = duration_hours

    def check_random_storm(self, prob_per_hour: float = 0.0002) -> bool:
        """Random storm check. ~0.5% chance per day."""
        if self.storm_active:
            return False
        if self.rng.random() < prob_per_hour:
            severity = self.rng.uniform(0.3, 0.8)
            duration = self.rng.randint(12, 72)
            self.trigger_storm(severity, duration)
            return True
        return False

    def weather_forecast(self, day: int, hour: int) -> str:
        """Generate natural language weather forecast for the agent."""
        conditions = self.get_conditions(day, hour)
        parts = []
        parts.append(f"Air temp: {conditions['air_temp']:.1f}C")
        if conditions['storm_active']:
            parts.append(f"STORM ACTIVE (severity: {self.storm_severity:.0%}, "
                        f"{self.storm_hours_remaining}h remaining)")
        elif conditions['cloud_cover'] < 0.3:
            parts.append("Clear skies")
        else:
            parts.append("Partly cloudy")
        if conditions['is_daytime']:
            parts.append(f"Daylight, solar: {conditions['solar_intensity']:.0f} W/m2")
        else:
            parts.append("Nighttime")
        return ". ".join(parts)
```

---

### Task 7: Create events.py — Event Scheduler

**Files:**
- Create: `src/agentic_rl/engine/events.py`

- [ ] **Step 1-5: Write events engine**

```python
# src/agentic_rl/engine/events.py
"""Event system — scheduled and random events that challenge the agent.

Event types:
1. Disease outbreak (SEIR trigger)
2. Storm (temperature drop, DO crash, equipment risk)
3. Equipment failure (aerator, biofilter, heater)
4. Algae bloom (DO supersaturation then crash)
5. Feed delivery delay (feed runs out)
6. Market price change (economic pressure)
7. Power outage (all systems down temporarily)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Event:
    """A scheduled or triggered event."""
    type: str           # 'disease', 'storm', 'equipment_failure', 'algae_bloom', 'feed_shortage', 'price_change', 'power_outage'
    trigger_hour: int   # hour in episode when event activates
    severity: float     # 0.0-1.0
    duration_hours: int # how long it lasts
    description: str    # human-readable description
    active: bool = False
    hours_remaining: int = 0

    # Equipment failure specifics
    equipment: str = ""  # 'aerator', 'biofilter', 'heater'

    # Price change specifics
    price_multiplier: float = 1.0


class EventScheduler:
    """Manages event scheduling and activation."""

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.scheduled_events: List[Event] = []
        self.active_events: List[Event] = []
        self.past_events: List[Event] = []
        self.current_hour: int = 0

    def reset(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.scheduled_events = []
        self.active_events = []
        self.past_events = []
        self.current_hour = 0

    def schedule(self, event: Event):
        """Add an event to the schedule."""
        self.scheduled_events.append(event)
        self.scheduled_events.sort(key=lambda e: e.trigger_hour)

    def step(self, hour: int) -> List[Event]:
        """Advance to given hour. Returns newly activated events."""
        self.current_hour = hour
        newly_activated = []

        # Check scheduled events
        remaining = []
        for event in self.scheduled_events:
            if event.trigger_hour <= hour:
                event.active = True
                event.hours_remaining = event.duration_hours
                self.active_events.append(event)
                newly_activated.append(event)
            else:
                remaining.append(event)
        self.scheduled_events = remaining

        # Update active events
        still_active = []
        for event in self.active_events:
            event.hours_remaining -= 1
            if event.hours_remaining <= 0:
                event.active = False
                self.past_events.append(event)
            else:
                still_active.append(event)
        self.active_events = still_active

        return newly_activated

    def has_active(self, event_type: str) -> bool:
        return any(e.type == event_type for e in self.active_events)

    def get_active_severity(self, event_type: str) -> float:
        for e in self.active_events:
            if e.type == event_type:
                return e.severity
        return 0.0

    def get_alerts(self) -> List[str]:
        """Get human-readable alert strings for active events."""
        return [e.description for e in self.active_events]

    def equipment_working(self, equipment: str) -> bool:
        """Check if a specific piece of equipment is functioning."""
        for e in self.active_events:
            if e.type == "equipment_failure" and e.equipment == equipment:
                return False
        return True
```

---

### Task 8: Create simulator.py — The Orchestrator

This is the brain that ties all subsystems together. One `step()` call advances the entire farm by 1 hour.

**Files:**
- Create: `src/agentic_rl/engine/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_simulator.py
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
```

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Write simulator.py**

```python
# src/agentic_rl/engine/simulator.py
"""Fish Farm Simulator — orchestrates all subsystems.

This is the central class that:
1. Holds all engine instances (water, fish, disease, economics, weather, events)
2. Processes agent actions
3. Advances all subsystems by 1 hour per step()
4. Returns complete state dicts

The cascade dynamics (overfeed → ammonia → DO crash → mortality)
emerge naturally from the coupled subsystem interactions.
"""

from typing import Dict, Any, Optional, List
from .water_quality import WaterQualityEngine
from .fish_biology import FishBiologyEngine
from .disease import DiseaseEngine
from .economics import EconomicsEngine
from .weather import WeatherEngine
from .events import EventScheduler, Event
from ..constants import SYSTEM, TILAPIA, WATER, ECONOMICS as ECON_CONST


class FishFarmSimulator:
    """Complete RAS fish farm simulation."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        import random
        self.rng = random.Random(seed)

        # Subsystems
        self.water = WaterQualityEngine(SYSTEM.tank_volume_m3, SYSTEM.tank_depth_m)
        self.fish = FishBiologyEngine()
        self.disease = DiseaseEngine()
        self.economics = EconomicsEngine()
        self.weather = WeatherEngine(seed)
        self.events = EventScheduler(seed)

        # Time tracking
        self.hour: int = 0
        self.day: int = 0
        self.total_hours: int = 0

        # Episode state
        self.harvested: bool = False
        self.catastrophe: bool = False
        self.feed_inventory_kg: float = 500.0  # starting feed stock

    def reset(
        self,
        initial_weight: float = TILAPIA.w_initial,
        initial_population: int = TILAPIA.N_initial,
        initial_temp: float = 28.0,
        initial_DO: float = 7.0,
        initial_TAN: float = 0.1,
        initial_pH: float = 7.5,
        day_of_year: int = 1,
        base_air_temp: float = 30.0,
        seed: Optional[int] = None,
        scheduled_events: Optional[List[Event]] = None,
    ) -> Dict[str, Any]:
        """Reset simulation to initial conditions."""
        if seed is not None:
            self.seed = seed
        import random
        self.rng = random.Random(self.seed)

        self.water.reset(initial_temp, initial_DO, initial_TAN, initial_pH, NO2=0.05)
        self.fish.reset(initial_weight, initial_population, day_of_year)
        self.disease.reset(initial_population)
        self.economics.reset()
        self.weather.reset(self.seed, base_air_temp)
        self.events.reset(self.seed)

        self.hour = 0
        self.day = 0
        self.total_hours = 0
        self.harvested = False
        self.catastrophe = False
        self.feed_inventory_kg = 500.0

        # Schedule events if provided
        if scheduled_events:
            for event in scheduled_events:
                self.events.schedule(event)

        return self.get_state()

    def step(
        self,
        feeding_rate: float,
        aeration_rate: float,
        heater_setting: float,
        water_exchange_rate: float,
        harvest: bool,
        treatment: str,
    ) -> Dict[str, Any]:
        """Advance simulation by 1 hour.

        Args:
            feeding_rate: 0.0-1.0 (fraction of max daily ration)
            aeration_rate: 0.0-1.0 (fraction of max aeration power)
            heater_setting: -1.0 to 1.0 (cool to heat)
            water_exchange_rate: 0.0-0.10 (fraction of volume per hour)
            harvest: True to harvest all fish (ends episode)
            treatment: 'none', 'antibiotics', 'salt', 'probiotics'
        Returns: full state dict
        """
        # Clamp inputs
        feeding_rate = max(0.0, min(1.0, feeding_rate))
        aeration_rate = max(0.0, min(1.0, aeration_rate))
        heater_setting = max(-1.0, min(1.0, heater_setting))
        water_exchange_rate = max(0.0, min(SYSTEM.max_exchange_rate, water_exchange_rate))

        # 1. Process events
        new_events = self.events.step(self.total_hours)

        # Equipment failures affect controls
        if not self.events.equipment_working("aerator"):
            aeration_rate = 0.0
        if not self.events.equipment_working("heater"):
            heater_setting = 0.0
        if not self.events.equipment_working("biofilter"):
            biofilter_eff = 0.1  # degraded
        else:
            biofilter_eff = WATER.biofilter_efficiency

        # Power outage kills everything
        if self.events.has_active("power_outage"):
            aeration_rate = 0.0
            heater_setting = 0.0
            biofilter_eff = 0.0

        # 2. Weather
        weather = self.weather.get_conditions(self.day + self.fish.day_of_year, self.hour)
        self.weather.step(self.hour)

        # 3. Feed inventory check
        biomass_kg = self.fish.biomass_kg
        feed_this_hour = (feeding_rate * TILAPIA.max_feeding_pct / 100.0
                         * biomass_kg / 24.0)
        if feed_this_hour > self.feed_inventory_kg:
            feed_this_hour = self.feed_inventory_kg
            feeding_rate = feed_this_hour / max(0.001, (TILAPIA.max_feeding_pct / 100.0 * biomass_kg / 24.0))
        self.feed_inventory_kg = max(0, self.feed_inventory_kg - feed_this_hour)

        # 4. Water quality update
        self.water.update_temperature(
            1.0, weather["air_temp"], heater_setting, SYSTEM.tank_volume_m3
        )
        self.water.step(
            dt_hours=1.0,
            fish_biomass_kg=biomass_kg,
            fish_weight_g=self.fish.weight_g,
            feeding_rate=feeding_rate,
            aeration_rate=aeration_rate,
            water_exchange_rate=water_exchange_rate,
            is_daytime=weather["is_daytime"],
            biofilter_efficiency=biofilter_eff,
        )

        # 5. Fish growth (daily — accumulated hourly)
        self.fish.grow(
            dt_hours=1.0,
            feeding_rate=feeding_rate,
            temperature=self.water.temperature,
            DO=self.water.DO,
            UIA=self.water.UIA,
            photoperiod_h=weather["photoperiod_hours"],
        )

        # 6. Mortality
        stocking_density = self.fish.population / SYSTEM.tank_volume_m3
        env_deaths = self.fish.apply_mortality(
            dt_hours=1.0,
            DO=self.water.DO,
            UIA=self.water.UIA,
            temperature=self.water.temperature,
            stocking_density=stocking_density,
        )

        # 7. Disease
        self.disease.check_stress_trigger(
            stress_level=self.fish.stress_level,
            DO=self.water.DO,
            UIA=self.water.UIA,
            temperature=self.water.temperature,
            stocking_density=stocking_density,
            rng_value=self.rng.random(),
        )

        if treatment != "none" and self.disease.is_active:
            self.disease.apply_treatment()

        disease_deaths = self.disease.step(1.0, self.fish.population)
        self.fish.population = max(0, self.fish.population - disease_deaths)
        self.fish.cumulative_mortality += disease_deaths
        self.disease.sync_population(self.fish.population)

        # 8. Economics
        self.economics.record_feed(feed_this_hour)
        if self.hour == 0:  # daily accounting
            self.economics.record_operating_day(
                aeration_hours=24.0 * aeration_rate,
                heater_hours=24.0 * abs(heater_setting),
                water_exchanged_m3=water_exchange_rate * SYSTEM.tank_volume_m3 * 24,
            )
        if treatment != "none" and self.disease.is_active:
            self.economics.record_treatment()

        # 9. Advance time
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day += 1
        self.total_hours += 1

        # 10. Harvest check
        if harvest:
            self.harvested = True

        # 11. Catastrophe check
        if self.fish.population <= 0:
            self.catastrophe = True
        elif self.fish.survival_rate < 0.2:
            self.catastrophe = True

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Return complete simulation state."""
        weather = self.weather.get_conditions(
            self.day + self.fish.day_of_year, self.hour
        )
        stocking_density = self.fish.population / SYSTEM.tank_volume_m3

        return {
            "fish": {
                "weight_g": round(self.fish.weight_g, 2),
                "population": self.fish.population,
                "biomass_kg": round(self.fish.biomass_kg, 2),
                "mortality_today": self.fish.mortality_today,
                "cumulative_mortality": self.fish.cumulative_mortality,
                "survival_rate": round(self.fish.survival_rate, 4),
                "stress_level": round(self.fish.stress_level, 3),
                "growth_rate_g_day": round(self.fish.growth_rate, 4),
                "fcr": round(self.fish.fcr, 3) if self.fish.fcr > 0 else 0.0,
                "feeding_response": self.fish.feeding_response(
                    self.water.temperature, self.water.DO,
                    self.water.UIA, self.fish.stress_level
                ),
                "stocking_density": round(stocking_density, 1),
            },
            "water": {
                "temperature": round(self.water.temperature, 2),
                "DO": round(self.water.DO, 2),
                "TAN": round(self.water.TAN, 4),
                "UIA": round(self.water.UIA, 5),
                "pH": round(self.water.pH, 2),
                "NO2": round(self.water.NO2, 4),
                "alkalinity": round(self.water.alkalinity, 1),
                "water_quality_score": round(self.water.get_water_quality_score(), 3),
            },
            "disease": {
                "active": self.disease.is_active,
                "infected": self.disease.infected,
                "exposed": self.disease.exposed,
                "recovered": self.disease.recovered,
                "treatment_active": self.disease.treatment_active,
                "total_disease_deaths": self.disease.total_disease_deaths,
            },
            "economics": {
                "total_feed_cost": round(self.economics.total_feed_cost, 2),
                "total_operating_cost": round(self.economics.total_operating_cost, 2),
                "total_cost": round(self.economics.total_cost, 2),
                "fish_value": round(self.economics.calculate_harvest_revenue(self.fish.biomass_kg), 2),
                "current_profit": round(self.economics.profit(self.fish.biomass_kg), 2),
                "feed_inventory_kg": round(self.feed_inventory_kg, 1),
            },
            "weather": {
                "air_temp": round(weather["air_temp"], 1),
                "is_daytime": weather["is_daytime"],
                "solar_intensity": round(weather["solar_intensity"], 0),
                "wind_speed": round(weather["wind_speed"], 1),
                "storm_active": weather["storm_active"],
                "forecast": self.weather.weather_forecast(
                    self.day + self.fish.day_of_year, self.hour
                ),
            },
            "time": {
                "hour": self.hour,
                "day": self.day,
                "total_hours": self.total_hours,
                "day_of_year": self.fish.day_of_year + self.day,
            },
            "events": {
                "active_events": self.events.get_alerts(),
                "equipment": {
                    "aerator": self.events.equipment_working("aerator"),
                    "biofilter": self.events.equipment_working("biofilter"),
                    "heater": self.events.equipment_working("heater"),
                },
            },
            "harvested": self.harvested,
            "catastrophe": self.catastrophe,
            "done": self.harvested or self.catastrophe,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/rahulrajpurohit/IdeaProjects/Agentic-Reinforcement-Learning && python -m pytest tests/test_simulator.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rl/engine/ tests/
git commit -m "feat: add complete fish farm simulator with weather, disease, economics, events"
```

---

## Chunk 3: Pydantic Models, Tasks, Graders

### Task 9: Rewrite models.py — FarmAction, FarmObservation, FarmState

**Files:**
- Rewrite: `src/agentic_rl/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write test**

```python
# tests/test_models.py
import pytest
from agentic_rl.models import FarmAction, FarmObservation, FarmState


class TestFarmAction:
    def test_default_action_valid(self):
        action = FarmAction()
        assert 0.0 <= action.feeding_rate <= 1.0
        assert 0.0 <= action.aeration_rate <= 1.0

    def test_action_clamping(self):
        action = FarmAction(feeding_rate=1.5, aeration_rate=-0.5)
        # Pydantic should clamp or reject
        assert action.feeding_rate <= 1.0
        assert action.aeration_rate >= 0.0

    def test_action_schema_has_descriptions(self):
        schema = FarmAction.model_json_schema()
        assert "feeding_rate" in schema["properties"]
        assert "description" in schema["properties"]["feeding_rate"]


class TestFarmObservation:
    def test_observation_has_required_fields(self):
        obs = FarmObservation(
            done=False, reward=0.5,
            avg_fish_weight=50.0, population=10000,
            temperature=28.0, dissolved_oxygen=7.0,
            ph=7.5, ammonia=0.1, nitrite=0.05,
            day_in_cycle=1, time_of_day=8,
        )
        assert obs.done is False
        assert obs.reward == 0.5

    def test_observation_includes_feedback(self):
        obs = FarmObservation(
            done=False, reward=0.0,
            avg_fish_weight=50.0, population=10000,
            temperature=28.0, dissolved_oxygen=7.0,
            ph=7.5, ammonia=0.1, nitrite=0.05,
            day_in_cycle=1, time_of_day=8,
            feedback="Fish are feeding eagerly."
        )
        assert "eagerly" in obs.feedback


class TestFarmState:
    def test_state_has_episode_id(self):
        state = FarmState(episode_id="test-123")
        assert state.episode_id == "test-123"
```

- [ ] **Step 2-3: Rewrite models.py**

```python
# src/agentic_rl/models.py
"""OpenEnv type definitions for the Fish Farm environment.

FarmAction: What the agent can do each hour
FarmObservation: What the agent sees (partial observability — no ground truth disease state)
FarmState: Full internal state (for grading)
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class FarmAction(Action):
    """Agent's hourly farm management decision."""

    feeding_rate: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Feeding intensity (0=none, 0.3=conservative, 0.5=normal, 1.0=maximum ration). "
                    "Higher feeding grows fish faster but produces more ammonia and consumes more oxygen."
    )
    aeration_rate: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Aerator power (0=off, 0.5=half, 1.0=full). "
                    "Increases dissolved oxygen but costs electricity."
    )
    heater_setting: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Temperature control (-1.0=max cooling, 0=off, 1.0=max heating). "
                    "Adjusts water temperature toward optimal growth range."
    )
    water_exchange_rate: float = Field(
        default=0.02, ge=0.0, le=0.10,
        description="Fresh water exchange (0=none, 0.05=5%/hour, 0.10=10%/hour max). "
                    "Dilutes ammonia and refreshes oxygen but costs water."
    )
    harvest_decision: bool = Field(
        default=False,
        description="Set True to harvest all fish and end the episode. "
                    "Revenue depends on total biomass and market price."
    )
    treatment: str = Field(
        default="none",
        description="Disease treatment: 'none', 'antibiotics' (speeds recovery), "
                    "'salt' (reduces nitrite toxicity), 'probiotics' (boosts biofilter). "
                    "Treatments cost money."
    )


class FarmObservation(Observation):
    """What the agent observes after each step.

    Note: Disease infection count is NOT directly visible — the agent must
    infer disease from behavioral indicators (feeding response, mortality spikes).
    """

    # Fish status
    avg_fish_weight: float = Field(default=5.0, description="Average individual fish weight (grams)")
    population: int = Field(default=10000, description="Total fish count in tank")
    mortality_today: int = Field(default=0, description="Fish deaths in the last 24 hours")
    stress_level: float = Field(default=0.0, description="Fish stress index (0.0=calm, 1.0=critical)")
    feeding_response: str = Field(default="normal", description="Fish appetite: eager/normal/sluggish/refusing")
    biomass_kg: float = Field(default=50.0, description="Total fish biomass in kg")

    # Water quality
    temperature: float = Field(default=28.0, description="Water temperature (Celsius)")
    dissolved_oxygen: float = Field(default=7.0, description="Dissolved oxygen (mg/L). Below 3=danger, below 1=lethal")
    ph: float = Field(default=7.5, description="Water pH (6.5-8.5 optimal)")
    ammonia: float = Field(default=0.1, description="Total ammonia nitrogen TAN (mg/L). Above 2=dangerous")
    ammonia_toxic: float = Field(default=0.005, description="Unionized ammonia UIA (mg/L). Above 0.05=toxic")
    nitrite: float = Field(default=0.05, description="Nitrite NO2 (mg/L). Above 0.5=stress")
    water_quality_score: float = Field(default=1.0, description="Composite water quality (0-1)")

    # System status
    aerator_working: bool = Field(default=True, description="Is the aerator functioning?")
    biofilter_working: bool = Field(default=True, description="Is the biofilter functioning?")
    heater_working: bool = Field(default=True, description="Is the heater functioning?")
    feed_remaining_kg: float = Field(default=500.0, description="Feed inventory remaining (kg)")

    # Economics
    current_fish_value: float = Field(default=0.0, description="Current market value of all fish ($)")
    total_cost_so_far: float = Field(default=0.0, description="Cumulative operating cost ($)")
    current_profit: float = Field(default=0.0, description="Revenue - costs if harvested now ($)")

    # Context
    weather_forecast: str = Field(default="", description="Current weather conditions")
    day_in_cycle: int = Field(default=0, description="Days since stocking")
    time_of_day: int = Field(default=0, description="Hour (0-23)")
    alerts: List[str] = Field(default_factory=list, description="Active alerts and warnings")

    # Env standard
    feedback: str = Field(default="", description="Narrative feedback on the current situation")


class FarmState(State):
    """Full internal state — used by graders, NOT visible to agent.

    Contains ground truth disease status, exact biochemistry, etc.
    """

    task_id: str = Field(default="", description="Current task ID")
    is_complete: bool = Field(default=False)
    final_score: float = Field(default=0.0)
    max_hours: int = Field(default=168)

    # Full simulator snapshot
    sim_state: Dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4-5: Run tests, commit**

---

### Task 10: Rewrite tasks.py — 12 Task Scenarios with Graders

Each task is a complete scenario configuration: initial conditions, event schedule, episode length, reward weights, and grading criteria.

**Files:**
- Rewrite: `src/agentic_rl/tasks.py`
- Create: `graders/farm_graders.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write task definitions**

```python
# src/agentic_rl/tasks.py
"""12 task scenarios for the Fish Farm environment.

Easy (3):    Single-concern, short episodes, forgiving thresholds
Medium (4):  Multi-concern, events, cascading risks
Hard (3):    Full lifecycle, compound events, multi-objective optimization
Extreme (2): Everything at once, frontier-model difficulty

Each task dict contains:
- initial_conditions: starting state overrides
- events: scheduled Event objects
- episode_hours: max episode length
- reward_weights: per-component weights for reward calculation
- grader: name of grading function
- description: natural language task briefing for the agent
- difficulty: easy/medium/hard/extreme
"""

from typing import Any, Dict, List
from .engine.events import Event


def _make_tasks() -> Dict[str, Dict[str, Any]]:
    return {
        # =====================================================================
        # EASY TASKS — Single concern, learn one control at a time
        # =====================================================================

        "feeding_basics": {
            "difficulty": "easy",
            "episode_hours": 7 * 24,  # 7 days
            "description": (
                "You manage a healthy tilapia tank for 7 days. Your goal: feed the fish "
                "to achieve steady growth without overfeeding. Fish start at 50g, target 55g+. "
                "Keep FCR below 2.0. Zero fish should die from starvation or overfeeding."
            ),
            "initial_conditions": {
                "weight_g": 50.0, "population": 10000, "temp": 30.0,
                "DO": 7.0, "TAN": 0.1, "pH": 7.5, "day_of_year": 90,
            },
            "events": [],
            "reward_weights": {"growth": 0.5, "fcr": 0.3, "survival": 0.2},
            "grader": "feeding_grader",
            "target_weight": 55.0,
        },

        "oxygen_management": {
            "difficulty": "easy",
            "episode_hours": 3 * 24,  # 3 days
            "description": (
                "It's a hot week (air temp 35C). Dissolved oxygen is dropping. "
                "Your job: keep DO above 5.0 mg/L at all times using the aerator. "
                "Fish are already stocked at moderate density. Score based on "
                "minimum DO maintained and time spent in safe zone."
            ),
            "initial_conditions": {
                "weight_g": 100.0, "population": 8000, "temp": 32.0,
                "DO": 6.0, "TAN": 0.3, "pH": 7.5, "day_of_year": 180,
                "base_air_temp": 35.0,
            },
            "events": [],
            "reward_weights": {"do_stability": 0.7, "efficiency": 0.3},
            "grader": "oxygen_grader",
        },

        "water_quality_balance": {
            "difficulty": "easy",
            "episode_hours": 7 * 24,  # 7 days
            "description": (
                "Manage all water quality parameters simultaneously: keep DO > 5, "
                "ammonia (UIA) < 0.05, pH 6.5-8.5, temperature 27-32C. "
                "You have full control: feeding, aeration, heater, water exchange. "
                "Score = time-averaged water quality composite score."
            ),
            "initial_conditions": {
                "weight_g": 80.0, "population": 9000, "temp": 29.0,
                "DO": 7.0, "TAN": 0.2, "pH": 7.5, "day_of_year": 100,
            },
            "events": [],
            "reward_weights": {"water_quality": 0.8, "efficiency": 0.2},
            "grader": "water_quality_grader",
        },

        # =====================================================================
        # MEDIUM TASKS — Multi-concern, events, cascading risks
        # =====================================================================

        "temperature_stress": {
            "difficulty": "medium",
            "episode_hours": 5 * 24,  # 5 days
            "description": (
                "ALERT: A heat wave is hitting. Air temperature will reach 38C for "
                "3 days starting hour 24. You must manage water temperature using "
                "heater (cooling mode), reduce feeding (fish eat less in heat), and "
                "increase aeration (warm water holds less oxygen). "
                "Goal: Keep fish alive and growing through the crisis."
            ),
            "initial_conditions": {
                "weight_g": 120.0, "population": 8000, "temp": 30.0,
                "DO": 7.0, "TAN": 0.2, "pH": 7.5, "day_of_year": 200,
                "base_air_temp": 33.0,
            },
            "events": [
                Event(type="heat_wave", trigger_hour=24, severity=0.7,
                      duration_hours=72, description="HEAT WAVE: Air temp reaching 38C"),
            ],
            "reward_weights": {"survival": 0.4, "growth": 0.3, "water_quality": 0.3},
            "grader": "stress_survival_grader",
        },

        "ammonia_crisis": {
            "difficulty": "medium",
            "episode_hours": 3 * 24,  # 3 days
            "description": (
                "EMERGENCY: The biofilter has partially failed (50% capacity). "
                "Ammonia is rising. You must: reduce feeding immediately, increase "
                "water exchange, maintain aeration. Goal: prevent ammonia (UIA) from "
                "reaching lethal levels (>0.6 mg/L) while keeping fish alive."
            ),
            "initial_conditions": {
                "weight_g": 150.0, "population": 7000, "temp": 30.0,
                "DO": 6.5, "TAN": 1.5, "pH": 7.8, "day_of_year": 150,
            },
            "events": [
                Event(type="equipment_failure", trigger_hour=0, severity=0.5,
                      duration_hours=48, description="BIOFILTER FAILURE: 50% capacity",
                      equipment="biofilter"),
            ],
            "reward_weights": {"ammonia_control": 0.4, "survival": 0.4, "efficiency": 0.2},
            "grader": "ammonia_crisis_grader",
        },

        "disease_outbreak": {
            "difficulty": "medium",
            "episode_hours": 10 * 24,  # 10 days
            "description": (
                "Fish are showing signs of stress — sluggish feeding, elevated mortality. "
                "A disease may be developing. Watch for: increasing mortality, "
                "feeding refusal, and behavioral changes. If you detect disease, "
                "apply treatment ('antibiotics'). Also manage water quality to "
                "reduce stress. Goal: contain the outbreak with <10% total mortality."
            ),
            "initial_conditions": {
                "weight_g": 200.0, "population": 6000, "temp": 31.0,
                "DO": 6.0, "TAN": 0.5, "pH": 7.6, "day_of_year": 120,
            },
            "events": [
                # Disease triggers at hour 12 (day 0.5)
                Event(type="disease", trigger_hour=12, severity=0.4,
                      duration_hours=0, description="Disease pathogen introduced"),
            ],
            "reward_weights": {"survival": 0.4, "treatment_timing": 0.3, "water_quality": 0.3},
            "grader": "disease_grader",
        },

        "growth_optimization": {
            "difficulty": "medium",
            "episode_hours": 14 * 24,  # 14 days
            "description": (
                "Optimize fish growth over 2 weeks. Fish start at 80g, target 120g+. "
                "Balance aggressive feeding (faster growth) against water quality "
                "degradation (ammonia, DO). Achieve the best FCR possible while "
                "maximizing weight gain. Minimize mortality."
            ),
            "initial_conditions": {
                "weight_g": 80.0, "population": 9000, "temp": 30.0,
                "DO": 7.0, "TAN": 0.1, "pH": 7.5, "day_of_year": 90,
            },
            "events": [],
            "reward_weights": {"growth": 0.4, "fcr": 0.3, "survival": 0.2, "water_quality": 0.1},
            "grader": "growth_optimization_grader",
            "target_weight": 120.0,
        },

        # =====================================================================
        # HARD TASKS — Full lifecycle, compound events, multi-objective
        # =====================================================================

        "full_growout": {
            "difficulty": "hard",
            "episode_hours": 60 * 24,  # 60 days
            "description": (
                "Complete grow-out cycle: take fish from 20g fingerlings to market "
                "weight (400g+). Manage all systems over 60 days. Random weather, "
                "possible disease, possible equipment issues. Score on: final weight, "
                "survival rate, FCR, and profit. Decide when to harvest for max value."
            ),
            "initial_conditions": {
                "weight_g": 20.0, "population": 10000, "temp": 28.0,
                "DO": 7.5, "TAN": 0.05, "pH": 7.5, "day_of_year": 60,
            },
            "events": [],  # random events will occur naturally
            "reward_weights": {"profit": 0.3, "growth": 0.25, "survival": 0.25, "fcr": 0.1, "water_quality": 0.1},
            "grader": "full_growout_grader",
            "target_weight": 400.0,
        },

        "storm_response": {
            "difficulty": "hard",
            "episode_hours": 5 * 24,  # 5 days
            "description": (
                "SEVERE STORM WARNING: A major storm hits at hour 24. Effects: "
                "temperature drops 8C, power outage for 12 hours (aerators, heater, "
                "biofilter ALL down), high winds. After power returns, biofilter "
                "needs 24 hours to recover. Your job: maximize survival through the crisis. "
                "Pre-position your systems before the storm hits."
            ),
            "initial_conditions": {
                "weight_g": 200.0, "population": 8000, "temp": 30.0,
                "DO": 7.5, "TAN": 0.2, "pH": 7.5, "day_of_year": 180,
            },
            "events": [
                Event(type="storm", trigger_hour=24, severity=0.9,
                      duration_hours=48, description="SEVERE STORM: Temp -8C, high winds"),
                Event(type="power_outage", trigger_hour=24, severity=1.0,
                      duration_hours=12, description="POWER OUTAGE: All equipment offline"),
                Event(type="equipment_failure", trigger_hour=36, severity=0.7,
                      duration_hours=24, description="BIOFILTER RECOVERY: Reduced capacity post-storm",
                      equipment="biofilter"),
            ],
            "reward_weights": {"survival": 0.6, "water_quality": 0.3, "efficiency": 0.1},
            "grader": "storm_grader",
        },

        "multi_objective": {
            "difficulty": "hard",
            "episode_hours": 30 * 24,  # 30 days
            "description": (
                "Multi-objective challenge: simultaneously maximize profit, maintain "
                "fish welfare (stress < 0.3), and minimize environmental impact "
                "(water discharge quality). Score is the product of all three objectives — "
                "neglecting any one dimension tanks your score. "
                "Fish start at 100g in a moderately stocked tank."
            ),
            "initial_conditions": {
                "weight_g": 100.0, "population": 8000, "temp": 29.0,
                "DO": 7.0, "TAN": 0.2, "pH": 7.5, "day_of_year": 100,
            },
            "events": [],
            "reward_weights": {"profit": 0.33, "welfare": 0.34, "environment": 0.33},
            "grader": "multi_objective_grader",
        },

        # =====================================================================
        # EXTREME TASKS — Everything at once, frontier-model difficulty
        # =====================================================================

        "catastrophe_prevention": {
            "difficulty": "extreme",
            "episode_hours": 14 * 24,  # 14 days
            "description": (
                "CRITICAL SITUATION: Multiple simultaneous challenges. "
                "Day 1: Algae bloom developing (DO supersaturation then crash). "
                "Day 3: Equipment degradation (aerator at 60% capacity). "
                "Day 5: Disease outbreak detected. "
                "Day 7: Market price drops 40%. "
                "Day 10: Feed delivery delayed (inventory running low). "
                "Prevent mass mortality, optimize harvest timing despite falling prices, "
                "manage disease while resources are constrained. "
                "This task separates frontier models from basic agents."
            ),
            "initial_conditions": {
                "weight_g": 250.0, "population": 7000, "temp": 31.0,
                "DO": 8.0, "TAN": 0.4, "pH": 7.8, "day_of_year": 200,
            },
            "events": [
                Event(type="algae_bloom", trigger_hour=12, severity=0.6,
                      duration_hours=48, description="ALGAE BLOOM: DO swinging wildly"),
                Event(type="equipment_failure", trigger_hour=72, severity=0.4,
                      duration_hours=96, description="AERATOR DEGRADED: 60% capacity",
                      equipment="aerator"),
                Event(type="disease", trigger_hour=120, severity=0.5,
                      duration_hours=0, description="DISEASE DETECTED: Mortality rising"),
                Event(type="price_change", trigger_hour=168, severity=0.4,
                      duration_hours=168, description="MARKET CRASH: Fish price -40%",
                      price_multiplier=0.6),
                Event(type="feed_shortage", trigger_hour=240, severity=0.7,
                      duration_hours=48, description="FEED DELAYED: Inventory critical"),
            ],
            "reward_weights": {"survival": 0.3, "profit": 0.25, "water_quality": 0.2,
                              "disease_control": 0.15, "timing": 0.1},
            "grader": "catastrophe_grader",
        },

        "season_management": {
            "difficulty": "extreme",
            "episode_hours": 90 * 24,  # 90 days (full season)
            "description": (
                "Full 90-day season. Fish from 10g to market weight. "
                "Seasonal temperature changes (summer peak). Random storms, "
                "random disease. Feed inventory must be managed (deliveries "
                "every 14 days). Decide optimal harvest timing — market price "
                "fluctuates weekly. Score = ROI (profit / total investment). "
                "The best agents will achieve >50% ROI."
            ),
            "initial_conditions": {
                "weight_g": 10.0, "population": 10000, "temp": 27.0,
                "DO": 7.5, "TAN": 0.05, "pH": 7.5, "day_of_year": 60,
            },
            "events": [],  # all events occur randomly
            "reward_weights": {"roi": 0.4, "growth": 0.2, "survival": 0.2, "fcr": 0.1, "welfare": 0.1},
            "grader": "season_grader",
        },
    }


TASKS = _make_tasks()


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_all_tasks() -> List[Dict[str, str]]:
    return [
        {
            "task_id": tid,
            "difficulty": task["difficulty"],
            "description": task["description"],
            "episode_hours": task["episode_hours"],
        }
        for tid, task in TASKS.items()
    ]
```

- [ ] **Step 2: Write graders**

```python
# graders/farm_graders.py
"""Task-specific graders for the Fish Farm environment.

Each grader takes the final simulator state and episode history,
returns a score between 0.0 and 1.0 with partial credit.
"""

from typing import Any, Dict, List
from .base_grader import BaseGrader, GradeResult


class FarmGrader(BaseGrader):
    """Universal grader that dispatches to task-specific scoring."""

    def grade(
        self,
        task_id: str,
        final_state: Dict[str, Any],
        episode_history: List[Dict[str, Any]],
        task_config: Dict[str, Any],
        **kwargs,
    ) -> GradeResult:
        grader_name = task_config.get("grader", "default")
        method = getattr(self, f"_{grader_name}", self._default_grader)
        return method(final_state, episode_history, task_config)

    def _feeding_grader(self, state, history, config) -> GradeResult:
        fish = state["fish"]
        target = config.get("target_weight", 55.0)

        # Weight achievement (0-0.4)
        weight_score = min(1.0, fish["weight_g"] / target) * 0.4

        # FCR score (0-0.3): FCR < 1.6 = perfect, > 2.5 = zero
        fcr = fish.get("fcr", 3.0)
        if fcr <= 1.6:
            fcr_score = 0.3
        elif fcr <= 2.5:
            fcr_score = 0.3 * (2.5 - fcr) / (2.5 - 1.6)
        else:
            fcr_score = 0.0

        # Survival (0-0.3): > 99% = perfect
        survival = fish.get("survival_rate", 0.0)
        survival_score = min(1.0, survival / 0.99) * 0.3

        score = weight_score + fcr_score + survival_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Weight: {fish['weight_g']:.1f}g (target {target}g), "
                    f"FCR: {fcr:.2f}, Survival: {survival:.1%}",
        )

    def _oxygen_grader(self, state, history, config) -> GradeResult:
        # Score based on time DO stayed above 5.0
        safe_hours = sum(1 for h in history if h["water"]["DO"] >= 5.0)
        total_hours = max(1, len(history))
        do_score = safe_hours / total_hours * 0.7

        # Min DO achieved
        min_do = min(h["water"]["DO"] for h in history) if history else 0
        if min_do >= 4.0:
            safety_score = 0.3
        elif min_do >= 3.0:
            safety_score = 0.15
        else:
            safety_score = 0.0

        score = do_score + safety_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"DO safe: {safe_hours}/{total_hours} hours. Min DO: {min_do:.2f} mg/L",
        )

    def _water_quality_grader(self, state, history, config) -> GradeResult:
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        score = avg_wq * 0.8

        # Bonus for no violations
        violations = sum(1 for h in history
                        if h["water"]["DO"] < 3.0 or h["water"]["UIA"] > 0.1)
        if violations == 0:
            score += 0.2
        else:
            score += 0.2 * max(0, 1.0 - violations / 20)

        return GradeResult(
            score=round(min(1.0, score), 3),
            passed=score >= 0.5,
            feedback=f"Avg water quality: {avg_wq:.3f}. Violations: {violations}",
        )

    def _stress_survival_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        growth = state["fish"]["weight_g"] - config["initial_conditions"]["weight_g"]

        survival_score = min(1.0, survival / 0.95) * 0.5
        growth_score = max(0, min(1.0, growth / 10.0)) * 0.3

        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.2

        score = survival_score + growth_score + wq_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Survival: {survival:.1%}, Growth: +{growth:.1f}g, WQ: {avg_wq:.3f}",
        )

    def _ammonia_crisis_grader(self, state, history, config) -> GradeResult:
        peak_uia = max(h["water"]["UIA"] for h in history) if history else 1.0
        survival = state["fish"]["survival_rate"]

        # UIA control (0-0.4): peak UIA < 0.3 = good, > 0.6 = fail
        if peak_uia < 0.3:
            uia_score = 0.4
        elif peak_uia < 0.6:
            uia_score = 0.4 * (0.6 - peak_uia) / 0.3
        else:
            uia_score = 0.0

        survival_score = min(1.0, survival / 0.90) * 0.4
        efficiency_score = 0.2 if state["economics"]["total_cost"] < 200 else 0.1

        score = uia_score + survival_score + efficiency_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Peak UIA: {peak_uia:.4f} mg/L, Survival: {survival:.1%}",
        )

    def _disease_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        disease_deaths = state["disease"]["total_disease_deaths"]
        initial_pop = config["initial_conditions"]["population"]

        # Survival (0-0.4)
        survival_score = min(1.0, survival / 0.90) * 0.4

        # Treatment timing (0-0.3): did agent treat before 20% infected?
        treatment_step = None
        for i, h in enumerate(history):
            if h.get("disease", {}).get("treatment_active", False):
                treatment_step = i
                break
        if treatment_step is not None and treatment_step < len(history) * 0.3:
            timing_score = 0.3
        elif treatment_step is not None:
            timing_score = 0.15
        else:
            timing_score = 0.0

        # Water quality during crisis (0-0.3)
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.3

        score = survival_score + timing_score + wq_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Survival: {survival:.1%}, Disease deaths: {disease_deaths}, "
                    f"Treatment started: step {treatment_step}",
        )

    def _growth_optimization_grader(self, state, history, config) -> GradeResult:
        fish = state["fish"]
        target = config.get("target_weight", 120.0)

        weight_score = min(1.0, fish["weight_g"] / target) * 0.4
        fcr = fish.get("fcr", 3.0)
        fcr_score = max(0, min(1.0, (2.5 - fcr) / 1.0)) * 0.3
        survival_score = min(1.0, fish["survival_rate"] / 0.98) * 0.2
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.1

        score = weight_score + fcr_score + survival_score + wq_score
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Weight: {fish['weight_g']:.1f}g/{target}g, FCR: {fcr:.2f}")

    def _full_growout_grader(self, state, history, config) -> GradeResult:
        fish = state["fish"]
        econ = state["economics"]
        target = config.get("target_weight", 400.0)

        profit_score = max(0, min(1.0, econ["current_profit"] / 5000)) * 0.3
        weight_score = min(1.0, fish["weight_g"] / target) * 0.25
        survival_score = min(1.0, fish["survival_rate"] / 0.85) * 0.25
        fcr = fish.get("fcr", 3.0)
        fcr_score = max(0, min(1.0, (2.5 - fcr) / 1.0)) * 0.1
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.1

        score = profit_score + weight_score + survival_score + fcr_score + wq_score
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Weight: {fish['weight_g']:.1f}g, Profit: ${econ['current_profit']:.0f}, "
                                  f"Survival: {fish['survival_rate']:.1%}")

    def _storm_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        survival_score = min(1.0, survival / 0.80) * 0.6
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.3
        efficiency_score = 0.1  # base credit

        score = survival_score + wq_score + efficiency_score
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Survival: {survival:.1%} through storm")

    def _multi_objective_grader(self, state, history, config) -> GradeResult:
        profit = max(0, state["economics"]["current_profit"])
        profit_norm = min(1.0, profit / 3000)

        avg_stress = sum(h["fish"]["stress_level"] for h in history) / max(1, len(history))
        welfare = max(0, 1.0 - avg_stress / 0.3)

        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))

        # Pareto product
        score = (profit_norm * welfare * avg_wq) ** (1/3)  # geometric mean
        return GradeResult(score=round(score, 3), passed=score >= 0.4,
                          feedback=f"Profit: ${profit:.0f}, Welfare: {welfare:.2f}, WQ: {avg_wq:.3f}")

    def _catastrophe_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        profit = state["economics"]["current_profit"]
        disease_deaths = state["disease"]["total_disease_deaths"]

        survival_score = min(1.0, survival / 0.70) * 0.3
        profit_score = max(0, min(1.0, (profit + 1000) / 3000)) * 0.25
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.2
        disease_score = max(0, 1.0 - disease_deaths / 500) * 0.15
        # Timing score: did they harvest before price crashed?
        timing_score = 0.1 if state["harvested"] else 0.0

        score = survival_score + profit_score + wq_score + disease_score + timing_score
        return GradeResult(score=round(score, 3), passed=score >= 0.3,
                          feedback=f"Catastrophe survival: {survival:.1%}, Profit: ${profit:.0f}")

    def _season_grader(self, state, history, config) -> GradeResult:
        econ = state["economics"]
        total_investment = econ["total_cost"] + config["initial_conditions"]["population"] * 0.05
        if total_investment > 0:
            roi = econ["current_profit"] / total_investment
        else:
            roi = 0

        roi_score = min(1.0, max(0, roi / 0.5)) * 0.4
        fish = state["fish"]
        growth_score = min(1.0, fish["weight_g"] / 400) * 0.2
        survival_score = min(1.0, fish["survival_rate"] / 0.80) * 0.2
        fcr = fish.get("fcr", 3.0)
        fcr_score = max(0, min(1.0, (2.5 - fcr) / 1.0)) * 0.1
        avg_stress = sum(h["fish"]["stress_level"] for h in history) / max(1, len(history))
        welfare_score = max(0, 1.0 - avg_stress / 0.3) * 0.1

        score = roi_score + growth_score + survival_score + fcr_score + welfare_score
        return GradeResult(score=round(score, 3), passed=score >= 0.3,
                          feedback=f"ROI: {roi:.1%}, Weight: {fish['weight_g']:.0f}g, "
                                  f"Survival: {fish['survival_rate']:.1%}")

    def _default_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        score = survival * 0.5
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        score += avg_wq * 0.5
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Default grader: survival={survival:.1%}, WQ={avg_wq:.3f}")
```

- [ ] **Step 3-5: Run tests, commit**

---

## Chunk 4: Environment, Server, Inference, Deployment

### Task 11: Rewrite environment.py — FishFarmEnvironment

**Files:**
- Rewrite: `src/agentic_rl/server/environment.py`
- Test: `tests/test_environment.py`

- [ ] **Step 1-5: Write environment**

The FishFarmEnvironment wraps the simulator and translates between OpenEnv's Action/Observation/State types and the simulator's dict-based interface.

Key behaviors:
- `reset(task_id)` → configures simulator with task's initial conditions and events
- `step(action)` → runs simulator, calculates task-specific reward, checks done
- `state` → returns FarmState with full internal snapshot
- Tracks episode history for grader

---

### Task 12: Rewrite app.py — FastAPI Server

**Files:**
- Rewrite: `src/agentic_rl/server/app.py`

Custom endpoints:
- `GET /tasks` → lists all 12 tasks with descriptions and action schema
- `POST /grader` → runs task-specific grader on completed episode
- `POST /baseline` → runs inference.py internally and returns scores

---

### Task 13: Create inference.py — LLM Agent

**Files:**
- Create: `inference.py` (root directory)

The inference script is where the LLM agent lives. It:
1. Connects to the environment via HTTP
2. For each task, calls `/reset` with `task_id`
3. Reads the observation and builds a natural language prompt
4. Calls the LLM via OpenAI client (using `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
5. Parses the LLM's response into a `FarmAction`
6. Posts the action to `/step`
7. Repeats until `done=True`
8. Records the final reward score

**LLM Prompt Design (critical for scoring):**

The prompt gives the agent:
- **Role**: "You are an expert aquaculture farm manager"
- **Task briefing**: The task description
- **Current readings**: Water temp, DO, pH, ammonia, nitrite (formatted like a sensor dashboard)
- **Fish status**: Weight, population, mortality, feeding response, stress
- **Economics**: Cost so far, fish value, profit estimate
- **Alerts**: Any active warnings
- **Available actions**: With descriptions and valid ranges
- **Decision history**: Last 3-5 observations for trend detection
- **Output format**: JSON matching FarmAction schema

**Prompt template** (abbreviated):

```
You are an expert tilapia aquaculture manager operating a Recirculating Aquaculture System (RAS).

TASK: {task_description}

CURRENT READINGS (Hour {hour}, Day {day}):
  Water Temperature: {temp}C (optimal: 27-32C)
  Dissolved Oxygen:  {DO} mg/L (danger below 3.0, lethal below 1.0)
  pH:                {pH} (optimal: 6.5-8.5)
  Ammonia (TAN):     {TAN} mg/L
  Toxic Ammonia:     {UIA} mg/L (toxic above 0.05)
  Nitrite:           {NO2} mg/L

FISH STATUS:
  Average Weight: {weight}g | Population: {pop} | Today's Mortality: {mort}
  Stress Level: {stress} | Feeding Response: {response}
  Biomass: {biomass}kg | FCR: {fcr}

ECONOMICS:
  Fish Value: ${value} | Cost So Far: ${cost} | Current Profit: ${profit}
  Feed Remaining: {feed}kg

WEATHER: {forecast}
ALERTS: {alerts}

EQUIPMENT: Aerator={aerator}, Biofilter={biofilter}, Heater={heater}

RECENT TREND (last 3 readings):
{history_summary}

Based on this data, decide your actions for the next hour.
Respond with ONLY a JSON object:
{
  "feeding_rate": 0.0-1.0,
  "aeration_rate": 0.0-1.0,
  "heater_setting": -1.0 to 1.0,
  "water_exchange_rate": 0.0-0.10,
  "harvest_decision": true/false,
  "treatment": "none"/"antibiotics"/"salt"/"probiotics"
}
```

Must complete all tasks in < 20 minutes on 2 vCPU / 8GB. Strategy: run easy tasks (short episodes) first, use batched requests, limit history window.

---

### Task 14: Fix openenv.yaml + Dockerfile + requirements.txt

- [ ] **Step 1: Fix openenv.yaml**

```yaml
spec_version: 1
name: fish_farm_env
version: "1.0.0"
type: space
runtime: fastapi
app: src.agentic_rl.server.app:app
port: 8000
description: "AI agent manages a Nile Tilapia RAS fish farm — bioenergetic growth, water quality, disease, economics"
```

- [ ] **Step 2: Update requirements.txt**

```
openenv-core>=0.2.2
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
httpx>=0.25.0
openai>=1.0.0
numpy>=1.24.0
huggingface-hub>=0.22.0
pytest>=7.0.0
```

- [ ] **Step 3: Update Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "src.agentic_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### Task 15: Write README.md

Full documentation covering:
- Why aquaculture (the $300B industry gap)
- How the simulation works (biological cascade)
- Action/Observation space definitions
- All 12 tasks with difficulty levels
- Setup and usage instructions
- Baseline scores
- Research citations

---

### Task 16: Deploy to HuggingFace Spaces

- [ ] **Step 1: Test locally**

```bash
cd /Users/rahulrajpurohit/IdeaProjects/Agentic-Reinforcement-Learning
docker build -t fishfarm .
docker run -p 8000:8000 fishfarm
```

- [ ] **Step 2: Verify endpoints**

```bash
curl http://localhost:8000/tasks
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "feeding_basics"}'
```

- [ ] **Step 3: Run openenv validate**

```bash
openenv validate
```

- [ ] **Step 4: Deploy**

```bash
openenv push --repo-id rahul24rajpurohit/fish-farm-env
```

- [ ] **Step 5: Verify HF Space responds**

---

## Implementation Order (Critical Path)

```
Day 1 (April 1):
├── Task 1:  constants.py          (30 min)   ← FOUNDATION
├── Task 2:  water_quality.py      (2 hours)  ← CORE PHYSICS
├── Task 3:  fish_biology.py       (2 hours)  ← CORE BIOLOGY
├── Task 4:  disease.py            (1 hour)
├── Task 5:  economics.py          (30 min)
├── Task 6:  weather.py            (30 min)
├── Task 7:  events.py             (30 min)
└── Task 8:  simulator.py          (2 hours)  ← INTEGRATION

Day 2 (April 2):
├── Task 9:  models.py             (1 hour)
├── Task 10: tasks.py + graders    (3 hours)  ← 12 TASKS
├── Task 11: environment.py        (2 hours)
└── Task 12: app.py                (1 hour)

Day 3 (April 3):
├── Task 13: inference.py          (3 hours)  ← LLM AGENT
├── Task 14: configs (yaml/docker) (30 min)
├── Task 15: README.md             (1 hour)
├── Task 16: Deploy + verify       (2 hours)
└── Buffer: Fix bugs, tune params  (2 hours)
```

Total estimated: ~22 hours of focused work across 3 days.

---

## Key Equations Reference (for implementation)

### Growth (every hour, applied to dt=1/24 day)
```
dW/dt = [h * pi * f * b * (1-a) * tau(T) * sigma(DO) * v(UIA)] * W^0.6277
      - [k_min * exp(s*(T-T_min))] * W^0.8373
```

### DO Mass Balance (every 6 minutes)
```
dDO/dt = P_photo - FR*biomass/V - 4.57*K_NR*TAN - DO_water
       + K_a*(DO_sat-DO) + A_mech + Q_ex*(DO_in-DO)
```

### TAN Mass Balance (every 6 minutes)
```
dTAN/dt = Feed*Protein*0.16*N_wasted*1.2/V - K_NR*TAN - Q_ex*TAN
```

### UIA (instantaneous)
```
UIA = TAN / (1 + 10^(pKa - pH))
pKa = 0.09018 + 2729.92 / (T + 273.15)
```

### Mortality
```
M = base_mortality * stress_multiplier + acute_lethal
stress_multiplier = 1 + 50*(max(0, stress-0.3))^2
```

### SEIR (every hour, applied to dt=1/24 day)
```
dS = -beta*S*I/N
dE = beta*S*I/N - sigma*E
dI = sigma*E - gamma*I - alpha*I
dR = gamma*I
```

---

## Success Criteria

| Criterion | Target | Weight |
|-----------|--------|--------|
| Real-world utility | Fills the #1 gap in aquaculture AI (30/30) | 30% |
| Task & grader quality | 12 tasks, 3+ difficulty levels, partial credit graders | 25% |
| Environment design | Biologically accurate cascade dynamics, clean state management | 20% |
| Code quality & spec compliance | openenv validate passes, Docker builds, baseline reproduces | 15% |
| Creativity & novelty | First-ever OpenEnv aquaculture env, 13 coupled state variables | 10% |

### Baseline to Beat
Q-learning achieved 79% less feed and zero mortality vs Bang-Bang control (Chahid et al. 2021). Our environment should reproduce this gap — a good LLM agent should beat a naive constant-feeding strategy.
