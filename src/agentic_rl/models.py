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
        description="Disease treatment: 'none', 'antibiotics' (speeds recovery, harms biofilter), "
                    "'salt' (reduces nitrite toxicity), 'probiotics' (boosts biofilter), "
                    "'vaccination' ($100, prevents 80% of future infections). "
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
    cumulative_mortality: int = Field(default=0, description="Total deaths since stocking")
    survival_rate: float = Field(default=1.0, description="Fraction of original population still alive")
    stress_level: float = Field(default=0.0, description="Fish stress index (0.0=calm, 1.0=critical)")
    feeding_response: str = Field(default="normal", description="Fish appetite: eager/normal/sluggish/refusing")
    biomass_kg: float = Field(default=50.0, description="Total fish biomass in kg")
    growth_rate_g_day: float = Field(default=0.0, description="Current growth rate (g/day)")
    fcr: float = Field(default=0.0, description="Feed conversion ratio (kg feed / kg gain). Target <2.0")
    sgr: float = Field(default=0.0, description="Specific growth rate (%/day)")
    stocking_density: float = Field(default=50.0, description="Fish per cubic meter")

    # Water quality
    temperature: float = Field(default=28.0, description="Water temperature (Celsius)")
    dissolved_oxygen: float = Field(default=7.0, description="Dissolved oxygen (mg/L). Below 3=danger, below 1=lethal")
    ph: float = Field(default=7.5, description="Water pH (6.5-8.5 optimal)")
    ammonia: float = Field(default=0.1, description="Total ammonia nitrogen TAN (mg/L). Above 2=dangerous")
    ammonia_toxic: float = Field(default=0.005, description="Unionized ammonia UIA (mg/L). Above 0.05=toxic")
    nitrite: float = Field(default=0.05, description="Nitrite NO2 (mg/L). Above 0.5=stress")
    nitrate: float = Field(default=0.0, description="Nitrate NO3 (mg/L). Product of nitrification")
    water_quality_score: float = Field(default=1.0, description="Composite water quality (0-1)")
    algae_bloom: bool = Field(default=False, description="Is algae bloom active (DO swings)")
    nighttime_do_risk: float = Field(default=0.0, description="Nighttime DO crash risk (0=safe, 1=imminent). Increase aeration if high.")

    # System status
    aerator_working: bool = Field(default=True, description="Is the aerator functioning?")
    biofilter_working: bool = Field(default=True, description="Is the biofilter functioning?")
    heater_working: bool = Field(default=True, description="Is the heater functioning?")
    feed_remaining_kg: float = Field(default=500.0, description="Feed inventory remaining (kg)")

    # Economics
    current_fish_value: float = Field(default=0.0, description="Current market value of all fish ($)")
    total_cost_so_far: float = Field(default=0.0, description="Cumulative operating cost ($)")
    current_profit: float = Field(default=0.0, description="Revenue - costs if harvested now ($)")
    feed_price_per_kg: float = Field(default=0.50, description="Current feed price (stochastic, $/kg)")
    market_price_multiplier: float = Field(default=1.0, description="Seasonal market price factor (1.0=normal)")
    marginal_cost_per_hour: float = Field(default=0.0, description="Cost of last hour of operation ($)")
    roi_pct: float = Field(default=0.0, description="Return on investment (%)")

    # Weather
    weather_forecast: str = Field(default="", description="Current weather conditions")
    is_daytime: bool = Field(default=True, description="Is it daytime (affects photosynthesis/DO)")
    storm_active: bool = Field(default=False, description="Is a storm currently active")
    humidity: float = Field(default=75.0, description="Relative humidity (%)")

    # Context
    day_in_cycle: int = Field(default=0, description="Days since stocking")
    time_of_day: int = Field(default=0, description="Hour (0-23)")
    day_of_year: int = Field(default=1, description="Calendar day (1-365, for seasonal context)")
    alerts: List[str] = Field(default_factory=list, description="Active alerts and warnings")

    # Disease signals (partial observability — no infection count, but behavioral indicators)
    disease_suspected: bool = Field(default=False, description="Behavioral signs suggest disease (mortality+appetite)")

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
