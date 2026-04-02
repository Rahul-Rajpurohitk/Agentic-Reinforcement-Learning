"""Economic model — feed cost, energy cost, treatment, fish valuation, profit/loss.

Source: KB-02 Sec 7, KB-03 Sec 6, KB-03 Sec 6.1-6.3

Models the real economic structure of RAS aquaculture:
- Feed = 50-70% of operating costs (the dominant lever)
- Energy (aeration + heating) = 15-25%
- Water = 5-10%
- Fixed overhead (labor, depreciation) = daily cost
- Treatment costs for disease management
- Fingerling stocking cost (upfront investment)
- Dead fish disposal cost
- Market price with size premium / weight-dependent pricing
- Harvest cost (logistics, processing)

Key insight: The agent must balance growth speed (more feed → more revenue)
against input costs (feed, energy, water) and risk costs (disease treatment,
mortality losses, disposal).
"""

import math
from ..constants import ECONOMICS, DISEASE, TILAPIA


class EconomicsEngine:
    """Tracks all economic flows in the fish farm simulation.

    Costs are tracked hourly for accuracy (not approximated at day boundaries).
    Revenue is calculated at harvest based on biomass and market price.

    Enhanced features:
    - Stochastic feed price (mean-reverting, soybean-linked, KB-03 Sec 6.2)
    - Seasonal market price variation (demand peaks around holidays)
    - Marginal cost tracking for harvest timing optimization
    """

    def __init__(self):
        # Cumulative cost trackers
        self.total_feed_cost: float = 0.0
        self.total_energy_cost: float = 0.0
        self.total_water_cost: float = 0.0
        self.total_fixed_cost: float = 0.0
        self.total_treatment_cost: float = 0.0
        self.total_fingerling_cost: float = 0.0
        self.total_disposal_cost: float = 0.0

        # Volume trackers
        self.total_feed_kg: float = 0.0
        self.total_energy_kwh: float = 0.0
        self.total_water_exchanged_m3: float = 0.0
        self.hours_operated: int = 0

        # Market state
        self.market_price_multiplier: float = 1.0  # for price events

        # Stochastic feed price (mean-reverting Ornstein-Uhlenbeck)
        # KB-03 Sec 6.2: Feed cost linked to soybean prices with volatility
        self.feed_price_current: float = ECONOMICS.feed_price_per_kg
        self._feed_price_mean: float = ECONOMICS.feed_price_per_kg
        self._feed_price_kappa: float = 0.02  # mean-reversion speed (per day)
        self._feed_price_sigma: float = 0.01  # daily volatility

        # Marginal cost tracking (for harvest timing signals)
        self._last_hour_cost: float = 0.0
        self._last_hour_revenue_delta: float = 0.0

    def reset(self, initial_population: int = 0):
        """Reset economics for new episode.

        Args:
            initial_population: Number of fingerlings stocked (for stocking cost).
        """
        self.total_feed_cost = 0.0
        self.total_energy_cost = 0.0
        self.total_water_cost = 0.0
        self.total_fixed_cost = 0.0
        self.total_treatment_cost = 0.0
        self.total_disposal_cost = 0.0
        self.total_feed_kg = 0.0
        self.total_energy_kwh = 0.0
        self.total_water_exchanged_m3 = 0.0
        self.hours_operated = 0
        self.market_price_multiplier = 1.0
        self.feed_price_current = ECONOMICS.feed_price_per_kg
        self._last_hour_cost = 0.0
        self._last_hour_revenue_delta = 0.0

        # Fingerling stocking cost (upfront investment)
        if initial_population > 0:
            self.total_fingerling_cost = initial_population * ECONOMICS.fingerling_cost

    def record_hourly_costs(
        self,
        feed_kg: float,
        aeration_rate: float,
        heater_setting: float,
        water_exchange_rate: float,
        tank_volume_m3: float,
        rng_value: float = 0.0,
    ):
        """Record all costs for a single hour of operation.

        Called every simulation step (1 hour) for accurate cost tracking.
        Feed price follows a mean-reverting stochastic process (KB-03 Sec 6.2).

        Args:
            feed_kg: kg of feed consumed this hour.
            aeration_rate: Aeration power level 0..1.
            heater_setting: Heater/chiller setting -1..1.
            water_exchange_rate: Fraction of tank volume exchanged this hour.
            tank_volume_m3: Tank volume (m³).
            rng_value: Random normal value for feed price stochastic update.
        """
        # Update stochastic feed price (Ornstein-Uhlenbeck, hourly step)
        # dP = kappa*(mu - P)*dt + sigma*dW
        dt_day = 1.0 / 24.0
        self.feed_price_current += (
            self._feed_price_kappa * (self._feed_price_mean - self.feed_price_current) * dt_day
            + self._feed_price_sigma * rng_value * math.sqrt(dt_day)
        )
        # Bound feed price to realistic range (±40% of mean)
        self.feed_price_current = max(
            self._feed_price_mean * 0.6,
            min(self._feed_price_mean * 1.4, self.feed_price_current)
        )

        # Feed cost (uses current stochastic price)
        feed_cost = feed_kg * self.feed_price_current
        self.total_feed_cost += feed_cost
        self.total_feed_kg += feed_kg

        # Energy cost (aeration + heating, per hour)
        aeration_kwh = aeration_rate * ECONOMICS.aeration_power_kw * 1.0  # 1 hour
        heater_kwh = abs(heater_setting) * ECONOMICS.heater_power_kw * 1.0
        energy_kwh = aeration_kwh + heater_kwh
        energy_cost = energy_kwh * ECONOMICS.electricity_cost_per_kwh
        self.total_energy_cost += energy_cost
        self.total_energy_kwh += energy_kwh

        # Water cost
        water_m3 = water_exchange_rate * tank_volume_m3
        water_cost = water_m3 * ECONOMICS.water_cost_per_m3
        self.total_water_cost += water_cost
        self.total_water_exchanged_m3 += water_m3

        # Fixed overhead (pro-rated hourly: daily cost / 24)
        hourly_fixed = ECONOMICS.fixed_cost_per_day / 24.0
        self.total_fixed_cost += hourly_fixed

        # Track marginal cost for harvest timing signals
        self._last_hour_cost = feed_cost + energy_cost + water_cost + hourly_fixed

        self.hours_operated += 1

    def record_treatment(self, treatment_type: str = "antibiotics"):
        """Record cost of one hour of disease treatment.

        Treatment costs (per day, pro-rated hourly):
        - Antibiotics: $50/day (DISEASE.treatment_cost_per_day)
        - Salt: $10/day (cheap, just NaCl)
        - Probiotics: $30/day (moderate cost)
        - Vaccination: $100 one-time (per application, not per day)
        """
        if treatment_type == "antibiotics":
            hourly_cost = DISEASE.treatment_cost_per_day / 24.0
        elif treatment_type == "salt":
            hourly_cost = 10.0 / 24.0
        elif treatment_type == "probiotics":
            hourly_cost = 30.0 / 24.0
        elif treatment_type == "vaccination":
            hourly_cost = 100.0  # one-time cost, not pro-rated
        else:
            return

        self.total_treatment_cost += hourly_cost

    def record_mortality(self, dead_fish_count: int, avg_weight_g: float):
        """Record disposal cost for dead fish.

        Dead fish must be removed from the tank (biosecurity)
        and disposed of properly. Cost depends on biomass.
        """
        if dead_fish_count <= 0:
            return
        dead_biomass_kg = dead_fish_count * avg_weight_g / 1000.0
        # Disposal cost: ~$0.20/kg dead fish (removal + rendering)
        disposal_cost = dead_biomass_kg * 0.20
        self.total_disposal_cost += disposal_cost

    def set_market_price(self, multiplier: float):
        """Set market price multiplier (for price change events).

        Args:
            multiplier: Price multiplier (1.0 = normal, 0.5 = crash, 1.5 = premium).
        """
        self.market_price_multiplier = max(0.1, min(3.0, multiplier))

    def apply_seasonal_price(self, day_of_year: int):
        """Apply seasonal market price variation.

        Tropical tilapia markets show demand peaks:
        - Christmas/New Year (days 355-10): +15% premium
        - Easter/Lent (days 80-100): +10% (high fish demand)
        - Mid-year (days 170-200): -5% (lower demand)

        This incentivizes agents to time harvests with market demand.
        """
        if day_of_year > 355 or day_of_year < 10:
            seasonal = 1.15  # holiday premium
        elif 80 <= day_of_year <= 100:
            seasonal = 1.10  # Lent/Easter
        elif 170 <= day_of_year <= 200:
            seasonal = 0.95  # mid-year dip
        else:
            seasonal = 1.0
        # Only apply seasonal if no event-driven price override
        if self.market_price_multiplier == 1.0:
            self.market_price_multiplier = seasonal

    def calculate_harvest_revenue(self, biomass_kg: float, avg_weight_g: float = 250.0) -> float:
        """Calculate revenue from harvesting all fish.

        Uses the same weight-dependent pricing curve as calculate_fish_value
        since actual harvest revenue should reflect size premiums.

        Base: $3.00/kg with premiums/discounts by size (KB-03 Sec 6.1).
        """
        return self.calculate_fish_value(biomass_kg, avg_weight_g)

    def calculate_fish_value(self, biomass_kg: float, avg_weight_g: float) -> float:
        """Current value of fish stock (potential revenue).

        Includes weight-dependent pricing:
        - < 200g: underweight, 80% of market price
        - 200-400g: normal price
        - 400-600g: premium, 110% of market price
        - > 600g: peak premium, 120% of market price
        """
        base_price = ECONOMICS.market_price_per_kg * self.market_price_multiplier

        if avg_weight_g < 200:
            weight_premium = 0.8
        elif avg_weight_g < 400:
            weight_premium = 0.8 + 0.2 * (avg_weight_g - 200) / 200
        elif avg_weight_g < 600:
            weight_premium = 1.0 + 0.1 * (avg_weight_g - 400) / 200
        else:
            weight_premium = 1.2

        return biomass_kg * base_price * weight_premium

    @property
    def total_operating_cost(self) -> float:
        """Total operating cost (energy + water + fixed overhead)."""
        return self.total_energy_cost + self.total_water_cost + self.total_fixed_cost

    @property
    def total_cost(self) -> float:
        """Total all-in cost including stocking, feed, operations, treatment, disposal."""
        return (self.total_feed_cost + self.total_operating_cost
                + self.total_treatment_cost + self.total_fingerling_cost
                + self.total_disposal_cost)

    def profit(self, biomass_kg: float, avg_weight_g: float = 250.0) -> float:
        """Calculate current profit/loss if harvested now.

        Profit = Revenue - Total Cost - Harvest Cost
        """
        revenue = self.calculate_fish_value(biomass_kg, avg_weight_g)
        harvest_cost = biomass_kg * ECONOMICS.harvest_cost_per_kg
        return revenue - self.total_cost - harvest_cost

    def roi(self, biomass_kg: float, avg_weight_g: float = 250.0) -> float:
        """Return on investment (%).

        ROI = (Profit / Total Cost) × 100
        """
        if self.total_cost > 0:
            return (self.profit(biomass_kg, avg_weight_g) / self.total_cost) * 100
        return 0.0

    @property
    def marginal_cost_per_hour(self) -> float:
        """Last hour's total cost — useful for harvest timing.

        When marginal cost exceeds marginal revenue from growth,
        it's optimal to harvest (KB-03 Sec 6.1: optimal stopping).
        """
        return self._last_hour_cost

    def cost_breakdown(self) -> dict:
        """Return detailed cost breakdown as dictionary."""
        total = max(0.01, self.total_cost)
        return {
            "feed": {"amount": round(self.total_feed_cost, 2),
                     "pct": round(self.total_feed_cost / total * 100, 1)},
            "energy": {"amount": round(self.total_energy_cost, 2),
                       "pct": round(self.total_energy_cost / total * 100, 1)},
            "water": {"amount": round(self.total_water_cost, 2),
                      "pct": round(self.total_water_cost / total * 100, 1)},
            "fixed": {"amount": round(self.total_fixed_cost, 2),
                      "pct": round(self.total_fixed_cost / total * 100, 1)},
            "treatment": {"amount": round(self.total_treatment_cost, 2),
                          "pct": round(self.total_treatment_cost / total * 100, 1)},
            "fingerlings": {"amount": round(self.total_fingerling_cost, 2),
                            "pct": round(self.total_fingerling_cost / total * 100, 1)},
            "disposal": {"amount": round(self.total_disposal_cost, 2),
                         "pct": round(self.total_disposal_cost / total * 100, 1)},
            "total": round(total, 2),
        }
