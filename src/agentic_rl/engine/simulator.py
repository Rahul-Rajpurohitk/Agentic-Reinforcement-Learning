"""Fish Farm Simulator — orchestrates all subsystems.

This is the central class that:
1. Holds all engine instances (water, fish, disease, economics, weather, events)
2. Processes agent actions every hour
3. Advances all subsystems with proper coupling
4. Wires events to their target subsystems
5. Manages feed inventory with delivery scheduling
6. Returns complete state dicts for observation/grading

The cascade dynamics (overfeed → ammonia → DO crash → stress → disease → mortality)
emerge naturally from the coupled subsystem interactions — this is the core design
principle of the simulation.

Event wiring:
- disease → DiseaseEngine.trigger_outbreak()
- storm → WeatherEngine.trigger_storm()
- equipment_failure → disables specific equipment
- power_outage → disables all equipment
- feed_shortage → reduces feed inventory replenishment
- price_change → EconomicsEngine.set_market_price()
- algae_bloom → boosts WaterQualityEngine.chlorophyll_a
"""

from typing import Dict, Any, Optional, List
from .water_quality import WaterQualityEngine
from .fish_biology import FishBiologyEngine
from .disease import DiseaseEngine
from .economics import EconomicsEngine
from .weather import WeatherEngine
from .events import EventScheduler, Event
from ..constants import SYSTEM, TILAPIA, WATER


class FishFarmSimulator:
    """Complete RAS fish farm simulation.

    Coordinates 6 subsystem engines with proper coupling order:
    1. Events → modify equipment/controls
    2. Weather → air temp, solar, wind
    3. Feed inventory → constrain feeding
    4. Water quality → DO, TAN, pH, temperature
    5. Fish biology → growth, stress, mortality
    6. Disease → SEIR epidemic, treatment effects
    7. Economics → cost tracking
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        import random
        self.rng = random.Random(seed)

        # Subsystems
        self.water = WaterQualityEngine(SYSTEM.tank_volume_m3, SYSTEM.tank_depth_m)
        self.fish = FishBiologyEngine(rng=self.rng)
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
        self.feed_inventory_kg: float = 500.0
        self.feed_delivery_interval_days: int = 7  # feed delivery every 7 days
        self.feed_delivery_amount_kg: float = 200.0

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
        """Reset simulation to initial conditions.

        Args:
            initial_weight: Starting fish weight (g).
            initial_population: Number of fish stocked.
            initial_temp: Starting water temperature (°C).
            initial_DO: Starting dissolved oxygen (mg/L).
            initial_TAN: Starting total ammonia nitrogen (mg/L).
            initial_pH: Starting pH.
            day_of_year: Calendar day (1-365) for photoperiod/season.
            base_air_temp: Average air temperature for location (°C).
            seed: Random seed (None = keep current).
            scheduled_events: Pre-defined events for this episode.

        Returns:
            Complete state dict.
        """
        if seed is not None:
            self.seed = seed
        import random
        self.rng = random.Random(self.seed)

        self.water.reset(initial_temp, initial_DO, initial_TAN, initial_pH, NO2=0.05)
        self.fish.reset(initial_weight, initial_population, day_of_year)
        self.disease.reset(initial_population)
        self.economics.reset(initial_population)
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

        Processing order ensures proper coupling:
        1. Events → disable equipment, trigger subsystems
        2. Weather → environmental conditions
        3. Feed → constrain by inventory
        4. Water quality → temperature, DO, TAN, pH, NO2
        5. Fish growth → bioenergetic model
        6. Mortality → stress-dependent + acute
        7. Disease → SEIR + treatment
        8. Economics → hourly cost tracking
        9. Time advance → day rollover, feed delivery

        Args:
            feeding_rate: 0.0-1.0 (fraction of max daily ration)
            aeration_rate: 0.0-1.0 (fraction of max aeration power)
            heater_setting: -1.0 to 1.0 (cool to heat)
            water_exchange_rate: 0.0-0.10 (fraction of volume per hour)
            harvest: True to harvest all fish (ends episode)
            treatment: 'none', 'antibiotics', 'salt', 'probiotics'

        Returns:
            Complete state dict.
        """
        # ---- Input clamping ----
        feeding_rate = max(0.0, min(1.0, feeding_rate))
        aeration_rate = max(0.0, min(1.0, aeration_rate))
        heater_setting = max(-1.0, min(1.0, heater_setting))
        water_exchange_rate = max(0.0, min(SYSTEM.max_exchange_rate, water_exchange_rate))

        # ================================================================
        # 1. PROCESS EVENTS — activate scheduled events, wire to subsystems
        # ================================================================
        new_events = self.events.step(self.total_hours)
        self._process_new_events(new_events)

        # Equipment failures modify controls
        if not self.events.equipment_working("aerator"):
            aeration_rate = 0.0
        if not self.events.equipment_working("heater"):
            heater_setting = 0.0

        # Biofilter efficiency: base × equipment × treatment effects
        if self.events.equipment_working("biofilter"):
            biofilter_eff = WATER.biofilter_efficiency
        else:
            biofilter_eff = 0.1  # degraded but not zero (residual bacteria)

        # Treatment side effects on biofilter
        biofilter_eff *= self.disease.get_biofilter_impact()

        # Power outage kills all equipment
        if self.events.has_active("power_outage"):
            aeration_rate = 0.0
            heater_setting = 0.0
            biofilter_eff = 0.0  # no flow through biofilter

        # Market price events
        price_mult = self.events.get_price_multiplier()
        if price_mult != 1.0:
            self.economics.set_market_price(price_mult)

        # ================================================================
        # 2. WEATHER — get current environmental conditions
        # ================================================================
        day_of_year = self.day + self.fish.day_of_year
        weather = self.weather.get_conditions(day_of_year, self.hour)
        self.weather.step(self.hour)

        # Heat wave event: boost air temperature while active
        heat_wave = self.events.get_active_event("heat_wave")
        if heat_wave is not None:
            weather["air_temp"] += heat_wave.severity * 10.0  # 0.7 severity → +7°C

        # Random storm check with seasonal modulation (only if no storm from events)
        if not self.weather.storm_active:
            self.weather.check_random_storm(day_of_year=day_of_year)

        # ================================================================
        # 3. FEED INVENTORY — constrain feeding by available stock
        # ================================================================
        biomass_kg = self.fish.biomass_kg
        max_feed_this_hour = (feeding_rate * TILAPIA.max_feeding_pct / 100.0
                              * biomass_kg / 24.0)

        # Feed shortage events reduce available feed
        shortage = self.events.get_feed_shortage_severity()
        if shortage > 0:
            max_feed_this_hour *= (1.0 - shortage)

        # Constrain by inventory
        feed_this_hour = min(max_feed_this_hour, self.feed_inventory_kg)
        if max_feed_this_hour > 0:
            effective_feeding_rate = feeding_rate * (feed_this_hour / max_feed_this_hour)
        else:
            effective_feeding_rate = 0.0
        self.feed_inventory_kg = max(0, self.feed_inventory_kg - feed_this_hour)

        # ================================================================
        # 4. WATER QUALITY — temperature, DO, TAN, pH, NO2
        # ================================================================

        # Temperature update (air exchange + heater + water exchange mixing)
        self.water.update_temperature(
            dt_hours=1.0,
            air_temp=weather["air_temp"],
            heater_setting=heater_setting,
            volume_m3=SYSTEM.tank_volume_m3,
            water_exchange_rate=water_exchange_rate,
        )

        # Pre-compute fish respiration rate using the tilapia-specific model
        # (KB-03 Sec 2.1 polynomial, R²=0.99) — shared with water quality
        fish_resp_rate = self.fish.respiration_rate(self.water.temperature)

        # Full water chemistry step
        self.water.step(
            dt_hours=1.0,
            fish_biomass_kg=biomass_kg,
            fish_weight_g=self.fish.weight_g,
            feeding_rate=effective_feeding_rate,
            aeration_rate=aeration_rate,
            water_exchange_rate=water_exchange_rate,
            is_daytime=weather["is_daytime"],
            biofilter_efficiency=biofilter_eff,
            solar_intensity=weather["solar_intensity"],
            wind_speed=weather["wind_speed"],
            fish_respiration_rate=fish_resp_rate,
            humidity=weather.get("humidity", 75.0),
        )

        # ================================================================
        # 5. FISH GROWTH — bioenergetic model
        # ================================================================
        self.fish.grow(
            dt_hours=1.0,
            feeding_rate=effective_feeding_rate,
            temperature=self.water.temperature,
            DO=self.water.DO,
            UIA=self.water.UIA,
            photoperiod_h=weather["photoperiod_hours"],
        )

        # ================================================================
        # 6. MORTALITY — environmental + stress-driven
        # ================================================================
        stocking_density = self.fish.population / SYSTEM.tank_volume_m3
        env_deaths = self.fish.apply_mortality(
            dt_hours=1.0,
            DO=self.water.DO,
            UIA=self.water.UIA,
            temperature=self.water.temperature,
            stocking_density=stocking_density,
        )

        # Record disposal cost for dead fish
        if env_deaths > 0:
            self.economics.record_mortality(env_deaths, self.fish.weight_g)

        # ================================================================
        # 7. DISEASE — SEIR model + treatment
        # ================================================================

        # Check if stress triggers new outbreak
        self.disease.check_stress_trigger(
            stress_level=self.fish.stress_level,
            DO=self.water.DO,
            UIA=self.water.UIA,
            temperature=self.water.temperature,
            stocking_density=stocking_density,
            rng_value=self.rng.random(),
        )

        # Apply treatment if requested
        # Vaccination works as prophylaxis even without active disease (KB-03 Sec 4.2)
        # Other treatments only apply when disease is active
        if treatment != "none":
            if treatment == "vaccination":
                # Vaccination is preventive — works anytime, moves S → R
                if not self.disease.treatment_active:
                    self.disease.apply_treatment(treatment)
                self.economics.record_treatment(treatment)
            elif self.disease.is_active:
                if not self.disease.treatment_active:
                    self.disease.apply_treatment(treatment)
                self.economics.record_treatment(treatment)

        # Advance disease model (temperature affects pathogen virulence)
        disease_deaths = self.disease.step(
            dt_hours=1.0,
            population=self.fish.population,
            stress_level=self.fish.stress_level,
            temperature=self.water.temperature,
        )

        # Apply disease deaths to population
        if disease_deaths > 0:
            self.fish.population = max(0, self.fish.population - disease_deaths)
            self.fish.cumulative_mortality += disease_deaths
            self.economics.record_mortality(disease_deaths, self.fish.weight_g)

        # Sync disease compartments with actual population
        self.disease.sync_population(self.fish.population)

        # ================================================================
        # 8. ECONOMICS + FEED TRACKING — hourly cost tracking
        # ================================================================
        # Record feed consumed in fish biology (single source of truth for FCR)
        self.fish.record_feed(feed_this_hour)

        self.economics.record_hourly_costs(
            feed_kg=feed_this_hour,
            aeration_rate=aeration_rate,
            heater_setting=heater_setting,
            water_exchange_rate=water_exchange_rate,
            tank_volume_m3=SYSTEM.tank_volume_m3,
            rng_value=self.rng.gauss(0, 1),  # for stochastic feed price
        )

        # Apply seasonal market price variation
        self.economics.apply_seasonal_price(day_of_year)

        # ================================================================
        # 9. TIME ADVANCE — day rollover, feed delivery
        # ================================================================
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day += 1
            self._daily_maintenance()
        self.total_hours += 1

        # ================================================================
        # 10. TERMINAL CONDITIONS
        # ================================================================
        if harvest:
            self.harvested = True

        if self.fish.population <= 0:
            self.catastrophe = True
        elif self.fish.survival_rate < 0.2:
            self.catastrophe = True

        return self.get_state()

    def _process_new_events(self, new_events: List[Event]):
        """Wire newly activated events to their target subsystems.

        This is where events become real — each event type triggers
        specific subsystem changes.
        """
        for event in new_events:
            if event.type == "disease":
                # Trigger disease outbreak
                initial = max(1, int(self.fish.population * event.severity * 0.01))
                self.disease.trigger_outbreak(initial_infected=initial)

            elif event.type == "storm":
                # Trigger weather storm
                self.weather.trigger_storm(
                    severity=event.severity,
                    duration_hours=event.duration_hours
                )

            elif event.type == "heat_wave":
                # Heat wave: handled via active event check in weather section
                # (no persistent state change needed — reverts when event ends)
                pass

            elif event.type == "algae_bloom":
                # Boost phytoplankton biomass → causes DO swings
                bloom_boost = 30.0 + event.severity * 100.0  # μg chl-a/L
                self.water.chlorophyll_a = min(
                    200.0, self.water.chlorophyll_a + bloom_boost
                )

            elif event.type == "feed_shortage":
                # Reduce feed inventory (delivery failure)
                reduction = event.severity * self.feed_inventory_kg * 0.5
                self.feed_inventory_kg = max(0, self.feed_inventory_kg - reduction)

            elif event.type == "price_change":
                # Adjust market price
                self.economics.set_market_price(event.price_multiplier)

            # equipment_failure and power_outage are handled by
            # EventScheduler.equipment_working() checks in step()

    def _daily_maintenance(self):
        """End-of-day maintenance tasks.

        Called when hour rolls over to 0 (midnight).
        Handles feed delivery and long-episode logistics.
        """
        # Feed delivery: replenish inventory on schedule
        if self.day > 0 and self.day % self.feed_delivery_interval_days == 0:
            # Only deliver if not in feed shortage
            if not self.events.has_active("feed_shortage"):
                self.feed_inventory_kg += self.feed_delivery_amount_kg

        # Warn if feed is critically low
        # (This is informational, agent sees it in state)

    def get_state(self) -> Dict[str, Any]:
        """Return complete simulation state.

        This is the ground-truth state used by graders. The observation
        endpoint in environment.py filters this for partial observability
        (e.g., hiding disease.infected count).
        """
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
                "sgr": round(self.fish.sgr, 3),
                "fcr": round(self.fish.fcr, 3) if self.fish.fcr > 0 else 0.0,
                "condition_factor": round(self.fish.condition_factor, 3),
                "weight_cv": round(self.fish.weight_cv, 3),
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
                "NO3": round(self.water.NO3, 3),
                "alkalinity": round(self.water.alkalinity, 1),
                "chlorophyll_a": round(self.water.chlorophyll_a, 1),
                "algae_bloom": self.water.algae_bloom_active,
                "water_quality_score": round(self.water.get_water_quality_score(), 3),
                "nighttime_do_risk": round(self.water.nighttime_do_risk, 3),
            },
            "disease": {
                "active": self.disease.is_active,
                "infected": self.disease.infected,
                "exposed": self.disease.exposed,
                "recovered": self.disease.recovered,
                "treatment_active": self.disease.treatment_active,
                "treatment_type": self.disease.treatment_type,
                "total_disease_deaths": self.disease.total_disease_deaths,
                "severity": round(self.disease.disease_severity, 3),
                "outbreak_count": self.disease.outbreak_count,
            },
            "economics": {
                "total_feed_cost": round(self.economics.total_feed_cost, 2),
                "total_energy_cost": round(self.economics.total_energy_cost, 2),
                "total_operating_cost": round(self.economics.total_operating_cost, 2),
                "total_treatment_cost": round(self.economics.total_treatment_cost, 2),
                "total_cost": round(self.economics.total_cost, 2),
                "fish_value": round(
                    self.economics.calculate_fish_value(
                        self.fish.biomass_kg, self.fish.weight_g
                    ), 2
                ),
                "current_profit": round(
                    self.economics.profit(
                        self.fish.biomass_kg, self.fish.weight_g
                    ), 2
                ),
                "feed_inventory_kg": round(self.feed_inventory_kg, 1),
                "market_price_multiplier": self.economics.market_price_multiplier,
                "feed_price_per_kg": round(self.economics.feed_price_current, 3),
                "marginal_cost_per_hour": round(self.economics.marginal_cost_per_hour, 3),
                "roi_pct": round(self.economics.roi(
                    self.fish.biomass_kg, self.fish.weight_g
                ), 2),
                "cost_breakdown": self.economics.cost_breakdown(),
            },
            "weather": {
                "air_temp": round(weather["air_temp"], 1),
                "is_daytime": weather["is_daytime"],
                "solar_intensity": round(weather["solar_intensity"], 0),
                "wind_speed": round(weather["wind_speed"], 1),
                "cloud_cover": round(weather["cloud_cover"], 2),
                "humidity": round(weather.get("humidity", 75), 1),
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
                "active_count": self.events.count_active(),
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
