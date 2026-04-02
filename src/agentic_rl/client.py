"""Official OpenEnv client for the Fish Farm environment.

Uses openenv.core.env_client.EnvClient with the 3 required abstract methods:
  _step_payload(action) -> dict
  _parse_result(payload) -> StepResult
  _parse_state(payload) -> State

Usage (sync):
    with FishFarmEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="feeding_basics")
        print(f"Fish weight: {result.observation.avg_fish_weight}g")
        result = env.step(FarmAction(
            feeding_rate=0.5, aeration_rate=0.5,
            water_exchange_rate=0.02,
        ))
        print(f"Reward: {result.reward}, DO: {result.observation.dissolved_oxygen}")

Usage (async):
    async with FishFarmEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="feeding_basics")
        result = await env.step(FarmAction(feeding_rate=0.5))
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import FarmAction, FarmObservation, FarmState


class FishFarmEnv(EnvClient[FarmAction, FarmObservation, FarmState]):
    """WebSocket client for the Fish Farm environment."""

    def _step_payload(self, action: FarmAction) -> Dict[str, Any]:
        """Serialize FarmAction to JSON for the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FarmObservation]:
        """Parse server response into StepResult[FarmObservation]."""
        obs_data = payload.get("observation", payload)
        observation = FarmObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            metadata=obs_data.get("metadata", {}),
            # Fish
            avg_fish_weight=obs_data.get("avg_fish_weight", 5.0),
            population=obs_data.get("population", 10000),
            mortality_today=obs_data.get("mortality_today", 0),
            cumulative_mortality=obs_data.get("cumulative_mortality", 0),
            survival_rate=obs_data.get("survival_rate", 1.0),
            stress_level=obs_data.get("stress_level", 0.0),
            feeding_response=obs_data.get("feeding_response", "normal"),
            biomass_kg=obs_data.get("biomass_kg", 50.0),
            growth_rate_g_day=obs_data.get("growth_rate_g_day", 0.0),
            fcr=obs_data.get("fcr", 0.0),
            sgr=obs_data.get("sgr", 0.0),
            stocking_density=obs_data.get("stocking_density", 50.0),
            # Water
            temperature=obs_data.get("temperature", 28.0),
            dissolved_oxygen=obs_data.get("dissolved_oxygen", 7.0),
            ph=obs_data.get("ph", 7.5),
            ammonia=obs_data.get("ammonia", 0.1),
            ammonia_toxic=obs_data.get("ammonia_toxic", 0.005),
            nitrite=obs_data.get("nitrite", 0.05),
            nitrate=obs_data.get("nitrate", 0.0),
            water_quality_score=obs_data.get("water_quality_score", 1.0),
            algae_bloom=obs_data.get("algae_bloom", False),
            # Equipment
            aerator_working=obs_data.get("aerator_working", True),
            biofilter_working=obs_data.get("biofilter_working", True),
            heater_working=obs_data.get("heater_working", True),
            feed_remaining_kg=obs_data.get("feed_remaining_kg", 500.0),
            # Economics
            current_fish_value=obs_data.get("current_fish_value", 0.0),
            total_cost_so_far=obs_data.get("total_cost_so_far", 0.0),
            current_profit=obs_data.get("current_profit", 0.0),
            feed_price_per_kg=obs_data.get("feed_price_per_kg", 0.50),
            market_price_multiplier=obs_data.get("market_price_multiplier", 1.0),
            marginal_cost_per_hour=obs_data.get("marginal_cost_per_hour", 0.0),
            roi_pct=obs_data.get("roi_pct", 0.0),
            # Weather
            weather_forecast=obs_data.get("weather_forecast", ""),
            is_daytime=obs_data.get("is_daytime", True),
            storm_active=obs_data.get("storm_active", False),
            humidity=obs_data.get("humidity", 75.0),
            # Context
            day_in_cycle=obs_data.get("day_in_cycle", 0),
            time_of_day=obs_data.get("time_of_day", 0),
            day_of_year=obs_data.get("day_of_year", 1),
            alerts=obs_data.get("alerts", []),
            disease_suspected=obs_data.get("disease_suspected", False),
            feedback=obs_data.get("feedback", ""),
        )
        return StepResult(
            observation=observation,
            reward=obs_data.get("reward"),
            done=obs_data.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FarmState:
        """Parse server state response into FarmState."""
        return FarmState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            is_complete=payload.get("is_complete", False),
            final_score=payload.get("final_score", 0.0),
            max_hours=payload.get("max_hours", 168),
            sim_state=payload.get("sim_state", {}),
        )
