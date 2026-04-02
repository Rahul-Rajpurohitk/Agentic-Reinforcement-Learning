"""Reward function implementations for the Fish Farm environment.

These provide dense shaping signals for GRPO training, rewarding
good aquaculture management decisions each step rather than only
at episode end.
"""

from typing import Any, Dict

from .base_reward import BaseReward


class SurvivalReward(BaseReward):
    """Reward for keeping fish alive. Penalizes mortality events."""

    def compute(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        prev_state: Dict[str, Any],
        **kwargs,
    ) -> float:
        mortality = state.get("fish", {}).get("mortality_today", 0)
        if mortality == 0:
            return 0.1  # small positive for zero deaths
        elif mortality < 10:
            return -0.1 * mortality
        else:
            return -1.0  # heavy penalty for mass mortality


class WaterQualityReward(BaseReward):
    """Reward for maintaining good water quality parameters.

    Rewards keeping DO, ammonia, pH, and temperature in safe ranges.
    """

    def compute(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        prev_state: Dict[str, Any],
        **kwargs,
    ) -> float:
        water = state.get("water", {})
        wq_score = water.get("water_quality_score", 0.5)
        # Scale to [-0.5, 0.5] so bad WQ is penalized
        return wq_score - 0.5


class GrowthEfficiencyReward(BaseReward):
    """Reward for efficient growth: weight gain with good FCR.

    Combines growth rate with feed conversion efficiency.
    """

    def compute(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        prev_state: Dict[str, Any],
        **kwargs,
    ) -> float:
        fish = state.get("fish", {})
        prev_fish = prev_state.get("fish", {})

        weight_gain = fish.get("weight_g", 0) - prev_fish.get("weight_g", 0)
        fcr = fish.get("fcr", 3.0)

        # Reward growth
        growth_reward = min(0.3, weight_gain * 0.1)

        # Reward good FCR (< 2.0 is efficient)
        if fcr <= 0:
            fcr_reward = 0.0  # no feeding yet
        elif fcr <= 1.6:
            fcr_reward = 0.2
        elif fcr <= 2.5:
            fcr_reward = 0.2 * (2.5 - fcr) / 0.9
        else:
            fcr_reward = -0.1

        return growth_reward + fcr_reward


class ProfitReward(BaseReward):
    """Reward for positive economic outcomes.

    Tracks marginal profit change (revenue growth minus cost increase).
    """

    def compute(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        prev_state: Dict[str, Any],
        **kwargs,
    ) -> float:
        profit = state.get("economics", {}).get("current_profit", 0)
        prev_profit = prev_state.get("economics", {}).get("current_profit", 0)
        delta = profit - prev_profit

        # Scale: $10 profit change = 0.1 reward
        return max(-0.5, min(0.5, delta / 100.0))


class CompositeReward(BaseReward):
    """Weighted combination of all reward components.

    Default weights match the multi-objective grader philosophy:
    survival > water quality > growth > profit.
    """

    def __init__(
        self,
        survival_weight: float = 0.35,
        wq_weight: float = 0.25,
        growth_weight: float = 0.25,
        profit_weight: float = 0.15,
    ):
        self.survival = SurvivalReward()
        self.water_quality = WaterQualityReward()
        self.growth = GrowthEfficiencyReward()
        self.profit = ProfitReward()
        self.weights = {
            "survival": survival_weight,
            "wq": wq_weight,
            "growth": growth_weight,
            "profit": profit_weight,
        }

    def compute(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        prev_state: Dict[str, Any],
        **kwargs,
    ) -> float:
        return (
            self.weights["survival"] * self.survival.compute(state, action, prev_state)
            + self.weights["wq"] * self.water_quality.compute(state, action, prev_state)
            + self.weights["growth"] * self.growth.compute(state, action, prev_state)
            + self.weights["profit"] * self.profit.compute(state, action, prev_state)
        )
