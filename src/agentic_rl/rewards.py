"""Reward functions for the Fish Farm environment.

Extracted as a standalone module (no openenv dependency) so it can be
imported by both the environment server and tests.

The reward function computes hourly reward signals based on task-specific
weights. Each weight key maps to a component of the simulation state:

  growth       — fish growth rate (g/day)
  survival     — zero mortality bonus / mortality penalty
  water_quality, do_stability, environment — water quality composite
  fcr          — feed conversion ratio (lower = better)
  profit, roi  — economic profit
  welfare      — inverse stress level
  ammonia_control — UIA concentration penalty
  efficiency   — cost rate penalty
"""

from typing import Any, Dict, Optional


def calculate_reward(
    sim_state: Dict[str, Any],
    prev_state: Optional[Dict[str, Any]],
    reward_weights: Dict[str, float],
) -> float:
    """Calculate hourly reward signal based on task-specific weights.

    Args:
        sim_state: Current simulation state dict.
        prev_state: Previous state (unused currently, for future delta rewards).
        reward_weights: Task-specific {component: weight} dict.

    Returns:
        Scalar reward (can be negative).
    """
    reward = 0.0
    fish = sim_state["fish"]
    water = sim_state["water"]

    # Growth reward: normalized by 5 g/day (excellent tilapia growth rate)
    if "growth" in reward_weights:
        growth_r = min(1.0, fish["growth_rate_g_day"] / 5.0) if fish["growth_rate_g_day"] > 0 else -0.1
        reward += reward_weights["growth"] * growth_r

    # Survival reward: +1 for zero deaths, penalty proportional to deaths
    if "survival" in reward_weights:
        surv_r = 1.0 if fish["mortality_today"] == 0 else max(-1.0, -fish["mortality_today"] / 100)
        reward += reward_weights["survival"] * surv_r

    # Water quality reward: rescaled from [0,1] to [-1,1]
    wq_key = None
    for k in ["water_quality", "do_stability", "environment"]:
        if k in reward_weights:
            wq_key = k
            break
    if wq_key:
        reward += reward_weights[wq_key] * (water["water_quality_score"] * 2 - 1)

    # FCR reward: target 2.0, penalize above, reward below
    if "fcr" in reward_weights:
        fcr = fish.get("fcr", 0)
        if fcr > 0:
            fcr_r = max(-1.0, min(1.0, (2.0 - fcr) / 1.0))
        else:
            fcr_r = 0.0
        reward += reward_weights["fcr"] * fcr_r

    # Profit reward: normalized by $1000
    if "profit" in reward_weights or "roi" in reward_weights:
        profit = sim_state["economics"]["current_profit"]
        profit_r = max(-1.0, min(1.0, profit / 1000))
        key = "profit" if "profit" in reward_weights else "roi"
        reward += reward_weights[key] * profit_r

    # Welfare reward: inverse of stress (low stress = high welfare)
    if "welfare" in reward_weights:
        welfare_r = 1.0 - min(1.0, fish["stress_level"] / 0.5)
        reward += reward_weights["welfare"] * welfare_r

    # Ammonia control: safe < 0.05, danger > 0.3
    if "ammonia_control" in reward_weights:
        uia = water["UIA"]
        if uia < 0.05:
            amm_r = 1.0
        elif uia < 0.3:
            amm_r = 1.0 - (uia - 0.05) / 0.25
        else:
            amm_r = -1.0
        reward += reward_weights["ammonia_control"] * amm_r

    # Efficiency: penalize high cost rate ($/hour)
    if "efficiency" in reward_weights:
        cost_rate = sim_state["economics"]["total_cost"] / max(1, sim_state["time"]["total_hours"])
        eff_r = max(0, 1.0 - cost_rate / 2.0)
        reward += reward_weights["efficiency"] * eff_r

    # Disease control: penalize active disease proportional to severity
    if "disease_control" in reward_weights:
        disease = sim_state.get("disease", {})
        if disease.get("active", False):
            severity = disease.get("severity", 0.5)
            disease_r = -severity
        else:
            disease_r = 0.5  # bonus for keeping disease-free
        reward += reward_weights["disease_control"] * disease_r

    # Timing reward: bonus for strategic harvest
    if "timing" in reward_weights:
        if sim_state.get("harvested", False):
            timing_r = 0.5  # harvested = some timing awareness
        else:
            timing_r = 0.0
        reward += reward_weights["timing"] * timing_r

    return round(reward, 4)
