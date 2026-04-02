"""Reward functions for the Fish Farm environment.

Extracted as a standalone module (no openenv dependency) so it can be
imported by both the environment server and tests.

The reward function computes hourly reward signals based on task-specific
weights. Each weight key maps to a component of the simulation state:

  growth           — fish growth rate (g/day)
  survival         — zero mortality bonus / mortality penalty
  water_quality    — water quality composite (also keys: do_stability, environment)
  fcr              — feed conversion ratio (lower = better)
  profit, roi      — economic profit / return on investment
  welfare          — inverse stress level
  ammonia_control  — UIA concentration penalty
  efficiency       — cost rate penalty
  disease_control  — penalize active disease, reward disease-free
  treatment_timing — reward early treatment response
  timing           — bonus for strategic harvest
"""

from typing import Any, Dict, Optional


def calculate_reward(
    sim_state: Dict[str, Any],
    prev_state: Optional[Dict[str, Any]],
    reward_weights: Dict[str, float],
) -> float:
    """Calculate hourly reward signal based on task-specific weights.

    Uses both absolute state values and, when prev_state is available,
    delta signals to reward improvement or penalize deterioration.

    Args:
        sim_state: Current simulation state dict.
        prev_state: Previous state for delta rewards.
        reward_weights: Task-specific {component: weight} dict.

    Returns:
        Scalar reward (can be negative).
    """
    reward = 0.0
    fish = sim_state["fish"]
    water = sim_state["water"]
    econ = sim_state["economics"]

    # ---- Growth reward ----
    # Normalized by 5 g/day (excellent tilapia growth rate).
    # Negative growth (weight loss) gets penalized harder.
    if "growth" in reward_weights:
        growth_rate = fish["growth_rate_g_day"]
        if growth_rate > 0:
            growth_r = min(1.0, growth_rate / 5.0)
        else:
            growth_r = max(-0.5, growth_rate / 5.0)  # gentle penalty for weight loss
        reward += reward_weights["growth"] * growth_r

    # ---- Survival reward ----
    # +1 for zero deaths, scaled penalty for deaths.
    # Mass mortality (>100) is catastrophic = -1.0.
    if "survival" in reward_weights:
        mort = fish["mortality_today"]
        if mort == 0:
            surv_r = 1.0
        elif mort < 10:
            surv_r = 0.5 - mort * 0.05  # gentle penalty
        else:
            surv_r = max(-1.0, -mort / 100)
        reward += reward_weights["survival"] * surv_r

    # ---- Water quality reward ----
    # Composite score rescaled from [0,1] to [-1,1].
    # Bonus for improvement if prev_state available.
    wq_key = None
    for k in ["water_quality", "do_stability", "environment"]:
        if k in reward_weights:
            wq_key = k
            break
    if wq_key:
        wq_score = water["water_quality_score"]
        base_r = wq_score * 2 - 1

        # Delta bonus: reward improving water quality
        if prev_state is not None:
            prev_wq = prev_state["water"]["water_quality_score"]
            delta = wq_score - prev_wq
            base_r += delta * 0.5  # small nudge toward improvement

        reward += reward_weights[wq_key] * max(-1.0, min(1.0, base_r))

    # ---- FCR reward ----
    # Target FCR 1.6 (tilapia optimal). Below = excellent, above 2.5 = bad.
    # FCR=0 means no net gain — slight negative to encourage growth.
    if "fcr" in reward_weights:
        fcr = fish.get("fcr", 0)
        if fcr <= 0:
            fcr_r = -0.1  # no growth = slight penalty (not as bad as high FCR)
        elif fcr <= 1.6:
            fcr_r = 1.0  # excellent
        elif fcr <= 2.0:
            fcr_r = 1.0 - (fcr - 1.6) / 0.4 * 0.3  # 1.0 → 0.7
        elif fcr <= 2.5:
            fcr_r = 0.7 - (fcr - 2.0) / 0.5 * 0.7  # 0.7 → 0.0
        else:
            fcr_r = max(-1.0, -(fcr - 2.5) / 2.0)  # increasingly negative
        reward += reward_weights["fcr"] * fcr_r

    # ---- Profit / ROI reward ----
    # Normalized by $1000 for profit, or use ROI percentage.
    if "profit" in reward_weights or "roi" in reward_weights:
        key = "profit" if "profit" in reward_weights else "roi"
        if key == "roi":
            roi = econ.get("roi_pct", 0)
            profit_r = max(-1.0, min(1.0, roi / 50.0))  # 50% ROI = max reward
        else:
            profit = econ["current_profit"]
            profit_r = max(-1.0, min(1.0, profit / 1000))
        reward += reward_weights[key] * profit_r

    # ---- Welfare reward ----
    # Low stress = high welfare. Threshold at 0.3 (below = good).
    if "welfare" in reward_weights:
        stress = fish["stress_level"]
        if stress <= 0.1:
            welfare_r = 1.0
        elif stress <= 0.3:
            welfare_r = 1.0 - (stress - 0.1) / 0.2 * 0.5  # 1.0 → 0.5
        elif stress <= 0.5:
            welfare_r = 0.5 - (stress - 0.3) / 0.2 * 0.5  # 0.5 → 0.0
        else:
            welfare_r = max(-1.0, -(stress - 0.5) / 0.5)  # 0.0 → -1.0
        reward += reward_weights["welfare"] * welfare_r

    # ---- Ammonia control ----
    # Safe < 0.02, chronic stress > 0.05, lethal > 0.6
    if "ammonia_control" in reward_weights:
        uia = water["UIA"]
        if uia < 0.02:
            amm_r = 1.0
        elif uia < 0.05:
            amm_r = 1.0 - (uia - 0.02) / 0.03 * 0.3  # 1.0 → 0.7
        elif uia < 0.3:
            amm_r = 0.7 - (uia - 0.05) / 0.25 * 1.7  # 0.7 → -1.0
        else:
            amm_r = -1.0
        reward += reward_weights["ammonia_control"] * amm_r

    # ---- Efficiency ----
    # Penalize high cost rate. Use marginal cost per hour if available.
    if "efficiency" in reward_weights:
        marginal = econ.get("marginal_cost_per_hour", 0)
        if marginal > 0:
            eff_r = max(0, 1.0 - marginal / 3.0)
        else:
            total_hours = max(1, sim_state["time"]["total_hours"])
            cost_rate = econ["total_cost"] / total_hours
            eff_r = max(0, 1.0 - cost_rate / 2.0)
        reward += reward_weights["efficiency"] * eff_r

    # ---- Disease control ----
    # Penalize active disease proportional to severity.
    # Bonus for disease-free or recovering.
    if "disease_control" in reward_weights:
        disease = sim_state.get("disease", {})
        if disease.get("active", False):
            severity = disease.get("severity", 0.5)
            if disease.get("treatment_active", False):
                disease_r = -severity * 0.5  # treating = half penalty
            else:
                disease_r = -severity  # untreated active disease
        else:
            disease_r = 0.5  # bonus for being disease-free
        reward += reward_weights["disease_control"] * disease_r

    # ---- Treatment timing ----
    # Reward early treatment, penalize late treatment.
    if "treatment_timing" in reward_weights:
        disease = sim_state.get("disease", {})
        if disease.get("treatment_active", False):
            # Treatment applied — reward if early (low total disease deaths)
            total_disease_deaths = disease.get("total_disease_deaths", 0)
            if total_disease_deaths < 50:
                timing_r = 0.8  # caught it early
            elif total_disease_deaths < 200:
                timing_r = 0.3  # somewhat late
            else:
                timing_r = 0.0  # very late
        elif disease.get("active", False):
            timing_r = -0.5  # active disease, no treatment = bad
        else:
            timing_r = 0.0  # no disease = neutral
        reward += reward_weights["treatment_timing"] * timing_r

    # ---- Nighttime DO risk ----
    # Penalize high nighttime DO crash risk; reward proactive aeration.
    # This incentivizes agents to act BEFORE a crash happens, not just react.
    if "do_risk" in reward_weights:
        nighttime_risk = water.get("nighttime_do_risk", 0.0)
        if nighttime_risk <= 0.2:
            risk_r = 1.0  # safe
        elif nighttime_risk <= 0.5:
            risk_r = 1.0 - (nighttime_risk - 0.2) / 0.3  # 1.0 → 0.0
        elif nighttime_risk <= 0.8:
            risk_r = -(nighttime_risk - 0.5) / 0.3  # 0.0 → -1.0
        else:
            risk_r = -1.0  # imminent crash

        # Delta bonus: reward reducing risk
        if prev_state is not None:
            prev_risk = prev_state["water"].get("nighttime_do_risk", 0.0)
            delta_risk = nighttime_risk - prev_risk
            risk_r -= delta_risk * 0.5  # rising risk = penalty, falling = bonus

        reward += reward_weights["do_risk"] * max(-1.0, min(1.0, risk_r))

    # ---- Harvest timing ----
    # Reward strategic harvest at market weight with good market price.
    if "timing" in reward_weights:
        if sim_state.get("harvested", False):
            weight = fish["weight_g"]
            price_mult = econ.get("market_price_multiplier", 1.0)
            if weight >= 500 and price_mult >= 1.0:
                timing_r = 1.0  # optimal harvest
            elif weight >= 400:
                timing_r = 0.6  # decent harvest
            elif weight >= 200:
                timing_r = 0.3  # early harvest (emergency?)
            else:
                timing_r = 0.1  # very early (probably losing)
        else:
            timing_r = 0.0
        reward += reward_weights["timing"] * timing_r

    return round(reward, 4)
