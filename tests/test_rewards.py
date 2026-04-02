"""Tests for the Fish Farm reward function.

The reward function _calculate_reward() in environment.py computes hourly reward
signals based on task-specific weights. It must:
1. Support all reward weight keys used across 12 tasks
2. Reward good management and penalize bad management
3. Return bounded values
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import from standalone rewards module (no openenv dependency)
from agentic_rl.rewards import calculate_reward as _calculate_reward
from agentic_rl.rewards import growth_stage_scale


def _make_sim_state(growth_rate=2.0, mortality=0, survival=0.98, stress=0.1,
                    do=6.5, uia=0.01, wq_score=0.85, fcr=1.6,
                    profit=500, total_cost=100, total_hours=24,
                    weight=100.0, roi_pct=50.0, marginal_cost=0.5,
                    market_price_multiplier=1.0, nighttime_do_risk=0.0):
    """Build a minimal sim state for reward testing."""
    return {
        "fish": {
            "weight_g": weight,
            "growth_rate_g_day": growth_rate,
            "mortality_today": mortality,
            "survival_rate": survival,
            "stress_level": stress,
            "fcr": fcr,
        },
        "water": {
            "DO": do,
            "UIA": uia,
            "water_quality_score": wq_score,
            "nighttime_do_risk": nighttime_do_risk,
        },
        "economics": {
            "current_profit": profit,
            "total_cost": total_cost,
            "roi_pct": roi_pct,
            "marginal_cost_per_hour": marginal_cost,
            "market_price_multiplier": market_price_multiplier,
        },
        "time": {
            "total_hours": total_hours,
        },
    }


class TestGrowthReward:
    def test_positive_growth_positive_reward(self):
        state = _make_sim_state(growth_rate=3.0)
        reward = _calculate_reward(state, None, {"growth": 1.0})
        assert reward > 0

    def test_negative_growth_negative_reward(self):
        state = _make_sim_state(growth_rate=-0.5)
        reward = _calculate_reward(state, None, {"growth": 1.0})
        assert reward < 0

    def test_higher_growth_higher_reward(self):
        r_high = _calculate_reward(_make_sim_state(growth_rate=4.0), None, {"growth": 1.0})
        r_low = _calculate_reward(_make_sim_state(growth_rate=1.0), None, {"growth": 1.0})
        assert r_high > r_low


class TestSurvivalReward:
    def test_zero_mortality_max_reward(self):
        state = _make_sim_state(mortality=0)
        reward = _calculate_reward(state, None, {"survival": 1.0})
        assert reward == 1.0

    def test_high_mortality_negative_reward(self):
        state = _make_sim_state(mortality=200)
        reward = _calculate_reward(state, None, {"survival": 1.0})
        assert reward < 0

    def test_moderate_mortality_partial_penalty(self):
        r_none = _calculate_reward(_make_sim_state(mortality=0), None, {"survival": 1.0})
        r_some = _calculate_reward(_make_sim_state(mortality=50), None, {"survival": 1.0})
        r_lots = _calculate_reward(_make_sim_state(mortality=200), None, {"survival": 1.0})
        assert r_none > r_some > r_lots


class TestWaterQualityReward:
    def test_high_wq_positive(self):
        state = _make_sim_state(wq_score=0.9)
        reward = _calculate_reward(state, None, {"water_quality": 1.0})
        assert reward > 0

    def test_low_wq_negative(self):
        state = _make_sim_state(wq_score=0.2)
        reward = _calculate_reward(state, None, {"water_quality": 1.0})
        assert reward < 0

    def test_do_stability_key_works(self):
        """do_stability should use water_quality_score too."""
        state = _make_sim_state(wq_score=0.9)
        reward = _calculate_reward(state, None, {"do_stability": 1.0})
        assert reward > 0

    def test_environment_key_works(self):
        """environment weight should use water_quality_score."""
        state = _make_sim_state(wq_score=0.9)
        reward = _calculate_reward(state, None, {"environment": 1.0})
        assert reward > 0


class TestFCRReward:
    def test_good_fcr_positive(self):
        state = _make_sim_state(fcr=1.4)
        reward = _calculate_reward(state, None, {"fcr": 1.0})
        assert reward > 0

    def test_bad_fcr_negative(self):
        state = _make_sim_state(fcr=3.5)
        reward = _calculate_reward(state, None, {"fcr": 1.0})
        assert reward < 0

    def test_zero_fcr_slight_penalty(self):
        """FCR=0 means no net biomass gain — should be slightly penalized."""
        state = _make_sim_state(fcr=0)
        reward = _calculate_reward(state, None, {"fcr": 1.0})
        assert -0.2 < reward < 0.0  # slight penalty, not catastrophic


class TestProfitReward:
    def test_positive_profit_positive_reward(self):
        state = _make_sim_state(profit=1000)
        reward = _calculate_reward(state, None, {"profit": 1.0})
        assert reward > 0

    def test_negative_profit_negative_reward(self):
        state = _make_sim_state(profit=-500)
        reward = _calculate_reward(state, None, {"profit": 1.0})
        assert reward < 0

    def test_roi_key_works(self):
        state = _make_sim_state(profit=1000)
        reward = _calculate_reward(state, None, {"roi": 1.0})
        assert reward > 0


class TestWelfareReward:
    def test_low_stress_high_welfare(self):
        state = _make_sim_state(stress=0.05)
        reward = _calculate_reward(state, None, {"welfare": 1.0})
        assert reward > 0.5

    def test_high_stress_low_welfare(self):
        state = _make_sim_state(stress=0.8)
        reward = _calculate_reward(state, None, {"welfare": 1.0})
        assert reward < 0.5


class TestAmmoniaControlReward:
    def test_safe_ammonia_max_reward(self):
        state = _make_sim_state(uia=0.01)
        reward = _calculate_reward(state, None, {"ammonia_control": 1.0})
        assert reward == 1.0

    def test_dangerous_ammonia_negative(self):
        state = _make_sim_state(uia=0.5)
        reward = _calculate_reward(state, None, {"ammonia_control": 1.0})
        assert reward < 0

    def test_moderate_ammonia_partial(self):
        r_safe = _calculate_reward(_make_sim_state(uia=0.01), None, {"ammonia_control": 1.0})
        r_mid = _calculate_reward(_make_sim_state(uia=0.15), None, {"ammonia_control": 1.0})
        r_bad = _calculate_reward(_make_sim_state(uia=0.5), None, {"ammonia_control": 1.0})
        assert r_safe > r_mid > r_bad


class TestEfficiencyReward:
    def test_low_cost_high_efficiency(self):
        state = _make_sim_state(total_cost=10, total_hours=24)
        reward = _calculate_reward(state, None, {"efficiency": 1.0})
        assert reward > 0.5

    def test_high_cost_low_efficiency(self):
        state = _make_sim_state(total_cost=200, total_hours=24, marginal_cost=2.5)
        reward = _calculate_reward(state, None, {"efficiency": 1.0})
        assert reward < 0.5


class TestCombinedWeights:
    def test_multi_weight_combines(self):
        """Multiple weights should combine additively."""
        state = _make_sim_state(growth_rate=3.0, mortality=0, wq_score=0.9)
        reward = _calculate_reward(state, None, {
            "growth": 0.5, "survival": 0.3, "water_quality": 0.2,
        })
        assert reward > 0

    def test_all_task_weight_keys_work(self):
        """Every weight key used across all 12 tasks should be handled."""
        from agentic_rl.tasks import TASKS

        all_keys = set()
        for task in TASKS.values():
            all_keys.update(task["reward_weights"].keys())

        state = _make_sim_state()
        # Test each key individually — should not raise
        for key in all_keys:
            reward = _calculate_reward(state, None, {key: 1.0})
            assert isinstance(reward, float), f"Key {key} returned non-float"


class TestDeltaReward:
    """Test delta-based reward improvements using prev_state."""

    def test_wq_improvement_bonus(self):
        """Improving water quality should give higher reward than staying flat."""
        prev = _make_sim_state(wq_score=0.7)
        curr_improving = _make_sim_state(wq_score=0.8)
        curr_flat = _make_sim_state(wq_score=0.7)

        r_improving = _calculate_reward(curr_improving, prev, {"water_quality": 1.0})
        r_flat = _calculate_reward(curr_flat, prev, {"water_quality": 1.0})
        assert r_improving > r_flat


class TestDiseaseControlReward:
    """Test disease control and treatment timing rewards."""

    def test_disease_free_positive(self):
        state = _make_sim_state()
        state["disease"] = {"active": False}
        reward = _calculate_reward(state, None, {"disease_control": 1.0})
        assert reward > 0

    def test_untreated_disease_negative(self):
        state = _make_sim_state()
        state["disease"] = {"active": True, "severity": 0.6, "treatment_active": False}
        reward = _calculate_reward(state, None, {"disease_control": 1.0})
        assert reward < 0

    def test_treated_disease_less_penalty(self):
        """Treating disease should reduce the penalty vs not treating."""
        state_treated = _make_sim_state()
        state_treated["disease"] = {"active": True, "severity": 0.6, "treatment_active": True}
        state_untreated = _make_sim_state()
        state_untreated["disease"] = {"active": True, "severity": 0.6, "treatment_active": False}

        r_treated = _calculate_reward(state_treated, None, {"disease_control": 1.0})
        r_untreated = _calculate_reward(state_untreated, None, {"disease_control": 1.0})
        assert r_treated > r_untreated


class TestHarvestTimingReward:
    """Test harvest timing rewards."""

    def test_harvest_at_market_weight_best(self):
        state = _make_sim_state(weight=550.0)
        state["harvested"] = True
        reward = _calculate_reward(state, None, {"timing": 1.0})
        assert reward >= 0.6

    def test_no_harvest_zero(self):
        state = _make_sim_state(weight=550.0)
        state["harvested"] = False
        reward = _calculate_reward(state, None, {"timing": 1.0})
        assert reward == 0.0


class TestNighttimeDORiskReward:
    """Test nighttime DO crash risk reward component."""

    def test_low_risk_positive(self):
        state = _make_sim_state(nighttime_do_risk=0.1)
        reward = _calculate_reward(state, None, {"do_risk": 1.0})
        assert reward > 0.5

    def test_high_risk_negative(self):
        state = _make_sim_state(nighttime_do_risk=0.9)
        reward = _calculate_reward(state, None, {"do_risk": 1.0})
        assert reward < 0

    def test_risk_gradient(self):
        """Higher risk should give lower reward."""
        r_safe = _calculate_reward(_make_sim_state(nighttime_do_risk=0.1), None, {"do_risk": 1.0})
        r_mid = _calculate_reward(_make_sim_state(nighttime_do_risk=0.4), None, {"do_risk": 1.0})
        r_danger = _calculate_reward(_make_sim_state(nighttime_do_risk=0.8), None, {"do_risk": 1.0})
        assert r_safe > r_mid > r_danger

    def test_risk_reduction_bonus(self):
        """Reducing nighttime risk should give higher reward than increasing it."""
        prev = _make_sim_state(nighttime_do_risk=0.6)
        curr_better = _make_sim_state(nighttime_do_risk=0.3)
        curr_worse = _make_sim_state(nighttime_do_risk=0.8)
        r_better = _calculate_reward(curr_better, prev, {"do_risk": 1.0})
        r_worse = _calculate_reward(curr_worse, prev, {"do_risk": 1.0})
        assert r_better > r_worse


class TestGrowthStageScale:
    """Test growth-stage-aware reward weight scaling (KB-03 Sec 10.1)."""

    def test_juvenile_boosts_water_quality(self):
        """Juvenile (<50g) should boost water_quality weight by 30%."""
        weights = {"water_quality": 1.0, "growth": 1.0}
        scaled = growth_stage_scale(30.0, weights)
        assert scaled["water_quality"] == 1.3
        assert scaled["growth"] == 0.8

    def test_juvenile_boosts_survival(self):
        weights = {"survival": 1.0}
        scaled = growth_stage_scale(20.0, weights)
        assert scaled["survival"] == 1.2

    def test_juvenile_do_stability_key(self):
        """do_stability key should also get juvenile boost."""
        weights = {"do_stability": 1.0}
        scaled = growth_stage_scale(10.0, weights)
        assert scaled["do_stability"] == 1.3

    def test_growout_boosts_fcr_and_efficiency(self):
        """Grow-out (50-300g) should boost fcr and efficiency by 20%."""
        weights = {"fcr": 1.0, "efficiency": 1.0, "growth": 1.0}
        scaled = growth_stage_scale(150.0, weights)
        assert scaled["fcr"] == 1.2
        assert scaled["efficiency"] == 1.2
        assert scaled["growth"] == 1.0  # unchanged in grow-out

    def test_preharvest_boosts_profit_and_timing(self):
        """Pre-harvest (>300g) should boost profit 30% and timing 50%."""
        weights = {"profit": 1.0, "timing": 1.0}
        scaled = growth_stage_scale(400.0, weights)
        assert scaled["profit"] == 1.3
        assert scaled["timing"] == 1.5

    def test_preharvest_roi_boost(self):
        weights = {"roi": 1.0}
        scaled = growth_stage_scale(350.0, weights)
        assert scaled["roi"] == 1.3

    def test_missing_keys_no_error(self):
        """Scaling should not crash if keys are absent."""
        weights = {"growth": 1.0}
        scaled = growth_stage_scale(30.0, weights)
        assert scaled["growth"] == 0.8
        assert "water_quality" not in scaled

    def test_does_not_mutate_original(self):
        """growth_stage_scale must not modify the input dict."""
        weights = {"water_quality": 1.0, "growth": 1.0}
        original_wq = weights["water_quality"]
        growth_stage_scale(30.0, weights)
        assert weights["water_quality"] == original_wq

    def test_integration_juvenile_higher_wq_reward(self):
        """Juvenile fish should get higher water quality reward due to scaling."""
        state_juvenile = _make_sim_state(wq_score=0.9, weight=30.0)
        state_adult = _make_sim_state(wq_score=0.9, weight=200.0)
        weights = {"water_quality": 1.0}
        r_juv = _calculate_reward(state_juvenile, None, dict(weights))
        r_adult = _calculate_reward(state_adult, None, dict(weights))
        assert r_juv > r_adult  # juvenile gets 1.3x boost

    def test_integration_preharvest_higher_profit_reward(self):
        """Pre-harvest fish should get higher profit reward due to scaling."""
        state_small = _make_sim_state(profit=500, weight=100.0)
        state_big = _make_sim_state(profit=500, weight=400.0)
        weights = {"profit": 1.0}
        r_small = _calculate_reward(state_small, None, dict(weights))
        r_big = _calculate_reward(state_big, None, dict(weights))
        assert r_big > r_small  # pre-harvest gets 1.3x boost
