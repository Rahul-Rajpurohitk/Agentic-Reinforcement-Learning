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


def _make_sim_state(growth_rate=2.0, mortality=0, survival=0.98, stress=0.1,
                    do=6.5, uia=0.01, wq_score=0.85, fcr=1.6,
                    profit=500, total_cost=100, total_hours=24,
                    weight=100.0):
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
        },
        "economics": {
            "current_profit": profit,
            "total_cost": total_cost,
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

    def test_zero_fcr_neutral(self):
        state = _make_sim_state(fcr=0)
        reward = _calculate_reward(state, None, {"fcr": 1.0})
        assert reward == 0.0


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
        state = _make_sim_state(total_cost=200, total_hours=24)
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
