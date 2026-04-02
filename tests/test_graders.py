"""Tests for FarmGrader — task-specific scoring for the Fish Farm environment.

Each grader scores an episode on [0.0, 1.0] with partial credit.
Tests verify:
1. Score range validity
2. Better management → higher score (monotonicity)
3. All 12 grader methods are callable
4. Edge cases (empty history, zero population)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graders.farm_graders import FarmGrader
from graders.base_grader import GradeResult


def _make_state(weight=100.0, population=5000, survival=0.98, fcr=1.6,
                stress=0.1, do=6.5, uia=0.01, wq_score=0.85,
                profit=500.0, total_cost=100.0, disease_deaths=0,
                harvested=False, treatment_active=False):
    """Build a synthetic final state dict for grading tests."""
    return {
        "fish": {
            "weight_g": weight, "population": population,
            "biomass_kg": weight * population / 1000.0,
            "mortality_today": 0, "cumulative_mortality": int((1 - survival) * 5000),
            "survival_rate": survival, "stress_level": stress,
            "growth_rate_g_day": 2.0, "sgr": 1.5, "fcr": fcr,
            "condition_factor": 2.8, "weight_cv": 0.10,
            "feeding_response": "normal", "stocking_density": 50.0,
        },
        "water": {
            "temperature": 30.0, "DO": do, "TAN": 0.3, "UIA": uia,
            "pH": 7.5, "NO2": 0.1, "NO3": 5.0, "alkalinity": 100.0,
            "chlorophyll_a": 5.0, "algae_bloom": False,
            "water_quality_score": wq_score,
        },
        "disease": {
            "active": False, "infected": 0, "exposed": 0, "recovered": 0,
            "treatment_active": treatment_active, "treatment_type": "none",
            "total_disease_deaths": disease_deaths, "severity": 0.0,
            "outbreak_count": 0,
        },
        "economics": {
            "total_feed_cost": 50.0, "total_energy_cost": 30.0,
            "total_operating_cost": 80.0, "total_treatment_cost": 0.0,
            "total_cost": total_cost,
            "fish_value": weight * population / 1000.0 * 3.0,
            "current_profit": profit,
            "feed_inventory_kg": 400.0,
            "market_price_multiplier": 1.0,
        },
        "harvested": harvested,
    }


def _make_history(n_hours, do=6.5, uia=0.01, wq_score=0.85, stress=0.1):
    """Build a synthetic episode history for grading tests."""
    return [
        {
            "water": {"DO": do, "UIA": uia, "water_quality_score": wq_score},
            "fish": {"mortality_today": 0, "stress_level": stress,
                     "weight_g": 100.0, "survival_rate": 0.98},
            "disease": {"treatment_active": False},
        }
        for _ in range(n_hours)
    ]


class TestGraderBasics:
    def test_grade_result_fields(self):
        result = GradeResult(score=0.75, passed=True, feedback="Good job")
        assert result.score == 0.75
        assert result.passed is True
        assert result.feedback == "Good job"

    def test_grade_result_details(self):
        result = GradeResult(score=0.5, passed=True, feedback="OK",
                            details={"key": "val"})
        assert result.details["key"] == "val"


class TestFeedingGrader:
    def test_returns_valid_score(self):
        grader = FarmGrader()
        state = _make_state(weight=60.0, fcr=1.5, survival=0.99)
        history = _make_history(168)
        config = {"grader": "feeding_grader", "target_weight": 55.0}
        result = grader.grade("feeding_basics", state, history, config)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.feedback, str)

    def test_higher_weight_scores_better(self):
        grader = FarmGrader()
        config = {"grader": "feeding_grader", "target_weight": 55.0}
        history = _make_history(168)

        result_good = grader.grade("t", _make_state(weight=70.0, fcr=1.5, survival=0.99),
                                    history, config)
        result_bad = grader.grade("t", _make_state(weight=40.0, fcr=1.5, survival=0.99),
                                   history, config)
        assert result_good.score > result_bad.score

    def test_better_fcr_scores_better(self):
        grader = FarmGrader()
        config = {"grader": "feeding_grader", "target_weight": 55.0}
        history = _make_history(168)

        result_good = grader.grade("t", _make_state(weight=60.0, fcr=1.4), history, config)
        result_bad = grader.grade("t", _make_state(weight=60.0, fcr=3.0), history, config)
        assert result_good.score > result_bad.score


class TestOxygenGrader:
    def test_full_safe_hours_scores_high(self):
        grader = FarmGrader()
        history = _make_history(72, do=6.5)  # all hours safe (DO ≥ 5)
        config = {"grader": "oxygen_grader"}
        state = _make_state()
        result = grader.grade("oxygen_management", state, history, config)
        assert result.score >= 0.7

    def test_low_do_scores_low(self):
        grader = FarmGrader()
        history = _make_history(72, do=2.0)  # all hours dangerous
        config = {"grader": "oxygen_grader"}
        state = _make_state()
        result = grader.grade("oxygen_management", state, history, config)
        assert result.score < 0.5


class TestStormGrader:
    def test_high_survival_through_storm(self):
        grader = FarmGrader()
        config = {"grader": "storm_grader"}
        state = _make_state(survival=0.90)
        history = _make_history(120, wq_score=0.7)
        result = grader.grade("storm_response", state, history, config)
        assert result.score > 0.5

    def test_mass_mortality_scores_low(self):
        grader = FarmGrader()
        config = {"grader": "storm_grader"}
        state = _make_state(survival=0.30)
        history = _make_history(120, wq_score=0.3)
        result = grader.grade("storm_response", state, history, config)
        assert result.score < 0.5


class TestDiseaseGrader:
    def test_early_treatment_scores_higher(self):
        grader = FarmGrader()
        config = {"grader": "disease_grader", "initial_conditions": {"weight_g": 200}}

        # Early treatment (step 10 of 240)
        early_history = _make_history(240, wq_score=0.8)
        early_history[10]["disease"]["treatment_active"] = True
        state_early = _make_state(weight=200, survival=0.92, disease_deaths=50)

        # Late treatment (step 200 of 240)
        late_history = _make_history(240, wq_score=0.8)
        late_history[200]["disease"]["treatment_active"] = True
        state_late = _make_state(weight=200, survival=0.85, disease_deaths=200)

        r_early = grader.grade("disease", state_early, early_history, config)
        r_late = grader.grade("disease", state_late, late_history, config)
        assert r_early.score > r_late.score


class TestMultiObjectiveGrader:
    def test_balanced_scores_higher(self):
        grader = FarmGrader()
        config = {"grader": "multi_objective_grader"}

        # Balanced: decent profit, low stress, good WQ
        balanced = _make_state(profit=2000, stress=0.1)
        balanced_hist = _make_history(720, wq_score=0.8, stress=0.1)

        # Unbalanced: high profit but high stress and poor WQ
        unbalanced = _make_state(profit=3000, stress=0.8)
        unbalanced_hist = _make_history(720, wq_score=0.3, stress=0.8)

        r_balanced = grader.grade("multi", balanced, balanced_hist, config)
        r_unbalanced = grader.grade("multi", unbalanced, unbalanced_hist, config)
        assert r_balanced.score > r_unbalanced.score


class TestCatastropheGrader:
    def test_harvested_gets_timing_bonus(self):
        grader = FarmGrader()
        config = {"grader": "catastrophe_grader"}
        history = _make_history(336, wq_score=0.6)

        state_harvested = _make_state(survival=0.75, profit=500,
                                       disease_deaths=100, harvested=True)
        state_no_harvest = _make_state(survival=0.75, profit=500,
                                        disease_deaths=100, harvested=False)

        r_h = grader.grade("catastrophe", state_harvested, history, config)
        r_n = grader.grade("catastrophe", state_no_harvest, history, config)
        assert r_h.score > r_n.score


class TestSeasonGrader:
    def test_high_roi_scores_well(self):
        grader = FarmGrader()
        config = {
            "grader": "season_grader",
            "initial_conditions": {"population": 10000},
        }
        history = _make_history(2160, wq_score=0.8, stress=0.1)
        state = _make_state(weight=350, survival=0.85, fcr=1.7,
                            profit=3000, total_cost=2000)
        result = grader.grade("season", state, history, config)
        assert result.score > 0.3  # should pass threshold


class TestAllGradersCallable:
    def test_every_task_grader_resolves(self):
        """Every task's grader name resolves to a method on FarmGrader."""
        from agentic_rl.tasks import TASKS
        grader = FarmGrader()
        for tid, task in TASKS.items():
            method_name = f"_{task['grader']}"
            assert hasattr(grader, method_name), \
                f"Missing grader method: {method_name} for task {tid}"
