"""Task-specific graders for the Fish Farm environment.

Each grader takes the final simulator state and episode history,
returns a score between 0.0 and 1.0 with partial credit.
"""

from typing import Any, Dict, List
from .base_grader import BaseGrader, GradeResult


class FarmGrader(BaseGrader):
    """Universal grader that dispatches to task-specific scoring."""

    def grade(
        self,
        task_id: str,
        final_state: Dict[str, Any],
        episode_history: List[Dict[str, Any]],
        task_config: Dict[str, Any],
        **kwargs,
    ) -> GradeResult:
        grader_name = task_config.get("grader", "default")
        method = getattr(self, f"_{grader_name}", self._default_grader)
        return method(final_state, episode_history, task_config)

    def _feeding_grader(self, state, history, config) -> GradeResult:
        fish = state["fish"]
        target = config.get("target_weight", 55.0)

        weight_score = min(1.0, fish["weight_g"] / target) * 0.4

        fcr = fish.get("fcr", 3.0)
        if fcr <= 0:
            # FCR=0 means no net biomass gain (fish lost weight or died) — no credit
            fcr_score = 0.0
        elif fcr <= 1.6:
            fcr_score = 0.3
        elif fcr <= 2.5:
            fcr_score = 0.3 * (2.5 - fcr) / (2.5 - 1.6)
        else:
            fcr_score = 0.0

        survival = fish.get("survival_rate", 0.0)
        survival_score = min(1.0, survival / 0.99) * 0.3

        score = weight_score + fcr_score + survival_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Weight: {fish['weight_g']:.1f}g (target {target}g), "
                    f"FCR: {fcr:.2f}, Survival: {survival:.1%}",
        )

    def _oxygen_grader(self, state, history, config) -> GradeResult:
        safe_hours = sum(1 for h in history if h["water"]["DO"] >= 5.0)
        total_hours = max(1, len(history))
        do_score = safe_hours / total_hours * 0.7

        min_do = min(h["water"]["DO"] for h in history) if history else 0
        if min_do >= 4.0:
            safety_score = 0.3
        elif min_do >= 3.0:
            safety_score = 0.15
        else:
            safety_score = 0.0

        score = do_score + safety_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"DO safe: {safe_hours}/{total_hours} hours. Min DO: {min_do:.2f} mg/L",
        )

    def _water_quality_grader(self, state, history, config) -> GradeResult:
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        score = avg_wq * 0.8

        violations = sum(1 for h in history
                        if h["water"]["DO"] < 3.0 or h["water"]["UIA"] > 0.1)
        if violations == 0:
            score += 0.2
        else:
            score += 0.2 * max(0, 1.0 - violations / 20)

        return GradeResult(
            score=round(min(1.0, score), 3),
            passed=score >= 0.5,
            feedback=f"Avg water quality: {avg_wq:.3f}. Violations: {violations}",
        )

    def _stress_survival_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        growth = state["fish"]["weight_g"] - config["initial_conditions"]["weight_g"]

        survival_score = min(1.0, survival / 0.95) * 0.5
        growth_score = max(0, min(1.0, growth / 10.0)) * 0.3

        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.2

        score = survival_score + growth_score + wq_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Survival: {survival:.1%}, Growth: +{growth:.1f}g, WQ: {avg_wq:.3f}",
        )

    def _ammonia_crisis_grader(self, state, history, config) -> GradeResult:
        peak_uia = max(h["water"]["UIA"] for h in history) if history else 1.0
        survival = state["fish"]["survival_rate"]

        if peak_uia < 0.3:
            uia_score = 0.4
        elif peak_uia < 0.6:
            uia_score = 0.4 * (0.6 - peak_uia) / 0.3
        else:
            uia_score = 0.0

        survival_score = min(1.0, survival / 0.90) * 0.4
        efficiency_score = 0.2 if state["economics"]["total_cost"] < 200 else 0.1

        score = uia_score + survival_score + efficiency_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Peak UIA: {peak_uia:.4f} mg/L, Survival: {survival:.1%}",
        )

    def _disease_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        disease_deaths = state["disease"]["total_disease_deaths"]

        survival_score = min(1.0, survival / 0.90) * 0.4

        treatment_step = None
        for i, h in enumerate(history):
            if h.get("disease", {}).get("treatment_active", False):
                treatment_step = i
                break
        if treatment_step is not None and treatment_step < len(history) * 0.3:
            timing_score = 0.3
        elif treatment_step is not None:
            timing_score = 0.15
        else:
            timing_score = 0.0

        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.3

        score = survival_score + timing_score + wq_score
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=f"Survival: {survival:.1%}, Disease deaths: {disease_deaths}, "
                    f"Treatment started: step {treatment_step}",
        )

    def _growth_optimization_grader(self, state, history, config) -> GradeResult:
        fish = state["fish"]
        target = config.get("target_weight", 120.0)

        weight_score = min(1.0, fish["weight_g"] / target) * 0.4
        fcr = fish.get("fcr", 3.0)
        fcr_score = max(0, min(1.0, (2.5 - fcr) / 1.0)) * 0.3
        survival_score = min(1.0, fish["survival_rate"] / 0.98) * 0.2
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.1

        score = weight_score + fcr_score + survival_score + wq_score
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Weight: {fish['weight_g']:.1f}g/{target}g, FCR: {fcr:.2f}")

    def _full_growout_grader(self, state, history, config) -> GradeResult:
        fish = state["fish"]
        econ = state["economics"]
        target = config.get("target_weight", 400.0)

        profit_score = max(0, min(1.0, econ["current_profit"] / 5000)) * 0.3
        weight_score = min(1.0, fish["weight_g"] / target) * 0.25
        survival_score = min(1.0, fish["survival_rate"] / 0.85) * 0.25
        fcr = fish.get("fcr", 3.0)
        fcr_score = max(0, min(1.0, (2.5 - fcr) / 1.0)) * 0.1
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.1

        score = profit_score + weight_score + survival_score + fcr_score + wq_score
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Weight: {fish['weight_g']:.1f}g, Profit: ${econ['current_profit']:.0f}, "
                                  f"Survival: {fish['survival_rate']:.1%}")

    def _storm_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        survival_score = min(1.0, survival / 0.80) * 0.6
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.3
        efficiency_score = 0.1

        score = survival_score + wq_score + efficiency_score
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Survival: {survival:.1%} through storm")

    def _multi_objective_grader(self, state, history, config) -> GradeResult:
        profit = max(0, state["economics"]["current_profit"])
        profit_norm = min(1.0, profit / 3000)

        avg_stress = sum(h["fish"]["stress_level"] for h in history) / max(1, len(history))
        welfare = max(0, 1.0 - avg_stress / 0.3)

        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))

        score = (profit_norm * welfare * avg_wq) ** (1/3)
        return GradeResult(score=round(score, 3), passed=score >= 0.4,
                          feedback=f"Profit: ${profit:.0f}, Welfare: {welfare:.2f}, WQ: {avg_wq:.3f}")

    def _catastrophe_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        profit = state["economics"]["current_profit"]
        disease_deaths = state["disease"]["total_disease_deaths"]

        survival_score = min(1.0, survival / 0.70) * 0.3
        profit_score = max(0, min(1.0, (profit + 1000) / 3000)) * 0.25
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.2
        disease_score = max(0, 1.0 - disease_deaths / 500) * 0.15
        timing_score = 0.1 if state["harvested"] else 0.0

        score = survival_score + profit_score + wq_score + disease_score + timing_score
        return GradeResult(score=round(score, 3), passed=score >= 0.3,
                          feedback=f"Catastrophe survival: {survival:.1%}, Profit: ${profit:.0f}")

    def _season_grader(self, state, history, config) -> GradeResult:
        econ = state["economics"]
        total_investment = econ["total_cost"] + config["initial_conditions"]["population"] * 0.05
        if total_investment > 0:
            roi = econ["current_profit"] / total_investment
        else:
            roi = 0

        roi_score = min(1.0, max(0, roi / 0.5)) * 0.4
        fish = state["fish"]
        growth_score = min(1.0, fish["weight_g"] / 400) * 0.2
        survival_score = min(1.0, fish["survival_rate"] / 0.80) * 0.2
        fcr = fish.get("fcr", 3.0)
        fcr_score = max(0, min(1.0, (2.5 - fcr) / 1.0)) * 0.1
        avg_stress = sum(h["fish"]["stress_level"] for h in history) / max(1, len(history))
        welfare_score = max(0, 1.0 - avg_stress / 0.3) * 0.1

        score = roi_score + growth_score + survival_score + fcr_score + welfare_score
        return GradeResult(score=round(score, 3), passed=score >= 0.3,
                          feedback=f"ROI: {roi:.1%}, Weight: {fish['weight_g']:.0f}g, "
                                  f"Survival: {fish['survival_rate']:.1%}")

    def _default_grader(self, state, history, config) -> GradeResult:
        survival = state["fish"]["survival_rate"]
        score = survival * 0.5
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        score += avg_wq * 0.5
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Default grader: survival={survival:.1%}, WQ={avg_wq:.3f}")
