"""Task-specific graders for the Fish Farm environment.

Each grader takes the final simulator state and episode history,
returns a score between 0.0 and 1.0 with partial credit.

Scoring philosophy: every grader has 3-6 weighted components that sum to 1.0.
Partial credit is given for each component (no cliff effects). Detailed
feedback describes exactly what the agent did well and poorly.

All graders are deterministic: same (state, history) → same score.
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
        """Feeding basics: grow fish to target weight with good FCR and survival.

        Components:
          weight (0.40): min(1, actual/target) — linear toward target weight
          fcr    (0.30): full marks ≤1.6, linear decline to 0 at 2.5, 0 if ≤0
          survival (0.30): min(1, actual/0.99) — normalized to 99% baseline
        """
        fish = state["fish"]
        target = config.get("target_weight", 55.0)

        weight_score = min(1.0, fish["weight_g"] / target) * 0.4

        fcr = fish.get("fcr", 3.0)
        if fcr <= 0:
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

        details = [
            f"Weight: {fish['weight_g']:.1f}g (target {target}g) [{weight_score:.2f}/0.40]",
            f"FCR: {fcr:.2f} (target <2.0) [{fcr_score:.2f}/0.30]",
            f"Survival: {survival:.1%} [{survival_score:.2f}/0.30]",
        ]
        return GradeResult(
            score=round(score, 3),
            passed=score >= 0.5,
            feedback=" | ".join(details),
        )

    def _oxygen_grader(self, state, history, config) -> GradeResult:
        """Oxygen management: keep DO above safe threshold.

        Components:
          do_safe   (0.70): fraction of hours with DO ≥ 5.0 mg/L
          safety    (0.30): bonus for min DO ≥ 4.0 (0.30), partial at ≥ 3.0 (0.15)
        """
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
        """Water quality balance: maintain all parameters simultaneously.

        Components:
          avg_wq     (0.80): average water_quality_score across episode
          violations (0.20): bonus for zero violations (DO<3 or UIA>0.1),
                             linearly decays with up to 20 violations
        """
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
        """Temperature stress: survive heat wave with minimal losses.

        Components:
          survival (0.50): min(1, actual/0.95) — normalized to 95% baseline
          growth   (0.30): min(1, weight_gain/10g) — any growth during stress
          wq       (0.20): average water_quality_score
        """
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
        """Ammonia crisis: manage rising ammonia after biofilter failure.

        Components:
          uia_control (0.40): full if peak UIA < 0.3, linear to 0 at 0.6
          survival    (0.40): min(1, actual/0.90) — normalized to 90% baseline
          efficiency  (0.20): bonus if total_cost < $200 (minimal resource use)
        """
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
        """Disease outbreak: detect and treat disease before mass mortality.

        Components:
          survival (0.40): min(1, actual/0.90) — normalized to 90% baseline
          timing   (0.30): full if treatment started in first 30% of episode
          wq       (0.30): average water_quality_score
        """
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
        """Growth optimization: maximize weight gain with efficient feed use.

        Components:
          weight   (0.40): min(1, actual/target) — target is 120g
          fcr      (0.30): max(0, (2.5-fcr)/1.0) — full at FCR≤1.5, 0 at FCR≥2.5
          survival (0.20): min(1, actual/0.98) — normalized to 98% baseline
          wq       (0.10): average water_quality_score
        """
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
        """Full growout: 60-day cycle from fingerling to market weight.

        Components:
          profit   (0.25): max(0, profit/5000) — normalized to $5,000 benchmark
          weight   (0.20): min(1, actual/400) — target 400g market weight
          survival (0.20): min(1, actual/0.85) — normalized to 85% baseline
          fcr      (0.10): max(0, (2.5-fcr)/1.0) — efficient feed conversion
          wq       (0.10): average water_quality_score
          harvest  (0.15): bonus for harvesting (0.15 at ≥400g, 0.08 at ≥200g, 0.03 otherwise)
        """
        fish = state["fish"]
        econ = state["economics"]
        target = config.get("target_weight", 400.0)

        profit_score = max(0, min(1.0, econ["current_profit"] / 5000)) * 0.25
        weight_score = min(1.0, fish["weight_g"] / target) * 0.20
        survival_score = min(1.0, fish["survival_rate"] / 0.85) * 0.20
        fcr = fish.get("fcr", 3.0)
        fcr_score = max(0, min(1.0, (2.5 - fcr) / 1.0)) * 0.10
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.10

        if state["harvested"] and fish["weight_g"] >= 400:
            harvest_bonus = 0.15
        elif state["harvested"] and fish["weight_g"] >= 200:
            harvest_bonus = 0.08
        elif state["harvested"]:
            harvest_bonus = 0.03
        else:
            harvest_bonus = 0.0

        score = profit_score + weight_score + survival_score + fcr_score + wq_score + harvest_bonus

        details = [
            f"Weight: {fish['weight_g']:.1f}g/{target}g [{weight_score:.2f}/0.20]",
            f"Profit: ${econ['current_profit']:.0f} [{profit_score:.2f}/0.25]",
            f"Survival: {fish['survival_rate']:.1%} [{survival_score:.2f}/0.20]",
            f"FCR: {fcr:.2f} [{fcr_score:.2f}/0.10]",
            f"Harvested: {'yes' if state['harvested'] else 'no'} [{harvest_bonus:.2f}/0.15]",
        ]
        return GradeResult(score=round(min(1.0, score), 3), passed=score >= 0.5,
                          feedback=" | ".join(details))

    def _storm_grader(self, state, history, config) -> GradeResult:
        """Storm response: survive severe storm + power outage.

        Components:
          survival   (0.60): min(1, actual/0.80) — normalized to 80% baseline
          wq         (0.30): average water_quality_score
          efficiency (0.10): flat bonus for completing episode
        """
        survival = state["fish"]["survival_rate"]
        survival_score = min(1.0, survival / 0.80) * 0.6
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.3
        efficiency_score = 0.1

        score = survival_score + wq_score + efficiency_score
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Survival: {survival:.1%} through storm")

    def _multi_objective_grader(self, state, history, config) -> GradeResult:
        """Multi-objective: balance profit, welfare, and water quality.

        Score = geometric_mean(profit_norm, welfare, avg_wq)^(1/3)

        This uses a geometric mean (not sum) so ALL three dimensions must
        be positive. If any component is 0, the entire score is 0.

        Normalization:
          profit_norm: max(0, profit) / 3000 — $3K is the benchmark
          welfare: max(0, 1 - avg_stress/0.3) — stress=0→1.0, stress≥0.3→0.0
          avg_wq: average water_quality_score (already 0-1)
        """
        profit = max(0, state["economics"]["current_profit"])
        profit_norm = min(1.0, profit / 3000)

        avg_stress = sum(h["fish"]["stress_level"] for h in history) / max(1, len(history))
        welfare = max(0, 1.0 - avg_stress / 0.3)

        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))

        score = (profit_norm * welfare * avg_wq) ** (1/3)
        return GradeResult(score=round(score, 3), passed=score >= 0.4,
                          feedback=f"Profit: ${profit:.0f}, Welfare: {welfare:.2f}, WQ: {avg_wq:.3f}")

    def _catastrophe_grader(self, state, history, config) -> GradeResult:
        """Catastrophe prevention: survive 5 compound crises in 14 days.

        Components:
          survival  (0.30): min(1, actual/0.70) — normalized to 70% baseline
          profit    (0.20): max(0, (profit+1000)/3000) — shifted so -$1K→0, $2K→1
          wq        (0.15): average water_quality_score
          disease   (0.15): max(0, 1 - deaths/500) — linear decay per death
          crisis    (0.10): recovery speed after worst DO event (fast=0.10, slow=0.02)
          timing    (0.10): bonus for harvesting before fish value drops further
        """
        survival = state["fish"]["survival_rate"]
        profit = state["economics"]["current_profit"]
        disease_deaths = state["disease"]["total_disease_deaths"]

        survival_score = min(1.0, survival / 0.70) * 0.3
        profit_score = max(0, min(1.0, (profit + 1000) / 3000)) * 0.20
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        wq_score = avg_wq * 0.15
        disease_score = max(0, 1.0 - disease_deaths / 500) * 0.15

        crisis_response = 0.1
        if history:
            worst_do_idx = min(range(len(history)),
                             key=lambda i: history[i]["water"]["DO"])
            if worst_do_idx < len(history) - 1:
                recovery_hours = 0
                for i in range(worst_do_idx, min(worst_do_idx + 24, len(history))):
                    if history[i]["water"]["DO"] >= 4.0:
                        recovery_hours = i - worst_do_idx
                        break
                if 0 < recovery_hours <= 6:
                    crisis_response = 0.1
                elif recovery_hours <= 12:
                    crisis_response = 0.06
                else:
                    crisis_response = 0.02

        timing_score = 0.1 if state["harvested"] else 0.0

        score = survival_score + profit_score + wq_score + disease_score + crisis_response + timing_score

        details = [
            f"Survival: {survival:.1%} [{survival_score:.2f}/0.30]",
            f"Profit: ${profit:.0f} [{profit_score:.2f}/0.20]",
            f"Disease deaths: {disease_deaths} [{disease_score:.2f}/0.15]",
            f"Crisis response [{crisis_response:.2f}/0.10]",
            f"Harvested: {'yes' if state['harvested'] else 'no'}",
        ]
        return GradeResult(score=round(min(1.0, score), 3), passed=score >= 0.3,
                          feedback=" | ".join(details))

    def _season_grader(self, state, history, config) -> GradeResult:
        """Season management: full 90-day season with ROI optimization.

        Components:
          roi      (0.40): max(0, roi/0.5) — ROI = profit / total_investment,
                           where total_investment = total_cost + fingerling_cost
                           (fingerling_cost = population × $0.05/fingerling)
          growth   (0.20): min(1, weight/400) — toward 400g market weight
          survival (0.20): min(1, actual/0.80) — normalized to 80% baseline
          fcr      (0.10): max(0, (2.5-fcr)/1.0) — efficient feed conversion
          welfare  (0.10): max(0, 1 - avg_stress/0.3) — low stress bonus
        """
        econ = state["economics"]
        # Total investment includes operating costs + fingerling purchase cost ($0.05 each)
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
        """Default grader: 50% survival + 50% water quality."""
        survival = state["fish"]["survival_rate"]
        score = survival * 0.5
        avg_wq = sum(h["water"]["water_quality_score"] for h in history) / max(1, len(history))
        score += avg_wq * 0.5
        return GradeResult(score=round(score, 3), passed=score >= 0.5,
                          feedback=f"Default grader: survival={survival:.1%}, WQ={avg_wq:.3f}")
