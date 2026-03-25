"""Reward function implementations for the Code Review environment.

These provide partial progress signals over the full review trajectory,
not just binary end-of-episode rewards.
"""

from typing import Dict, List, Optional

from .base_reward import BaseReward


class RecallReward(BaseReward):
    """Reward based on recall: fraction of ground truth issues found."""

    def __init__(self, keyword_threshold: float = 0.3):
        self.keyword_threshold = keyword_threshold

    def compute(
        self,
        issues_found: List[Dict[str, str]],
        ground_truth: List[Dict[str, str]],
        **kwargs,
    ) -> float:
        if not ground_truth:
            return 1.0

        matched = 0
        for truth in ground_truth:
            keywords = truth.get("keywords", [])
            for found in issues_found:
                text = f"{found.get('description', '')} {found.get('suggestion', '')}".lower()
                matches = sum(1 for kw in keywords if kw.lower() in text)
                if keywords and matches / len(keywords) >= self.keyword_threshold:
                    matched += 1
                    break

        return matched / len(ground_truth)


class PrecisionReward(BaseReward):
    """Reward penalizing false positives (noise in reviews)."""

    def compute(
        self,
        issues_found: List[Dict[str, str]],
        ground_truth: List[Dict[str, str]],
        **kwargs,
    ) -> float:
        if not issues_found:
            return 0.0

        matched = 0
        for found in issues_found:
            text = f"{found.get('description', '')} {found.get('suggestion', '')}".lower()
            for truth in ground_truth:
                keywords = truth.get("keywords", [])
                matches = sum(1 for kw in keywords if kw.lower() in text)
                if keywords and matches / len(keywords) >= 0.3:
                    matched += 1
                    break

        return matched / len(issues_found)


class SeverityWeightedReward(BaseReward):
    """Reward that weights critical issues more heavily than minor ones."""

    SEVERITY_WEIGHTS = {"critical": 3.0, "major": 2.0, "minor": 1.0}

    def compute(
        self,
        issues_found: List[Dict[str, str]],
        ground_truth: List[Dict[str, str]],
        **kwargs,
    ) -> float:
        if not ground_truth:
            return 1.0

        total_weight = sum(
            self.SEVERITY_WEIGHTS.get(t.get("severity", "minor"), 1.0)
            for t in ground_truth
        )

        earned_weight = 0.0
        for truth in ground_truth:
            keywords = truth.get("keywords", [])
            weight = self.SEVERITY_WEIGHTS.get(truth.get("severity", "minor"), 1.0)
            for found in issues_found:
                text = f"{found.get('description', '')} {found.get('suggestion', '')}".lower()
                matches = sum(1 for kw in keywords if kw.lower() in text)
                if keywords and matches / len(keywords) >= 0.3:
                    earned_weight += weight
                    break

        return earned_weight / total_weight if total_weight > 0 else 0.0
