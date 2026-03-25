"""Tests for reward functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rewards import RecallReward, PrecisionReward, SeverityWeightedReward


GROUND_TRUTH = [
    {"severity": "critical", "keywords": ["zero", "division", "empty"]},
    {"severity": "major", "keywords": ["resource", "leak", "close"]},
]


class TestRecallReward:
    def setup_method(self):
        self.reward = RecallReward()

    def test_all_found(self):
        issues = [
            {"description": "zero division on empty list", "suggestion": "check"},
            {"description": "resource leak, file not closed", "suggestion": "use with"},
        ]
        score = self.reward.compute(issues_found=issues, ground_truth=GROUND_TRUTH)
        assert score == 1.0

    def test_none_found(self):
        score = self.reward.compute(issues_found=[], ground_truth=GROUND_TRUTH)
        assert score == 0.0

    def test_partial_found(self):
        issues = [{"description": "zero division on empty list", "suggestion": "check"}]
        score = self.reward.compute(issues_found=issues, ground_truth=GROUND_TRUTH)
        assert score == 0.5

    def test_no_ground_truth(self):
        score = self.reward.compute(issues_found=[], ground_truth=[])
        assert score == 1.0


class TestPrecisionReward:
    def setup_method(self):
        self.reward = PrecisionReward()

    def test_all_correct(self):
        issues = [
            {"description": "zero division on empty list", "suggestion": "check"},
        ]
        score = self.reward.compute(issues_found=issues, ground_truth=GROUND_TRUTH)
        assert score == 1.0

    def test_all_false_positives(self):
        issues = [
            {"description": "bad variable name", "suggestion": "rename"},
        ]
        score = self.reward.compute(issues_found=issues, ground_truth=GROUND_TRUTH)
        assert score == 0.0


class TestSeverityWeightedReward:
    def setup_method(self):
        self.reward = SeverityWeightedReward()

    def test_critical_worth_more(self):
        # Only find the critical issue
        critical_only = [
            {"description": "zero division on empty list", "suggestion": "check"},
        ]
        score_critical = self.reward.compute(
            issues_found=critical_only, ground_truth=GROUND_TRUTH
        )

        # Only find the major issue
        major_only = [
            {"description": "resource leak, file not closed", "suggestion": "use with"},
        ]
        score_major = self.reward.compute(
            issues_found=major_only, ground_truth=GROUND_TRUTH
        )

        # Critical should be worth more
        assert score_critical > score_major
