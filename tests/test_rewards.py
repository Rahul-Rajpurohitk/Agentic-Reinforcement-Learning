"""Tests for reward functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rewards import ExactMatchReward, PartialMatchReward, MultiObjectiveReward


class TestExactMatchReward:
    def setup_method(self):
        self.reward = ExactMatchReward()

    def test_exact_match(self):
        assert self.reward.compute("q", "paris", target="paris") == 1.0

    def test_case_insensitive(self):
        assert self.reward.compute("q", "Paris", target="paris") == 1.0

    def test_no_match(self):
        assert self.reward.compute("q", "london", target="paris") == 0.0

    def test_no_target(self):
        assert self.reward.compute("q", "paris") == 0.0

    def test_strips_whitespace(self):
        assert self.reward.compute("q", "  paris  ", target="paris") == 1.0


class TestPartialMatchReward:
    def setup_method(self):
        self.reward = PartialMatchReward()

    def test_exact_match_gives_1(self):
        score = self.reward.compute("q", "paris", target="paris")
        assert score == 1.0

    def test_no_overlap_gives_0(self):
        score = self.reward.compute("q", "xyz", target="paris")
        assert score == 0.0

    def test_partial_overlap(self):
        score = self.reward.compute("q", "the capital is paris", target="paris france")
        assert 0.0 < score < 1.0


class TestMultiObjectiveReward:
    def test_weighted_combination(self):
        reward = MultiObjectiveReward(
            rewards=[ExactMatchReward(), PartialMatchReward()],
            weights=[0.5, 0.5],
        )
        score = reward.compute("q", "paris", target="paris")
        assert score == 1.0

    def test_mixed_scores(self):
        reward = MultiObjectiveReward(
            rewards=[ExactMatchReward(), PartialMatchReward()],
            weights=[0.7, 0.3],
        )
        # Exact match fails, partial should give something
        score = reward.compute("q", "the answer is paris", target="paris")
        assert 0.0 < score < 1.0
