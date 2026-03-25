"""Tests for grader implementations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graders import ExactMatchGrader, RubricGrader


class TestExactMatchGrader:
    def setup_method(self):
        self.grader = ExactMatchGrader()

    def test_correct_answer(self):
        result = self.grader.grade(
            task_id="t1", target="paris", history=["paris"], final_score=1.0
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_wrong_answer(self):
        result = self.grader.grade(
            task_id="t1", target="paris", history=["london"], final_score=0.0
        )
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_history(self):
        result = self.grader.grade(
            task_id="t1", target="paris", history=[], final_score=0.0
        )
        assert result.passed is False

    def test_correct_among_multiple(self):
        result = self.grader.grade(
            task_id="t1", target="paris", history=["london", "berlin", "paris"], final_score=0.5
        )
        assert result.passed is True


class TestRubricGrader:
    def test_single_criterion(self):
        grader = RubricGrader(
            criteria={
                "correctness": {
                    "fn": lambda h, t, **kw: 1.0 if t in h else 0.0,
                    "weight": 1.0,
                    "description": "Is the answer correct?",
                },
            }
        )
        result = grader.grade(
            task_id="t1", target="paris", history=["paris"], final_score=1.0
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_multi_criteria(self):
        grader = RubricGrader(
            criteria={
                "correctness": {
                    "fn": lambda h, t, **kw: 1.0 if t in h else 0.0,
                    "weight": 0.7,
                    "description": "Correct answer",
                },
                "efficiency": {
                    "fn": lambda h, t, **kw: max(0, 1.0 - len(h) * 0.1),
                    "weight": 0.3,
                    "description": "Fewer attempts",
                },
            }
        )
        result = grader.grade(
            task_id="t1", target="paris", history=["paris"], final_score=1.0
        )
        assert result.passed is True
        assert result.score > 0.5
