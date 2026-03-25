"""Tests for grader implementations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graders import KeywordMatchGrader, StrictGrader


SAMPLE_GROUND_TRUTH = [
    {
        "line": "6",
        "severity": "critical",
        "category": "bug",
        "description": "ZeroDivisionError",
        "keywords": ["zero", "division", "empty", "len"],
    },
]


class TestKeywordMatchGrader:
    def setup_method(self):
        self.grader = KeywordMatchGrader()

    def test_perfect_match(self):
        result = self.grader.grade(
            task_id="t1",
            ground_truth=SAMPLE_GROUND_TRUTH,
            agent_issues=[{
                "line": "6",
                "severity": "critical",
                "category": "bug",
                "description": "ZeroDivisionError when empty list, len is zero",
                "suggestion": "Check for empty list",
            }],
        )
        assert result.score > 0.5
        assert result.passed is True

    def test_no_match(self):
        result = self.grader.grade(
            task_id="t1",
            ground_truth=SAMPLE_GROUND_TRUTH,
            agent_issues=[{
                "line": "1",
                "severity": "minor",
                "category": "style",
                "description": "Bad variable name",
                "suggestion": "Rename it",
            }],
        )
        assert result.score < 0.3
        assert result.passed is False

    def test_empty_agent_issues(self):
        result = self.grader.grade(
            task_id="t1",
            ground_truth=SAMPLE_GROUND_TRUTH,
            agent_issues=[],
        )
        assert result.score == 0.0

    def test_scores_in_range(self):
        result = self.grader.grade(
            task_id="t1",
            ground_truth=SAMPLE_GROUND_TRUTH,
            agent_issues=[{"description": "something", "suggestion": ""}],
        )
        assert 0.0 <= result.score <= 1.0

    def test_no_ground_truth(self):
        result = self.grader.grade(
            task_id="t1", ground_truth=[], agent_issues=[]
        )
        assert result.score == 1.0
        assert result.passed is True


class TestStrictGrader:
    def setup_method(self):
        self.grader = StrictGrader()

    def test_all_found_passes(self):
        result = self.grader.grade(
            task_id="t1",
            ground_truth=SAMPLE_GROUND_TRUTH,
            agent_issues=[{
                "description": "ZeroDivisionError when empty, len is zero",
                "suggestion": "Check division",
            }],
        )
        assert result.score == 1.0
        assert result.passed is True

    def test_missing_issue_fails(self):
        result = self.grader.grade(
            task_id="t1",
            ground_truth=SAMPLE_GROUND_TRUTH,
            agent_issues=[{
                "description": "Unrelated issue",
                "suggestion": "Unrelated fix",
            }],
        )
        assert result.score == 0.0
        assert result.passed is False
