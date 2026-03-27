"""Tests for custom HTTP endpoints: /tasks, /grader, /baseline."""

import sys
import os

# Ensure project root is on path for graders import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from src.agentic_rl.server.app import app

client = TestClient(app)


class TestTasksEndpoint:
    """Tests for GET /tasks."""

    def test_returns_200(self):
        resp = client.get("/tasks")
        assert resp.status_code == 200

    def test_returns_tasks_list(self):
        data = client.get("/tasks").json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
        assert len(data["tasks"]) >= 3  # at least 3 tasks required

    def test_task_has_required_fields(self):
        tasks = client.get("/tasks").json()["tasks"]
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task

    def test_returns_action_schema(self):
        data = client.get("/tasks").json()
        assert "action_schema" in data
        schema = data["action_schema"]
        assert "properties" in schema
        assert "issues_found" in schema["properties"]


class TestGraderEndpoint:
    """Tests for POST /grader."""

    def test_returns_200(self):
        resp = client.post("/grader", json={
            "task_id": "easy_001",
            "issues_found": [],
        })
        assert resp.status_code == 200

    def test_returns_score_in_range(self):
        data = client.post("/grader", json={
            "task_id": "easy_001",
            "issues_found": [],
        }).json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_good_answer_scores_higher(self):
        # Empty answer
        empty = client.post("/grader", json={
            "task_id": "easy_001",
            "issues_found": [],
        }).json()

        # Answer with relevant keywords
        good = client.post("/grader", json={
            "task_id": "easy_001",
            "issues_found": [
                {
                    "line": "6",
                    "severity": "critical",
                    "category": "bug",
                    "description": "ZeroDivisionError when list is empty, len returns zero",
                    "suggestion": "Check if len(numbers) is zero before division",
                }
            ],
        }).json()

        assert good["score"] > empty["score"]

    def test_unknown_task_returns_404(self):
        resp = client.post("/grader", json={
            "task_id": "nonexistent_task",
            "issues_found": [],
        })
        assert resp.status_code == 404

    def test_grader_not_static(self):
        """Grader must not always return the same score (disqualification criteria)."""
        scores = set()
        test_cases = [
            {"task_id": "easy_001", "issues_found": []},
            {"task_id": "easy_001", "issues_found": [
                {"description": "zero division empty len", "suggestion": "check zero"}
            ]},
            {"task_id": "hard_001", "issues_found": []},
        ]
        for case in test_cases:
            data = client.post("/grader", json=case).json()
            scores.add(round(data["score"], 4))

        assert len(scores) > 1, "Grader returns the same score for all inputs — will be disqualified"


class TestBaselineEndpoint:
    """Tests for POST /baseline."""

    def test_single_task(self):
        resp = client.post("/baseline", json={"task_id": "easy_001"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["task_id"] == "easy_001"

    def test_all_tasks(self):
        resp = client.post("/baseline", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tasks"] >= 3
        assert "avg_grader_score" in data

    def test_result_has_scores(self):
        data = client.post("/baseline", json={"task_id": "easy_001"}).json()
        result = data["results"][0]
        assert "grader_score" in result
        assert "grader_passed" in result
        assert 0.0 <= result["grader_score"] <= 1.0

    def test_unknown_task_returns_404(self):
        resp = client.post("/baseline", json={"task_id": "nonexistent"})
        assert resp.status_code == 404
