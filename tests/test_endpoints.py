"""Tests for custom HTTP endpoints: /tasks, /grader, /baseline.

These endpoints are required by the OpenEnv hackathon spec:
- /tasks — list all 12 tasks with action schema
- /grader — grade a completed episode
- /baseline — run heuristic baseline on task(s)
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from src.agentic_rl.server.app import app
    client = TestClient(app)
    HAS_OPENENV = True
except ImportError:
    HAS_OPENENV = False
    client = None


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestTasksEndpoint:
    """Tests for GET /tasks."""

    def test_returns_200(self):
        resp = client.get("/tasks")
        assert resp.status_code == 200

    def test_returns_12_tasks(self):
        data = client.get("/tasks").json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
        assert len(data["tasks"]) == 12

    def test_task_has_required_fields(self):
        tasks = client.get("/tasks").json()["tasks"]
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task
            assert "description" in task
            assert "episode_hours" in task

    def test_difficulty_levels_present(self):
        tasks = client.get("/tasks").json()["tasks"]
        difficulties = {t["difficulty"] for t in tasks}
        assert difficulties == {"easy", "medium", "hard", "extreme"}

    def test_returns_action_schema(self):
        data = client.get("/tasks").json()
        assert "action_schema" in data
        schema = data["action_schema"]
        assert "properties" in schema
        # Verify key action fields are documented
        for field in ("feeding_rate", "aeration_rate", "heater_setting",
                      "water_exchange_rate", "harvest_decision", "treatment"):
            assert field in schema["properties"], f"Missing action field: {field}"

    def test_action_schema_has_descriptions(self):
        schema = client.get("/tasks").json()["action_schema"]
        for field_name, field_info in schema["properties"].items():
            assert "description" in field_info, \
                f"Action field {field_name} missing description"


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestGraderEndpoint:
    """Tests for POST /grader."""

    def _make_fake_state(self):
        """Create a minimal valid state dict for grading."""
        return {
            "fish": {
                "weight_g": 60.0, "population": 4900, "biomass_kg": 294.0,
                "mortality_today": 5, "cumulative_mortality": 100,
                "survival_rate": 0.98, "stress_level": 0.1,
                "growth_rate_g_day": 2.0, "sgr": 1.5, "fcr": 1.6,
                "condition_factor": 2.8, "weight_cv": 0.10,
                "feeding_response": "normal", "stocking_density": 49.0,
            },
            "water": {
                "temperature": 30.0, "DO": 6.5, "TAN": 0.3, "UIA": 0.01,
                "pH": 7.5, "NO2": 0.1, "NO3": 5.0, "alkalinity": 100.0,
                "chlorophyll_a": 5.0, "algae_bloom": False,
                "water_quality_score": 0.85,
            },
            "disease": {
                "active": False, "infected": 0, "exposed": 0, "recovered": 0,
                "treatment_active": False, "treatment_type": "none",
                "total_disease_deaths": 0, "severity": 0.0, "outbreak_count": 0,
            },
            "economics": {
                "total_feed_cost": 50.0, "total_energy_cost": 30.0,
                "total_operating_cost": 80.0, "total_treatment_cost": 0.0,
                "total_cost": 80.0, "fish_value": 882.0,
                "current_profit": 802.0, "feed_inventory_kg": 400.0,
                "market_price_multiplier": 1.0,
            },
            "weather": {
                "air_temp": 30.0, "is_daytime": True, "solar_intensity": 500,
                "wind_speed": 3.0, "cloud_cover": 0.3, "humidity": 75.0,
                "storm_active": False, "forecast": "Clear skies",
            },
            "time": {"hour": 12, "day": 7, "total_hours": 168, "day_of_year": 97},
            "events": {
                "active_events": [], "active_count": 0,
                "equipment": {"aerator": True, "biofilter": True, "heater": True},
            },
            "harvested": False, "catastrophe": False, "done": False,
        }

    def test_returns_200(self):
        state = self._make_fake_state()
        resp = client.post("/grader", json={
            "task_id": "feeding_basics",
            "final_state": state,
            "episode_history": [state],
        })
        assert resp.status_code == 200

    def test_returns_score_in_range(self):
        state = self._make_fake_state()
        data = client.post("/grader", json={
            "task_id": "feeding_basics",
            "final_state": state,
            "episode_history": [state],
        }).json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_returns_feedback(self):
        state = self._make_fake_state()
        data = client.post("/grader", json={
            "task_id": "feeding_basics",
            "final_state": state,
            "episode_history": [state],
        }).json()
        assert "feedback" in data
        assert isinstance(data["feedback"], str)
        assert len(data["feedback"]) > 0

    def test_returns_passed_flag(self):
        state = self._make_fake_state()
        data = client.post("/grader", json={
            "task_id": "feeding_basics",
            "final_state": state,
            "episode_history": [state],
        }).json()
        assert "passed" in data
        assert isinstance(data["passed"], bool)

    def test_unknown_task_returns_404(self):
        resp = client.post("/grader", json={
            "task_id": "nonexistent_task_xyz",
            "final_state": {},
            "episode_history": [],
        })
        assert resp.status_code == 404

    def test_grader_not_static(self):
        """Grader must not always return the same score (disqualification criteria)."""
        scores = set()

        # Good state
        good = self._make_fake_state()
        good["fish"]["weight_g"] = 100.0
        good["fish"]["fcr"] = 1.4
        good["fish"]["survival_rate"] = 0.99

        # Bad state
        bad = self._make_fake_state()
        bad["fish"]["weight_g"] = 30.0
        bad["fish"]["fcr"] = 4.0
        bad["fish"]["survival_rate"] = 0.50

        for state in [good, bad]:
            data = client.post("/grader", json={
                "task_id": "feeding_basics",
                "final_state": state,
                "episode_history": [state],
            }).json()
            scores.add(round(data["score"], 4))

        assert len(scores) > 1, "Grader returns the same score for all inputs — will be disqualified"


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestBaselineEndpoint:
    """Tests for POST /baseline."""

    def test_single_task(self):
        resp = client.post("/baseline", json={"task_id": "feeding_basics"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["task_id"] == "feeding_basics"

    def test_all_tasks(self):
        resp = client.post("/baseline", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tasks"] == 12
        assert "avg_grader_score" in data

    def test_result_has_scores(self):
        data = client.post("/baseline", json={"task_id": "feeding_basics"}).json()
        result = data["results"][0]
        assert "grader_score" in result
        assert "grader_passed" in result
        assert "grader_feedback" in result
        assert 0.0 <= result["grader_score"] <= 1.0

    def test_result_has_hours(self):
        data = client.post("/baseline", json={"task_id": "oxygen_management"}).json()
        result = data["results"][0]
        assert "hours_simulated" in result
        assert result["hours_simulated"] > 0

    def test_unknown_task_returns_404(self):
        resp = client.post("/baseline", json={"task_id": "nonexistent"})
        assert resp.status_code == 404

    def test_baseline_scores_vary_by_difficulty(self):
        """Different tasks should produce different baseline scores."""
        scores = {}
        for tid in ["feeding_basics", "ammonia_crisis"]:
            data = client.post("/baseline", json={"task_id": tid}).json()
            scores[tid] = data["results"][0]["grader_score"]

        # Not all equal
        assert len(set(round(s, 3) for s in scores.values())) > 1


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv-core not installed locally")
class TestHeuristicEndToEnd:
    """End-to-end: heuristic agent running through FishFarmEnvironment."""

    def test_heuristic_beats_baseline_on_feeding(self):
        """Heuristic agent should score well on feeding_basics."""
        from inference import heuristic_action
        from src.agentic_rl.server.environment import FishFarmEnvironment
        from src.agentic_rl.models import FarmAction

        env = FishFarmEnvironment()
        obs = env.reset(task_id="feeding_basics")
        obs_dict = obs.model_dump()

        for step in range(168):
            action_dict = heuristic_action(obs_dict, "feeding_basics", step, 168)
            action = FarmAction(**action_dict)
            obs = env.step(action)
            obs_dict = obs.model_dump()
            if obs.done:
                break

        final_score = obs.reward or 0
        assert final_score > 0.5, f"Heuristic scored only {final_score:.3f} on feeding_basics"

    def test_heuristic_on_oxygen_management(self):
        """Heuristic should maintain DO on oxygen_management task."""
        from inference import heuristic_action
        from src.agentic_rl.server.environment import FishFarmEnvironment
        from src.agentic_rl.models import FarmAction

        env = FishFarmEnvironment()
        obs = env.reset(task_id="oxygen_management")
        obs_dict = obs.model_dump()

        for step in range(72):
            action_dict = heuristic_action(obs_dict, "oxygen_management", step, 72)
            action = FarmAction(**action_dict)
            obs = env.step(action)
            obs_dict = obs.model_dump()
            if obs.done:
                break

        final_score = obs.reward or 0
        assert final_score >= 0.7, f"Heuristic scored {final_score:.3f} on oxygen_management"

    def test_heuristic_on_ammonia_crisis(self):
        """Heuristic should handle ammonia crisis with aggressive dilution."""
        from inference import heuristic_action
        from src.agentic_rl.server.environment import FishFarmEnvironment
        from src.agentic_rl.models import FarmAction

        env = FishFarmEnvironment()
        obs = env.reset(task_id="ammonia_crisis")
        obs_dict = obs.model_dump()

        for step in range(72):
            action_dict = heuristic_action(obs_dict, "ammonia_crisis", step, 72)
            action = FarmAction(**action_dict)
            obs = env.step(action)
            obs_dict = obs.model_dump()
            if obs.done:
                break

        final_score = obs.reward or 0
        assert final_score >= 0.4, f"Heuristic scored {final_score:.3f} on ammonia_crisis"

    def test_metadata_has_author(self):
        """Metadata endpoint should return rich metadata."""
        resp = client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Fish Farm OpenEnv"
        assert data["author"] == "Rahul Rajpurohit"
        assert "Nile Tilapia" in data["description"]

    def test_schema_endpoint_returns_all_schemas(self):
        """Schema endpoint should return action, observation, and state schemas."""
        resp = client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data
        # Action schema should have our 6 controls
        props = data["action"].get("properties", {})
        assert "feeding_rate" in props
        assert "treatment" in props
