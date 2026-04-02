"""Tests for task definitions and grader integration."""
import pytest
from agentic_rl.tasks import TASKS, list_all_tasks, get_task
from agentic_rl.engine.simulator import FishFarmSimulator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from graders.farm_graders import FarmGrader


class TestTaskDefinitions:
    def test_12_tasks_loaded(self):
        assert len(TASKS) == 12

    def test_all_tasks_have_required_fields(self):
        required = {"difficulty", "episode_hours", "description", "initial_conditions",
                    "events", "reward_weights", "grader"}
        for tid, task in TASKS.items():
            missing = required - set(task.keys())
            assert not missing, f"Task {tid} missing fields: {missing}"

    def test_difficulty_distribution(self):
        diffs = [t["difficulty"] for t in TASKS.values()]
        assert diffs.count("easy") == 3
        assert diffs.count("medium") == 4
        assert diffs.count("hard") == 3
        assert diffs.count("extreme") == 2

    def test_get_task_valid(self):
        task = get_task("feeding_basics")
        assert task["difficulty"] == "easy"

    def test_get_task_invalid(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task")

    def test_list_all_tasks_returns_12(self):
        tasks = list_all_tasks()
        assert len(tasks) == 12


class TestGraderIntegration:
    def _run_task(self, task_id, feed=0.4, aeration=0.5, max_hours=None):
        task = get_task(task_id)
        sim = FishFarmSimulator(seed=42)
        ic = task["initial_conditions"]
        sim.reset(
            initial_weight=ic["weight_g"],
            initial_population=ic["population"],
            initial_temp=ic["temp"],
            initial_DO=ic["DO"],
            initial_TAN=ic["TAN"],
            initial_pH=ic["pH"],
            day_of_year=ic["day_of_year"],
            base_air_temp=ic.get("base_air_temp", 30.0),
            scheduled_events=task["events"][:],
        )
        hours = max_hours or min(task["episode_hours"], 72)  # cap for test speed
        history = []
        for _ in range(hours):
            state = sim.step(feed, aeration, 0.0, 0.02, False, "none")
            history.append(state)
        return state, history, task

    def test_feeding_grader_returns_score(self):
        state, history, task = self._run_task("feeding_basics")
        grader = FarmGrader()
        result = grader.grade("feeding_basics", state, history, task)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.feedback, str)

    def test_oxygen_grader_returns_score(self):
        state, history, task = self._run_task("oxygen_management", aeration=0.8)
        grader = FarmGrader()
        result = grader.grade("oxygen_management", state, history, task)
        assert 0.0 <= result.score <= 1.0

    def test_good_management_beats_bad(self):
        """Good management (moderate feed + high aeration) should score higher than neglect.
        Runs for 168h (7 days) so the biological cascade (overfeed → ammonia → DO crash →
        stress → mortality) has time to fully develop and differentiate the two strategies."""
        state_good, hist_good, task = self._run_task("feeding_basics", feed=0.5, aeration=0.7, max_hours=168)
        state_bad, hist_bad, _ = self._run_task("feeding_basics", feed=1.0, aeration=0.0, max_hours=168)

        grader = FarmGrader()
        good_score = grader.grade("feeding_basics", state_good, hist_good, task).score
        bad_score = grader.grade("feeding_basics", state_bad, hist_bad, task).score
        assert good_score > bad_score

    def test_all_graders_callable(self):
        """Every task's grader name resolves to a method."""
        grader = FarmGrader()
        for tid, task in TASKS.items():
            method_name = f"_{task['grader']}"
            assert hasattr(grader, method_name), f"Missing grader method: {method_name}"
