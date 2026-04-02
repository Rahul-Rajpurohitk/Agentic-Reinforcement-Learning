"""Fish Farm Environment — core game logic.

A real-world OpenEnv environment where an AI agent manages a Nile Tilapia
Recirculating Aquaculture System (RAS) fish farm.

Extends openenv.core.env_server.Environment with the official interface:
  reset(seed, episode_id, **kwargs) -> Observation
  step(action, timeout_s, **kwargs) -> Observation
  state -> State
"""

import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from ..models import FarmAction, FarmObservation, FarmState
from ..tasks import get_task
from ..engine.simulator import FishFarmSimulator
from ..engine.events import Event
from ..rewards import calculate_reward


def _build_feedback(sim_state: Dict[str, Any], task_desc: str, hour: int) -> str:
    """Generate narrative feedback from the current simulation state."""
    fish = sim_state["fish"]
    water = sim_state["water"]
    econ = sim_state["economics"]

    parts = []

    # Time context
    time = sim_state["time"]
    parts.append(f"Day {time['day']}, Hour {time['hour']:02d}:00.")

    # Fish status
    resp = fish["feeding_response"]
    if resp == "refusing":
        parts.append("WARNING: Fish are refusing to eat!")
    elif resp == "sluggish":
        parts.append("Fish are feeding sluggishly — possible stress.")
    elif resp == "eager":
        parts.append("Fish are feeding eagerly — good appetite.")

    # Water alerts
    if water["DO"] < 3.0:
        parts.append(f"DANGER: Dissolved oxygen critically low at {water['DO']:.1f} mg/L!")
    elif water["DO"] < 5.0:
        parts.append(f"Warning: DO below optimal at {water['DO']:.1f} mg/L.")

    if water["UIA"] > 0.3:
        parts.append(f"DANGER: Toxic ammonia very high at {water['UIA']:.4f} mg/L!")
    elif water["UIA"] > 0.05:
        parts.append(f"Warning: Toxic ammonia elevated at {water['UIA']:.4f} mg/L.")

    if water["temperature"] > 35 or water["temperature"] < 22:
        parts.append(f"DANGER: Water temperature at {water['temperature']:.1f}C — outside safe range!")

    # Mortality
    if fish["mortality_today"] > 50:
        parts.append(f"ALERT: {fish['mortality_today']} fish died in the last period!")
    elif fish["mortality_today"] > 10:
        parts.append(f"Note: {fish['mortality_today']} fish deaths recorded.")

    # Disease
    disease = sim_state["disease"]
    if disease["active"]:
        parts.append("Disease outbreak is active! Consider treatment.")

    # Events
    events = sim_state.get("events", {})
    for alert in events.get("active_events", []):
        parts.append(f"EVENT: {alert}")

    # Economics milestone
    if fish["weight_g"] >= 400 and hour > 0:
        parts.append(f"Fish have reached market weight ({fish['weight_g']:.0f}g). Consider harvesting.")

    if not parts[1:]:  # no alerts after the time line
        parts.append("All systems nominal.")

    return " ".join(parts)


def _calculate_reward(
    sim_state: Dict[str, Any],
    prev_state: Optional[Dict[str, Any]],
    reward_weights: Dict[str, float],
) -> float:
    """Calculate hourly reward signal. Delegates to rewards module."""
    return calculate_reward(sim_state, prev_state, reward_weights)


class FishFarmEnvironment(Environment[FarmAction, FarmObservation, FarmState]):
    """OpenEnv Fish Farm environment.

    AI agents manage a Nile Tilapia RAS fish farm, making hourly decisions
    about feeding, aeration, temperature control, water exchange, disease
    treatment, and harvest timing across 12 tasks of escalating difficulty.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._sim = FishFarmSimulator()
        self._state = FarmState()
        self._task_config: Dict[str, Any] = {}
        self._episode_history: List[Dict[str, Any]] = []
        self._prev_sim_state: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "feeding_basics",
        **kwargs: Any,
    ) -> FarmObservation:
        """Start a new episode with the given task."""
        task = get_task(task_id)
        eid = episode_id or str(uuid.uuid4())
        actual_seed = seed or 42

        ic = task["initial_conditions"]
        sim_state = self._sim.reset(
            initial_weight=ic["weight_g"],
            initial_population=ic["population"],
            initial_temp=ic["temp"],
            initial_DO=ic["DO"],
            initial_TAN=ic["TAN"],
            initial_pH=ic["pH"],
            day_of_year=ic["day_of_year"],
            base_air_temp=ic.get("base_air_temp", 30.0),
            seed=actual_seed,
            scheduled_events=[
                Event(
                    type=e.type, trigger_hour=e.trigger_hour,
                    severity=e.severity, duration_hours=e.duration_hours,
                    description=e.description, equipment=e.equipment,
                    price_multiplier=e.price_multiplier,
                ) for e in task["events"]
            ],
        )

        self._task_config = task
        self._episode_history = []
        self._prev_sim_state = None

        self._state = FarmState(
            episode_id=eid,
            step_count=0,
            task_id=task_id,
            is_complete=False,
            final_score=0.0,
            max_hours=task["episode_hours"],
            sim_state=sim_state,
        )

        return self._make_observation(sim_state, done=False, reward=None,
                                      feedback=f"TASK: {task['description']}")

    def step(
        self,
        action: FarmAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FarmObservation:
        """Process the agent's action and advance simulation by 1 hour."""
        if self._state.is_complete:
            return self._terminal_observation()

        self._state.step_count += 1

        sim_state = self._sim.step(
            feeding_rate=action.feeding_rate,
            aeration_rate=action.aeration_rate,
            heater_setting=action.heater_setting,
            water_exchange_rate=action.water_exchange_rate,
            harvest=action.harvest_decision,
            treatment=action.treatment,
        )

        self._episode_history.append(sim_state)

        # Calculate reward
        reward = _calculate_reward(
            sim_state, self._prev_sim_state,
            self._task_config["reward_weights"],
        )
        self._prev_sim_state = sim_state

        # Check done
        done = sim_state["done"] or self._state.step_count >= self._state.max_hours

        if done:
            self._state.is_complete = True
            # Run grader for final score
            from graders.farm_graders import FarmGrader
            grader = FarmGrader()
            grade_result = grader.grade(
                self._state.task_id, sim_state,
                self._episode_history, self._task_config,
            )
            self._state.final_score = grade_result.score
            reward = grade_result.score  # final reward is the grader score

        self._state.sim_state = sim_state

        feedback = _build_feedback(sim_state, self._task_config["description"],
                                   self._state.step_count)

        return self._make_observation(sim_state, done=done, reward=reward,
                                      feedback=feedback)

    @property
    def state(self) -> FarmState:
        """Return current internal state (for grading)."""
        return self._state

    def _make_observation(
        self, sim_state: Dict[str, Any],
        done: bool, reward: Optional[float], feedback: str,
    ) -> FarmObservation:
        """Build observation from simulator state."""
        fish = sim_state["fish"]
        water = sim_state["water"]
        econ = sim_state["economics"]
        weather = sim_state["weather"]
        events = sim_state.get("events", {})
        equip = events.get("equipment", {})

        return FarmObservation(
            done=done,
            reward=reward,
            metadata={"episode_id": self._state.episode_id,
                      "step": self._state.step_count},
            # Fish
            avg_fish_weight=fish["weight_g"],
            population=fish["population"],
            mortality_today=fish["mortality_today"],
            stress_level=fish["stress_level"],
            feeding_response=fish["feeding_response"],
            biomass_kg=fish["biomass_kg"],
            # Water
            temperature=water["temperature"],
            dissolved_oxygen=water["DO"],
            ph=water["pH"],
            ammonia=water["TAN"],
            ammonia_toxic=water["UIA"],
            nitrite=water["NO2"],
            water_quality_score=water["water_quality_score"],
            # Equipment
            aerator_working=equip.get("aerator", True),
            biofilter_working=equip.get("biofilter", True),
            heater_working=equip.get("heater", True),
            feed_remaining_kg=econ["feed_inventory_kg"],
            # Economics
            current_fish_value=econ["fish_value"],
            total_cost_so_far=econ["total_cost"],
            current_profit=econ["current_profit"],
            # Context
            weather_forecast=weather["forecast"],
            day_in_cycle=sim_state["time"]["day"],
            time_of_day=sim_state["time"]["hour"],
            alerts=events.get("active_events", []),
            # Feedback
            feedback=feedback,
        )

    def _terminal_observation(self) -> FarmObservation:
        """Return observation for an already-completed episode."""
        sim_state = self._state.sim_state or self._sim.get_state()
        return self._make_observation(
            sim_state, done=True, reward=self._state.final_score,
            feedback=f"Episode ended. Final score: {self._state.final_score:.3f}",
        )
