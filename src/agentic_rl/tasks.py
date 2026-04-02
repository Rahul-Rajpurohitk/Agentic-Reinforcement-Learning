"""12 task scenarios for the Fish Farm environment.

Easy (3):    Single-concern, short episodes, forgiving thresholds
Medium (4):  Multi-concern, events, cascading risks
Hard (3):    Full lifecycle, compound events, multi-objective optimization
Extreme (2): Everything at once, frontier-model difficulty

Each task dict contains:
- initial_conditions: starting state overrides
- events: scheduled Event objects
- episode_hours: max episode length
- reward_weights: per-component weights for reward calculation
- grader: name of grading function
- description: natural language task briefing for the agent
- difficulty: easy/medium/hard/extreme
"""

from typing import Any, Dict, List
from .engine.events import Event


def _make_tasks() -> Dict[str, Dict[str, Any]]:
    return {
        # =====================================================================
        # EASY TASKS — Single concern, learn one control at a time
        # =====================================================================

        "feeding_basics": {
            "difficulty": "easy",
            "episode_hours": 7 * 24,
            "description": (
                "You manage a healthy tilapia tank for 7 days. Your goal: feed the fish "
                "to achieve steady growth without overfeeding. Fish start at 50g, target 55g+. "
                "Keep FCR below 2.0. Zero fish should die from starvation or overfeeding."
            ),
            "initial_conditions": {
                "weight_g": 50.0, "population": 5000, "temp": 30.0,
                "DO": 7.0, "TAN": 0.1, "pH": 7.5, "day_of_year": 90,
            },
            "events": [],
            "reward_weights": {"growth": 0.5, "fcr": 0.3, "survival": 0.2},
            "grader": "feeding_grader",
            "target_weight": 55.0,
        },

        "oxygen_management": {
            "difficulty": "easy",
            "episode_hours": 3 * 24,
            "description": (
                "It's a hot week (air temp 35C). Dissolved oxygen is dropping. "
                "Your job: keep DO above 5.0 mg/L at all times using the aerator. "
                "Fish are already stocked at moderate density. Score based on "
                "minimum DO maintained and time spent in safe zone."
            ),
            "initial_conditions": {
                "weight_g": 100.0, "population": 4000, "temp": 32.0,
                "DO": 6.0, "TAN": 0.3, "pH": 7.5, "day_of_year": 180,
                "base_air_temp": 35.0,
            },
            "events": [],
            "reward_weights": {"do_stability": 0.5, "do_risk": 0.2, "efficiency": 0.3},
            "grader": "oxygen_grader",
        },

        "water_quality_balance": {
            "difficulty": "easy",
            "episode_hours": 7 * 24,
            "description": (
                "Manage all water quality parameters simultaneously: keep DO > 5, "
                "ammonia (UIA) < 0.05, pH 6.5-8.5, temperature 27-32C. "
                "You have full control: feeding, aeration, heater, water exchange. "
                "Score = time-averaged water quality composite score."
            ),
            "initial_conditions": {
                "weight_g": 80.0, "population": 5000, "temp": 29.0,
                "DO": 7.0, "TAN": 0.2, "pH": 7.5, "day_of_year": 100,
            },
            "events": [],
            "reward_weights": {"water_quality": 0.8, "efficiency": 0.2},
            "grader": "water_quality_grader",
        },

        # =====================================================================
        # MEDIUM TASKS — Multi-concern, events, cascading risks
        # =====================================================================

        "temperature_stress": {
            "difficulty": "medium",
            "episode_hours": 5 * 24,
            "description": (
                "ALERT: A heat wave is hitting. Air temperature will reach 38C for "
                "3 days starting hour 24. You must manage water temperature using "
                "heater (cooling mode), reduce feeding (fish eat less in heat), and "
                "increase aeration (warm water holds less oxygen). "
                "Goal: Keep fish alive and growing through the crisis."
            ),
            "initial_conditions": {
                "weight_g": 120.0, "population": 8000, "temp": 30.0,
                "DO": 7.0, "TAN": 0.2, "pH": 7.5, "day_of_year": 200,
                "base_air_temp": 33.0,
            },
            "events": [
                Event(type="heat_wave", trigger_hour=24, severity=0.7,
                      duration_hours=72, description="HEAT WAVE: Air temp reaching 38C"),
            ],
            "reward_weights": {"survival": 0.4, "growth": 0.3, "water_quality": 0.3},
            "grader": "stress_survival_grader",
        },

        "ammonia_crisis": {
            "difficulty": "medium",
            "episode_hours": 3 * 24,
            "description": (
                "EMERGENCY: The biofilter has partially failed (50% capacity). "
                "Ammonia is rising. You must: reduce feeding immediately, increase "
                "water exchange, maintain aeration. Goal: prevent ammonia (UIA) from "
                "reaching lethal levels (>0.6 mg/L) while keeping fish alive."
            ),
            "initial_conditions": {
                "weight_g": 150.0, "population": 7000, "temp": 30.0,
                "DO": 6.5, "TAN": 1.5, "pH": 7.8, "day_of_year": 150,
            },
            "events": [
                Event(type="equipment_failure", trigger_hour=0, severity=0.5,
                      duration_hours=48, description="BIOFILTER FAILURE: 50% capacity",
                      equipment="biofilter"),
            ],
            "reward_weights": {"ammonia_control": 0.4, "survival": 0.4, "efficiency": 0.2},
            "grader": "ammonia_crisis_grader",
        },

        "disease_outbreak": {
            "difficulty": "medium",
            "episode_hours": 10 * 24,
            "description": (
                "Fish are showing signs of stress — sluggish feeding, elevated mortality. "
                "A disease may be developing. Watch for: increasing mortality, "
                "feeding refusal, and behavioral changes. If you detect disease, "
                "apply treatment ('antibiotics'). Also manage water quality to "
                "reduce stress. Goal: contain the outbreak with <10% total mortality."
            ),
            "initial_conditions": {
                "weight_g": 200.0, "population": 6000, "temp": 31.0,
                "DO": 6.0, "TAN": 0.5, "pH": 7.6, "day_of_year": 120,
            },
            "events": [
                Event(type="disease", trigger_hour=12, severity=0.4,
                      duration_hours=0, description="Disease pathogen introduced"),
            ],
            "reward_weights": {"survival": 0.4, "treatment_timing": 0.3, "water_quality": 0.3},
            "grader": "disease_grader",
        },

        "growth_optimization": {
            "difficulty": "medium",
            "episode_hours": 14 * 24,
            "description": (
                "Optimize fish growth over 2 weeks. Fish start at 80g, target 120g+. "
                "Balance aggressive feeding (faster growth) against water quality "
                "degradation (ammonia, DO). Achieve the best FCR possible while "
                "maximizing weight gain. Minimize mortality."
            ),
            "initial_conditions": {
                "weight_g": 80.0, "population": 6000, "temp": 30.0,
                "DO": 7.0, "TAN": 0.1, "pH": 7.5, "day_of_year": 90,
            },
            "events": [],
            "reward_weights": {"growth": 0.4, "fcr": 0.3, "survival": 0.2, "water_quality": 0.1},
            "grader": "growth_optimization_grader",
            "target_weight": 120.0,
        },

        # =====================================================================
        # HARD TASKS — Full lifecycle, compound events, multi-objective
        # =====================================================================

        "full_growout": {
            "difficulty": "hard",
            "episode_hours": 60 * 24,
            "description": (
                "Complete grow-out cycle: take fish from 20g fingerlings to market "
                "weight (400g+). Manage all systems over 60 days. Random weather, "
                "possible disease, possible equipment issues. Score on: final weight, "
                "survival rate, FCR, and profit. Decide when to harvest for max value."
            ),
            "initial_conditions": {
                "weight_g": 20.0, "population": 7000, "temp": 28.0,
                "DO": 7.5, "TAN": 0.05, "pH": 7.5, "day_of_year": 60,
            },
            "events": [],
            "reward_weights": {"profit": 0.3, "growth": 0.25, "survival": 0.25, "fcr": 0.1, "water_quality": 0.1},
            "grader": "full_growout_grader",
            "target_weight": 400.0,
        },

        "storm_response": {
            "difficulty": "hard",
            "episode_hours": 5 * 24,
            "description": (
                "SEVERE STORM WARNING: A major storm hits at hour 24. Effects: "
                "temperature drops 8C, power outage for 12 hours (aerators, heater, "
                "biofilter ALL down), high winds. After power returns, biofilter "
                "needs 24 hours to recover. Your job: maximize survival through the crisis. "
                "Pre-position your systems before the storm hits."
            ),
            "initial_conditions": {
                "weight_g": 200.0, "population": 8000, "temp": 30.0,
                "DO": 7.5, "TAN": 0.2, "pH": 7.5, "day_of_year": 180,
            },
            "events": [
                Event(type="storm", trigger_hour=24, severity=0.9,
                      duration_hours=48, description="SEVERE STORM: Temp -8C, high winds"),
                Event(type="power_outage", trigger_hour=24, severity=1.0,
                      duration_hours=12, description="POWER OUTAGE: All equipment offline"),
                Event(type="equipment_failure", trigger_hour=36, severity=0.7,
                      duration_hours=24, description="BIOFILTER RECOVERY: Reduced capacity post-storm",
                      equipment="biofilter"),
            ],
            "reward_weights": {"survival": 0.6, "water_quality": 0.3, "efficiency": 0.1},
            "grader": "storm_grader",
        },

        "multi_objective": {
            "difficulty": "hard",
            "episode_hours": 30 * 24,
            "description": (
                "Multi-objective challenge: simultaneously maximize profit, maintain "
                "fish welfare (stress < 0.3), and minimize environmental impact "
                "(water discharge quality). Score is the product of all three objectives — "
                "neglecting any one dimension tanks your score. "
                "Fish start at 100g in a moderately stocked tank."
            ),
            "initial_conditions": {
                "weight_g": 100.0, "population": 8000, "temp": 29.0,
                "DO": 7.0, "TAN": 0.2, "pH": 7.5, "day_of_year": 100,
            },
            "events": [],
            "reward_weights": {"profit": 0.33, "welfare": 0.34, "environment": 0.33},
            "grader": "multi_objective_grader",
        },

        # =====================================================================
        # EXTREME TASKS — Everything at once, frontier-model difficulty
        # =====================================================================

        "catastrophe_prevention": {
            "difficulty": "extreme",
            "episode_hours": 14 * 24,
            "description": (
                "CRITICAL SITUATION: Multiple simultaneous challenges. "
                "Day 1: Algae bloom developing (DO supersaturation then crash). "
                "Day 3: Equipment degradation (aerator at 60% capacity). "
                "Day 5: Disease outbreak detected. "
                "Day 7: Market price drops 40%. "
                "Day 10: Feed delivery delayed (inventory running low). "
                "Prevent mass mortality, optimize harvest timing despite falling prices, "
                "manage disease while resources are constrained. "
                "This task separates frontier models from basic agents."
            ),
            "initial_conditions": {
                "weight_g": 250.0, "population": 7000, "temp": 31.0,
                "DO": 8.0, "TAN": 0.4, "pH": 7.8, "day_of_year": 200,
            },
            "events": [
                Event(type="algae_bloom", trigger_hour=12, severity=0.6,
                      duration_hours=48, description="ALGAE BLOOM: DO swinging wildly"),
                Event(type="equipment_failure", trigger_hour=72, severity=0.4,
                      duration_hours=96, description="AERATOR DEGRADED: 60% capacity",
                      equipment="aerator"),
                Event(type="disease", trigger_hour=120, severity=0.5,
                      duration_hours=0, description="DISEASE DETECTED: Mortality rising"),
                Event(type="price_change", trigger_hour=168, severity=0.4,
                      duration_hours=168, description="MARKET CRASH: Fish price -40%",
                      price_multiplier=0.6),
                Event(type="feed_shortage", trigger_hour=240, severity=0.7,
                      duration_hours=48, description="FEED DELAYED: Inventory critical"),
            ],
            "reward_weights": {"survival": 0.3, "profit": 0.25, "water_quality": 0.2,
                              "disease_control": 0.15, "timing": 0.1},
            "grader": "catastrophe_grader",
        },

        "season_management": {
            "difficulty": "extreme",
            "episode_hours": 90 * 24,
            "description": (
                "Full 90-day season. Fish from 10g to market weight. "
                "Seasonal temperature changes (summer peak). Random storms, "
                "random disease. Feed inventory must be managed (deliveries "
                "every 14 days). Decide optimal harvest timing — market price "
                "fluctuates weekly. Score = ROI (profit / total investment). "
                "The best agents will achieve >50% ROI."
            ),
            "initial_conditions": {
                "weight_g": 10.0, "population": 10000, "temp": 27.0,
                "DO": 7.5, "TAN": 0.05, "pH": 7.5, "day_of_year": 60,
            },
            "events": [],
            "reward_weights": {"roi": 0.4, "growth": 0.2, "survival": 0.2, "fcr": 0.1, "welfare": 0.1},
            "grader": "season_grader",
        },
    }


TASKS = _make_tasks()


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_all_tasks() -> List[Dict[str, str]]:
    return [
        {
            "task_id": tid,
            "difficulty": task["difficulty"],
            "description": task["description"],
            "episode_hours": task["episode_hours"],
        }
        for tid, task in TASKS.items()
    ]
