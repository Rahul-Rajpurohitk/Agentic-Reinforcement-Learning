"""Fish Farm LLM Agent — inference script for OpenEnv evaluation.

Connects to the Fish Farm environment via HTTP, runs all tasks using an LLM
to make management decisions, and reports scores.

Required environment variables:
    API_BASE_URL: LLM API endpoint (e.g., https://api.openai.com/v1)
    MODEL_NAME: Model to use (e.g., gpt-4o, claude-3-sonnet)
    HF_TOKEN: HuggingFace token for authentication

Runtime constraint: < 20 minutes on 2 vCPU / 8GB RAM
Strategy: Run easy tasks first (shortest episodes), limit history window,
batch requests where possible.
"""

import json
import os
import time
import re
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


# ---- Configuration ----
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_HISTORY = 5  # recent observations to include in prompt


# ---- LLM Client ----

def get_llm_client() -> OpenAI:
    """Create OpenAI-compatible client."""
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy"),
    )


SYSTEM_PROMPT = """You are an expert Nile Tilapia aquaculture manager operating a Recirculating Aquaculture System (RAS).

Your fish need:
- Water temperature: 27-32°C optimal (below 20°C or above 36°C = danger)
- Dissolved oxygen: >5 mg/L optimal (below 3 = danger, below 1 = lethal)
- pH: 6.5-8.5 optimal
- Ammonia (UIA): <0.02 mg/L safe, >0.05 toxic, >0.6 lethal
- Feeding: Higher feeding = faster growth BUT more ammonia and less oxygen

Key trade-offs:
- Feeding increases ammonia production and oxygen consumption
- Aeration adds oxygen but costs electricity
- Water exchange dilutes ammonia but costs water
- Disease needs early treatment (antibiotics) before mortality spikes
- Harvest timing: bigger fish = more revenue, but risk of die-off

You must respond with ONLY a valid JSON object matching the action schema. No explanation, no markdown, just JSON."""


def build_observation_prompt(
    task_description: str,
    obs: Dict[str, Any],
    recent_history: List[Dict[str, Any]],
) -> str:
    """Build the observation prompt for the LLM."""
    prompt = f"""TASK: {task_description}

CURRENT READINGS (Day {obs.get('day_in_cycle', 0)}, Hour {obs.get('time_of_day', 0):02d}:00):
  Water Temperature: {obs.get('temperature', 0):.1f}°C (optimal: 27-32°C)
  Dissolved Oxygen:  {obs.get('dissolved_oxygen', 0):.1f} mg/L (danger below 3.0, lethal below 1.0)
  pH:                {obs.get('ph', 0):.2f} (optimal: 6.5-8.5)
  Ammonia (TAN):     {obs.get('ammonia', 0):.3f} mg/L
  Toxic Ammonia:     {obs.get('ammonia_toxic', 0):.5f} mg/L (toxic above 0.05)
  Nitrite:           {obs.get('nitrite', 0):.3f} mg/L
  Water Quality:     {obs.get('water_quality_score', 0):.3f}/1.000

FISH STATUS:
  Average Weight: {obs.get('avg_fish_weight', 0):.1f}g | Population: {obs.get('population', 0):,}
  Today's Mortality: {obs.get('mortality_today', 0)} | Stress: {obs.get('stress_level', 0):.2f}
  Feeding Response: {obs.get('feeding_response', 'unknown')}
  Biomass: {obs.get('biomass_kg', 0):.1f}kg

ECONOMICS:
  Fish Value: ${obs.get('current_fish_value', 0):.0f} | Cost So Far: ${obs.get('total_cost_so_far', 0):.0f}
  Current Profit: ${obs.get('current_profit', 0):.0f}
  Feed Remaining: {obs.get('feed_remaining_kg', 0):.0f}kg

WEATHER: {obs.get('weather_forecast', 'N/A')}
EQUIPMENT: Aerator={'OK' if obs.get('aerator_working', True) else 'FAILED'}, Biofilter={'OK' if obs.get('biofilter_working', True) else 'FAILED'}, Heater={'OK' if obs.get('heater_working', True) else 'FAILED'}"""

    # Alerts
    alerts = obs.get('alerts', [])
    if alerts:
        prompt += f"\nALERTS: {'; '.join(alerts)}"

    # Feedback
    feedback = obs.get('feedback', '')
    if feedback:
        prompt += f"\nSTATUS: {feedback}"

    # Recent trend
    if recent_history:
        prompt += "\n\nRECENT TREND (last readings):"
        for i, h in enumerate(recent_history[-3:]):
            prompt += f"\n  [{-len(recent_history[-3:])+i}h] DO={h.get('dissolved_oxygen', 0):.1f}, " \
                     f"TAN={h.get('ammonia', 0):.3f}, Temp={h.get('temperature', 0):.1f}, " \
                     f"Mort={h.get('mortality_today', 0)}"

    prompt += """

Based on this data, decide your actions for the next hour.
Respond with ONLY a JSON object:
{
  "feeding_rate": 0.0-1.0,
  "aeration_rate": 0.0-1.0,
  "heater_setting": -1.0 to 1.0,
  "water_exchange_rate": 0.0-0.10,
  "harvest_decision": false,
  "treatment": "none"
}"""

    return prompt


def parse_action(response_text: str) -> Dict[str, Any]:
    """Parse LLM response into action dict, with fallback defaults."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group())
            except json.JSONDecodeError:
                action = {}
        else:
            action = {}

    # Apply defaults and clamp
    return {
        "feeding_rate": max(0.0, min(1.0, float(action.get("feeding_rate", 0.3)))),
        "aeration_rate": max(0.0, min(1.0, float(action.get("aeration_rate", 0.5)))),
        "heater_setting": max(-1.0, min(1.0, float(action.get("heater_setting", 0.0)))),
        "water_exchange_rate": max(0.0, min(0.10, float(action.get("water_exchange_rate", 0.02)))),
        "harvest_decision": bool(action.get("harvest_decision", False)),
        "treatment": str(action.get("treatment", "none")),
    }


def run_task(
    client: OpenAI,
    env_client: httpx.Client,
    task_id: str,
    task_description: str,
    max_hours: int,
) -> Dict[str, Any]:
    """Run a single task and return results."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id} ({max_hours} hours)")
    print(f"{'='*60}")

    # Reset environment
    reset_resp = env_client.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    history: List[Dict[str, Any]] = []
    total_reward = 0.0
    steps = 0
    start_time = time.time()

    # Decide LLM call frequency based on episode length
    # Short episodes: call every step. Long: every 4 hours (batch constant actions)
    llm_interval = 1 if max_hours <= 168 else 4

    current_action = {
        "feeding_rate": 0.3, "aeration_rate": 0.5,
        "heater_setting": 0.0, "water_exchange_rate": 0.02,
        "harvest_decision": False, "treatment": "none",
    }

    while not obs.get("done", False) and steps < max_hours:
        # Call LLM at interval
        if steps % llm_interval == 0:
            prompt = build_observation_prompt(task_description, obs, history)
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                response_text = completion.choices[0].message.content or "{}"
                current_action = parse_action(response_text)
            except Exception as e:
                print(f"  [Step {steps}] LLM error: {e}, using previous action")

        # Step environment
        step_resp = env_client.post(
            f"{ENV_URL}/step",
            json=current_action,
        )
        step_resp.raise_for_status()
        obs = step_resp.json()

        reward = obs.get("reward", 0) or 0
        total_reward += reward
        history.append(obs)
        steps += 1

        # Progress logging every 24 hours
        if steps % 24 == 0:
            print(f"  Day {steps//24}: Weight={obs.get('avg_fish_weight', 0):.1f}g, "
                  f"Pop={obs.get('population', 0)}, DO={obs.get('dissolved_oxygen', 0):.1f}, "
                  f"Profit=${obs.get('current_profit', 0):.0f}")

    elapsed = time.time() - start_time
    final_reward = obs.get("reward", 0) or 0

    print(f"  Result: score={final_reward:.3f}, steps={steps}, time={elapsed:.1f}s")

    return {
        "task_id": task_id,
        "final_reward": final_reward,
        "total_reward": total_reward,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "final_weight": obs.get("avg_fish_weight", 0),
        "final_population": obs.get("population", 0),
        "final_profit": obs.get("current_profit", 0),
    }


def main():
    """Run inference on all tasks."""
    print("Fish Farm LLM Agent — Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API: {API_BASE_URL}")
    print(f"  Env: {ENV_URL}")

    client = get_llm_client()
    env_client = httpx.Client(timeout=30.0)

    # Get task list
    tasks_resp = env_client.get(f"{ENV_URL}/tasks")
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json()["tasks"]

    # Sort by episode length (shortest first for time budget)
    tasks.sort(key=lambda t: t["episode_hours"])

    results = []
    total_start = time.time()

    for task in tasks:
        # Time budget check: stop if > 18 minutes elapsed (2 min buffer)
        elapsed_total = time.time() - total_start
        if elapsed_total > 18 * 60:
            print(f"\n  TIME BUDGET: Skipping remaining tasks ({elapsed_total:.0f}s elapsed)")
            break

        result = run_task(
            client, env_client,
            task["task_id"],
            task["description"],
            task["episode_hours"],
        )
        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    avg_score = sum(r["final_reward"] for r in results) / len(results) if results else 0
    print(f"  Tasks completed: {len(results)}/{len(tasks)}")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Total time: {total_elapsed:.0f}s")
    for r in results:
        print(f"    {r['task_id']:25} score={r['final_reward']:.3f} ({r['elapsed_s']}s)")

    env_client.close()


if __name__ == "__main__":
    main()
