"""Fish Farm LLM Agent — inference script for OpenEnv evaluation.

Connects to the Fish Farm environment via HTTP, runs all tasks using an LLM
to make management decisions, and reports scores.

Architecture:
1. LLM-based agent with domain-expert system prompt
2. Rule-based heuristic fallback when LLM is unavailable or too slow
3. Adaptive call frequency: more LLM calls during crises
4. Smart harvest timing based on weight, market price, and days remaining

Required environment variables:
    API_BASE_URL: LLM API endpoint (e.g., https://api.openai.com/v1)
    MODEL_NAME: Model to use (e.g., gpt-4o, claude-3-sonnet)
    HF_TOKEN: HuggingFace token for authentication

Runtime constraint: < 20 minutes on 2 vCPU / 8GB RAM
Strategy: Run easy tasks first (shortest episodes), limit history window,
batch requests where possible, heuristic fallback for speed.
"""

import json
import os
import time
import re
import math
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


# ---- Configuration ----
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_HISTORY = 8  # recent observations to include in prompt


# ---- LLM Client ----

def get_llm_client() -> OpenAI:
    """Create OpenAI-compatible client."""
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy"),
    )


# ---- Expert System Prompt ----

SYSTEM_PROMPT = """You are an expert Nile Tilapia aquaculture manager running a 100m³ Recirculating Aquaculture System (RAS). You have deep knowledge of fish biology, water chemistry, disease management, and farm economics.

## Species: Nile Tilapia (Oreochromis niloticus)
- Optimal temperature: 27-32°C (growth stops <18.7°C or >39.7°C, lethal <11°C or >42°C)
- Dissolved oxygen: >5 mg/L optimal (below 3 = stress, below 1 = lethal)
- pH: 6.5-8.5 optimal
- Unionized ammonia (UIA): <0.02 safe, >0.05 chronic stress, >0.6 lethal
- Nitrite (NO2): <0.1 safe, >0.5 stress, >5.0 lethal
- Target market weight: 400-700g (peak price at ~700g)
- Growth: ~2.93%/day SGR under optimal conditions
- Feed conversion ratio (FCR) target: <2.0 (lower = more efficient)

## Critical Cascade to Prevent
Overfeeding → ammonia rises → nitrification consumes O2 → DO drops → stress increases → disease risk → mass mortality. This is the #1 failure mode.

## Decision Framework by Priority
1. SURVIVAL: Keep DO >5, UIA <0.05, temp 27-32°C. If any parameter is critical, address it FIRST.
2. DISEASE: If mortality spikes or feeding becomes sluggish/refusing, apply 'antibiotics' immediately. If no disease yet and >30 days remain, consider 'vaccination' for prevention.
3. WATER QUALITY: Balance feeding against ammonia. Use water exchange (0.02-0.05) for ammonia control. Use aeration for DO.
4. GROWTH: Feed aggressively (0.5-0.7) only when water quality is good. Reduce feed (0.2-0.3) if UIA >0.03 or DO <5.
5. ECONOMICS: Monitor feed price (it fluctuates). Harvest when fish reach 400g+ AND market_price_multiplier >1.0 if possible. Don't harvest if price is crashed (<0.8x) unless episode is ending.

## Treatment Guide
- 'antibiotics': Use during active disease. Doubles recovery rate but harms biofilter (reduces TAN removal).
- 'salt': Use for nitrite stress. Mild recovery boost (1.3x), cheap ($10/day).
- 'probiotics': Boosts biofilter AND recovery (1.5x). Good for prevention/mild issues ($30/day).
- 'vaccination': One-time $100. Prevents 80% of future infections. Best used early when >30 days remain.
- 'none': Default. Don't over-treat — antibiotics harm the biofilter.

## Heater Strategy
- If temp <27°C: heat (positive setting 0.3-1.0 proportional to deficit)
- If temp >33°C: cool (negative setting -0.3 to -1.0 proportional to excess)
- If temp 27-33°C: off (0.0) — save electricity

## Night vs Day
- Daytime: photosynthesis adds DO, can feed normally
- Nighttime: no photosynthesis, DO drops faster — increase aeration, reduce feeding

## Feeding Response Signals
- 'eager': Fish are healthy, can increase feed
- 'normal': Maintain current feeding
- 'sluggish': Reduce feed to 0.2-0.3, something is wrong (check water quality)
- 'refusing': Stop feeding (0.0), fish are stressed — fix water quality and check for disease

Respond with ONLY a valid JSON object. No explanation, no markdown, just JSON."""


# ---- Heuristic Fallback Agent ----

def heuristic_action(obs: Dict[str, Any], task_id: str, step: int, max_hours: int) -> Dict[str, Any]:
    """Rule-based fallback when LLM is unavailable or too slow.

    Implements the decision framework from the system prompt using
    simple threshold logic. Not as nuanced as an LLM but handles
    the critical cascades correctly.
    """
    DO = obs.get("dissolved_oxygen", 7.0)
    UIA = obs.get("ammonia_toxic", 0.005)
    TAN = obs.get("ammonia", 0.1)
    temp = obs.get("temperature", 28.0)
    stress = obs.get("stress_level", 0.0)
    mortality = obs.get("mortality_today", 0)
    feeding_resp = obs.get("feeding_response", "normal")
    weight = obs.get("avg_fish_weight", 50.0)
    population = obs.get("population", 10000)
    feed_remaining = obs.get("feed_remaining_kg", 500.0)
    biofilter_ok = obs.get("biofilter_working", True)
    aerator_ok = obs.get("aerator_working", True)
    disease_suspected = obs.get("disease_suspected", False)
    is_daytime = obs.get("is_daytime", True)
    market_mult = obs.get("market_price_multiplier", 1.0)
    nighttime_do_risk = obs.get("nighttime_do_risk", 0.0)
    hours_left = max_hours - step

    # ---- Aeration ----
    if DO < 3.0:
        aeration = 1.0  # emergency
    elif DO < 5.0:
        aeration = 0.8
    elif nighttime_do_risk > 0.6:
        aeration = 0.9  # preemptive: high nighttime crash risk
    elif not is_daytime:
        aeration = 0.6  # nighttime needs more
    else:
        aeration = 0.4  # daytime with photosynthesis

    if not aerator_ok:
        aeration = 0.0  # broken

    # ---- Feeding ----
    if feeding_resp == "refusing" or DO < 2.0 or UIA > 0.3:
        feeding = 0.0  # don't waste feed on stressed fish
    elif feeding_resp == "sluggish" or DO < 4.0 or UIA > 0.05:
        feeding = 0.2  # minimal
    elif stress > 0.5:
        feeding = 0.2
    elif feed_remaining < 20.0:
        feeding = 0.15  # conserve inventory
    elif is_daytime:
        feeding = 0.5  # normal daytime
    else:
        feeding = 0.3  # reduced at night

    # ---- Water exchange ----
    if UIA > 0.1 or TAN > 2.0:
        exchange = 0.08  # emergency dilution
    elif UIA > 0.05 or TAN > 1.0:
        exchange = 0.05
    elif not biofilter_ok:
        exchange = 0.04  # compensate for broken biofilter
    else:
        exchange = 0.02  # maintenance

    # ---- Heater ----
    if temp < 25.0:
        heater = min(1.0, (27.0 - temp) / 5.0)
    elif temp < 27.0:
        heater = 0.3
    elif temp > 35.0:
        heater = max(-1.0, (32.0 - temp) / 5.0)
    elif temp > 33.0:
        heater = -0.3
    else:
        heater = 0.0

    # ---- Treatment ----
    treatment = "none"
    if disease_suspected and mortality > 10:
        treatment = "antibiotics"
    elif disease_suspected:
        treatment = "probiotics"  # milder, doesn't harm biofilter
    elif not biofilter_ok and stress < 0.3:
        treatment = "probiotics"  # helps biofilter recover

    # Vaccination: early prevention for long episodes
    if hours_left > 30 * 24 and step < 48 and not disease_suspected and weight < 100:
        treatment = "vaccination"

    # ---- Harvest ----
    harvest = False

    # Harvest if fish at market weight near episode end
    if weight >= 400 and hours_left <= 24:
        harvest = True
    # Harvest if market premium and fish are ready
    elif weight >= 500 and market_mult >= 1.1:
        harvest = True
    # Emergency harvest if population crashing
    elif population < 1000 and weight > 200:
        harvest = True
    # Harvest on last step
    elif hours_left <= 1 and weight > 100:
        harvest = True

    return {
        "feeding_rate": round(feeding, 2),
        "aeration_rate": round(aeration, 2),
        "heater_setting": round(heater, 2),
        "water_exchange_rate": round(exchange, 3),
        "harvest_decision": harvest,
        "treatment": treatment,
    }


# ---- Observation Prompt Builder ----

def build_observation_prompt(
    task_description: str,
    obs: Dict[str, Any],
    recent_history: List[Dict[str, Any]],
    step: int,
    max_hours: int,
) -> str:
    """Build the observation prompt for the LLM with full situational awareness."""
    hours_left = max_hours - step
    days_left = hours_left / 24.0

    prompt = f"""TASK: {task_description}

TIME: Day {obs.get('day_in_cycle', 0)}, {obs.get('time_of_day', 0):02d}:00 ({'DAY' if obs.get('is_daytime', True) else 'NIGHT'}) | {hours_left}h remaining ({days_left:.1f} days)
Day of Year: {obs.get('day_of_year', 1)} | Storm: {'ACTIVE' if obs.get('storm_active', False) else 'No'} | Humidity: {obs.get('humidity', 75):.0f}%

WATER QUALITY:
  Temperature:     {obs.get('temperature', 0):.1f}°C  (optimal 27-32°C)
  Dissolved Oxygen: {obs.get('dissolved_oxygen', 0):.1f} mg/L  (danger <3.0, lethal <1.0)
  pH:              {obs.get('ph', 0):.2f}  (optimal 6.5-8.5)
  TAN:             {obs.get('ammonia', 0):.3f} mg/L
  UIA (toxic):     {obs.get('ammonia_toxic', 0):.5f} mg/L  (toxic >0.05, lethal >0.6)
  Nitrite (NO2):   {obs.get('nitrite', 0):.3f} mg/L  (stress >0.5)
  Nitrate (NO3):   {obs.get('nitrate', 0):.1f} mg/L
  WQ Score:        {obs.get('water_quality_score', 0):.3f}/1.000
  Algae Bloom:     {'YES — DO will swing!' if obs.get('algae_bloom', False) else 'No'}
  Night DO Risk:   {obs.get('nighttime_do_risk', 0):.2f}  {'⚠ HIGH — boost aeration!' if obs.get('nighttime_do_risk', 0) > 0.5 else '(0=safe, 1=crash imminent)'}

FISH:
  Weight: {obs.get('avg_fish_weight', 0):.1f}g | Population: {obs.get('population', 0):,} | Biomass: {obs.get('biomass_kg', 0):.1f}kg
  Growth: {obs.get('growth_rate_g_day', 0):.2f} g/day | SGR: {obs.get('sgr', 0):.2f}%/day | FCR: {obs.get('fcr', 0):.2f}
  Deaths Today: {obs.get('mortality_today', 0)} | Total Deaths: {obs.get('cumulative_mortality', 0)} | Survival: {obs.get('survival_rate', 1):.1%}
  Stress: {obs.get('stress_level', 0):.2f} | Appetite: {obs.get('feeding_response', 'unknown')}
  Density: {obs.get('stocking_density', 0):.1f} fish/m³
  Disease Signs: {'YES — consider treatment!' if obs.get('disease_suspected', False) else 'None'}

ECONOMICS:
  Fish Value: ${obs.get('current_fish_value', 0):.0f} | Costs: ${obs.get('total_cost_so_far', 0):.0f} | Profit: ${obs.get('current_profit', 0):.0f}
  Feed Price: ${obs.get('feed_price_per_kg', 0.5):.3f}/kg | Market Multiplier: {obs.get('market_price_multiplier', 1.0):.2f}x
  Marginal Cost: ${obs.get('marginal_cost_per_hour', 0):.2f}/hr | ROI: {obs.get('roi_pct', 0):.1f}%
  Feed Stock: {obs.get('feed_remaining_kg', 0):.0f}kg

EQUIPMENT: Aerator={'OK' if obs.get('aerator_working', True) else 'FAILED'} | Biofilter={'OK' if obs.get('biofilter_working', True) else 'FAILED'} | Heater={'OK' if obs.get('heater_working', True) else 'FAILED'}"""

    # Alerts
    alerts = obs.get('alerts', [])
    if alerts:
        prompt += f"\nACTIVE EVENTS: {'; '.join(alerts)}"

    # Feedback
    feedback = obs.get('feedback', '')
    if feedback:
        prompt += f"\nSITUATION: {feedback}"

    # Recent trend (shows trajectory for key parameters)
    if recent_history:
        prompt += "\n\nTREND (recent hours):"
        for i, h in enumerate(recent_history[-5:]):
            idx = step - len(recent_history[-5:]) + i
            prompt += (
                f"\n  [h{idx}] DO={h.get('dissolved_oxygen', 0):.1f}, "
                f"UIA={h.get('ammonia_toxic', 0):.4f}, "
                f"Temp={h.get('temperature', 0):.1f}, "
                f"Wt={h.get('avg_fish_weight', 0):.1f}g, "
                f"Mort={h.get('mortality_today', 0)}, "
                f"Feed={h.get('feeding_response', '?')}"
            )

    # Harvest advisory
    weight = obs.get('avg_fish_weight', 0)
    if weight >= 400:
        prompt += f"\n\nHARVEST ADVISORY: Fish at {weight:.0f}g (market weight). "
        if obs.get('market_price_multiplier', 1.0) >= 1.1:
            prompt += "Market price is PREMIUM — good time to harvest!"
        elif hours_left <= 48:
            prompt += "Episode ending soon — consider harvesting."
        elif obs.get('market_price_multiplier', 1.0) < 0.8:
            prompt += "Market price is CRASHED — wait if possible."

    prompt += """

Decide actions for the next hour. Respond with ONLY a JSON object:
{
  "feeding_rate": 0.0-1.0,
  "aeration_rate": 0.0-1.0,
  "heater_setting": -1.0 to 1.0,
  "water_exchange_rate": 0.0-0.10,
  "harvest_decision": false,
  "treatment": "none"
}
Treatment options: "none", "antibiotics", "salt", "probiotics", "vaccination"."""

    return prompt


# ---- Response Parser ----

def parse_action(response_text: str) -> Dict[str, Any]:
    """Parse LLM response into action dict, with fallback defaults."""
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


# ---- Adaptive LLM Call Frequency ----

def should_call_llm(obs: Dict[str, Any], step: int, last_llm_step: int,
                    base_interval: int) -> bool:
    """Decide whether to call the LLM this step.

    Calls more frequently during crises, less during stable periods.
    """
    steps_since_llm = step - last_llm_step

    # Always call on first step
    if step == 0:
        return True

    # Crisis detection: call every step during emergencies
    DO = obs.get("dissolved_oxygen", 7.0)
    UIA = obs.get("ammonia_toxic", 0.005)
    mortality = obs.get("mortality_today", 0)
    disease = obs.get("disease_suspected", False)
    storm = obs.get("storm_active", False)

    nighttime_do_risk = obs.get("nighttime_do_risk", 0.0)

    is_crisis = (
        DO < 3.0
        or UIA > 0.1
        or mortality > 20
        or disease
        or storm
        or nighttime_do_risk > 0.7
    )

    if is_crisis:
        return steps_since_llm >= 1  # every step during crisis

    # Elevated concern: call every 2 steps
    is_concern = (
        DO < 5.0
        or UIA > 0.04
        or mortality > 5
        or obs.get("feeding_response", "normal") in ("sluggish", "refusing")
    )

    if is_concern:
        return steps_since_llm >= 2

    # Normal: use base interval
    return steps_since_llm >= base_interval


# ---- Task Runner ----

def run_task(
    client: OpenAI,
    env_client: httpx.Client,
    task_id: str,
    task_description: str,
    max_hours: int,
    time_budget_s: float,
) -> Dict[str, Any]:
    """Run a single task and return results."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id} ({max_hours}h = {max_hours/24:.0f} days)")
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
    llm_calls = 0
    heuristic_calls = 0
    last_llm_step = -10  # force first LLM call

    # Decide base LLM call interval based on episode length and time budget
    # Shorter episodes get more frequent calls (every step)
    # Longer episodes get less frequent (every 4-6 hours)
    if max_hours <= 72:
        base_interval = 1
    elif max_hours <= 168:
        base_interval = 2
    elif max_hours <= 720:
        base_interval = 4
    else:
        base_interval = 6

    # Per-step time budget: stop using LLM if running out of time
    per_task_budget = time_budget_s * 0.85  # 85% of allocated time

    current_action = heuristic_action(obs, task_id, 0, max_hours)

    while not obs.get("done", False) and steps < max_hours:
        elapsed = time.time() - start_time
        use_llm = (
            elapsed < per_task_budget
            and should_call_llm(obs, steps, last_llm_step, base_interval)
        )

        if use_llm:
            prompt = build_observation_prompt(
                task_description, obs, history, steps, max_hours,
            )
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                )
                response_text = completion.choices[0].message.content or "{}"
                current_action = parse_action(response_text)
                llm_calls += 1
                last_llm_step = steps
            except Exception as e:
                print(f"  [Step {steps}] LLM error: {e}")
                current_action = heuristic_action(obs, task_id, steps, max_hours)
                heuristic_calls += 1
        else:
            # Use heuristic when LLM isn't called
            if steps - last_llm_step > base_interval:
                current_action = heuristic_action(obs, task_id, steps, max_hours)
                heuristic_calls += 1

        # Force harvest check on last steps regardless of LLM
        hours_left = max_hours - steps
        weight = obs.get("avg_fish_weight", 0)
        if hours_left <= 1 and weight > 100:
            current_action["harvest_decision"] = True
        elif hours_left <= 24 and weight >= 400:
            current_action["harvest_decision"] = True

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

        # Keep history bounded to save memory
        if len(history) > MAX_HISTORY * 2:
            history = history[-MAX_HISTORY:]

        # Progress logging every 24 hours
        if steps % 24 == 0:
            print(
                f"  Day {steps//24}: Wt={obs.get('avg_fish_weight', 0):.1f}g, "
                f"Pop={obs.get('population', 0)}, DO={obs.get('dissolved_oxygen', 0):.1f}, "
                f"UIA={obs.get('ammonia_toxic', 0):.4f}, "
                f"Profit=${obs.get('current_profit', 0):.0f}, "
                f"FCR={obs.get('fcr', 0):.2f}"
            )

    elapsed = time.time() - start_time
    final_reward = obs.get("reward", 0) or 0

    print(f"  Result: score={final_reward:.3f}, steps={steps}, "
          f"time={elapsed:.1f}s, LLM={llm_calls}, heuristic={heuristic_calls}")

    return {
        "task_id": task_id,
        "final_reward": final_reward,
        "total_reward": total_reward,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "final_weight": obs.get("avg_fish_weight", 0),
        "final_population": obs.get("population", 0),
        "final_profit": obs.get("current_profit", 0),
        "llm_calls": llm_calls,
        "heuristic_calls": heuristic_calls,
    }


# ---- Main Entry Point ----

def main():
    """Run inference on all tasks within the 20-minute budget."""
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

    # Sort by episode length (shortest first for time budget efficiency)
    tasks.sort(key=lambda t: t["episode_hours"])

    # Time budget allocation: 18 minutes total (2 min buffer)
    total_budget_s = 18 * 60
    total_episode_hours = sum(t["episode_hours"] for t in tasks)

    results = []
    total_start = time.time()

    for task in tasks:
        elapsed_total = time.time() - total_start

        # Stop if past budget
        if elapsed_total > total_budget_s:
            print(f"\n  TIME BUDGET EXCEEDED: Skipping remaining tasks ({elapsed_total:.0f}s)")
            break

        # Allocate time proportional to episode length
        remaining_budget = total_budget_s - elapsed_total
        remaining_hours = sum(
            t["episode_hours"] for t in tasks
            if t["task_id"] not in {r["task_id"] for r in results}
        )
        if remaining_hours > 0:
            task_budget = remaining_budget * (task["episode_hours"] / remaining_hours)
        else:
            task_budget = remaining_budget

        result = run_task(
            client, env_client,
            task["task_id"],
            task["description"],
            task["episode_hours"],
            time_budget_s=task_budget,
        )
        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    avg_score = sum(r["final_reward"] for r in results) / len(results) if results else 0
    total_llm = sum(r["llm_calls"] for r in results)
    total_heur = sum(r["heuristic_calls"] for r in results)
    print(f"  Tasks completed: {len(results)}/{len(tasks)}")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Total time: {total_elapsed:.0f}s")
    print(f"  LLM calls: {total_llm} | Heuristic: {total_heur}")
    for r in results:
        print(f"    {r['task_id']:25} score={r['final_reward']:.3f} "
              f"({r['elapsed_s']}s, LLM={r['llm_calls']})")

    env_client.close()


if __name__ == "__main__":
    main()
