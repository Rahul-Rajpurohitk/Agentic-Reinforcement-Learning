"""Fish Farm LLM Agent — local/direct inference (no HTTP, no WebSocket).

Runs the agent directly against FishFarmEnvironment in-process.
This is the fastest way to test and works without a running server.

Usage:
    # Heuristic only (no API key needed)
    python inference_local.py

    # With LLM
    API_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-xxx python inference_local.py

    # Single task
    python inference_local.py --task feeding_basics

    # Heuristic only (skip LLM even if key is set)
    python inference_local.py --heuristic-only
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from src.agentic_rl.server.environment import FishFarmEnvironment
from src.agentic_rl.models import FarmAction
from src.agentic_rl.tasks import TASKS, list_all_tasks
from inference import (
    heuristic_action,
    build_observation_prompt,
    parse_action,
    should_call_llm,
    SYSTEM_PROMPT,
)

# ---- Configuration ----
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
MAX_HISTORY = 8


def get_llm_client():
    """Create LLM client. Tries OpenAI API, falls back to HF InferenceClient."""
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # Try OpenAI-compatible endpoint first
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        print(f"  LLM: OpenAI-compatible ({API_BASE_URL})")
        return client
    except Exception as e:
        print(f"  OpenAI API failed ({e}), trying HuggingFace InferenceClient...")

    # Fallback: HuggingFace InferenceClient
    try:
        from inference import HFInferenceWrapper
        wrapper = HFInferenceWrapper(model=MODEL_NAME, token=api_key)
        wrapper.create(messages=[{"role": "user", "content": "test"}], max_tokens=5)
        print(f"  LLM: HuggingFace InferenceClient ({MODEL_NAME})")
        return wrapper
    except Exception as e2:
        print(f"  HF InferenceClient also failed ({e2}). Using heuristic only.")
        return None


def run_task_local(
    task_id: str,
    llm_client=None,
    time_budget_s: float = 120.0,
) -> Dict[str, Any]:
    """Run a single task using direct FishFarmEnvironment (no HTTP)."""
    task_info = None
    for t in list_all_tasks():
        if t["task_id"] == task_id:
            task_info = t
            break

    if not task_info:
        raise ValueError(f"Unknown task: {task_id}")

    max_hours = task_info["episode_hours"]
    description = task_info["description"]

    print(f"\n{'='*60}")
    print(f"  Task: {task_id} ({max_hours}h = {max_hours/24:.0f} days)")
    print(f"  Mode: {'LLM + Heuristic' if llm_client else 'Heuristic only'}")
    print(f"{'='*60}")

    # Create environment and reset
    env = FishFarmEnvironment()
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()

    history: List[Dict[str, Any]] = []
    steps = 0
    start_time = time.time()
    llm_calls = 0
    heuristic_calls = 0
    last_llm_step = -10

    # LLM call interval based on episode length
    if max_hours <= 72:
        base_interval = 1
    elif max_hours <= 168:
        base_interval = 2
    elif max_hours <= 720:
        base_interval = 4
    else:
        base_interval = 6

    per_task_budget = time_budget_s * 0.85
    current_action_dict = heuristic_action(obs_dict, task_id, 0, max_hours)

    while not obs_dict.get("done", False) and steps < max_hours:
        elapsed = time.time() - start_time

        use_llm = (
            llm_client is not None
            and elapsed < per_task_budget
            and should_call_llm(obs_dict, steps, last_llm_step, base_interval)
        )

        if use_llm:
            prompt = build_observation_prompt(
                description, obs_dict, history, steps, max_hours,
            )
            try:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                )
                response_text = completion.choices[0].message.content or "{}"
                current_action_dict = parse_action(response_text)
                llm_calls += 1
                last_llm_step = steps
            except Exception as e:
                print(f"  [Step {steps}] LLM error: {e}")
                current_action_dict = heuristic_action(obs_dict, task_id, steps, max_hours)
                heuristic_calls += 1
        else:
            if steps - last_llm_step > base_interval or llm_client is None:
                current_action_dict = heuristic_action(obs_dict, task_id, steps, max_hours)
                heuristic_calls += 1

        # Force harvest on last steps
        hours_left = max_hours - steps
        weight = obs_dict.get("avg_fish_weight", 0)
        if hours_left <= 1 and weight > 100:
            current_action_dict["harvest_decision"] = True
        elif hours_left <= 24 and weight >= 400:
            current_action_dict["harvest_decision"] = True

        # Step
        action = FarmAction(**current_action_dict)
        obs = env.step(action)
        obs_dict = obs.model_dump()

        history.append(obs_dict)
        steps += 1

        if len(history) > MAX_HISTORY * 2:
            history = history[-MAX_HISTORY:]

        # Progress logging every 24 hours
        if steps % 24 == 0:
            print(
                f"  Day {steps//24}: Wt={obs_dict.get('avg_fish_weight', 0):.1f}g, "
                f"Pop={obs_dict.get('population', 0)}, DO={obs_dict.get('dissolved_oxygen', 0):.1f}, "
                f"UIA={obs_dict.get('ammonia_toxic', 0):.4f}, "
                f"Profit=${obs_dict.get('current_profit', 0):.0f}, "
                f"FCR={obs_dict.get('fcr', 0):.2f}"
            )

    elapsed = time.time() - start_time
    final_score = obs_dict.get("reward", 0) or 0

    print(f"  Result: score={final_score:.3f}, steps={steps}, "
          f"time={elapsed:.1f}s, LLM={llm_calls}, heuristic={heuristic_calls}")
    print(f"  Final: weight={obs_dict.get('avg_fish_weight', 0):.1f}g, "
          f"pop={obs_dict.get('population', 0)}, "
          f"profit=${obs_dict.get('current_profit', 0):.0f}")

    return {
        "task_id": task_id,
        "score": final_score,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "final_weight": obs_dict.get("avg_fish_weight", 0),
        "final_population": obs_dict.get("population", 0),
        "final_profit": obs_dict.get("current_profit", 0),
        "llm_calls": llm_calls,
        "heuristic_calls": heuristic_calls,
    }


def main():
    parser = argparse.ArgumentParser(description="Fish Farm Agent — Local Inference")
    parser.add_argument("--task", type=str, default=None, help="Single task to run (default: all)")
    parser.add_argument("--heuristic-only", action="store_true", help="Skip LLM, heuristic only")
    args = parser.parse_args()

    print("Fish Farm Agent — Local/Direct Inference")
    print(f"  Model: {MODEL_NAME}")

    llm_client = None if args.heuristic_only else get_llm_client()
    if llm_client:
        print(f"  Mode: LLM ({API_BASE_URL}) + Heuristic fallback")
    else:
        print("  Mode: Heuristic only (no API key or --heuristic-only)")

    # Determine tasks to run
    if args.task:
        task_ids = [args.task]
    else:
        all_tasks = list_all_tasks()
        all_tasks.sort(key=lambda t: t["episode_hours"])
        task_ids = [t["task_id"] for t in all_tasks]

    # Time budget: 18 min total
    total_budget_s = 18 * 60
    total_hours = sum(
        next(t["episode_hours"] for t in list_all_tasks() if t["task_id"] == tid)
        for tid in task_ids
    )

    results = []
    total_start = time.time()

    for tid in task_ids:
        elapsed_total = time.time() - total_start
        if elapsed_total > total_budget_s:
            print(f"\n  TIME BUDGET EXCEEDED ({elapsed_total:.0f}s), skipping remaining tasks")
            break

        remaining = total_budget_s - elapsed_total
        task_hours = next(t["episode_hours"] for t in list_all_tasks() if t["task_id"] == tid)
        remaining_hours = sum(
            next(t["episode_hours"] for t in list_all_tasks() if t["task_id"] == t_id)
            for t_id in task_ids if t_id not in {r["task_id"] for r in results}
        )
        budget = remaining * (task_hours / remaining_hours) if remaining_hours > 0 else remaining

        result = run_task_local(tid, llm_client, time_budget_s=budget)
        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    total_llm = sum(r["llm_calls"] for r in results)
    total_heur = sum(r["heuristic_calls"] for r in results)
    print(f"  Tasks completed: {len(results)}/{len(task_ids)}")
    print(f"  Average score:   {avg_score:.3f}")
    print(f"  Total time:      {total_elapsed:.0f}s")
    print(f"  LLM calls:       {total_llm}")
    print(f"  Heuristic calls: {total_heur}")
    print()
    for r in results:
        status = "PASS" if r["score"] >= 0.7 else "WEAK" if r["score"] >= 0.4 else "FAIL"
        print(f"  [{status}] {r['task_id']:25} score={r['score']:.3f}  "
              f"wt={r['final_weight']:.0f}g  pop={r['final_population']}  "
              f"({r['elapsed_s']}s)")

    return results


if __name__ == "__main__":
    main()
