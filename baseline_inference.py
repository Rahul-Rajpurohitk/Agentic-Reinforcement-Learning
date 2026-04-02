"""Baseline inference script for the Fish Farm environment.

Runs the heuristic agent against all 12 tasks and reports reproducible
baseline scores. This is a required submission artifact that demonstrates
the graders produce meaningful, varying signal across difficulty levels.

Usage:
    # Heuristic baseline (no API key needed, no server needed)
    python baseline_inference.py

    # Save results to file
    python baseline_inference.py --output baseline_results.json

    # Single task
    python baseline_inference.py --task feeding_basics
"""

import argparse
import json
import time

from src.agentic_rl.server.environment import FishFarmEnvironment
from src.agentic_rl.models import FarmAction
from src.agentic_rl.tasks import list_all_tasks
from inference import heuristic_action


def run_baseline(task_ids=None, output_file=None):
    """Run heuristic baseline on all tasks and report scores."""
    all_tasks = list_all_tasks()
    all_tasks.sort(key=lambda t: t["episode_hours"])

    if task_ids:
        all_tasks = [t for t in all_tasks if t["task_id"] in task_ids]

    print(f"{'='*60}")
    print("Baseline Inference — Fish Farm Environment")
    print("Agent: Heuristic (rule-based, deterministic)")
    print(f"Tasks: {len(all_tasks)}")
    print(f"{'='*60}\n")

    results = []
    total_start = time.time()

    for task_info in all_tasks:
        task_id = task_info["task_id"]
        max_hours = task_info["episode_hours"]

        print(f"--- {task_id} ({task_info['difficulty']}, {max_hours}h) ---")

        env = FishFarmEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        obs_dict = obs.model_dump()

        steps = 0
        while not obs_dict.get("done", False) and steps < max_hours:
            action_dict = heuristic_action(obs_dict, task_id, steps, max_hours)
            action = FarmAction(**action_dict)
            obs = env.step(action)
            obs_dict = obs.model_dump()
            steps += 1

        score = obs_dict.get("reward", 0) or 0

        print(f"  Score: {score:.3f}  |  Weight: {obs_dict.get('avg_fish_weight', 0):.0f}g  "
              f"|  Pop: {obs_dict.get('population', 0)}  "
              f"|  Profit: ${obs_dict.get('current_profit', 0):.0f}")

        results.append({
            "task_id": task_id,
            "difficulty": task_info["difficulty"],
            "score": score,
            "steps": steps,
            "final_weight": obs_dict.get("avg_fish_weight", 0),
            "final_population": obs_dict.get("population", 0),
            "final_profit": obs_dict.get("current_profit", 0),
        })

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")

    by_difficulty = {"easy": [], "medium": [], "hard": [], "extreme": []}
    for r in results:
        by_difficulty[r["difficulty"]].append(r["score"])

    total_scores = []
    for difficulty in ["easy", "medium", "hard", "extreme"]:
        scores = by_difficulty[difficulty]
        if scores:
            avg = sum(scores) / len(scores)
            total_scores.extend(scores)
            print(f"  {difficulty.upper():8s}: {avg:.3f} avg ({len(scores)} tasks)")

    overall_avg = sum(total_scores) / len(total_scores) if total_scores else 0.0
    print(f"  {'OVERALL':8s}: {overall_avg:.3f} avg ({len(total_scores)} tasks)")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"{'='*60}")

    # Save results
    output = {
        "agent": "heuristic",
        "seed": 42,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "summary": {
            "overall_avg": round(overall_avg, 3),
            "total_tasks": len(results),
            "elapsed_s": round(total_elapsed, 1),
            "by_difficulty": {
                k: round(sum(v) / len(v), 3) if v else 0.0
                for k, v in by_difficulty.items()
            },
        },
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Farm Baseline Inference")
    parser.add_argument("--task", type=str, nargs="+", default=None,
                        help="Specific task(s) to run")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                        help="Output file (default: baseline_results.json)")
    args = parser.parse_args()

    run_baseline(task_ids=args.task, output_file=args.output)
