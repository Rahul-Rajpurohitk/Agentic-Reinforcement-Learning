"""GRPO Training for the Fish Farm environment.

Fine-tune an LLM to manage a Nile Tilapia RAS farm using Group Relative
Policy Optimization, with reward signals from the fish farm simulator.

Usage:
    # Start environment server first
    uvicorn src.agentic_rl.server.app:app --port 8000

    # Run training (requires GPU + training extras)
    pip install -e ".[training]"
    python -m training.train_grpo --model_name "Qwen/Qwen2.5-0.5B"
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rewards import CompositeReward, SurvivalReward, WaterQualityReward


SYSTEM_PROMPT = """You are an expert Nile Tilapia aquaculture manager. Given the current state of a 100m³ RAS fish farm, decide the next hour's actions.

Respond with ONLY a JSON object:
{
  "feeding_rate": 0.0-1.0,
  "aeration_rate": 0.0-1.0,
  "heater_setting": -1.0 to 1.0,
  "water_exchange_rate": 0.0-0.10,
  "harvest_decision": false,
  "treatment": "none"
}

Key priorities: keep DO > 5 mg/L, UIA < 0.05 mg/L, temp 27-32°C. Feed based on growth stage.
Treatment options: "none", "antibiotics", "salt", "probiotics", "vaccination"."""


def build_state_prompt(obs: dict) -> str:
    """Convert observation dict to a concise text prompt for the LLM."""
    return (
        f"WATER: DO={obs.get('dissolved_oxygen', 0):.1f} mg/L, "
        f"UIA={obs.get('ammonia_toxic', 0):.4f}, "
        f"Temp={obs.get('temperature', 0):.1f}°C, "
        f"pH={obs.get('ph', 0):.2f}, "
        f"WQ={obs.get('water_quality_score', 0):.3f}\n"
        f"FISH: {obs.get('avg_fish_weight', 0):.1f}g, "
        f"Pop={obs.get('population', 0)}, "
        f"Stress={obs.get('stress_level', 0):.2f}, "
        f"FCR={obs.get('fcr', 0):.2f}, "
        f"Appetite={obs.get('feeding_response', 'normal')}\n"
        f"ECON: Profit=${obs.get('current_profit', 0):.0f}, "
        f"Feed=${obs.get('feed_price_per_kg', 0.5):.2f}/kg\n"
        f"Decide actions for next hour. JSON only."
    )


def parse_action(completion: str) -> dict:
    """Parse an LLM completion into a FarmAction-compatible dict."""
    defaults = {
        "feeding_rate": 0.3,
        "aeration_rate": 0.5,
        "heater_setting": 0.0,
        "water_exchange_rate": 0.02,
        "harvest_decision": False,
        "treatment": "none",
    }
    try:
        text = completion.strip()
        if "```" in text:
            text = text.split("```")[1].split("```")[0]
            if text.startswith("json"):
                text = text[4:]
        action = json.loads(text)
        for key, default in defaults.items():
            if key not in action:
                action[key] = default
        return action
    except (json.JSONDecodeError, IndexError):
        return defaults


def main():
    parser = argparse.ArgumentParser(description="GRPO Training — Fish Farm Environment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_fish_farm")
    parser.add_argument("--env_url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["feeding_basics", "oxygen_management", "water_quality_balance"],
        help="Task IDs to train on (start with easy tasks)",
    )
    args = parser.parse_args()

    print("=== GRPO Training — Fish Farm Environment ===")
    print(f"Model: {args.model_name}")
    print(f"Environment: {args.env_url}")
    print(f"Tasks: {args.tasks}")
    print()

    # Verify rewards work
    reward_fn = CompositeReward()
    test_state = {
        "fish": {"mortality_today": 0, "weight_g": 50, "fcr": 1.5, "survival_rate": 1.0},
        "water": {"water_quality_score": 0.9, "DO": 8.0, "UIA": 0.01},
        "economics": {"current_profit": 100},
    }
    test_action = {"feeding_rate": 0.4, "aeration_rate": 0.5}
    test_reward = reward_fn.compute(test_state, test_action, test_state)
    print(f"Reward function test: {test_reward:.3f} (expected ~0.1-0.2)")

    # ---------------------------------------------------------------
    # Full GRPO training loop (requires GPU + trl/transformers)
    # ---------------------------------------------------------------
    #
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from trl import GRPOConfig, GRPOTrainer
    # from datasets import Dataset
    # import httpx
    #
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    #
    # # Collect state prompts from environment
    # client = httpx.Client(base_url=args.env_url, timeout=30.0)
    # prompts = []
    #
    # for task_id in args.tasks:
    #     resp = client.post("/reset", json={"task_id": task_id})
    #     obs = resp.json()["observation"]
    #
    #     # Collect observations over 24 hours
    #     for step in range(24):
    #         prompt = build_state_prompt(obs)
    #         prompts.append({"prompt": SYSTEM_PROMPT + "\n\n" + prompt})
    #
    #         # Step with default action
    #         resp = client.post("/step", json={
    #             "feeding_rate": 0.3, "aeration_rate": 0.5,
    #             "heater_setting": 0.0, "water_exchange_rate": 0.02,
    #             "harvest_decision": False, "treatment": "none",
    #         })
    #         data = resp.json()
    #         obs = data["observation"]
    #         if data.get("done"):
    #             break
    #
    # dataset = Dataset.from_list(prompts * (args.num_episodes // len(prompts) + 1))
    # dataset = dataset.select(range(args.num_episodes))
    #
    # # GRPO reward: run action through simulator, score with CompositeReward
    # def reward_fn_batch(prompts, completions, **kwargs):
    #     scores = []
    #     reward = CompositeReward()
    #     for prompt_text, completion in zip(prompts, completions):
    #         action = parse_action(completion)
    #         resp = client.post("/step", json=action)
    #         data = resp.json()
    #         state = data["observation"]
    #         # Approximate prev_state from prompt (simplified)
    #         score = reward.compute(
    #             state={"fish": state, "water": state, "economics": state},
    #             action=action,
    #             prev_state={"fish": state, "water": state, "economics": state},
    #         )
    #         scores.append(score)
    #     return scores
    #
    # config = GRPOConfig(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     learning_rate=args.learning_rate,
    #     num_train_epochs=3,
    #     logging_steps=10,
    # )
    #
    # trainer = GRPOTrainer(
    #     model=model,
    #     config=config,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset,
    #     reward_funcs=[reward_fn_batch],
    # )
    # trainer.train()
    # trainer.save_model(args.output_dir)
    # print(f"Model saved to {args.output_dir}")

    print()
    print("Training template ready.")
    print("To run full training, install GPU dependencies:")
    print("  pip install -e '.[training]'")
    print("Then uncomment the training loop in this file.")


if __name__ == "__main__":
    main()
