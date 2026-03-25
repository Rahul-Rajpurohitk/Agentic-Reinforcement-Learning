"""GRPO Training Template for the Code Review environment.

Fine-tune an LLM to review code using Group Relative Policy Optimization
with reward signals from the Code Review environment.

Usage:
    # Start environment server first
    uvicorn src.agentic_rl.server.app:app --port 8000

    # Run training (requires GPU)
    python -m training.train_grpo --model_name "Qwen/Qwen2.5-0.5B"
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rewards import RecallReward, SeverityWeightedReward


def create_reward_functions():
    """Create reward functions for GRPO training."""
    return {
        "recall": RecallReward(),
        "severity": SeverityWeightedReward(),
    }


def parse_review_from_completion(completion: str) -> dict:
    """Parse an LLM completion into a ReviewAction-compatible dict."""
    try:
        # Try to extract JSON from the completion
        if "```json" in completion:
            completion = completion.split("```json")[1].split("```")[0].strip()
        elif "```" in completion:
            completion = completion.split("```")[1].split("```")[0].strip()
        return json.loads(completion)
    except (json.JSONDecodeError, IndexError):
        return {"issues_found": [], "overall_assessment": "comment", "confidence": 0.0}


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Code Review")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_run")
    parser.add_argument("--env_url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    print(f"=== GRPO Training — Code Review Environment ===")
    print(f"Model: {args.model_name}")
    print(f"Environment: {args.env_url}")
    print()

    # ---------------------------------------------------------------
    # Uncomment below when GPU is available
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
    # # Collect prompts from environment
    # client = httpx.Client(base_url=args.env_url, timeout=30.0)
    # task_ids = ["easy_001", "easy_002", "easy_003",
    #             "medium_001", "medium_002", "medium_003",
    #             "hard_001", "hard_002", "hard_003"]
    #
    # prompts = []
    # for task_id in task_ids:
    #     obs = client.post("/reset", json={"task_id": task_id}).json()
    #     prompt = (
    #         f"Review this {obs['language']} code for bugs and security issues.\n"
    #         f"Context: {obs['context']}\n\n"
    #         f"```{obs['language']}\n{obs['code_snippet']}\n```\n\n"
    #         f"Respond with JSON: {{\"issues_found\": [...], "
    #         f"\"overall_assessment\": \"...\"}}"
    #     )
    #     prompts.append(prompt)
    #
    # dataset = Dataset.from_dict({"prompt": prompts * (args.num_episodes // len(prompts))})
    #
    # # Reward function for GRPO
    # reward_fns = create_reward_functions()
    # def reward_fn(prompts, completions, **kwargs):
    #     scores = []
    #     for completion in completions:
    #         review = parse_review_from_completion(completion)
    #         # Submit to environment and get score
    #         resp = client.post("/step", json=review)
    #         scores.append(resp.json()["reward"])
    #     return scores
    #
    # config = GRPOConfig(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     learning_rate=args.learning_rate,
    # )
    #
    # trainer = GRPOTrainer(
    #     model=model, config=config, tokenizer=tokenizer,
    #     train_dataset=dataset, reward_funcs=[reward_fn],
    # )
    # trainer.train()
    # trainer.save_model(args.output_dir)

    print("Training template ready. Uncomment code when GPU is available.")


if __name__ == "__main__":
    main()
