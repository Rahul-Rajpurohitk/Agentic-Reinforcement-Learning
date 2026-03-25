"""GRPO Training Template using TRL + OpenEnv.

This template shows how to fine-tune an LLM using Group Relative Policy
Optimization (GRPO) with reward signals from an OpenEnv environment.

Requirements:
    - GPU with sufficient VRAM (A100 40GB recommended for larger models)
    - Running OpenEnv server: uvicorn src.agentic_rl.server.app:app --port 8000

Usage:
    python -m training.train_grpo --model_name "Qwen/Qwen2.5-0.5B" --num_episodes 100
"""

import argparse
from typing import List

from rewards import ExactMatchReward, PartialMatchReward, MultiObjectiveReward


def create_reward_function():
    """Create the composite reward function for GRPO training.

    Customize this to match your problem statement.
    """
    return MultiObjectiveReward(
        rewards=[ExactMatchReward(), PartialMatchReward()],
        weights=[0.7, 0.3],
    )


def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """TRL-compatible reward function.

    This is called by the GRPO trainer to score each (prompt, completion) pair.
    """
    reward = create_reward_function()
    targets = kwargs.get("targets", [None] * len(prompts))

    scores = []
    for prompt, completion, target in zip(prompts, completions, targets):
        # Extract just the assistant's response if formatted as chat
        score = reward.compute(prompt=prompt, completion=completion, target=target)
        scores.append(score)

    return scores


def main():
    parser = argparse.ArgumentParser(description="GRPO Training with OpenEnv")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_run")
    parser.add_argument("--env_url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    print(f"=== GRPO Training Template ===")
    print(f"Model: {args.model_name}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Environment: {args.env_url}")
    print()

    # ---------------------------------------------------------------
    # Uncomment below once you have GPU access and dependencies ready
    # ---------------------------------------------------------------
    #
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from trl import GRPOConfig, GRPOTrainer
    # from datasets import Dataset
    #
    # # 1. Load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    #
    # # 2. Prepare dataset from environment
    # # Collect episodes by interacting with the environment
    # from src.agentic_rl.client import AgenticRLClient
    # client = AgenticRLClient(base_url=args.env_url)
    #
    # prompts = []
    # targets = []
    # for _ in range(args.num_episodes):
    #     obs = client.reset()
    #     prompts.append(obs.prompt)
    #     state = client.get_state()
    #     targets.append(state.target)
    #
    # dataset = Dataset.from_dict({"prompt": prompts, "target": targets})
    #
    # # 3. Configure GRPO trainer
    # config = GRPOConfig(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     num_generations=args.num_generations,
    #     max_new_tokens=args.max_new_tokens,
    #     learning_rate=args.learning_rate,
    #     logging_steps=10,
    #     save_steps=50,
    # )
    #
    # # 4. Create trainer with reward function
    # trainer = GRPOTrainer(
    #     model=model,
    #     config=config,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset,
    #     reward_funcs=[reward_fn],
    # )
    #
    # # 5. Train!
    # trainer.train()
    # trainer.save_model(args.output_dir)
    # print(f"Model saved to {args.output_dir}")

    print("Training template ready. Uncomment the training code when GPU is available.")
    print("See comments in train_grpo.py for the full GRPO training pipeline.")


if __name__ == "__main__":
    main()
