"""Baseline inference script for the Code Review environment.

Runs an LLM (via OpenAI-compatible API) against all tasks and reports
reproducible baseline scores. This is a required submission artifact.

Usage:
    # Start the environment server first:
    uvicorn src.agentic_rl.server.app:app --port 8000

    # Run baseline (requires OPENAI_API_KEY env var):
    python baseline_inference.py

    # Or specify a different model/base URL:
    python baseline_inference.py --model gpt-4o-mini --base-url http://localhost:8000
"""

import argparse
import json
import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("ERROR: httpx package not installed. Run: pip install httpx")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert code reviewer. You will be given a code snippet and must identify all bugs, logic errors, security vulnerabilities, and style issues.

For each issue you find, provide:
- line: the line number where the issue occurs
- severity: "critical", "major", or "minor"
- category: "bug", "security", "style", "performance", or "logic"
- description: a clear explanation of what's wrong
- suggestion: how to fix it

Also provide an overall_assessment: "approve", "request_changes", or "comment".

Respond ONLY with valid JSON in this exact format:
{
  "issues_found": [
    {
      "line": "5",
      "severity": "critical",
      "category": "bug",
      "description": "Description of the issue",
      "suggestion": "How to fix it"
    }
  ],
  "overall_assessment": "request_changes",
  "confidence": 0.9
}"""


def call_llm(client: OpenAI, model: str, code: str, context: str) -> dict:
    """Send code to LLM for review and parse the response."""
    user_message = f"""Review this code for bugs, logic errors, and security vulnerabilities.

Context: {context}

```python
{code}
```

Respond with JSON only."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # Deterministic for reproducibility
        max_tokens=2000,
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON from potential markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


def run_baseline(env_url: str, model: str, openai_base_url: str = None):
    """Run baseline inference on all tasks and report scores."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    # Initialize clients
    llm_kwargs = {"api_key": api_key}
    if openai_base_url:
        llm_kwargs["base_url"] = openai_base_url
    llm_client = OpenAI(**llm_kwargs)
    env_client = httpx.Client(base_url=env_url, timeout=30.0)

    # Get all tasks
    tasks_resp = env_client.get("/tasks")
    tasks_resp.raise_for_status()
    all_tasks = tasks_resp.json()["tasks"]

    print(f"{'='*60}")
    print(f"Baseline Inference — Code Review Environment")
    print(f"Model: {model}")
    print(f"Environment: {env_url}")
    print(f"Tasks: {len(all_tasks)}")
    print(f"{'='*60}\n")

    results = []

    for task_info in all_tasks:
        task_id = task_info["task_id"]
        difficulty = task_info["difficulty"]

        print(f"--- Task: {task_id} ({difficulty}) ---")

        # Reset environment
        reset_resp = env_client.post("/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        code = obs["code_snippet"]
        context = obs["context"]

        # Get LLM review
        try:
            review = call_llm(llm_client, model, code, context)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  LLM Error: {e}")
            review = {
                "issues_found": [],
                "overall_assessment": "comment",
                "confidence": 0.0,
            }

        # Submit to environment
        step_resp = env_client.post("/step", json=review)
        step_resp.raise_for_status()
        result = step_resp.json()

        score = result["reward"]
        feedback = result["feedback"]

        print(f"  Score: {score:.3f}")
        print(f"  Feedback: {feedback}")
        print(f"  Issues reported: {len(review.get('issues_found', []))}")
        print()

        results.append({
            "task_id": task_id,
            "difficulty": difficulty,
            "score": score,
            "issues_found": len(review.get("issues_found", [])),
            "feedback": feedback,
        })

    # Summary
    print(f"{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")

    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for r in results:
        by_difficulty[r["difficulty"]].append(r["score"])

    total_scores = []
    for difficulty in ["easy", "medium", "hard"]:
        scores = by_difficulty[difficulty]
        if scores:
            avg = sum(scores) / len(scores)
            total_scores.extend(scores)
            print(f"  {difficulty.upper():8s}: {avg:.3f} avg ({len(scores)} tasks)")

    overall_avg = sum(total_scores) / len(total_scores) if total_scores else 0.0
    print(f"  {'OVERALL':8s}: {overall_avg:.3f} avg ({len(total_scores)} tasks)")
    print(f"{'='*60}")

    # Save results
    output = {
        "model": model,
        "environment": env_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "summary": {
            "overall_avg": overall_avg,
            "by_difficulty": {
                k: sum(v) / len(v) if v else 0.0
                for k, v in by_difficulty.items()
            },
        },
    }

    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to baseline_results.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline inference")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Env server URL")
    parser.add_argument("--openai-base-url", default=None, help="OpenAI API base URL")
    args = parser.parse_args()

    run_baseline(
        env_url=args.base_url,
        model=args.model,
        openai_base_url=args.openai_base_url,
    )
