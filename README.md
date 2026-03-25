# Agentic Reinforcement Learning

**Meta PyTorch OpenEnv Hackathon** | Scaler School of Technology

Building RL environments using Meta's [OpenEnv](https://github.com/raun/openenv-course) framework for training AI agents with reinforcement learning.

## Hackathon Context

- **Event**: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology
- **Partners**: Meta, HuggingFace, PyTorch Foundation
- **Round 1**: March 25 - April 8, 2026 (Online)
- **Finale**: April 25-26, 2026 (Bangalore, In-person)
- **Goal**: Build mini-RL environments with defined tasks, automated graders, and reward logic

## Project Structure

```
├── src/agentic_rl/           # Core OpenEnv environment
│   ├── models.py             # Action, Observation, State types (Pydantic)
│   ├── client.py             # HTTP client for the environment
│   └── server/
│       ├── environment.py    # Core logic: reset(), step(), state
│       ├── app.py            # FastAPI server
│       └── Dockerfile        # Container spec
├── rewards/                  # Reward function library
│   ├── base_reward.py        # BaseReward interface
│   └── example_rewards.py    # ExactMatch, PartialMatch, MultiObjective
├── graders/                  # Evaluation/grading system
│   ├── base_grader.py        # BaseGrader interface
│   └── example_graders.py    # ExactMatch, Rubric graders
├── training/                 # GRPO training pipeline
│   └── train_grpo.py         # TRL + OpenEnv training template
├── tests/                    # Test suite
└── docs/                     # Quick reference docs
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the environment server
uvicorn src.agentic_rl.server.app:app --reload --port 8000

# 3. Run tests
pytest tests/ -v

# 4. Interact with the environment
python -c "
from src.agentic_rl.client import AgenticRLClient
client = AgenticRLClient()
obs = client.reset()
print(obs.prompt)
obs = client.step('your answer here')
print(obs.feedback, obs.score)
"
```

## OpenEnv 3-Component Pattern

Every OpenEnv environment follows this pattern:

1. **Types** (`models.py`): Pydantic models for Action, Observation, State
2. **Server** (`server/environment.py`): Game logic with `reset()`, `step()`, `state`
3. **Client** (`client.py`): HTTP/WebSocket client extending `EnvClient`

## Customization Guide

When problem statements are revealed (April 1st):

1. Update `models.py` with problem-specific Action/Observation/State fields
2. Implement task logic in `server/environment.py`
3. Create reward functions in `rewards/`
4. Create graders in `graders/`
5. Wire up GRPO training in `training/train_grpo.py`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action, get observation |
| `/state` | GET | Get internal state (for grading) |
| `/docs` | GET | Interactive API docs (Swagger) |

## Tech Stack

- **OpenEnv**: Meta's RL environment framework
- **FastAPI**: Server framework
- **Pydantic**: Type-safe data models
- **TRL**: Transformer Reinforcement Learning (GRPO training)
- **PyTorch**: Deep learning backend

## Author

**Rahul Rajpurohit** — rahulrajpurohit2024@gmail.com
