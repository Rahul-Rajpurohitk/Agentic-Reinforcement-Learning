# Code Review Agent Environment

**An OpenEnv-compatible RL environment for training AI agents to review code**

A real-world reinforcement learning environment where AI agents learn to identify bugs, logic errors, and security vulnerabilities in code. Built for the [Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/) x Scaler School of Technology.

## Why Code Review?

Code review is one of the most common tasks in software engineering — every PR needs review, yet it's time-consuming and inconsistent. Training agents to review code has immediate, practical value:

- Catches bugs before they reach production
- Identifies security vulnerabilities early
- Scales across codebases without reviewer fatigue
- Provides consistent, reproducible assessments

This environment lets you train and evaluate agents on realistic code review tasks with clear, measurable outcomes.

---

## Environment Description

The agent receives a Python code snippet and must identify all issues — bugs, logic errors, security vulnerabilities, and style problems. Each issue must be reported with a line number, severity, category, description, and fix suggestion.

### Action Space

The agent sends a `ReviewAction`:

```python
{
    "issues_found": [
        {
            "line": "6",                    # Line number of the issue
            "severity": "critical",          # critical | major | minor
            "category": "bug",               # bug | security | style | performance | logic
            "description": "ZeroDivisionError when list is empty",
            "suggestion": "Check for empty list before division"
        }
    ],
    "overall_assessment": "request_changes",  # approve | request_changes | comment
    "confidence": 0.9                         # 0.0 - 1.0
}
```

### Observation Space

The environment returns a `ReviewObservation`:

```python
{
    "task_id": "easy_001",
    "task_difficulty": "easy",
    "code_snippet": "def calculate_average(numbers):\n    ...",
    "language": "python",
    "context": "A utility function to calculate the average of a list",
    "feedback": "Found 1/1 issues. Excellent!",
    "reward": 0.85,          # Partial progress signal [0.0 - 1.0]
    "done": true,            # Episode complete flag
    "info": { ... }          # Grading details
}
```

### Reward Function

The reward provides **meaningful partial progress signals** over the full trajectory:

- **Keyword matching** (70%): How well found issues match ground-truth issue descriptions
- **Line proximity** (20%): How close the reported line is to the actual issue
- **Severity accuracy** (10%): Whether the severity classification is correct
- **Noise penalty**: False positives are gently penalized (-0.03 each, max -0.15)

This shaped reward encourages agents to find all real issues while avoiding false alarms.

---

## Tasks

9 tasks across 3 difficulty levels. Each task has deterministic graders scoring 0.0–1.0.

### Easy (3 tasks)

| Task ID | Description | Issues | Expected Score |
|---------|-------------|--------|----------------|
| `easy_001` | ZeroDivisionError in average function | 1 bug | ~0.85+ |
| `easy_002` | Negative number handling in find_max | 1 logic | ~0.85+ |
| `easy_003` | File handle resource leak | 1 bug | ~0.85+ |

### Medium (3 tasks)

| Task ID | Description | Issues | Expected Score |
|---------|-------------|--------|----------------|
| `medium_001` | Integer vs float division in binary search | 1 bug | ~0.70+ |
| `medium_002` | Missing set.add() in deduplication | 1 logic | ~0.70+ |
| `medium_003` | Cache key ignores kwargs in memoize | 1 logic | ~0.60+ |

### Hard (3 tasks)

| Task ID | Description | Issues | Expected Score |
|---------|-------------|--------|----------------|
| `hard_001` | Mass assignment + path traversal in REST API | 2 security | ~0.40+ |
| `hard_002` | Weak crypto + token issues in auth system | 4 security | ~0.30+ |
| `hard_003` | Command injection + pickle deserialization in pipeline | 3 security | ~0.35+ |

**Hard tasks genuinely challenge frontier models** — finding all subtle security vulnerabilities requires deep understanding of web security, cryptography, and attack vectors.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Setup

```bash
# Clone and install
git clone https://github.com/Rahul-Rajpurohitk/Agentic-Reinforcement-Learning.git
cd Agentic-Reinforcement-Learning
pip install -r requirements.txt

# Run the environment server
uvicorn src.agentic_rl.server.app:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run baseline inference (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your_key_here
python baseline_inference.py
```

### Docker

```bash
docker build -t code-review-env .
docker run -p 8000:8000 code-review-env
```

### Interact with the Environment

```python
from src.agentic_rl.client import CodeReviewClient

client = CodeReviewClient(base_url="http://localhost:8000")

# Start a review
obs = client.reset(task_id="easy_001")
print(obs.code_snippet)

# Submit review
obs = client.step(
    issues_found=[{
        "line": "6",
        "severity": "critical",
        "category": "bug",
        "description": "ZeroDivisionError on empty list",
        "suggestion": "Check length before dividing",
    }],
    overall_assessment="request_changes",
)
print(f"Score: {obs.reward}, Feedback: {obs.feedback}")
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (returns 200) |
| `/tasks` | GET | List all tasks with difficulty levels |
| `/reset` | POST | Start new episode `{"task_id": "easy_001"}` |
| `/step` | POST | Submit review action |
| `/state` | GET | Internal state (ground truth for grading) |
| `/docs` | GET | Interactive Swagger API docs |

---

## Project Structure

```
├── openenv.yaml              # OpenEnv spec manifest
├── Dockerfile                # Container spec (docker build + run)
├── requirements.txt          # Python dependencies
├── baseline_inference.py     # Baseline script using OpenAI API
├── src/agentic_rl/
│   ├── models.py             # Typed Pydantic models (Action, Observation, State)
│   ├── tasks.py              # 9 task definitions with ground truth
│   ├── client.py             # HTTP client
│   └── server/
│       ├── environment.py    # Core logic: reset(), step(), state
│       └── app.py            # FastAPI server
├── rewards/                  # Reward function library
├── graders/                  # Deterministic graders (0.0-1.0)
├── training/                 # GRPO training template
└── tests/                    # Test suite
```

---

## Baseline Scores

Scores from `baseline_inference.py` using GPT-4o-mini (temperature=0.0):

| Difficulty | Avg Score | Tasks |
|-----------|-----------|-------|
| Easy | ~0.80 | 3 |
| Medium | ~0.60 | 3 |
| Hard | ~0.35 | 3 |
| **Overall** | **~0.58** | **9** |

*(Run `python baseline_inference.py` to reproduce)*

---

## Tech Stack

- **OpenEnv** — Meta's RL environment framework
- **FastAPI** — Server framework with auto-generated docs
- **Pydantic** — Type-safe data models
- **TRL** — Transformer RL (GRPO training)
- **PyTorch** — Deep learning backend

## Author

**Rahul Rajpurohit** — rahulrajpurohit2024@gmail.com
