# OpenEnv Hackathon — Complete Course Cheatsheet
> Everything from 5 modules in one doc. Master this and you're ready for April 1st.

---

## MODULE 1: WHY OPENENV?

### The RL Loop (memorize this)
```python
while not done:
    observation = env.observe()
    action = policy.choose(observation)
    reward = env.step(action)
    policy.learn(reward)
```

### Why Not Gymnasium?
| Problem | Gymnasium | OpenEnv |
|---------|-----------|---------|
| Type Safety | `obs[0][3]` mystery | `obs.info_state` typed |
| Isolation | Same process (crash risk) | Docker containers |
| Deployment | "Works on my machine" | Same container everywhere |
| Scaling | Hard to distribute | Deploy to K8s |
| Language | Python only | Any language via HTTP |

**Core Philosophy: RL environments should be microservices.**

### Architecture
```
YOUR TRAINING CODE
  env = MyEnv(base_url="https://...")
  result = env.reset()       ← Type-safe
  result = env.step(action)  ← Type-safe
       │
       │  WebSocket / HTTP
       │
  DOCKER CONTAINER (HF Space / local / cloud)
    FastAPI Server → Environment (reset, step, state)
```

### The 3-Method Interface (EVERY environment, always)
| Method | Does | Returns |
|--------|------|---------|
| `reset()` | Start new episode | StepResult (obs, reward, done) |
| `step(action)` | Take action | StepResult (obs, reward, done) |
| `state()` | Get metadata | State (episode_id, step_count) |

### The 3-Component Pattern (EVERY environment, always)
```
my_env/
├── models.py          ← Pydantic: Action, Observation, State
├── client.py          ← EnvClient (what users import)
├── openenv.yaml       ← Manifest
└── server/
    ├── environment.py ← Game logic (reset/step/state)
    ├── app.py         ← FastAPI (one-liner)
    └── Dockerfile     ← Container
```

---

## MODULE 2: USING EXISTING ENVIRONMENTS

### Environment Hub
Every HF Space gives you 3 things:
1. **Server** endpoint: `https://<user>-<space>.hf.space`
2. **Pip package**: `pip install git+https://huggingface.co/spaces/<space>`
3. **Docker image**: `docker pull registry.hf.space/<space>:latest`

### Type-Safe Models (key pattern!)
```python
from openenv.core.env_server import Action, Observation, State

class MyAction(Action):
    action_id: int

class MyObservation(Observation):
    # done: bool and reward: Optional[float] are INHERITED
    info_state: List[float]
    legal_actions: List[int]

class MyState(State):
    # episode_id and step_count are INHERITED
    target_word: str = ""
```

**TRICK**: `done` and `reward` are already in `Observation` base class. Don't re-declare them!

### Policy Pattern
```python
def my_policy(obs):
    # obs is typed — use obs.legal_actions, obs.info_state, etc.
    return some_action_id
```

### Switching Games = Just Change URL
```python
# Same client class, different endpoint
env = OpenSpielEnv(base_url="https://openenv-openspiel-catch.hf.space")
env = OpenSpielEnv(base_url="https://openenv-openspiel-tictactoe.hf.space")
```

---

## MODULE 3: DEPLOYING ENVIRONMENTS

### Local Dev (fastest iteration)
```bash
git clone https://huggingface.co/spaces/openenv/echo-env
cd echo-env
uv sync
uv run server              # OR: uvicorn echo_env.server.app:app --reload
```

### Docker
```bash
docker build -t my-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 -e WORKERS=4 -e MAX_CONCURRENT_ENVS=100 my-env:latest
```

### Deploy to HF Spaces (ONE COMMAND!)
```bash
openenv push --repo-id username/my-env
```
This creates: API endpoint + Web UI + API docs + health check.

### openenv.yaml
```yaml
name: my_env
version: "1.0.0"
description: My custom environment
```

### Environment Variables
| Var | Default | Use |
|-----|---------|-----|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions |

### Scaling Quick Facts
- Free HF Space: ~128 concurrent sessions (2 vCPU, 16GB RAM)
- Single container (8 workers): ~2,048 sessions
- WebSocket overhead: ~0.1ms/frame (vs 10-50ms HTTP)

---

## MODULE 4: BUILDING YOUR OWN ENVIRONMENT ⭐ (MOST IMPORTANT)

### Step 1: Types (`models.py`)
```python
from openenv.core.env_server import Action, Observation, State

class WordGameAction(Action):
    guess: str

class WordGameObservation(Observation):
    # done and reward inherited!
    masked_word: str
    guessed_letters: List[str]
    attempts_remaining: int
    message: str

class WordGameState(State):
    # episode_id and step_count inherited!
    target_word: str = ""
    max_attempts: int = 10
```

### Step 2: Environment (`server/environment.py`)
```python
from openenv.core.env_server import Environment

class WordGameEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True  # ← Don't forget!

    def reset(self, seed=None, episode_id=None, **kwargs) -> WordGameObservation:
        # Initialize state, return initial observation
        ...

    def step(self, action: WordGameAction, timeout_s=None, **kwargs) -> WordGameObservation:
        # Process action, check win/lose, return observation with reward
        ...

    @property
    def state(self) -> WordGameState:
        return self._state
```

**TRICKS**:
- `reset()` signature: `(self, seed=None, episode_id=None, **kwargs)`
- `step()` signature: `(self, action, timeout_s=None, **kwargs)`
- Set `SUPPORTS_CONCURRENT_SESSIONS = True` for multi-player support
- Return `reward=None` for intermediate steps, `reward=1.0/0.0` on done

### Step 3: Client (`client.py`)
```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

class WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):
    def _step_payload(self, action: WordGameAction) -> dict:
        return {"guess": action.guess}  # Serialize to JSON

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=WordGameObservation(...),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WordGameState:
        return WordGameState(**payload)
```

**TRICK**: `EnvClient` is generic — parameterized with `[Action, Observation, State]`

### Step 4: FastAPI App (`server/app.py`) — ONE LINE!
```python
from openenv.core.env_server import create_fastapi_app
from environment import WordGameEnvironment

app = create_fastapi_app(WordGameEnvironment)
# Auto-generates: /ws, /reset, /step, /state, /health, /web, /docs
```

### Step 5: Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Fast Path: Scaffold
```bash
openenv init word_game    # Creates full directory structure
cd word_game
# Edit: models.py, server/environment.py, client.py
uv run server             # Test locally
openenv push --repo-id user/word-game  # Deploy
```

**~100 lines of meaningful code for a complete environment!**

---

## MODULE 5: TRAINING WITH GRPO + TRL

### What is GRPO?
1. Generate **group** of completions for same prompt
2. **Score** each with reward functions
3. Use **relative ranking** within group to update policy
4. **No value model** needed (unlike PPO) — the group IS the baseline

### Training Pipeline
```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct, reward_greens, reward_yellows],
    rollout_func=rollout_func,   # Your env interaction
    train_dataset=dataset,
    args=grpo_config,
)
trainer.train()
```

### Reward Functions (give richer gradient signal!)
```python
# Binary: did you solve it?
def reward_correct(completions, **kwargs):
    return [1.0 if solved else 0.0 for ...]

# Shaped: how many greens? (partial progress!)
def reward_greens(completions, **kwargs):
    return [num_greens / 5.0 for ...]

# Shaped: how many yellows?
def reward_yellows(completions, **kwargs):
    return [num_yellows / 5.0 for ...]

# Penalty: repeated guesses
def reward_repetition(completions, **kwargs):
    return [1.0 if no_repeats else 0.0 for ...]
```

**TRICK**: Multiple reward functions > single binary reward. Partial signals help the model learn faster.

### Rollout Function Pattern
```python
def rollout_once(trainer, env, tokenizer, prompt, system_prompt, max_turns):
    result = env.reset()
    observation = result.observation

    for turn in range(max_turns):
        if result.done:
            break
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_game_state(observation)},
        ]
        rollout = generate_rollout_completions(trainer, [messages])
        guess = extract_guess(rollout["text"])
        result = env.step(Action(message=guess))
        observation = result.observation

    return {"prompt_ids": ..., "completion_ids": ..., "logprobs": ..., "rewards": ...}
```

### GRPO Config (key settings)
```python
GRPOConfig(
    learning_rate=5e-6,
    gradient_accumulation_steps=64,    # Effective batch without OOM
    per_device_train_batch_size=1,
    num_generations=2,                 # Group size
    max_completion_length=8,           # Short for games
    use_vllm=True,                     # Fast generation
    vllm_mode="colocate",              # Share GPU: train + generate
    gradient_checkpointing=True,       # Save memory
)
```

### Hardware
- A100 40GB, ~90 min training, ~37GB peak memory

---

## HACKATHON WINNING FORMULA

### What Judges Want (by weight)
1. **Real-world utility (30%)**: NOT games/toys. Email triage, code review, data cleaning, scheduling, customer support, content moderation.
2. **Task & grader quality (25%)**: 3+ tasks, easy→medium→hard, deterministic graders 0.0-1.0, hard task challenges frontier models.
3. **Environment design (20%)**: Clean reset(), sensible action/obs types, shaped rewards (not binary), proper episode boundaries.
4. **Code quality & spec (15%)**: `openenv validate` passes, Docker works, HF Space responds, baseline reproduces.
5. **Creativity (10%)**: Novel domain, interesting reward design, clever mechanics.

### Pre-Submission Must-Pass Checklist
- [ ] HF Space pings → 200 + responds to reset()
- [ ] `openenv validate` passes
- [ ] `docker build && docker run` works
- [ ] Baseline inference script runs, produces scores
- [ ] 3+ tasks with graders, all scores in 0.0-1.0

### The Build Sprint (April 1-7)
```
Day 1: Pick problem statement → openenv init → define models.py
Day 2: Implement environment.py (reset/step/state) + tasks
Day 3: Graders + reward functions + tests
Day 4: Client + baseline inference script
Day 5: Docker + local testing + fix bugs
Day 6: Deploy to HF Spaces + openenv validate
Day 7: Polish README + final testing + submit HF URL
```

### Critical Imports Cheatsheet
```python
# Server-side
from openenv.core.env_server import Environment, Action, Observation, State
from openenv.core.env_server import create_fastapi_app

# Client-side
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

# Training
from trl import GRPOTrainer, GRPOConfig
```
