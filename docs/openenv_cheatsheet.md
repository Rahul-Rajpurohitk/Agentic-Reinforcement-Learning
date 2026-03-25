# OpenEnv Quick Reference

## CLI Commands

```bash
# Scaffold a new environment
openenv init <env_name>

# Run locally
cd <env_name> && uv run server

# Deploy to HuggingFace Spaces
openenv push --repo-id <username>/<env-name>
```

## 3-Component Pattern Files

| File | Purpose |
|------|---------|
| `models.py` | Pydantic types: Action, Observation, State |
| `server/environment.py` | Game logic: reset(), step(), state |
| `client.py` | EnvClient subclass with HTTP/WS methods |
| `server/app.py` | FastAPI wiring via `create_fastapi_app()` |
| `server/Dockerfile` | Container spec for deployment |

## Key Classes

```python
# models.py - Define your types
class MyAction(BaseModel):
    response: str

class MyObservation(BaseModel):
    prompt: str
    done: bool

class MyState(BaseModel):
    target: str
```

```python
# environment.py - Core logic
class MyEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self) -> MyObservation: ...
    def step(self, action: MyAction) -> MyObservation: ...
    @property
    def state(self) -> MyState: ...
```

```python
# client.py - Client methods
class MyClient(EnvClient):
    def _step_payload(self, action) -> dict: ...
    def _parse_result(self, payload) -> StepResult: ...
    def _parse_state(self, payload) -> MyState: ...
```

```python
# app.py - One-liner wiring
from openenv.core.env_server import create_fastapi_app
app = create_fastapi_app(MyEnvironment)
```

## GRPO Training (TRL)

```python
from trl import GRPOConfig, GRPOTrainer

def reward_fn(prompts, completions, **kwargs) -> list[float]:
    # Return list of reward scores
    return [score_completion(p, c) for p, c in zip(prompts, completions)]

config = GRPOConfig(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_generations=4,
    max_new_tokens=256,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[reward_fn],
)
trainer.train()
```

## Deployment Flow

1. **Local**: `uvicorn server.app:app --reload`
2. **Docker**: `docker build -t myenv . && docker run -p 8000:8000 myenv`
3. **HF Spaces**: `openenv push --repo-id user/env-name`

## Useful Links

- [OpenEnv Course](https://github.com/raun/openenv-course)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [FastAPI Docs](https://fastapi.tiangolo.com)
