# OpenEnv Quick Reference

## CLI Commands

```bash
openenv init <env_name>          # Scaffold new environment
openenv validate                 # Validate openenv.yaml + spec compliance
openenv push --repo-id user/env  # Deploy to HuggingFace Spaces
```

## openenv.yaml Format

```yaml
spec_version: 1
name: code_review_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

## 3-Component Pattern

| File | Purpose |
|------|---------|
| `models.py` | Pydantic BaseModel: Action, Observation, State |
| `server/environment.py` | Core logic: reset(), step(), state property |
| `client.py` | EnvClient for HTTP/WebSocket interaction |
| `server/app.py` | `create_fastapi_app(Env, Action, Obs, State)` |
| `Dockerfile` | Container spec |

## Environment Interface

```python
class MyEnvironment:
    def reset(self) -> Observation:      # Start new episode
    def step(self, action: Action) -> Observation:  # Process action
    @property
    def state(self) -> State:            # Internal state for grading
```

## Observation Must Include

```python
class MyObservation(BaseModel):
    reward: float    # [0.0, 1.0]
    done: bool       # Episode complete?
    info: dict       # Metadata
```

## Deployment Flow

1. `uvicorn server.app:app --reload` (local)
2. `docker build -t env . && docker run -p 8000:8000 env` (container)
3. `openenv push --repo-id user/env` (HF Spaces)

## Pre-Submission Checklist

- [ ] `openenv validate` passes
- [ ] `docker build && docker run` works
- [ ] HF Space returns 200 on ping + responds to reset()
- [ ] Baseline inference script runs and produces scores
- [ ] 3+ tasks with graders, scores in 0.0-1.0
