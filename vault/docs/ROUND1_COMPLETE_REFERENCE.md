# Round 1 — Complete Reference

> Combined from dashboard screenshots (March 27) + live dashboard fetch (April 1, 2026)

---

## THE TASK

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

---

## KEY REQUIREMENTS AT A GLANCE

- Must simulate a real-world task (not games or toys)
- Implement full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml
- Minimum 3 tasks with agent graders (easy -> medium -> hard, scores 0.0-1.0)
- Meaningful reward function with partial progress signals
- Baseline inference script with reproducible scores
- Deploy to Hugging Face Spaces + working Dockerfile
- README with environment description, action/observation spaces, setup instructions

---

## Detailed Requirements

### FUNCTIONAL REQUIREMENTS

#### Real-world task simulation

The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

#### OpenEnv spec compliance

Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) -> returns observation, reward, done, info. reset() -> returns initial observation. state() -> returns current state. openenv.yaml with metadata. Tested via openenv validate.

#### Minimum 3 tasks with agent graders

Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0-1.0). Tasks should range: easy -> medium -> hard. Graders must have clear, deterministic success/failure criteria.

#### Meaningful reward function

Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).

#### Baseline inference script

Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.

---

### NON-FUNCTIONAL REQUIREMENTS

#### Deploys to a Hugging Face Space

Environment must run as a containerized HF Space tagged with openenv.

#### Containerized execution

Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.

#### Documentation

README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.

---

## INFERENCE SCRIPT REQUIREMENTS (NEW — from live dashboard April 1)

### Script naming and location

The inference script must be named `inference.py` and placed in the root directory.

### Required environment variables

Participants must use OpenAI Client for all LLM calls using these variables:

| Variable | Purpose |
|----------|---------|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Hugging Face / API key |

### Infrastructure restrictions

- Runtime of inference script should be less than 20 minutes
- Make sure your env and inference can run on a machine with vcpu=2, memory=8gb

---

## Evaluation Criteria

| PARAMETER | WEIGHT | DESCRIPTION |
|-----------|--------|-------------|
| Real-world utility | 30% | Does the environment model a genuine task? Would someone actually use this to train or evaluate agents? |
| Task & grader quality | 25% | Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression? |
| Environment design | 20% | Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries. |
| Code quality & spec compliance | 15% | Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works. |
| Creativity & novelty | 10% | Novel problem domain, interesting mechanics, clever reward design, original approach. |

---

## SCORING BREAKDOWN

### Real-world utility (30%)

- 0-5: Toy/artificial problem with no practical application
- 6-15: Valid domain but shallow modeling of the real task
- 16-25: Good domain modeling, would be useful for agent evaluation
- 26-30: Excellent — fills a real gap, immediate value for the RL/agent community

### Task & grader quality (25%)

- 3+ tasks with difficulty range?
- Graders produce scores between 0.0-1.0?
- Graders deterministic and reproducible?
- Hard task genuinely challenges frontier models?

### Environment design (20%)

- reset() produces clean state?
- Action/observation types well-designed and documented?
- Reward function provides useful varying signal (not just sparse)?
- Episode boundaries sensible?

### Code quality & spec compliance (15%)

- openenv validate passes?
- docker build && docker run works?
- HF Space deploys and responds?
- Baseline script runs and reproduces scores?

### Creativity & novelty (10%)

- Domain we haven't seen in OpenEnv before?
- Reward design has interesting properties?
- Clever mechanics that make the environment engaging?

---

## How Judging works

### Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

### Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

### Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

---

## DISQUALIFICATION CRITERIA

- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script

---

## Pre-Submission Checklist — all must pass or you're disqualified

| Check | What They Do |
|-------|-------------|
| HF Space deploys | Automated ping to the Space URL — must return 200 and respond to reset() |
| OpenEnv spec compliance | Validate openenv.yaml, typed models, step()/reset()/state() endpoints |
| Dockerfile builds | Automated docker build on the submitted repo |
| Baseline reproduces | Run the submitted inference script — must complete without error and produce scores |
| 3+ tasks with graders | Enumerate tasks, run each grader, verify scores in 0.0-1.0 range |
| Additional Endpoints to Expose | `/baseline` - Trigger inference script and returns baseline score for all 3 tasks |
| | `/grader` - Returns grader score after an episode is completed |
| | `/tasks` - Returns list of tasks and the action schema (fields required for an action in a step) |
| Validator | Run the pre-submission validation script before submitting |

---

## Problem Statements

When Round 1 opens, you'll choose 1 of 4-5 problem statements and build an OpenEnv environment around it.

### Example of what a problem statement looks like

> "Build a mini-game RL environment with clearly defined tasks, automated graders, and reward logic using the OpenEnv framework."

- -> Create a mini-game an AI agent can play
- -> Define tasks with increasing difficulty
- -> Write graders that verify task completion
- -> Define reward logic for scoring
- -> Package using OpenEnv for automated evaluation

### Evaluation Criteria

| Criterion | Description |
|-----------|-------------|
| Runtime correctness | Runs without errors |
| Interface compliance | Follows OpenEnv standard |
| Task design | Clear, realistic, testable |
| Grading logic | Reward system makes sense |

20,000 -> 3,000 teams advance

---

## Team / Solo Declaration

### Solo Warrior
Compete individually. You'll work and submit on your own.

### Team Up
2-3 members. Only the team lead fills this form.

### Team Selection Rules
- Only team lead submits the form; teammates added by email
- Automatic updates if already added to a team
- Each teammate needs individual account
- Once confirmed, teams cannot be changed. Solo is locked for Round 1 only.

---

## Bootcamp Event

**OpenEnv Round 1 Bootcamp: Build Your First RL Environment**
- Timing: 8:00 PM Onwards, Wednesday, 1st April
- Host: Ben Burtenshaw (Community Education in AI at Hugging Face) and Pulkit Aneja (Scaler Instructor)
- Description: Live walkthrough to submit a strong Round 1 entry

---

## Preparatory Course (Study Material)

4 modules, ~3.5 hours total:

| Module | Topic | Duration | Priority |
|--------|-------|----------|----------|
| Module 1 | Why OpenEnv? | 45 min | Essential for Round 1 |
| Module 2 | Using Existing Environments | 50 min | Essential for Round 1 |
| Module 3 | Deploying Environments | 45 min | Essential for Round 1 |
| Module 4 | Building Your Own Environment | 60 min | Most Important for Round 1 |

Full course repository on GitHub.

---

## Prerequisites

Install before April 1st.

### Required

**Python 3.10+**
Install 3.10, 3.11, or 3.12.
```
$ python --version
```

**Git + GitHub account**
Push your submission to GitHub or HF.
```
$ git --version
```

**Hugging Face CLI**
Deploy to HF Spaces.
```
$ pip install huggingface_hub
$ huggingface-cli login
```

**OpenEnv**
The framework.
```
$ pip install openenv-core
```

**Google Colab**
Prep course runs in Colab. Free tier works.
```
$ pip install openenv-core
```
```
-> colab.research.google.com
```

**Docker**
Isolated container testing.
```
docker --version
```

### Recommended

**VS Code** — Best Python + Docker support

---

## How to Submit

When Round 1 starts on 1 April:

**STEP 1: Application Form**
Choose 1 of the 4-5 problem statements revealed on the platform.

**STEP 2: Scaffold**
```
$ openenv init my_env
```
Generate project structure.

**STEP 3: Build**
Define your environment in the generated files.

**STEP 4: Test locally**
```
$ uv run server
```

**STEP 5: Deploy**
```
$ openenv push --repo-id your-username/your-env
```

**STEP 6: Submit**
Paste your HF Spaces URL here before the deadline.

---

## Submission Details

- Submission window opened: 28th March
- Deadline: **8 Apr 11:59 PM**
- Only team leaders can make the final submission
- Problem Statement is live — build and submit

---

## FAQs (Questions from dashboard)

- How does the team/solo declaration work?
- Who should fill the team form?
- What if someone already added me to their team?
- Can I change my team or switch to solo after confirming?
- Do I need to complete the prep course?
- What happens during Round 1?
- Can I update my submission?
- How are submissions evaluated?
- What framework must be used?
- What happens after Round 1?
- What do I need to submit?
- Where can I get help?

(Answers are behind interactive toggles on the dashboard)

---

## Support

Need help? Reach out: help_openenvhackathon@scaler.com
Join Discord for community support.

---

SUBMISSION WINDOW OPENS ON 28TH MARCH
