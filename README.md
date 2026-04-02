# Fish Farm RL Environment

**The world's first OpenEnv-compatible aquaculture farming environment**

An AI agent manages a Nile Tilapia Recirculating Aquaculture System (RAS) — making hourly decisions about feeding, aeration, temperature control, water exchange, disease treatment, and harvest timing. Built on real aquaculture science: bioenergetic growth models, coupled DO/ammonia/pH dynamics, SEIR disease epidemiology, and realistic economic trade-offs.

Built for the [Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/) x Scaler School of Technology.

## Why Aquaculture?

Aquaculture is a **$300B global industry** producing 50%+ of the world's fish. Yet:
- **No Gymnasium/OpenEnv-compatible aquaculture environment exists** — this is the #1 identified gap in aquaculture AI research
- Real farms lose **$10B+ annually** to preventable die-offs from water quality failures, disease outbreaks, and suboptimal feeding
- The biological cascade (overfeed → ammonia → DO crash → stress → disease → mass mortality) creates a naturally rich RL problem with **13 coupled state variables**
- Q-learning already achieved **79% less feed and zero mortality** vs traditional control (Chahid et al. 2021) — proving RL has massive real-world impact here

---

## The Biological Cascade

The core challenge: **everything is connected.**

```
Overfeeding ──→ Ammonia ↑ ──→ DO ↓ ──→ Fish Stress ↑ ──→ Disease ──→ Mass Mortality
     ↑              ↑           ↓           ↓               ↓              ↓
  Feed Cost      pH shift    Growth ↓   Feeding ↓       Treatment $    Revenue = 0
```

An agent that feeds aggressively grows fish faster but risks catastrophic ammonia spikes. An agent that plays it safe grows slowly and loses money. The optimal policy requires **balancing 6 continuous controls across 13 coupled state variables** — a challenge that scales from easy single-concern tasks to extreme multi-crisis scenarios.

---

## Environment Design

### Action Space (6 continuous controls)

```json
{
  "feeding_rate": 0.0-1.0,       // Feed intensity (growth vs ammonia trade-off)
  "aeration_rate": 0.0-1.0,      // Oxygen injection (DO vs electricity cost)
  "heater_setting": -1.0-1.0,    // Temperature control (growth vs energy)
  "water_exchange_rate": 0.0-0.1, // Fresh water (dilution vs water cost)
  "harvest_decision": true/false,  // Harvest all fish (ends episode)
  "treatment": "none/antibiotics"  // Disease treatment (recovery vs cost)
}
```

### Observation Space (partial observability)

The agent sees sensor readings (temperature, DO, pH, ammonia, nitrite), fish status (weight, population, mortality, feeding behavior, stress), economics (costs, fish value, profit), weather, equipment status, and alerts. **Disease infection count is hidden** — the agent must infer disease from behavioral indicators.

### State Variables (13 coupled)

| Variable | Source | Coupling |
|----------|--------|----------|
| Fish weight | Bioenergetic ODE | Temperature, DO, UIA, feeding |
| Population | Mortality model | Stress, acute lethal events |
| DO | Mass balance ODE | Fish respiration, aeration, temperature, nitrification |
| TAN | Mass balance ODE | Feeding, biofilter, water exchange |
| UIA | Chemical equilibrium | TAN, pH, temperature |
| pH | Alkalinity buffer | Nitrification acid production |
| NO2 | Nitrification intermediate | Biofilter capacity |
| Temperature | Thermal model | Air temp, heater, volume |
| Stress | Weighted composite | DO, UIA, temperature, density |
| Disease (SEIR) | Compartmental model | Stress triggers, treatment |
| Feed inventory | Logistics | Consumption, deliveries |
| Costs | Accounting | All actions have costs |
| Weather | Stochastic | Diel cycle, seasons, storms |

### Key Equations

**Growth** (bioenergetic, from KAUST/FAO research):
```
dW/dt = [h·π·f·b·(1-a)·τ(T)·σ(DO)·v(UIA)] × W^0.6277 - [k_min·e^(s·(T-T_min))] × W^0.8373
```

**DO mass balance** (10 sub-steps/hour for stability):
```
dDO/dt = P_photo - FR·biomass/V - 4.57·K_NR·TAN - DO_water + K_a·(DO_sat-DO) + A_mech + Q_ex·(DO_in-DO)
```

**Ammonia toxicity**:
```
UIA = TAN / (1 + 10^(pKa - pH)),  pKa = 0.09018 + 2729.92/(T + 273.15)
```

---

## 12 Tasks (Easy → Extreme)

### Easy (3 tasks) — Learn one control
| Task | Hours | Challenge |
|------|-------|-----------|
| `feeding_basics` | 168 | Feed fish to 55g+ with FCR < 2.0, zero deaths |
| `oxygen_management` | 72 | Keep DO > 5.0 during hot weather (35°C air) |
| `water_quality_balance` | 168 | Maintain all water parameters simultaneously |

### Medium (4 tasks) — Multi-concern + events
| Task | Hours | Challenge |
|------|-------|-----------|
| `temperature_stress` | 120 | Survive a 3-day heat wave (38°C) |
| `ammonia_crisis` | 72 | Biofilter failure — manage rising ammonia |
| `disease_outbreak` | 240 | Detect and treat disease before 10% mortality |
| `growth_optimization` | 336 | Maximize growth while maintaining water quality |

### Hard (3 tasks) — Full lifecycle + compound events
| Task | Hours | Challenge |
|------|-------|-----------|
| `full_growout` | 1440 | 60-day grow-out: 20g → 400g market weight |
| `storm_response` | 120 | Severe storm + 12h power outage + biofilter recovery |
| `multi_objective` | 720 | Pareto-optimize profit × welfare × environment |

### Extreme (2 tasks) — Frontier-model difficulty
| Task | Hours | Challenge |
|------|-------|-----------|
| `catastrophe_prevention` | 336 | 5 compound crises in 14 days (algae bloom → aerator failure → disease → market crash → feed shortage) |
| `season_management` | 2160 | Full 90-day season with random events, ROI optimization |

---

## Quick Start

### Setup

```bash
git clone https://github.com/Rahul-Rajpurohitk/Agentic-Reinforcement-Learning.git
cd Agentic-Reinforcement-Learning
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/test_simulator.py tests/test_tasks_grader.py -v

# Start environment server
uvicorn src.agentic_rl.server.app:app --port 8000
```

### Docker

```bash
docker build -t fish-farm-env .
docker run -p 8000:8000 fish-farm-env
```

### Run Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export OPENAI_API_KEY=your_key
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List all 12 tasks with action schema |
| `/reset` | POST | Start episode `{"task_id": "feeding_basics"}` |
| `/step` | POST | Submit action, get observation |
| `/state` | GET | Internal state (ground truth for grading) |
| `/grader` | POST | Grade a completed episode |
| `/baseline` | POST | Run constant-action baseline |
| `/docs` | GET | Interactive Swagger docs |

---

## Project Structure

```
├── openenv.yaml              # OpenEnv spec (spec_version: 1, type: space)
├── inference.py              # LLM agent (< 20 min on 2 vCPU/8GB)
├── Dockerfile                # Container spec
├── requirements.txt
├── src/agentic_rl/
│   ├── constants.py          # All biological/physical/economic constants
│   ├── models.py             # FarmAction, FarmObservation, FarmState
│   ├── tasks.py              # 12 task scenarios
│   ├── engine/
│   │   ├── water_quality.py  # DO, TAN, UIA, pH, temperature dynamics
│   │   ├── fish_biology.py   # Bioenergetic growth, stress, mortality
│   │   ├── disease.py        # SEIR epidemic model
│   │   ├── economics.py      # Feed cost, operating cost, profit
│   │   ├── weather.py        # Diel cycle, seasons, storms
│   │   ├── events.py         # Event scheduler (equipment, disease, storms)
│   │   └── simulator.py      # Orchestrator (ties all subsystems together)
│   └── server/
│       ├── environment.py    # FishFarmEnvironment (OpenEnv interface)
│       └── app.py            # FastAPI server
├── graders/
│   ├── base_grader.py        # BaseGrader + GradeResult
│   └── farm_graders.py       # 12 task-specific graders with partial credit
├── tests/
│   ├── test_water_quality.py # 12 tests
│   ├── test_fish_biology.py  # 13 tests
│   ├── test_simulator.py     # 9 integration tests
│   └── test_tasks_grader.py  # 10 tests
└── docs/
    └── knowledge-base/       # 4,400+ lines of aquaculture research
```

---

## Research Foundation

Built on 4,400+ lines of research across 40+ citations:
- **Growth model**: FAO bioenergetic equations for Nile Tilapia (Oreochromis niloticus)
- **Water chemistry**: DO mass balance, ammonia equilibrium (Emerson et al. 1975)
- **Disease**: SEIR compartmental model with environmental triggers
- **Economics**: Real industry cost structures (feed = 50-70% of OpEx)
- **RL baseline**: Chahid et al. 2021 (Q-learning: 79% feed reduction, zero mortality)

---

## Author

**Rahul Rajpurohit** — Solo entry, Meta PyTorch OpenEnv Hackathon 2026
