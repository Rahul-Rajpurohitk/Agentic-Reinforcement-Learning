# Fish Farm RL Environment

**The world's first OpenEnv-compatible aquaculture farming environment**

An AI agent manages a Nile Tilapia Recirculating Aquaculture System (RAS) — making hourly decisions about feeding, aeration, temperature control, water exchange, disease treatment, and harvest timing. Built on real aquaculture science: bioenergetic growth models, coupled DO/ammonia/pH dynamics, SEIR disease epidemiology, stochastic economics, and realistic multi-objective trade-offs.

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
                    ↑           ↓
              Evaporation   Nitrification ──→ NO2 ──→ NO3
              concentrates   consumes O2      ↑
                                           Biofilter
```

An agent that feeds aggressively grows fish faster but risks catastrophic ammonia spikes. An agent that plays it safe grows slowly and loses money. The optimal policy requires **balancing 6 continuous controls across 13 coupled state variables** — a challenge that scales from easy single-concern tasks to extreme multi-crisis scenarios.

---

## Simulation Engine Highlights

### 6 Deeply Coupled Subsystem Engines

| Engine | Key Features |
|--------|-------------|
| **Water Quality** | 10 sub-steps/hour DO mass balance, two-stage nitrification (AOB + NOB), denitrification under anoxic conditions, Smith-Talling photosynthesis, Penman evaporation model, Beer-Lambert light attenuation, nighttime DO crash risk tracking |
| **Fish Biology** | FAO bioenergetic growth ODE with stochastic noise (Wiener process σ=2%), dual respiration model (tilapia polynomial R²=0.99 + allometric fallback), size-dependent feeding rates, feeding response behavior |
| **Disease** | SEIR compartmental model with immunity waning (R→S at 1/30 per day), temperature-dependent pathogen virulence, stress-triggered outbreaks, 4 treatment options + prophylactic vaccination (works without active disease) |
| **Economics** | Ornstein-Uhlenbeck stochastic feed pricing, seasonal market multipliers (Christmas +15%, Lent +10%, mid-year dip -5%), marginal cost tracking, weight-dependent fish valuation with market premium curve, detailed cost breakdown (7 categories) |
| **Weather** | Diel temperature/solar cycle, seasonal storm probability (3× during monsoon), Beaufort wind scale, humidity-driven evaporation |
| **Events** | Equipment failures, power outages, algae blooms, feed shortages, market crashes — all wired to appropriate subsystems |

### Key Equations

**Growth** (bioenergetic, from KAUST/FAO research):
```
dW/dt = [h·π·f·b·(1-a)·τ(T)·σ(DO)·v(UIA)] × W^0.6277 - [k_min·e^(s·(T-T_min))] × W^0.8373
```

**DO mass balance** (10 sub-steps/hour for stability):
```
dDO/dt = P_photo - FR·biomass/V - 4.57·K_NR·TAN - DO_water + K_a·(DO_sat-DO) + A_mech + Q_ex·(DO_in-DO)
```

**Tilapia respiration polynomial** (R²=0.99, valid 20-200g, 24-32°C):
```
FR = 2014.45 + 2.75W - 165.2T + 0.007W² + 3.93T² - 0.21WT
```

**Ammonia toxicity**:
```
UIA = TAN / (1 + 10^(pKa - pH)),  pKa = 0.09018 + 2729.92/(T + 273.15)
```

**Stochastic feed price** (Ornstein-Uhlenbeck):
```
dp = κ(μ - p)dt + σ·dW,  bounded to ±40% of mean
```

---

## Environment Design

### Action Space (6 continuous controls)

```json
{
  "feeding_rate": 0.0-1.0,        // Feed intensity (growth vs ammonia trade-off)
  "aeration_rate": 0.0-1.0,       // Oxygen injection (DO vs electricity cost)
  "heater_setting": -1.0-1.0,     // Temperature control (growth vs energy)
  "water_exchange_rate": 0.0-0.1,  // Fresh water (dilution vs water cost)
  "harvest_decision": true/false,   // Harvest all fish (ends episode)
  "treatment": "none/antibiotics/salt/probiotics/vaccination"
}
```

### Observation Space (47 fields, partial observability)

The agent sees sensor readings (temperature, DO, pH, TAN, UIA, NO2, NO3, water quality score, nighttime DO crash risk), fish status (weight, population, mortality, feeding response, stress, FCR, SGR, growth rate, stocking density, survival rate), economics (costs, fish value, profit, feed price, market multiplier, ROI, marginal cost), weather (forecast, daytime, storm, humidity, day of year), equipment status, disease behavioral signals, and event alerts.

**Disease infection count is hidden** — the agent must infer disease from behavioral indicators (mortality spikes + feeding refusal + elevated stress → `disease_suspected` flag).

### Inference Agent Architecture

The LLM inference agent uses a **dual-mode architecture**:
1. **LLM mode**: Domain-expert system prompt with full situational awareness (all 30+ observation fields, trend analysis, harvest advisories)
2. **Heuristic fallback**: Rule-based agent that handles the critical cascades correctly when LLM is unavailable or time-constrained
3. **Adaptive call frequency**: More LLM calls during crises (every step), fewer during stable periods (every 4-6 hours)
4. **Smart time budgeting**: Proportional allocation across tasks, automatic fallback to heuristic when time runs low

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

# Run tests (276 tests)
pytest tests/ -v

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
├── inference.py              # LLM agent + heuristic fallback (< 20 min on 2 vCPU/8GB)
├── Dockerfile                # Container spec
├── requirements.txt
├── src/agentic_rl/
│   ├── constants.py          # All biological/physical/economic constants (frozen dataclasses)
│   ├── models.py             # FarmAction (6 controls), FarmObservation (30+ fields), FarmState
│   ├── tasks.py              # 12 task scenarios (easy → extreme)
│   ├── rewards.py            # Task-weighted reward function (10 component keys)
│   ├── engine/
│   │   ├── water_quality.py  # DO mass balance, nitrification, denitrification, evaporation, photosynthesis
│   │   ├── fish_biology.py   # Bioenergetic growth, dual respiration, stress, mortality
│   │   ├── disease.py        # SEIR epidemic, immunity waning, temperature virulence, vaccination
│   │   ├── economics.py      # OU feed pricing, seasonal markets, cost breakdown, ROI
│   │   ├── weather.py        # Diel cycle, seasonal storms, wind/humidity
│   │   ├── events.py         # Event scheduler (equipment, disease, storms, prices)
│   │   └── simulator.py      # Orchestrator (9-step coupling order)
│   └── server/
│       ├── environment.py    # FishFarmEnvironment (OpenEnv interface)
│       └── app.py            # FastAPI server + custom endpoints
├── graders/
│   ├── base_grader.py        # BaseGrader + GradeResult
│   └── farm_graders.py       # 12 task-specific graders with partial credit
├── tests/                    # 276 tests (2.3s)
│   ├── test_water_quality.py # DO, TAN, UIA, denitrification, evaporation, temperature
│   ├── test_fish_biology.py  # Growth, mortality, stress, respiration, size-feeding
│   ├── test_disease.py       # SEIR dynamics, treatments, vaccination, immunity, temperature
│   ├── test_economics.py     # Costs, stochastic pricing, seasonal markets, ROI, breakdown
│   ├── test_simulator.py     # Integration, observations, heuristic, stochastic growth, nighttime DO risk, vaccination prophylaxis, cost breakdown, harvest revenue
│   ├── test_constants.py     # Parameter sanity, utility functions (32 tests)
│   ├── test_tasks_grader.py  # Task definitions, all 12 graders
│   ├── test_rewards.py       # All reward component keys, delta rewards, disease/harvest, nighttime DO risk
│   ├── test_models.py        # Action/Observation/State model validation
│   └── test_endpoints.py     # /tasks, /grader, /baseline API endpoints
└── docs/
    └── knowledge-base/       # 4,400+ lines of aquaculture research (40+ citations)
```

---

## Research Foundation

Built on 4,400+ lines of research across 40+ citations:
- **Growth model**: FAO bioenergetic equations for Nile Tilapia (Oreochromis niloticus)
- **Respiration**: Tilapia-specific polynomial (R²=0.99) from controlled feeding experiments
- **Water chemistry**: DO mass balance, ammonia equilibrium (Emerson et al. 1975), two-stage nitrification
- **Disease**: SEIR compartmental model with temperature-dependent virulence and immunity waning
- **Economics**: Ornstein-Uhlenbeck stochastic pricing, seasonal demand curves, real industry cost structures (feed = 50-70% of OpEx)
- **RL baseline**: Chahid et al. 2021 (Q-learning: 79% feed reduction, zero mortality)

---

## Author

**Rahul Rajpurohit** — Solo entry, Meta PyTorch OpenEnv Hackathon 2026
