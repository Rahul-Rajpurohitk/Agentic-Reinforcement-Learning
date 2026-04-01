# Aquaculture / Smart Fish Farm — Knowledge Base Index

> Complete research foundation for the OpenEnv Hackathon environment
> Domain: Autonomous Fish Farm Management
> Compiled: April 1, 2026

---

## Documents

| # | File | Lines | Description |
|---|------|-------|-------------|
| 01 | [01-BIOLOGY-AND-SCIENCE.md](./01-BIOLOGY-AND-SCIENCE.md) | 1,140 | PhD-level aquaculture biology: growth models (SGR, TGC, von Bertalanffy), water chemistry (DO, ammonia, nitrogen cycle, pH), feeding science (FCR, feeding rates), disease pathology (12+ diseases with triggers), behavioral welfare indicators, species-specific data (salmon, tilapia, shrimp, catfish, trout), RAS design, biofloc technology. **Includes all equations, thresholds, and units.** |
| 02 | [02-REAL-WORLD-OPERATIONS.md](./02-REAL-WORLD-OPERATIONS.md) | 918 | How commercial fish farms ACTUALLY operate: daily routines, sensor systems and refresh rates, automated feeding tech, DO management (aerators, paddlewheels), economics (cost structures, margins by species/country), mass mortality events and causes, Norwegian/Scottish salmon operations, infrastructure specs, regulations, harvest processes. **Real industry numbers and operational reality.** |
| 03 | [03-MATHEMATICAL-MODELS.md](./03-MATHEMATICAL-MODELS.md) | 1,444 | Control systems and mathematical models: bioenergetic growth ODEs, DO mass balance equations, TAN production models, disease SIR/SEIR equations, population dynamics, bioeconomic optimization, PID/MPC/Bang-Bang control, MDP formulations, stochastic models, multi-objective Pareto, DEB theory, thermal models, agent-based simulation. **Ready-to-implement state/action/reward specs for OpenEnv.** |
| 04 | [04-RL-AND-AI-RESEARCH.md](./04-RL-AND-AI-RESEARCH.md) | 350+ | Every RL paper applied to aquaculture: Q-learning (KAUST 2021), DDPG series (6 papers, 2025), RAG-LLM+DQN hybrid, disease prediction RL, existing environments and tools (gym_fishing, FishMet, Fish Gym), reward functions tried, computer vision systems, autonomous robotics, AQUA-7B LLM, state-of-art gaps. **40 citations with full URLs.** |
| 05 | [../ENVIRONMENT_RESEARCH.md](../ENVIRONMENT_RESEARCH.md) | 450+ | Broader domain research: OpenEnv ecosystem status, all existing environments, SF hackathon winners, other hackathon winners, hardest benchmarks for frontier models, domain gap analysis, physical reality-based ideas evaluation, task-rich autonomous systems deep dive. |

---

## Total Knowledge Base: ~4,300+ lines across 5 documents

---

## Quick Reference: Key Numbers for Environment Design

### Water Quality Thresholds (Nile Tilapia — our primary species)

| Parameter | Optimal | Acceptable | Stress | Lethal |
|-----------|---------|------------|--------|--------|
| Temperature | 27-32°C | 22-34°C | <20°C or >36°C | <11°C or >42°C |
| Dissolved O₂ | >5 mg/L | 3-5 mg/L | 1-3 mg/L | <1 mg/L |
| pH | 6.5-8.5 | 6.0-9.0 | <5.5 or >9.5 | <4 or >11 |
| NH₃ (unionized) | <0.02 mg/L | 0.02-0.05 | 0.05-0.3 | >2.0 mg/L |
| Nitrite | <0.1 mg/L | 0.1-0.5 | 0.5-1.0 | >5.0 mg/L |

### Growth Model
```
dW/dt = [Ψ(f,T,DO) × v(UIA) × W^0.67] - [k(T) × W^0.81]
SGR = 2.93 %/day at 32°C (tilapia)
FCR = 1.5-2.0 (tilapia)
```

### Key RL Result (Baseline to Beat)
Q-learning achieved **79% less feed** and **zero mortality** vs Bang-Bang control (Chahid et al. 2021)

### Gap We're Filling
**No Gymnasium/OpenEnv-compatible aquaculture farming RL environment exists.** This is the #1 identified gap in the entire aquaculture AI research landscape.

---

## Decision: Environment Configuration

- **Species**: Nile Tilapia (Oreochromis niloticus) — best-studied, fastest growth, most RL data
- **System**: Recirculating Aquaculture System (RAS) — most controllable, most sensors, most relevant for AI
- **Scale**: Single-tank → multi-tank progression across tasks
- **Time step**: 1 hour (balances biological dynamics with decision frequency)
- **Episode length**: Variable by task (1 day for feeding → full season for harvest timing)
