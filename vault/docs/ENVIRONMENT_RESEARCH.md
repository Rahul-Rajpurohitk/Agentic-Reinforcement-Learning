# OpenEnv Environment Research — Comprehensive Domain Analysis

> Research conducted April 1, 2026 for Meta PyTorch OpenEnv Hackathon (Round 1)
> Deadline: April 8, 2026 11:59 PM

---

## Table of Contents

1. [OpenEnv Ecosystem Status](#openenv-ecosystem-status)
2. [Existing Environments Catalog](#existing-environments-catalog)
3. [SF Hackathon Winners (March 2026)](#sf-hackathon-winners)
4. [Other Hackathon Winners (2025-2026)](#other-hackathon-winners)
5. [Hardest Tasks for Frontier Models](#hardest-tasks-for-frontier-models)
6. [Domain Gap Analysis](#domain-gap-analysis)
7. [Physical Reality-Based Ideas — Full Analysis](#physical-reality-based-ideas)
8. [Task-Rich Autonomous Systems — Deep Dive](#task-rich-autonomous-systems)
9. [Final Shortlist](#final-shortlist)

---

## OpenEnv Ecosystem Status

- **Framework by**: Meta (PyTorch), Hugging Face, Unsloth
- **GitHub**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **HF Hub**: [huggingface.co/openenv](https://huggingface.co/openenv)
- **Core package**: `pip install "openenv-core[core]>=0.2.1"`
- **Interface**: Gymnasium-compatible — `reset()`, `step(action)`, `state()`
- **Transport**: FastAPI server, WebSocket protocol, Docker containers
- **Status**: Experimental, ~13 published environments, mostly games
- **Training integrations**: TRL (GRPO), TorchForge, SkyRL, Unsloth, VeRL (planned)

---

## Existing Environments Catalog

### Official OpenEnv Spaces (HuggingFace `openenv/` org)

| Environment | Domain | Status |
|------------|--------|--------|
| Echo Environment | Testing | Running |
| Coding Environment | Code Execution (smolagents) | Running |
| REPL Environment (x2) | Code Execution | Running |
| OpenSpiel | Games (Catch, Tic-Tac-Toe, 2048) | Running |
| Atari | Arcade Games | Running |
| TextArena - Wordle | Word Games | Running |
| TextArena - Sudoku | Logic Puzzles | Running |
| Chat Environment | Conversational | Running |
| BrowserGym | Web Browsing (MiniWoB++) | Running |
| TB2 (Terminal-Bench 2) | Terminal/CLI | Running |
| SUMO-RL | Traffic Simulation | Running |

### OpenEnv Catalog (30+ entries)

| Environment | Domain |
|------------|--------|
| Chess, Connect4, Snake | Games |
| Grid World, Maze | Classic RL |
| Calendar Gym (Turing) | Tool Use / Productivity |
| CARLA | Autonomous Driving |
| FinRL, FinQL | Finance |
| dm_control | Robotics |
| Unity ML-Agents | Simulation |
| Web Search | Information Retrieval |
| OpenApp | Mobile |
| Julia | Code Execution |
| Git | Version Control |
| Reasoning Gym | Logic |
| kernrl | Systems |
| Wildfire | Emergency |
| DIPG Safety | AI Safety |

### Community / Hackathon Entries

| Environment | Domain | Status |
|------------|--------|--------|
| browsergym_env | Web Browsing | Running |
| browser-gym | Web Agent | Running |
| Football Play-Calling | Sports Strategy | Unknown |
| Stack Doctor | Unknown | Build Error |

### Domain Coverage Assessment

| Domain | Coverage | Maturity |
|--------|----------|----------|
| Games/Puzzles | HIGH | Most environments |
| Code Execution | MEDIUM | Solid |
| Web/Browsing | MEDIUM | Functional |
| Autonomous Driving | MEDIUM | CARLA blog post |
| Finance | LOW-MEDIUM | Listed, sparse docs |
| Tool Use | MEDIUM | Calendar Gym well-documented |
| Traffic | LOW | SUMO-RL running |
| Healthcare | VERY LOW | Hackathon only |
| Scientific Research | VERY LOW | Hackathon only |
| Cybersecurity | LOW | DIPG Safety listed |
| **Physical Robotics** | **NONE** | **MASSIVE GAP** |
| **Autonomous Delivery** | **NONE** | **MASSIVE GAP** |
| **Industrial Automation** | **NONE** | **MASSIVE GAP** |

---

## SF Hackathon Winners (March 7-8, 2026)

Cerebral Valley + PyTorch, $100K+ prize pool, 5 themed tracks.

| Project | Prize | Domain | Description |
|---------|-------|--------|-------------|
| **ZeroShot Cancer** | $10K (1st) | Biology | RL env for biological experiment planning (Scanpy, GSEApy, Biopython) |
| **ClinKriya** | Finalist | Healthcare | EHR simulation for clinical workflow navigation (Epic/Cerner) |
| **RL Voice Agents** | Finalist | Customer Service | Nested RL environments with simulated customers for IVR replacement |

**Key insight**: Winners were creative real-world domains, NOT games or code tools.

---

## Other Hackathon Winners (2025-2026)

### Microsoft AI Agents Hackathon (18,000 devs, 570 submissions)
- **RiskWise** (Best Overall): Supply chain risk analysis agent
- **DeepStudy**: AI-driven virtual classroom for K-12
- **Nuroxa**: Dementia risk analysis assistant
- **Agent Groot**: Multi-modal retail agent

### Nous Research RL Hackathon (May 2025)
47 new environments including:
- Sanskrit Poetry (RL for verse composition)
- Pokemon Showdown (strategic battles)
- Protein Design (molecular optimization)
- Philosophy RLAIF
- Poker (incomplete information)
- Lean Proving (theorem proving)

### Lablab.ai Winners
- **GameForge AI**: 4 AI agents → fully playable browser game in <60 seconds
- **Stylin'**: Fashion identification + outfit building in 30 seconds
- **Emergency Triage Automation**: $3K winner

### Pattern: Winners combine creative domain + strong engineering + clear real-world value

---

## Hardest Tasks for Frontier Models

### Sub-1% Accuracy
- **ARC-AGI-3**: Interactive reasoning — Gemini 3.1 Pro: 0.37%, GPT-5.4: 0.26%, Claude Opus 4.6: 0.25%. Humans: 100%.

### Under 15%
- **CUB (Computer Use Benchmark)**: Real-world desktop workflows — top score 10.4%
- **TheAgentCompany** (CMU): Workplace tasks — best agent 24% (Claude 3.5 Sonnet)

### 30-50%
- **Humanity's Last Exam**: Expert questions — GPT-5.4: 41.6%
- **FrontierMath**: Research math — GPT-5.4: 47.6%
- **Tau-Bench**: Customer service — pass^8 reliability ~25%
- **SWE-Bench Pro**: Multi-language SWE — Claude Opus 4.5: 45.9%

### Key Failure Pattern
Agent with 85% per-step accuracy on 10-step workflow succeeds only ~20% of the time (compound probability).

---

## Domain Gap Analysis

### Completely Untouched in RL

| Domain | RL Env Exists? | Real-World Systems Exist? |
|--------|---------------|--------------------------|
| Sidewalk delivery robots | NO | Starship (9M+ deliveries), Serve, Coco |
| Drone medical delivery | NO | Zipline (1M+ deliveries) |
| Autonomous mining trucks | NO | Caterpillar, Komatsu (production in AU/CL) |
| Construction site inspection | NO | Boston Dynamics Spot (daily use) |
| Autonomous ships | NO | Mayflower, Yara Birkeland |
| Warehouse inventory drones | NO | Verity, Gather AI |
| Underwater ROV inspection | NO | Saab Sabertooth, production use |
| Container port operations | NO (custom one-offs only) | Singapore Tuas, LA Port |
| Space station management | NO | NASA ISS |
| Nuclear plant operations | NO (as gym) | 440+ reactors worldwide |
| Autonomous greenhouse | Partial (gym-DSSAT, 2-3 tasks) | Many commercial operations |
| Airport turnaround | NO | Every airport worldwide |
| Underground mining | NO | Production in AU, CA, CL |
| Fish farm / aquaculture | NO | Norway, Chile, global industry |

---

## Physical Reality-Based Ideas — Full Analysis

### 1. Sidewalk Delivery Robot Navigation

**Real systems**: Starship (9M+ deliveries, 12 cameras, radar, ultrasonics, L4 autonomy), Serve Robotics (Jetson Orin AGX, 360° LiDAR, 100K+ deliveries), Coco (Niantic VPS from 30B Pokemon Go images, 1000+ robots)

**Real-world challenges documented**:
- 40 dangerous near-misses in 5 days on one campus
- Pedestrians breach safety buffers constantly
- Construction zones, cracked sidewalks, missing curb cuts
- GPS signal loss in urban canyons
- Weather degrading sensors and traction
- ADA accessibility conflicts (blocking wheelchair users)
- Road crossing decisions (125,000/day for Starship)

**RL env exists?**: NO. Closest: HMP-DRL (research paper, not packaged gym), TwinWalk (simulation, not RL), AutoVRL (1/10th-scale car, not sidewalk)

**Text observation format**:
```
Position: (12, 8) | Heading: East | Speed: 3km/h | Battery: 71%
Weather: Rain | Sidewalk: Narrow | Time: 14:32
Ahead (0-5m): Pedestrian (adult, head-on, 4km/h)
Ahead (5-15m): Construction barrier, right half blocked
Road crossing in 20m (2-lane, moderate traffic, no signal)
Deliveries: 1/3 done | Next: 247 Oak Ave (180m) | ETA deadline: 14:45
```

### 2. Zipline Medical Drone Delivery

**Real system**: Zipline — 1M+ deliveries, blood/vaccines/medicine, 80km+ range, parachute drops, operating in Rwanda, Ghana, US

**Tasks**: Single delivery → multi-drop route → storm mid-flight → emergency blood priority → fleet coordination

### 3. Autonomous Mining Trucks

**Real systems**: Caterpillar 797F (400-ton), Komatsu 930E — operating autonomously in Australian iron ore mines

**Tasks**: Single point-to-point → fleet no-collision → night shift → blast rerouting → rain + grade limits

### 4. Construction Site Inspection Robot (Spot-style)

**Tasks**: Single floor walk → active work zone navigation → safety violation detection → multi-floor inspection → night + battery constraints

### 5. Autonomous Ship Navigation

**Real systems**: Mayflower Autonomous Ship, Yara Birkeland (world's first autonomous cargo ship)

**Tasks**: Open water → coastal traffic → narrow channel + current → storm rerouting → full port approach

### 6. Warehouse Inventory Drone Scanner

**Real systems**: Verity, Gather AI (formerly Ware) — drones scanning barcodes in warehouses

**Tasks**: Single aisle → zone scan + forklift avoidance → find missing pallet → overnight audit + battery swaps → multi-drone coordination

### 7. Underwater ROV Pipeline Inspection

**Tasks**: Straight pipeline → subsea structure navigation → defect detection in murky water → current change → emergency gas leak localization

---

## Task-Rich Autonomous Systems — Deep Dive

### Tier 1: Richest Task Spaces (12-16+ distinct task types)

#### Container Port Terminal Operations — ~16 task types
1. Berth allocation
2. Quay crane scheduling
3. Stowage planning
4. Container stacking/yard management
5. Pre-marshalling
6. Container relocation
7. AGV/truck dispatch & routing
8. Yard crane scheduling
9. Gate management
10. Traffic flow optimization
11. Predictive maintenance
12. Energy management (AGV charging)
13. Collision avoidance
14. Inventory/dwell forecasting
15. Resource portfolio allocation
16. Anomaly detection & self-optimization

**Cascade**: Berth → crane → yard stacking → gate → truck dispatch. Poor stacking → relocations → yard congestion → crane idle → missed vessel cutoffs.

**RL papers**: 71+ published. Q-learning, Dueling DDQN, PPO proven tractable.

#### Nuclear Power Plant — ~14 control systems
1. Control rod positioning (72+ rods)
2. Boron concentration management
3. Pressurizer heater/spray
4. Reactor coolant pump
5. Feedwater flow control
6. Steam generator management
7. Steam dump valve control
8. Turbine speed/load control
9. Chemical Volume Control System
10. Containment systems
11. Safety injection systems
12. Axial flux distribution
13. Grid synchronization
14. Emergency response (SCRAM)

**Modes**: Startup, power ascent, steady state, load following, controlled shutdown, emergency SCRAM — each has completely different control strategies.

**Delayed consequences**: Xenon-135 buildup creates time-delayed reactivity effects that constrain power changes over HOURS.

#### Autonomous Greenhouse / Vertical Farm — ~13 control variables
1. Air temperature
2. Humidity/VPD management
3. CO2 concentration
4. Light intensity
5. Light spectrum (red/blue/far-red)
6. Photoperiod scheduling
7. Nutrient concentration (EC)
8. Nutrient pH
9. Irrigation timing/volume
10. Root zone temperature
11. Airflow/ventilation
12. Crop steering (generative vs. vegetative)
13. Pest/disease management

**Coupling**: Temperature ↔ humidity (inverse) → VPD → transpiration → nutrient uptake. CO2 enrichment needs sufficient light. Light raises leaf temp → changes local VPD. High humidity → fungal disease; low humidity → stomatal closure → reduced CO2 uptake.

**Multi-timescale**: Seconds (stomata) → hours (transpiration) → days (growth) → weeks (flowering).

### Tier 2: Very Rich (10-13 task types)

#### Underground Mining — ~12 task types
Drill → blast → scale → support → muck → haul per face, with shared equipment contention, ventilation routing, water management, gas monitoring across dozens of working faces. NP-hard scheduling.

#### Offshore Drilling Platform — ~12 decision types
Mud weight, WOB, RPM, flow rate, bit selection, casing points, cementing, trajectory, BOP, formation evaluation, managed pressure drilling, contingency response.

#### Airport Turnaround — ~14 task types
Gate assignment, jet bridge, deboarding, baggage unload, cleaning, catering, fueling, lavatory, water, baggage load, boarding, pushback, de-icing, GPU/PCA. 150+ activities across 30 actors per turnaround.

### Tier 3: Rich (8-11 task types)

#### Aquaculture / Fish Farm — ~10 tasks
Feeding, dissolved oxygen, temperature, pH, ammonia, salinity, disease detection, biomass estimation, equipment health, harvest timing.

#### F1 Race Strategy — ~10 decisions
Tire compound, pit timing, fuel/energy, battery recovery, DRS, overtaking, safety car response, weather changes, traffic, car setup. PPO agents show 8.6s improvement over fixed strategies.

#### Data Center Operations — ~9 controls
Supply airflow, temperature, humidity, workload placement, VM migration, server power, cooling tower, aisle pressure, UPS. Meta achieved 20% fan energy reduction with offline RL. Google/DeepMind: 9-13% energy savings.

---

## Existing RL Environments Per Domain (what's already built)

| Domain | Existing Env | Tasks | Compute |
|--------|-------------|-------|---------|
| Power Grid | Grid2Op / RL2Grid | 39 tasks, 5 levels | 2 vCPU OK (small grids) |
| Air Traffic | BlueSky-Gym | 7 environments | 2 vCPU OK (headless) |
| Warehouse | RWARE, Storehouse | Pick-deliver only | 2 vCPU OK |
| Building Energy | CityLearn, Sinergym | HVAC/energy only | 2 vCPU OK (CityLearn) |
| Agriculture | gym-DSSAT | 2-3 tasks | 2 vCPU OK |
| Traffic | CityFlow, SUMO-RL | Signals only | 2 vCPU OK |
| Surgical | SurRoL | 14 tasks | Needs GPU ❌ |
| Port Operations | NONE | — | Custom build needed |
| Nuclear | NONE (as gym) | — | Custom build needed |
| Space Station | NONE | — | Custom build needed |
| Airport | NONE | — | Custom build needed |
| Mining | NONE | — | Custom build needed |
| Delivery Robots | NONE | — | Custom build needed |
| Drone Delivery | NONE | — | Custom build needed |

---

## Final Shortlist

### Physical + Reality-Based + Task-Rich + Novel + Buildable in 2-3 days

| Rank | Idea | Tasks | Novel? | "Anyone Gets It?" | Build Time |
|------|------|-------|--------|-------------------|------------|
| 1 | **Autonomous Greenhouse / Vertical Farm** | 13 controls, 8+ tasks | Yes (gym-DSSAT covers only 2) | Yes (food, sustainability) | 2-3 days |
| 2 | **Sidewalk Delivery Robot** | 8+ tasks | Yes (zero RL env) | Yes (everyone sees them) | 2 days |
| 3 | **Airport Turnaround Ops** | 14 types, 10+ tasks | Yes (zero RL env) | Yes (everyone flies) | 2-3 days |
| 4 | **Drone Medical Delivery (Zipline)** | 7+ tasks | Yes (zero RL env) | Yes (life-saving) | 2 days |
| 5 | **Container Port Terminal** | 16 types, 10+ tasks | Yes (zero RL env) | Somewhat (industry-specific) | 3 days |
| 6 | **Data Center Ops** | 9 types, 6+ tasks | Partial (Meta did offline RL) | Somewhat | 2 days |
| 7 | **F1 Race Strategy** | 10 types, 8+ tasks | Partial (PPO papers exist) | Yes (exciting!) | 2-3 days |
| 8 | **Aquaculture / Fish Farm** | 10 types, 7+ tasks | Yes (zero RL env) | Somewhat | 2 days |

---

## Key Sources

### OpenEnv & Hackathon
- [OpenEnv Blog](https://huggingface.co/blog/openenv)
- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Docs](https://meta-pytorch.org/OpenEnv/)
- [TRL OpenEnv Integration](https://huggingface.co/docs/trl/openenv)
- [SF Hackathon (Cerebral Valley)](https://cerebralvalley.ai/e/openenv-hackathon-sf)
- [India Hackathon (Scaler)](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)
- [Centific Analysis](https://www.centific.com/blog/openenv-hackathon-rl-environments)

### Delivery Robots
- [Starship Technologies](https://www.starship.xyz/faq/)
- [Serve Robotics + NVIDIA](https://www.nvidia.com/en-us/case-studies/serve-robotics/)
- [Coco + Niantic VPS](https://www.nianticspatial.com/en/blog/coco-robotics)
- [HMP-DRL Paper](https://arxiv.org/html/2512.24651v1)
- [Robust Route Planning](https://arxiv.org/html/2507.12067)
- [TwinWalk Simulator](https://yorkspace.library.yorku.ca)

### Autonomous Systems
- [Grid2Op](https://github.com/rte-france/Grid2Op) | [RL2Grid](https://github.com/...)
- [BlueSky-Gym](https://github.com/...) — ATC
- [gym-DSSAT](https://github.com/...) — Agriculture
- [CityLearn](https://github.com/...) — Building Energy
- [RWARE](https://github.com/semitable/robotic-warehouse)
- [SimFire](https://arxiv.org/html/2311.15925) — Wildfire

### Benchmarks
- [ARC-AGI-3](https://arcprize.org/arc-agi/3)
- [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany)
- [PaperBench](https://arxiv.org/abs/2504.01848)
- [EVMbench](https://openai.com/index/introducing-evmbench/)
- [Tau-Bench](https://taubench.com/)
- [GAIA2](https://huggingface.co/blog/gaia2)

### Industrial Systems
- [Container Port RL (71+ papers)](https://link.springer.com/article/10.1007/s10696-025-09643-4)
- [Nuclear RL](https://www.sciencedirect.com/science/article/pii/S1738573324000032)
- [Greenhouse Control](https://onlinelibrary.wiley.com/doi/full/10.1002/fes3.70026)
- [F1 Race Strategy RL](https://arxiv.org/html/2512.21570)
- [Meta Data Center RL](https://engineering.fb.com/2024/09/10/data-center-engineering/simulator-based-reinforcement-learning-for-data-center-cooling-optimization/)
- [Underground Mining](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131003)
- [Airport Turnaround](https://www.aviationpros.com/ground-support-worldwide/ground-handling/article/55319018/reimagining-ramp-operations-how-ai-is-transforming-the-turnaround)
- [Aquaculture AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC8435764/)
