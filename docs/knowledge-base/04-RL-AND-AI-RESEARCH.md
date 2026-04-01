# Reinforcement Learning & AI Applied to Aquaculture
## Comprehensive Research — Every Paper, Approach, Result, and Gap

> Compiled April 2026 from 70+ sources across arXiv, Springer, Nature, IEEE, ScienceDirect, GitHub

---

## Table of Contents

1. [RL Approaches in Aquaculture](#1-rl-approaches-in-aquaculture)
2. [Model Predictive Control (Baseline)](#2-model-predictive-control--the-primary-baseline)
3. [Existing RL Environments and Simulation Tools](#3-existing-rl-environments-and-simulation-tools)
4. [Reward Functions That Have Been Tried](#4-reward-functions-that-have-been-tried)
5. [Water Quality Prediction (Deep Learning)](#5-water-quality-prediction)
6. [Computer Vision for Fish Monitoring](#6-computer-vision-for-fish-monitoring)
7. [Autonomous and Robotic Systems](#7-autonomous-and-robotic-systems)
8. [Foundation Models for Aquaculture](#8-aqua-llm--foundation-model)
9. [State of the Art Summary and Gaps](#9-state-of-the-art-summary)
10. [Key Research Groups](#10-key-research-groups)
11. [Complete Source List](#11-complete-source-list)

---

## 1. RL Approaches in Aquaculture

### 1.1 Q-Learning for Fish Growth Trajectory Tracking (FOUNDATIONAL PAPER)

**Paper:** "Fish Growth Trajectory Tracking via Reinforcement Learning in Precision Aquaculture"
**Authors:** Abderrazak Chahid, Ibrahima N'Doye, John E. Majoris, Michael L. Berumen, Taous-Meriem Laleg-Kirati (KAUST)
**Year:** 2021 (Aquaculture journal; arXiv:2103.07251)

Formulates fish growth trajectory tracking as sampled-data optimal control using discrete MDPs. Two Q-learning algorithms learn optimal feeding policies across full life cycle (juveniles → market weight) using bioenergetic growth model for Nile tilapia.

**State space:** Fish weight, population density
**Action space:** Discrete feeding rates (0.1 to 1.0 relative feeding)
**Reward function:**
```
r(s,a) = -[(w(s) - w_d(t)) / w_d(t))^2 + λ * f^2]
```
Penalizes both weight deviation from target trajectory AND excess feeding.

**Results:**
- 1.7% tracking error (tank), 6.6% tracking error (floating cage)
- Only 707g feed consumption vs 2,488-3,480g for MPC/PID/Bang-Bang
- **Zero fish mortality** even under unionized ammonia (UIA) spikes
- MPC/PID/Bang-Bang all lost 1/10 fish

### 1.2 Model-Based vs Model-Free Comparison

**Paper:** "Model-based versus model-free feeding control and water quality monitoring for fish growth tracking in aquaculture systems"
**Authors:** Fahad Aljehani, Ibrahima N'Doye, Taous-Meriem Laleg-Kirati (KAUST)
**Year:** 2023

Most rigorous comparison between control paradigms:

| Approach | Feed Consumed | RMSE | Fish Mortality |
|----------|--------------|------|----------------|
| Bang-Bang | 3,480g | 27.7% | 1/10 fish |
| PID | 3,246g | 22.8% | 1/10 fish |
| MPC (feeding only) | 2,872g | 19.9% | 1/10 fish |
| **Q-Learning** | **707g** | **12.1%** | **0/10 fish** |
| MPC (integrated WQ) | 2,488g | 2.7% | 0/10 fish |

**Bioenergetic growth model used:**
```
dw/dt = [Ψ(f,T,DO) × v(UIA) × w^m] - [k(T) × w^n]
```
Where:
- Anabolism: `Ψ(f,T,DO) = h × ρ × f × b × (1-a) × τ(T) × σ(DO)` — captures feeding, temperature, dissolved oxygen
- Catabolism: `k(T) = k_min × exp[j(T - T_min)]` — maintenance metabolism
- Allometric exponents: m=0.67 (anabolism), n=0.81 (catabolism)

**Recommendation:** Authors propose "MPC ∩ RL synergy" — combining MPC's constraint satisfaction with Q-learning's robustness to environmental uncertainties.

### 1.3 DDPG for RAS Control (2025 Paper Series — 6 Papers)

A cluster of 6 related papers published in 2025 represents the most comprehensive RL work in aquaculture:

#### Paper 1: Core DDPG for RAS
"A deep deterministic policy gradient approach for optimizing feeding rates and water quality management in recirculating aquaculture systems"
- Published: Aquaculture International (2025)
- Algorithm: DDPG with continuous action space
- Dual-component reward structure; rewards converge to 250 units mean
- Outperforms MPC, PID, Bang-Bang in tracking accuracy, feed consumption, stability
- Tested at three scales: 1,000L (10 fish/m³), 10,000L (50 fish/m³), 50,000L (100 fish/m³)

#### Paper 2: Edge Computing DDPG
"Lightweight deep deterministic policy gradient for edge computing in recirculating aquaculture systems"
- Published: Scientific Reports (2025)
- Edge-DDPG: **85% computational complexity reduction**, maintaining 92% performance
- Compact neural networks + memory-efficient replay buffers for ARM-based edge devices
- Real-time response under 50ms constraint

#### Paper 3: Interpretable Decision Trees from DDPG
"Enhancing interpretability and explainability for fish farmers: decision tree approximation of DDPG for RAS control"
- Published: Aquaculture International (2025)
- Extracts decision tree rules from trained DDPG neural network
- 92.7% policy fidelity with only 5.8% performance drop
- Makes AI decisions transparent for non-technical fish farmers

#### Paper 4: Cloud + AWS IoT Integration
"Intelligent cloud-based RAS management: integration of DDPG reinforcement learning with AWS IoT"
- Published: Scientific Reports (2025)
- AWS IoT Core + AWS Greengrass for sensor connectivity and edge intelligence
- 99.97% IoT message delivery rate, 98.7% reliability in critical parameter control
- Failsafe operation during network disruptions up to 72 hours

#### Paper 5: Transfer Learning Across Species
"Enhanced transfer learning and federated intelligence for cross-species adaptability in intelligent RAS"
- Published: Aquaculture International (2025)
- Modular neural network separating general knowledge from species-specific knowledge
- Adapts to new species in **14 days** vs 45-60 day traditional retraining
- 87.3% of optimal performance with just 14 days of data
- 76% cost reduction; 23.5% collective improvement via federated learning across facilities
- ROI of 4-14 months

#### Paper 6: Multi-Objective Hierarchical DDPG
"Adaptive multi-objective reinforcement learning with interpretable visualization for integrated RAS management across growth cycles"
- Published: Aquaculture International (2025)
- Hierarchical DDPG with growth stage-specific policies
- Dynamically adjusts priorities (feeding efficiency vs. water quality vs. growth) across life stages
- Interpretable visualization framework for practitioners

### 1.4 RAG-LLM + Deep Q-Network Hybrid

**Paper:** "An integrating RAG-LLM and deep Q-network framework for intelligent fish control systems"
**Authors:** Danvirutai, Charoenwattanasak et al.
**Year:** 2025 (Scientific Reports)

Novel fusion of LLM reasoning with RL decision-making:
- IoT system: ESP-32 microcontroller with ammonia, DO, pH, turbidity, temperature sensors
- RAG-LLM provides expert knowledge retrieval from Q&A vector database
- DQN provides real-time adaptive control
- Ensemble learning via majority voting between DQN and RAG-LLM policies
- Results: 2% error rate, 1.8% fish growth improvement over expert-managed farms

### 1.5 RL for Disease Prediction

#### BIGRU-A3C for Mycobacteriosis
**Paper:** "Explainable deep reinforcement learning with BIGRU-A3C for early mycobacteriosis prediction in smart aquaculture"
**Year:** 2025 (Aquaculture International)
- Bidirectional GRU + Asynchronous Advantage Actor-Critic (A3C)
- Uses Mycobacteriosis Disease Water Quality Index (MWQI)
- Classifies water conditions from DO, pH, temperature, ammonia
- Addresses temporal dependencies in water quality

#### GR-DQN for White Spot Disease
**Paper:** "A Reinforcement Learning based Hybrid GR-DQN Model for Predicting Ichthyophthiriosis Disease in Aquaculture"
**Year:** 2025 (Procedia Computer Science)
- GRU + DQN hybrid
- RL rewards correct disease predictions
- Analyzes water quality state transitions

#### GIS + Multi-Armed Bandit for Virus Transmission
**Paper:** "An Integrated GIS-Based Reinforcement Learning Approach for Efficient Prediction of Disease Transmission in Aquaculture"
**Year:** 2023 (Information, MDPI)
- GIS + Multi-Armed Bandit (MAB) for virus transmission tracking
- Applied to Greek aquaculture regions
- MAB achieves 96% accuracy
- Calculates disease transmission intervals between cages

### 1.6 RL for Fish School Control

**Paper:** "Controlling Fish Schools via Reinforcement Learning of Virtual Fish Movement"
**Authors:** Yusuke Nishii, Hiroaki Kawashima (Kyoto University)
**Year:** 2026 (arXiv:2603.16384)
- Q-learning trains virtual fish displayed on screens to guide real fish schools
- State: 100 states (10×10 grid of real fish position × virtual fish position)
- Action: discrete cell displacements [-2, +2]
- Reward: [-1, +1] based on real fish proximity to target
- Tested with Rummy-nose tetras; statistically significant (p=1.46×10⁻⁵¹)

### 1.7 RL for Fish Survival Forecasting

**Paper:** "Deep reinforcement learning for forecasting fish survival in open aquaculture ecosystem"
**Year:** 2023 (Environmental Monitoring and Assessment)
- Q-learning + deep feed-forward neural networks
- Reduces reliance on labeled data through RL capability

### 1.8 Aquaculture Simulator with PPO

**Paper:** "Development of simulator for efficient aquaculture of Sillago japonica using reinforcement learning"
**Authors:** Haruki Kuroki, Hiroshi Ikeoka, Koichi Isawa
**Year:** 2020 (IEEE ICIP)
- Species: Sillago japonica (Japanese whiting)
- PPO for feeder control
- Image recognition + RL feeder control
- Custom aquaculture simulator

---

## 2. Model Predictive Control — The Primary Baseline

**Paper:** "Model Predictive Control Paradigms for Fish Growth Reference Tracking in Precision Aquaculture"
**Authors:** Chahid, N'Doye, Laleg-Kirati (KAUST)
**Year:** 2021 (Journal of Process Control)

Four MPC formulations:
1. Track desired growth trajectory while penalizing feed, temperature, DO
2. Directly optimize Feed Conversion Ratio (FCR)
3. Tradeoff between growth tracking, dynamic energy, and food cost
4. Maximize fish growth while minimizing costs

MPC advantage: predicts controlled variable behavior, optimizes over prediction horizon.
MPC limitation: requires accurate mathematical model (bioenergetic equations).

---

## 3. Existing RL Environments and Simulation Tools

### 3.1 gym_fishing (boettiger-lab) — Fisheries Management
- GitHub: https://github.com/boettiger-lab/gym_fishing
- `pip install gym_fishing`
- 4 environments: fishing-v0 (discrete), v1 (continuous), v2 (tipping points), v4 (parameter uncertainty)
- Logistic growth population dynamics
- Observation: fish biomass; Action: harvest quota
- **NOTE: This is for WILD fisheries management, NOT aquaculture farming**

### 3.2 Fish Gym (dongfangliu) — Physics-Based Swimming
- GitHub: https://github.com/dongfangliu/gym-fish
- Physics-based simulation for articulated underwater agents
- Environments: Cruising, Pose Control, Two-fish Schooling, Path Following
- First physics-based agent-fluid interaction environment
- Linux only

### 3.3 MARL-Aquarium — Predator-Prey
- GitHub: https://github.com/michaelkoelle/marl-aquarium
- PettingZoo multi-agent RL
- Predator: radius 30px, max velocity 5.0, FOV 150°, reward +10/catch
- Prey: radius 20px, max velocity 4.0, FOV 120°, reward +1/survival, -1000 if caught
- 16 discrete actions

### 3.4 FishMet Digital Twin (University of Bergen)
- Website: https://fishmet.uib.no/
- Authors: Sergey Budaev, Giovanni Cusimano, Ivar Ronnestad
- Year: 2025
- Mechanistic, process-based simulation at 1-second time steps
- Models: stomach processing (water uptake + transport), midgut digestion (Michaelis-Menten kinetics), energy budget (RE = DE - losses), appetite (3-component: stomach, midgut, energy), stress effects
- Species: Atlantic salmon, rainbow trout (validated)
- Interfaces: CLI, GUI, R, HTTP/iBOSS server

### 3.5 Fish Bioenergetics 4.0
- Website: http://fishbioenergetics.org/
- R-based, open-source (Wisconsin Model)
- 105 bioenergetic models for 72 aquatic species
- Shiny GUI
- Models energy intake, metabolism, growth from temperature, DO, salinity

### 3.6 Fish-PrFEQ (1998, Legacy)
- Authors: C. Young Cho, Dominique P. Bureau
- Predicts growth, energy/nitrogen/phosphorus retention, waste output
- Includes oxygen requirement module

### 3.7 AnyLogic RAS Model
- Agent-based simulation on AnyLogic Cloud
- Used for sea lice control modeling

### 3.8 Aqua-Sim NG (NS-3)
- Underwater sensor network simulator
- Acoustic communication, MAC/routing protocols, localization

### CRITICAL GAP: NO GYMNASIUM-COMPATIBLE AQUACULTURE FARMING ENVIRONMENT EXISTS
- gym_fishing = wild fisheries (population management), NOT farming
- Fish Gym = swimming physics, NOT farming operations
- MARL-Aquarium = predator-prey game, NOT farming
- FishMet = biological model, NOT RL environment
- **Building an OpenEnv-compatible aquaculture farming environment fills the #1 gap**

---

## 4. Reward Functions That Have Been Tried

| Paper/System | Reward Function | Components |
|---|---|---|
| Chahid et al. 2021 (Q-learning) | `r = -[(w-w_d)/w_d)² + λ×f²]` | Weight tracking error + feed penalty |
| Aljehani et al. 2023 | Same + water quality terms | Weight error + feed + temp + DO + UIA |
| DDPG-RAS 2025 | Dual-component (converges to 250) | Feeding efficiency + WQ stability |
| Multi-obj DDPG 2025 | Hierarchical, stage-adaptive | Stage-specific weights for feeding/WQ/growth |
| RAG-LLM + DQN 2025 | Majority voting ensemble | Correct action consensus |
| BIGRU-A3C 2025 | Disease classification accuracy | Correct MWQI classification |
| GR-DQN 2025 | Disease prediction accuracy | Correct disease detection |
| GIS-MAB 2023 | Transmission prediction | Correct interval prediction |
| Fish school 2026 | `r ∈ [-1, +1]` proximity | Centroid distance to target |
| MPC (baseline) | Cost function minimization | Growth tracking + feed cost + energy |

### Reward Design Principles from Literature:
1. **Multi-component**: Always combine growth tracking + resource efficiency + safety
2. **Quadratic penalty**: Weight deviation uses squared error for smooth gradients
3. **Regularization**: Feed consumption term (λ×f²) prevents overfeeding
4. **Safety constraint**: UIA/DO violations should trigger large negative reward
5. **Stage-adaptive**: Different life stages need different reward weightings
6. **Multi-objective**: Pareto front between growth, cost, welfare, environment

---

## 5. Water Quality Prediction (Deep Learning)

Key models providing observation space for RL:

| Model | Task | Key Feature |
|---|---|---|
| Enhanced LSTM | DO prediction | Adapts to changing conditions |
| Pearson-LSTM-AM | DO prediction (freshwater) | Attention-enhanced, feature selection |
| CNN-BiLSTM + attention | DO prediction | Spatial + temporal dependencies |
| GRU-N-Beats | DO forecasting | Hybrid architecture |
| PID error corrector + DL | DO + temperature | Control theory + deep learning |
| L-PIGRU | Fish growth prediction | 0.68% MAPE (vs 4.01% ARIMAX, 2.03% LSTM) |

---

## 6. Computer Vision for Fish Monitoring

| System | Task | Performance |
|---|---|---|
| YOLOv5 + SORT | Underwater fish tracking | 75.6 FPS |
| U-YOLOv7 | Underwater organism detection | Specialized architecture |
| ResNet34-CBAM | Tilapia feeding behavior | 99.72% accuracy |
| MIT 2026 system | River herring counting | 42,510 fish counted |
| Pacific deployment | Finfish species ID | 600+ species, 80K+ specimens |

Four CV domains in aquaculture:
1. **FDR** — Fish Detection/Recognition
2. **FBE** — Fish Biomass Estimation
3. **FBC** — Fish Behavior Classification
4. **FHA** — Fish Health Analysis

---

## 7. Autonomous and Robotic Systems

- **AGISTAR USV** — Solar-powered autonomous surface vessel for automated fish feeding
- **Multi-AUV MARL** — Cooperative underwater tracking with 30,000× speedup (GPU-accelerated)
- **SINTEF vision** — Fully unmanned fish farms with robotic inspection, cleaning, monitoring
- **3D tracking** — Monocular depth + acoustic sensing for UUV mapping
- Challenges: battery life, connectivity, implementation costs

---

## 8. AQUA LLM — Foundation Model for Aquaculture

**Paper:** "AQUA: A Large Language Model for Aquaculture & Fisheries"
**Year:** 2025 (arXiv:2507.20520)
- AQUA-7B: 7-billion parameter model fine-tuned on ~3 million QA pairs
- AQUADAPT framework: agentic data generation + automated evaluation
- First LLM purpose-built for aquaculture
- HuggingFace: KurmaAI/AQUA-7B
- Covers: species-specific farming, hatchery ops, water quality, disease management

---

## 9. State of the Art Summary

### What Works Now
- DDPG for continuous feeding rate control in RAS (superior to MPC/PID/Bang-Bang)
- Q-learning for discrete feeding optimization with bioenergetic growth models
- Edge deployment with 85% computation reduction
- Cross-species transfer learning (14 days vs 60 days)
- Cloud-IoT integration (AWS) with 99.97% reliability
- Disease prediction via RL (MAB, DQN, A3C variants)
- Digital twins (FishMet) for appetite/growth simulation

### GAPS — What Nobody Has Built
1. **No standardized RL environment for aquaculture farming** — unlike Atari/MuJoCo, no gym exists. THIS IS THE #1 GAP.
2. **No open-source aquaculture simulator for RL training** — FishMet is closest but not an RL environment
3. **Sim-to-real transfer untested** — no domain randomization or sim-to-real papers
4. **Safety/constraint RL not applied** — offline safe RL and constrained RL unexplored despite obvious need
5. **Multi-objective reward design is ad hoc** — no systematic study
6. **Limited species coverage** — mostly Nile tilapia and Atlantic salmon
7. **No long-term deployment data** — cloud/edge papers lack >1 year operational results
8. **PPO/SAC/TD3 largely untested** — nearly all work uses Q-learning or DDPG
9. **Shrimp/prawn farming** — almost no RL work despite being massive industry
10. **Fish welfare as explicit RL objective** — not in any reward function
11. **Multi-agent for multi-cage** — no MARL for multi-tank/cage coordination
12. **Scalability** — most work validated on single tanks; commercial scale (hundreds of tanks) unaddressed

---

## 10. Key Research Groups

| Group | Institution | Focus |
|---|---|---|
| Chahid, N'Doye, Laleg-Kirati | KAUST, Saudi Arabia | Q-learning, MPC, bioenergetic models (CORE group) |
| DDPG-RAS series authors | Unknown (6 papers, 2025) | DDPG, edge, cloud IoT, transfer learning |
| Budaev, Ronnestad | University of Bergen, Norway | FishMet digital twin |
| Danvirutai et al. | Thailand | RAG-LLM + DQN hybrid |
| Nishii, Kawashima | Kyoto University, Japan | Fish school control via RL |
| Kuroki, Ikeoka, Isawa | Japan | PPO-based aquaculture simulator |
| Narisetty et al. | KurmaAI | AQUA-7B LLM |
| Carl Boettiger lab | UC Berkeley | gym_fishing environment |

---

## 11. Complete Source List

### Core RL Papers
1. Chahid et al. (2021) — Q-learning fish growth tracking — [arXiv:2103.07251](https://arxiv.org/abs/2103.07251)
2. Aljehani et al. (2023) — Model-based vs model-free — [arXiv:2306.09915](https://arxiv.org/html/2306.09915)
3. Chahid et al. (2021) — MPC paradigms — [arXiv:2102.00004](https://arxiv.org/abs/2102.00004)
4. KAUST (2023) — Feeding control survey — [arXiv:2306.09920](https://arxiv.org/html/2306.09920)

### 2025 DDPG-RAS Series
5. DDPG for RAS feeding/WQ — [Springer](https://link.springer.com/article/10.1007/s10499-025-01914-z)
6. Edge-DDPG lightweight — [Nature SR](https://www.nature.com/articles/s41598-025-21677-0)
7. Decision tree from DDPG — [Springer](https://link.springer.com/article/10.1007/s10499-025-02325-w)
8. Cloud DDPG + AWS IoT — [Nature SR](https://www.nature.com/articles/s41598-025-33736-7)
9. Transfer learning cross-species — [Springer](https://link.springer.com/article/10.1007/s10499-025-02212-4)
10. Multi-objective DDPG — [Springer](https://link.springer.com/article/10.1007/s10499-025-02320-1)

### Disease Prediction
11. BIGRU-A3C mycobacteriosis — [Springer](https://link.springer.com/article/10.1007/s10499-025-02233-z)
12. GR-DQN ichthyophthiriosis — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050925013766)
13. GIS-MAB disease transmission — [MDPI](https://www.mdpi.com/2078-2489/14/11/583)

### Other RL
14. Fish school control (2026) — [arXiv:2603.16384](https://arxiv.org/html/2603.16384)
15. DRL fish survival — [Springer](https://link.springer.com/article/10.1007/s10661-023-11937-9)
16. Sillago japonica simulator — [IEEE](https://ieeexplore.ieee.org/abstract/document/9367369/)
17. RAG-LLM + DQN — [Nature SR](https://www.nature.com/articles/s41598-025-05892-3)

### Environments & Tools
18. gym_fishing — [GitHub](https://github.com/boettiger-lab/gym_fishing)
19. Fish Gym — [Docs](https://gym-fish.readthedocs.io/en/latest/)
20. MARL-Aquarium — [GitHub](https://github.com/michaelkoelle/marl-aquarium)
21. FishMet — [Website](https://fishmet.uib.no/) | [Manual](https://fishmet.uib.no/doc/manual.html)
22. Fish Bioenergetics 4.0 — [Paper](https://www.tandfonline.com/doi/full/10.1080/03632415.2017.1377558)
23. Fish-PrFEQ — [ResearchGate](https://www.researchgate.net/publication/248858239)
24. Aqua-Sim NG — [GitHub](https://github.com/rmartin5/aqua-sim-ng)

### Foundation Models & AI
25. AQUA-7B LLM — [arXiv:2507.20520](https://arxiv.org/abs/2507.20520)
26. L-PIGRU growth prediction — [MisPeces](https://www.mispeces.com/en/news/A-digital-twin-for-aquaculture-AI-predicts-fish-growth-in-real-time/)

### Reviews & Surveys
27. AI in aquaculture systematic review (2025) — [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/jwas.13107)
28. Generative AI in aquaculture (2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0144860925001268)
29. Deep learning for sustainable aquaculture (2025) — [MDPI](https://www.mdpi.com/2071-1050/17/11/5084)
30. Autonomous robotics editorial (2025) — [Frontiers](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1740881/full)
31. Digital twins in intensive aquaculture (2024) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S016816992400067X)
32. AI-driven aquaculture review (2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2589721725000182)
33. AI mortality prediction DSS (2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0144860925001104)
34. Smart Aquaculture IoT market — [FarmXpert](https://www.farmxpertgroup.com/2025/07/smart-aquaculture-2025-iot-ai.html)
35. Precision aquaculture IoT + CV — [arXiv:2409.08695](https://arxiv.org/html/2409.08695v2)

### Biological Models
36. Nile tilapia DEB model (2025) — [Springer](https://link.springer.com/article/10.1007/s10641-025-01675-x)
37. Bioenergetic growth model — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.18.706619v1.full)
38. FishMet digital twin (2025) — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/aff2.70064)

### Robotics
39. Multi-AUV cooperative MARL — [arXiv:2404.13654](https://arxiv.org/html/2404.13654v3)
40. AnyLogic RAS model — [AnyLogic Cloud](https://cloud.anylogic.com/model/8064c0b1-717d-4fa9-ac12-79812dd31734)
