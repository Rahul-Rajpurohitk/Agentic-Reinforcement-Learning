# Aquaculture Control Systems, Mathematical Models & Simulation Approaches
## Comprehensive Research for OpenEnv Environment Design

---

# Table of Contents
1. [Fish Growth Models](#1-fish-growth-models)
2. [Water Quality Dynamics](#2-water-quality-dynamics)
3. [Feeding Optimization](#3-feeding-optimization)
4. [Disease Transmission Models](#4-disease-transmission-models)
5. [Population Dynamics & Mortality](#5-population-dynamics--mortality)
6. [Economic / Bioeconomic Models](#6-economic--bioeconomic-models)
7. [Control Systems (PID, MPC, Bang-Bang)](#7-control-systems)
8. [MDP / Reinforcement Learning Formulations](#8-mdp--reinforcement-learning-formulations)
9. [Stochastic Models & Uncertainty](#9-stochastic-models--uncertainty)
10. [Multi-Objective Optimization](#10-multi-objective-optimization)
11. [Established Simulation Tools (AQUATOX, etc.)](#11-established-simulation-tools)
12. [Thermal / Pond Models](#12-thermal--pond-models)
13. [Agent-Based & Discrete Event Simulation](#13-agent-based--discrete-event-simulation)
14. [Dynamic Energy Budget (DEB) Theory](#14-dynamic-energy-budget-deb-theory)
15. [Unified OpenEnv State-Action-Reward Specification](#15-unified-openenv-state-action-reward-specification)

---

# 1. Fish Growth Models

## 1.1 Scope-for-Growth (Bioenergetic) Model

The foundational model for fish growth in aquaculture. Growth = anabolism - catabolism.

### Core Equation (von Bertalanffy generalized)

```
dW/dt = H * W^m  -  k(T) * W^n
```

Where:
- `W` = fish weight (g)
- `H` = anabolism coefficient, function of food, temperature, environment
- `k(T)` = catabolism coefficient (temperature-dependent)
- `m` = anabolism exponent (typically 0.63-0.72)
- `n` = catabolism exponent (typically 0.53-0.84)

### Expanded Anabolism (FAO Model)

```
H = a * b * R
R = h * f * E * W^m
E = pi * tau(T)
```

Where:
- `a` = fraction of assimilated food used for feeding catabolism (0-1)
- `b` = food assimilation efficiency (0-1)
- `R` = daily ration (g/d)
- `h` = food consumption coefficient
- `f` = food availability ratio (0-1), relative feeding level
- `pi` = photoperiod scalar (photoperiod_hours / 12)
- `tau(T)` = temperature scalar

### Full Bioenergetic Growth (from arxiv:2306.09915)

```
dw/dt = Psi(f, T, DO) * v(UIA) * w^m  -  k(T) * w^n
```

Where:
```
Psi(f, T, DO) = h * rho * f * b * (1-a) * tau(T) * sigma(DO)
k(T) = k_min * exp(j * (T - T_min))
```

### Parameters by Species (FAO calibration data):

| Parameter | Unit | Nile Tilapia | Tambaqui | Pacu | Common Carp |
|-----------|------|-------------|----------|------|-------------|
| b (assimilation) | - | 0.7108 | 0.6695 | 0.7719 | 0.7129 |
| m (anabolism exp) | - | 0.6277 | 0.6855 | 0.7154 | 0.6722 |
| h (consumption coeff) | - | 0.4768 | 0.2863 | 0.2415 | 0.3282 |
| a (feeding catabolism) | - | 0.0559 | 0.1057 | 0.0529 | 0.0786 |
| n (catabolism exp) | - | 0.8373 | 0.5336 | 0.5332 | 0.5166 |
| k_min | g^(1-n)/d | 0.0104 | 0.0146 | 0.0094 | 0.0104 |
| s (temp sensitivity) | 1/C | 0.0288 | 0.0110 | 0.0290 | 0.0027 |
| T_min | C | 18.7 | 14.4 | 17.5 | 10.1 |
| T_max | C | 39.7 | 38.6 | 31.4 | 36.2 |
| T_opt | C | 32.4 | 29.0 | 28.1 | 30.6 |

Source: [FAO Annex 3 Fish Growth Model](https://www.fao.org/4/W5268E/W5268E09.htm)

## 1.2 Environmental Response Functions

### Temperature Factor (Bell-shaped)

```
tau(T) = exp{-4.6 * ((T_opt - T) / (T_opt - T_min))^4}   if T < T_opt
tau(T) = exp{-4.6 * ((T - T_opt) / (T_max - T_opt))^4}   if T >= T_opt
```

### Dissolved Oxygen Factor (Piecewise linear)

```
sigma(DO) = 1.0                              if DO > DO_crit
sigma(DO) = (DO - DO_min) / (DO_crit - DO_min)  if DO_min <= DO <= DO_crit
sigma(DO) = 0.0                              if DO < DO_min
```

**Parameters (Tilapia):**
- DO_crit = 5.0 mg/L (growth reduction begins)
- DO_min = 3.0 mg/L (growth ceases)

### Unionized Ammonia Factor

```
v(UIA) = 1.0                                     if UIA < UIA_crit
v(UIA) = (UIA_max - UIA) / (UIA_max - UIA_crit)  if UIA_crit <= UIA <= UIA_max
v(UIA) = 0.0                                     if UIA > UIA_max
```

**Parameters:**
- UIA_crit = 0.025-0.06 mg/L (sublethal effects begin)
- UIA_max = 0.6-1.4 mg/L (lethal)

### Catabolism Temperature Dependence

```
k(T) = k_min * exp(s * (T - T_min))
```

Source: [arxiv:2306.09915](https://arxiv.org/html/2306.09915)

## 1.3 Von Bertalanffy Growth Function (VBGF)

### Length form
```
L(t) = L_inf * (1 - exp(-K * (t - t_0)))
```

### Weight form
```
W(t) = W_inf * (1 - exp(-K * (t - t_0)))^b
```

Where:
- `L_inf` = asymptotic maximum length (cm)
- `K` = growth rate coefficient (1/year)
- `t_0` = theoretical age at zero length (year)
- `b` = allometric exponent in W = a*L^b (typically ~3.0)
- `W_inf` = asymptotic maximum weight (g)

**Typical values:**
- Tilapia: L_inf ~ 35-50 cm, K ~ 0.3-0.6/yr
- Salmon: L_inf ~ 80-120 cm, K ~ 0.1-0.3/yr
- Catfish: L_inf ~ 60-90 cm, K ~ 0.2-0.4/yr

### Temperature-modified VBGF

Growth rate K can be modified by temperature using Q10:
```
K(T) = K_ref * Q10^((T - T_ref) / 10)
```

Where Q10 typically ranges 2.0-3.0 for fish metabolism.

## 1.4 Specific Growth Rate (SGR)

```
SGR = (ln(W_t) - ln(W_0)) * 100 / t
```

Units: % body weight/day. Typical values:
- Tilapia grow-out: 3-4%/day
- Salmon grow-out: 1.5-2.0%/day

## 1.5 Feed Conversion Ratio (FCR)

```
FCR = Total feed consumed (kg) / Total weight gain (kg)
```

Good values: 1.0-1.75 (species dependent)
- Tilapia: 1.2-1.8
- Salmon: 1.0-1.3
- Catfish: 1.5-2.2

## 1.6 Wisconsin Bioenergetics Energy Balance

```
G = C - (R + SDA + F + U)
```

Where:
- `G` = growth (cal/g/d)
- `C` = consumption = C_max * p * f(T)
- `R` = respiration (standard + active + SDA)
- `SDA` = specific dynamic action (cost of digestion, ~15% of C)
- `F` = egestion (fecal losses, ~10-20% of C)
- `U` = excretion (urinary/branchial, ~5-10% of C)

**Consumption model:**
```
C = C_max * p * f(T) * W^(-0.3)
```
where p = realized fraction of C_max (0-1), f(T) = temperature function

**Respiration model:**
```
R = R_a * W^R_b * e^(R_Q * T) * ACT
```
where R_a, R_b, R_Q are species parameters, ACT = activity multiplier

Source: [Fish Bioenergetics 4.0](https://www.tandfonline.com/doi/full/10.1080/03632415.2017.1377558)

---

# 2. Water Quality Dynamics

## 2.1 Dissolved Oxygen (DO) Model

### Mass Balance Equation

```
dDO/dt = P_photo - R_phyto - R_fish - R_water - R_sediment + K_a * (DO_sat - DO) + A_mech
```

Where:
- `P_photo` = photosynthetic oxygen production (mg O2/L/h)
- `R_phyto` = phytoplankton respiration = 0.10 * P_photo * 1.08^(T-20)
- `R_fish` = fish respiration (mg O2/L/h)
- `R_water` = water column microbial respiration (mg O2/L/h)
- `R_sediment` = sediment oxygen demand (mg O2/L/h)
- `K_a` = reaeration coefficient (1/h)
- `DO_sat` = saturation DO concentration (mg/L)
- `A_mech` = mechanical aeration input (mg O2/L/h)

### Fish Respiration Rate

```
FR = 10^X * 1000   (mg O2 / kg fish / h)

X = 0.40 + 0.016*T - 0.0006*T^2 - 0.016*ln(W)
```

Where T = temperature (C), W = fish weight (g)

**Alternatively (from Tilapia farm model):**
```
FR = 2014.45 + 2.75*W - 165.2*T + 0.007*W^2 + 3.93*T^2 - 0.21*W*T
```
Units: mg O2/kg fish/h; R^2 = 0.99; Valid range: 20-200g fish, 24-32C

### DO Consumption via Fish Respiration

```
DO_fish = FR * SD / 1000
```
Where SD = stocking density (kg/m^3)

### Photosynthetic Production (Smith-Talling Model)

```
P = (alpha * I_k * P_max) / sqrt(alpha^2 * I^2 + P_max^2) * (I_z / I_0)
```

Light attenuation (Beer-Lambert):
```
I_z = I_0 * exp(-K_ext * z)
```
Where K_ext = extinction coefficient (m^-1)

### DO Saturation (temperature-dependent)

```
DO_sat(T) = 468 / (31.6 + T)   [approximate, mg/L, freshwater]
```

More precise values:
- 25C freshwater: 8.24 mg/L
- 30C freshwater: 7.54 mg/L
- 35C freshwater: 6.94 mg/L

### Nitrification Oxygen Demand

```
DO_nitrif = 4.57 * K_NR * N_r / V
K_NR = 0.11 * 1.08^(T-20)
N_r = (0.03 * Fr * W * NF) / (24 * 1000)
```
- 4.57 g O2 consumed per g TAN oxidized to nitrate
- K_NR = nitrification rate coefficient
- N_r = ammonia production rate (g TAN/h)

### Nighttime Oxygen Budget (Catfish pond, 15,000 kg/ha)

| Component | O2 (kg/ha/12h) |
|-----------|----------------|
| Fish respiration | 60 |
| Water column organisms | 120 |
| Sediment organisms | 42 |
| **Total demand** | **222** |
| Water column at dusk | 100 |
| Atmospheric diffusion | 10 |
| **Total supply** | **110** |
| **Aeration needed** | **112** |

Source: [Scientific Reports - Tilapia DO model](https://pmc.ncbi.nlm.nih.gov/articles/PMC8677810/), [DO Fishpond Model](https://scialert.net/fulltext/?doi=ijar.2008.83.97)

## 2.2 Ammonia / Total Ammonia Nitrogen (TAN) Model

### TAN Production Rate

```
TAN_produced (kg/d) = Feed_rate (kg/d) * Protein% * 0.16 * N_wasted% * 1.2
```

Where:
- Protein%: typically 0.30-0.45 (30-45% protein in feed)
- 0.16: g nitrogen per g protein (protein is ~16% N)
- N_wasted%: fraction of consumed N not retained (typically 0.50-0.80)
- 1.2: conversion factor g TAN per g N

**Example:** 60 kg feed/d * 0.40 * 0.16 * 0.50 * 1.2 = 2.3 kg TAN/d

### TAN Mass Balance in Tank

```
dTAN/dt = TAN_excretion + TAN_feed_decay - TAN_nitrification - TAN_water_exchange - TAN_volatilization
```

**Simplified for RAS:**
```
dTAN/dt = P_TAN - Q_biofilter * TAN * eta_nitrif - Q_exchange * TAN / V
```

Where:
- P_TAN = fish excretion rate (g TAN/h)
- Q_biofilter = biofilter flow rate (m^3/h)
- eta_nitrif = biofilter removal efficiency
- Q_exchange = water exchange flow (m^3/h)
- V = tank volume (m^3)

### Unionized Ammonia Fraction (UIA)

```
UIA = TAN * F(pH, T)
```

Multiplication factor F depends on pH and temperature:
- pH 7.0, 20C: F = 0.0039
- pH 8.0, 20C: F = 0.0381
- pH 9.0, 20C: F = 0.2836
- pH 7.0, 30C: F = 0.0082
- pH 8.0, 30C: F = 0.0783

**Toxicity thresholds (UIA in mg/L):**
- Chronic effects begin: 0.025-0.05
- Tissue damage: > 0.05
- LC50 warmwater fish: 0.7-3.0
- LC50 coldwater fish: 0.3-0.9
- Lethal: 2.0

### Biofilter Sizing

```
V_biofilter (m^3) = TAN_production (g/d) / VTR (g TAN/m^3/d)
```

Volumetric TAN Removal Rates:
- Trickling filter: 90 g TAN/m^3/d
- Moving-bed reactor: 350 g TAN/m^3/d
- Rotating biological contactor: 442 g TAN/m^2/d

### Alkalinity Consumption

For every 1 g NH3-N converted to NO3-N:
- 7.14 g alkalinity as CaCO3 consumed
- ~0.25 kg baking soda per kg feed

Source: [Global Seafood Alliance - Biofilter Sizing](https://www.globalseafood.org/advocate/estimating-biofilter-size-for-ras-systems/)

## 2.3 Oxygen Consumption Rates by Species

| Species | O2 Rate (mg/kg/h) | Temperature (C) | Notes |
|---------|-------------------|-----------------|-------|
| Salmon (juvenile) | 83-400 | varied | Starved to fed |
| Salmon (average) | 245.5 | varied | Mean |
| Tilapia | 31-75 g O2/m^3/h | 24-32 | RAS system |
| Goldfish | 90.5 | baseline | Mesor |
| Q10 general fish | 2.0-5.0 | - | Doubles per 10C |

---

# 3. Feeding Optimization

## 3.1 Model Predictive Control for Feeding (MPC)

### Feeding-Only Objective (MPC1)

```
min J = integral[t_k to t_k+N] [ ||( w_hat(t) - w_d(t) ) / w_d(t)||^2  +  lambda * ||f(t)||^2 ] dt
```

Subject to:
```
dw_hat/dt = g(w_hat(t), f(t))          # growth model
f_min <= f(t) <= f_max                  # feeding bounds
w_0 <= w_hat(t) <= w_end               # weight bounds
w_hat(t_k) = w(t_k)                    # initial condition
```

**Parameters:**
- N = 6 (prediction horizon, days)
- lambda = 0.002 (feeding regularization)
- f_min = 0.1, f_max = 1.0 (fraction of C_max)

### Joint Feeding + Water Quality (MPC2)

```
min J = integral[ ||(w - w_d)/w_d||^2 + lambda_1*||f||^2 + lambda_2*||T - T_d||^2 + lambda_3*||DO - DO_d||^2 + lambda_4*||(UIA - UIA_d)/UIA_d||^2 ] dt
```

**Parameters:**
- N = 5 (prediction horizon)
- lambda_1 = 0.001 (feeding weight)
- lambda_2 = 0.2 (temperature weight)
- lambda_3 = 0.5 (DO weight)
- lambda_4 = 0.5 (ammonia weight)

### Optimal Feeding Trajectory

Research shows optimal feeding follows a pattern:
- Continuous decline from stocking to minimum value
- Slight increase toward harvest size
- Marginal revenue of feeding = marginal cost at harvest

Source: [arxiv:2306.09915](https://arxiv.org/html/2306.09915)

## 3.2 Daily Feeding Amount Calculation

```
Feed (kg/d) = (Fr * W_n * N_fish) / 100,000
```

Where:
```
Fr = 17.02 * exp(ln(W) + 1.142) - 19.52   [% body weight/day]
```

---

# 4. Disease Transmission Models

## 4.1 SIR Model for Fish Disease

### Differential Equations

```
dS/dt = -beta * S * I / N  +  mu * N  -  mu * S
dI/dt =  beta * S * I / N  -  gamma * I  -  alpha * I  -  mu * I
dR/dt =  gamma * I  -  mu * R
```

Where:
- S = susceptible fish count
- I = infected fish count
- R = recovered/removed fish count
- N = S + I + R = total population
- beta = transmission coefficient
- gamma = recovery rate (1/infectious period)
- alpha = disease-induced mortality rate
- mu = natural mortality rate

### Basic Reproduction Number

```
R_0 = beta / (gamma + alpha + mu)
```

R_0 > 1: epidemic, R_0 < 1: disease dies out

### SEIR Extension (with exposed/latent period)

```
dS/dt = -beta * S * I / N
dE/dt =  beta * S * I / N  -  sigma * E
dI/dt =  sigma * E  -  gamma * I  -  alpha * I
dR/dt =  gamma * I
```

Where sigma = 1/latent_period

## 4.2 Fish Disease Parameters (Furunculosis in Salmon)

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Transmission coeff | beta | 0.01-0.40 | 1/d | Density-dependent |
| Latent period | 1/sigma | 5 | days | SEIR |
| Recovery period | 1/gamma | 10 | days | |
| Disease mortality | alpha | 0.02-0.44 | fish/d | Density-dependent |
| Natural mortality | mu | 0.0001-0.005 | fish/d | Background |
| Daily mortality rate | | 5% | | At peak infection |

### Density-Dependent Transmission (Chinook salmon, furunculosis)

| Fish Density (fish/L) | beta | Disease Mortality (fish/d) |
|----------------------|-------|---------------------------|
| 9.13 | 0.01 | 0.42 |
| 4.56 | 0.019 | 0.44 |
| 0.72 | 0.0051 | 0.18 |
| 0.36 | 0.0076 | 0.02 |
| 0.19 | 0.0001 | ~0 |

### Vaccination Impact (from SEIR simulation)

| Immunized % | Estimated Mortality (fish) |
|-------------|--------------------------|
| 0% | 42,900 |
| 50% | 12,500 |
| 80% | 900 |
| 95% | 500 |

### Mortality Model (Logistic function of UIA)

```
k_1(UIA) = Z / (1 + exp{-beta_m * (UIA - eta)})
```

**Parameters:**
- Z = 99.41
- beta_m = 10.36
- eta = 0.80

### Spatial Transmission (between farms)

Distance-dependent transmission uses scaling parameter:
```
P_transmission ~ distance^(ScalingInf)
```
- High transmission: ScalingInf = -1.0
- Moderate: ScalingInf = -1.8
- Lower: ScalingInf = -2.6

Source: [PMC Aquaculture Epidemic](https://pmc.ncbi.nlm.nih.gov/articles/PMC10527373/), [ResearchGate Fish Disease](https://www.researchgate.net/publication/230801108)

---

# 5. Population Dynamics & Mortality

## 5.1 Population Model

```
dN/dt = -M(t) * N(t)  +  S(t)  -  H(t)
```

Where:
- N(t) = fish population count
- M(t) = total mortality rate (natural + disease)
- S(t) = stocking rate (fish/day)
- H(t) = harvest rate (fish/day)

### Size-Dependent Natural Mortality

Natural mortality is inversely related to fish length:
```
M_nat = a_M * L^(-b_M)
```

### Density-Dependent Growth

At high stocking densities, growth rates decrease:
```
g(N) = g_max * (1 - N/K_pop)
```

## 5.2 Exponential Mortality (standard aquaculture)

```
N(t) = N_0 * exp(-m * t)
```

**Parameters (salmon farming):**
- m = 0.10 (10% total mortality over production cycle)
- N_0 = 10,000 fish initial stocking

## 5.3 Population State (discrete time, integer)

```
p(t+1) = p(t) + stocking_rate - INT(p(t) * k_1(UIA))
```

---

# 6. Economic / Bioeconomic Models

## 6.1 Profit Maximization

### Objective Function

```
max Pi = P_fish * B(T_h) - C_feed(T_h) - C_harvest(T_h) - C_fixed * T_h
```

Where:
- Pi = total profit
- P_fish = market price per kg (potentially weight/size dependent)
- B(T_h) = biomass at harvest time T_h
- C_feed = cumulative feed cost
- C_harvest = harvesting cost
- C_fixed = fixed operating cost per day
- T_h = harvest time (decision variable)

### Optimal Harvest Condition

At optimum, marginal revenue of continuing = marginal cost + discount:
```
d/dt [P * B(t)] = r * [P * B(t) - C_cum(t)] + dC/dt
```
Where r = discount rate

## 6.2 Stochastic Aquaculture Valuation (Real Options)

### Fish Growth (Bertalanffy)
```
w(t) = w_inf * (a - b*exp(-c*t))^3
```
Parameters (salmon): a = 1.113, b = 1.097, c = 1.43, w_inf = 6 kg

### Biomass
```
B(t) = N(t) * w(t) = N_0 * exp(-m*t) * w(t)
```

### Salmon Spot Price (Schwartz 2-factor model)
```
dS/S = (r - delta) dt + sigma_1 dW_1
d_delta = kappa*(alpha - delta) dt + sigma_2 dW_2
corr(dW_1, dW_2) = rho
```

**Calibrated parameters (salmon):**
- sigma_1 = 0.23 (spot volatility)
- sigma_2 = 0.75 (convenience yield vol)
- kappa = 2.6 (mean reversion speed)
- alpha = 0.02 (long-term mean)
- rho = 0.9 (correlation)
- r = 0.0303 (risk-free rate)

### Feed Cost (stochastic, soybean-linked)
```
F_t = F_0 * S_t^soy / S_0^soy
```
where F_0 = 0.25 * PC (NOK/kg/year), c_FCR = 1.1 (kg feed/kg fish)

### Optimal Stopping Problem
```
W_0 = sup_tau E^Q[ exp(-r*tau) * (S_tau * B(tau) - C_harvest(tau)) - C_feed_cumulative(tau) ]
```

State vector: x = (S_salmon, delta_salmon, S_soy, delta_soy) -- 4D

### Harvesting Cost
```
C_harvest(t) = H_0 * B(t),   H_0 = 0.1 * PC (NOK/kg)
```

### Monte Carlo: 100,000 simulation paths
Solved via Longstaff-Schwartz LSMC or deep neural network classifier.

**Key result:** Stochastic feeding costs improve harvest timing by up to 11.6% value.

Source: [arxiv:2309.02970](https://ar5iv.labs.arxiv.org/html/2309.02970)

## 6.3 Feed Cost Fraction

Feed typically represents 30-60% of total aquaculture operating costs, making feed optimization the single highest-impact economic lever.

---

# 7. Control Systems

## 7.1 PID Controller for Feeding

```
f(t) = K_p * e(t) + K_i * integral(e) dt + K_d * de/dt
```

Where e(t) = w_d(t) - w(t) (tracking error from desired growth trajectory)

**Tuned gains (from arxiv:2306.09915):**
- K_p = 0.1
- K_i = 12
- K_d = 0.01

## 7.2 Bang-Bang Controller

```
f = 1.0   if (w_d - w) > 0   (underfed, increase to max)
f = 0.1   if (w_d - w) <= 0  (overfed, reduce to maintenance)
```

## 7.3 PID for Dissolved Oxygen

DO PID controllers in RAS:
- **Proportional**: adjusts aeration based on DO deviation from setpoint
- **Integral**: eliminates steady-state DO offset
- **Derivative**: predicts DO trends, prevents overshoot

Advanced: DE-RBF-PID (Differential Evolution optimized Radial Basis Function Neural Network PID)
- First stage: find optimal initial PID gains via differential evolution
- Second stage: RBF neural network adjusts PID online

**DO setpoint:** typically 5-7 mg/L
**Actuator:** aeration blower speed / valve position

### DO Control Performance
- Control accuracy achievable: +/- 0.25 mg/L
- pH control accuracy: +/- 0.23

Source: [ScienceDirect PID RAS](https://www.sciencedirect.com/science/article/abs/pii/S1537511021001197)

## 7.4 Fuzzy PID Control

Variable universe fuzzy PID with cascade architecture:
- Outer loop: DO concentration setpoint tracking
- Inner loop: aeration actuator control
- Fuzzy rules: adjust PID gains based on error magnitude and rate

## 7.5 Performance Comparison (10-fish population)

| Controller | Mortality | RMSE (growth) | Feed Used (g) |
|-----------|-----------|---------------|---------------|
| Bang-Bang | 0/10 | 14.1% | 3479.7 |
| PID | 0/10 | 15.6% | 3245.5 |
| MPC (feed only) | 0/10 | 14.9% | 2871.9 |
| Q-Learning | 0/10 | **11.7%** | **718.3** |

Q-Learning achieves 79% less feed consumption with better growth tracking.

---

# 8. MDP / Reinforcement Learning Formulations

## 8.1 Fish Growth as MDP

### State Space

```
s_t = (w_t, T_t, DO_t, UIA_t, t)
```

Or extended:
```
s_t = (w_t, N_t, TAN_t, DO_t, T_t, pH_t, day_of_year)
```

### Action Space

```
a_t = f_t in [0, 1]   (feeding rate as fraction of C_max)
```

Or multi-dimensional:
```
a_t = (f_t, aeration_t, water_exchange_t, harvest_decision_t)
```

### Transition Dynamics

```
w_{t+1} = w_t + dw/dt * dt    (via bioenergetic growth model)
DO_{t+1} = DO_t + dDO/dt * dt  (via DO mass balance)
TAN_{t+1} = TAN_t + dTAN/dt * dt (via ammonia balance)
```

### Reward Function

```
r_t(s_t, a_t) = -[ ((w(s_t) - w_d(t)) / w_d(t))^2  +  lambda * f^2 ]
```

Or multi-objective:
```
r_t = -[ w_growth_error^2 + lambda_1*f^2 + lambda_2*(T-T_d)^2 + lambda_3*(DO-DO_d)^2 + lambda_4*(UIA/UIA_d)^2 ]
```

### Q-Learning Update

```
Q(s_t, a_t) <- Q(s_t, a_t) + alpha * [ r_t + gamma * max_a' Q(s_{t+1}, a') - Q(s_t, a_t) ]
```

**Hyperparameters:**
- alpha: learning rate
- gamma: discount factor (0.95-0.99)
- lambda: feeding regularization (varies by case)

### Policy
```
a* = argmax_a' Q(s, a')
```

Source: [ScienceDirect Q-Learning Aquaculture](https://www.sciencedirect.com/science/article/pii/S0044848621015015)

## 8.2 DDPG for RAS Control

Deep Deterministic Policy Gradient for continuous action spaces:

**State:** (DO, T, pH, TAN, NO2, NO3, fish_weight, biomass, day)
**Actions:** (feeding_rate, aeration_level, water_exchange_rate) -- continuous

**Reward function (hierarchical by growth stage):**
```
r = w_1 * growth_reward + w_2 * water_quality_reward + w_3 * efficiency_reward
```

Dynamic weights adjust importance across:
- Juvenile phase: prioritize water quality stability
- Grow-out phase: prioritize feeding efficiency
- Pre-harvest: prioritize growth rate

**Performance:** DDPG achieves 85% computational reduction vs full model while maintaining 92% performance.

Source: [Springer DDPG RAS](https://link.springer.com/article/10.1007/s10499-025-01914-z)

## 8.3 Gym Fishing Environment (OpenAI Gym)

### Environment: `fishing-v0`

**Growth model (logistic):**
```
X_{t+1} = X_t + r * X_t * (1 - X_t/K) - harvest_t + noise
noise ~ N(0, sigma^2)
```

**State:** fish population biomass X_t
**Action:** harvest quota (discrete: 0 to n_actions)
```
quota = action / n_actions * K
```
**Reward:** harvest value (quota taken)

**Parameters:**
- r: intrinsic growth rate
- K: carrying capacity
- sigma: process noise

**Optimal policy:** Constant escapement (Reed 1979)

**Variant fishing-v2:** Includes tipping point dynamics.

Source: [GitHub gym_fishing](https://github.com/boettiger-lab/gym_fishing)

---

# 9. Stochastic Models & Uncertainty

## 9.1 Sources of Stochasticity in Aquaculture

1. **Growth uncertainty**: individual variation in growth rates
2. **Mortality uncertainty**: random disease events, environmental stress
3. **Price uncertainty**: market price fluctuations
4. **Feed cost uncertainty**: commodity price volatility
5. **Environmental uncertainty**: temperature, weather, water quality
6. **Biological uncertainty**: feed conversion efficiency variation

## 9.2 Stochastic Growth Model

```
dW = [H*W^m - k*W^n] dt + sigma_W * W dZ
```

Where sigma_W = growth volatility, dZ = Wiener process

## 9.3 Stochastic Mortality

```
dN = -m*N dt + sigma_N * N dZ_N
```

With jump processes for disease events:
```
dN = -m*N dt - J*N dQ
```
Where dQ = Poisson process, J = jump size (fraction killed)

## 9.4 Monte Carlo Simulation

Standard approach: 100,000 paths with backward induction.

Three key stochastic variables in bioeconomic models:
1. Survival rate (random)
2. Individual growth rate (random)
3. Feed consumption (random)

## 9.5 Least-Squares Monte Carlo (LSMC)

For optimal stopping (harvest timing):
- Regress continuation value on polynomial basis functions
- Compare with immediate exercise value
- Problem: dimensionality curse when adding stochastic growth + mortality + price

**Alternative:** Deep neural network classifier (2-4 seconds training vs LSMC)

---

# 10. Multi-Objective Optimization

## 10.1 Pareto Framework for Aquaculture

### Objectives (typically conflicting):

1. **Maximize profit**: max sum(P_fish * harvest_weight - C_feed - C_ops)
2. **Minimize environmental impact**: min (TAN_discharge + feed_waste + chemical_use)
3. **Maximize product quality**: max (fish_size_uniformity + flesh_quality)
4. **Minimize resource use**: min (water_use + energy_use + land_use)

### Multi-Objective Formulation

```
min F(x) = [f_1(x), f_2(x), ..., f_k(x)]
```
subject to:
```
g_i(x) <= 0,  i = 1,...,m  (inequality constraints)
h_j(x) = 0,   j = 1,...,p  (equality constraints)
x_min <= x <= x_max
```

### Solution Methods Used in Aquaculture

- **NSGA-II** (Non-dominated Sorting Genetic Algorithm)
- **Particle Swarm Optimization (PSO)**
- **Multi-objective RL**: Dynamic reward weights across growth stages
- **Weighted sum scalarization**: r = w_1*f_1 + w_2*f_2 + ... + w_k*f_k

### Adaptive Multi-Objective RL (recent work)

Hierarchical DDPG with stage-dependent reward weighting:
- Stage 1 (juvenile): water quality stability dominates
- Stage 2 (grow-out): feeding efficiency dominates
- Stage 3 (pre-harvest): growth rate and biomass dominate

Source: [Springer Multi-Objective RL](https://link.springer.com/article/10.1007/s10499-025-02320-1)

---

# 11. Established Simulation Tools

## 11.1 AQUATOX (EPA)

- Process-based aquatic ecosystem model
- State variables: fish populations, invertebrates, aquatic plants, nutrients, organic chemicals
- Uses differential equations with 1-day time step
- Tracks organic matter with user-settable stoichiometry
- Includes allometric bioenergetics for invertebrates
- Size-class modeling (oysters, crabs)
- Release 3.2 (latest), SQLite parameter databases

Source: [EPA AQUATOX](https://www.epa.gov/hydrowq/aquatox)

## 11.2 WASP (Water Analysis Simulation Program)

EPA model for water quality including:
- Nutrient cycling (N, P)
- Dissolved oxygen dynamics
- Eutrophication
- Temperature corrections using Q10 factors (1.047-1.072)
- Algal growth via Monod kinetics

## 11.3 Fish Bioenergetics 4.0

R-based modeling application:
- 73 fish species, 105 published models
- Species-specific parameter files
- Temperature-dependent consumption and respiration
- Growth estimation from field data

Source: [Fish Bioenergetics 4.0](https://www.tandfonline.com/doi/full/10.1080/03632415.2017.1377558)

## 11.4 DTU-DADS-Aqua

Stochastic spatiotemporal hybrid simulation:
- Combines compartmental models + agent-based models
- Farm-site hyperconnectivity based on distance
- Used for aquaculture disease spread assessment

---

# 12. Thermal / Pond Models

## 12.1 Pond Heat Balance

### Energy Balance Equation

```
dT_w/dt = (Q_solar + Q_sky - Q_rad - Q_evap - Q_conv + Q_sed + Q_mech) / (rho * c_p * V)
```

Where:
- Q_solar = solar radiation absorbed (41% of total heat input)
- Q_sky = longwave sky radiation (33%)
- Q_rad = pond surface radiation (21%)
- Q_evap = evaporative heat loss
- Q_conv = convective heat exchange
- rho = water density (kg/m^3)
- c_p = specific heat of water (4186 J/kg/C)
- V = pond volume (m^3)

### Thermal Stratification Model

Water column discretized into horizontal volume elements:
```
dT_i/dt = (K_turb / dz^2) * (T_{i-1} - 2*T_i + T_{i+1}) + Q_net_i / (rho * c_p * dz)
```

Where K_turb = turbulent diffusion coefficient, function of:
- Wind speed
- Depth
- Density gradient (temperature dependent)

### Key Observations

- Stratification peaks 14:00-16:00 daily
- Surface-bottom temp difference: 1.3-4.0C (0.8-2.0m deep ponds)
- Phytoplankton density significantly affects stratification

### PHATR Model

Pond Heat and Temperature Regulation:
- 1st order ODE solved by 4th order Runge-Kutta
- Predicts temperature for earthen outdoor ponds
- Determines energy transfer mechanism sizing

Source: [Academia - Pond Heat Balance](https://www.academia.edu/8305074/SIMULATION_MODEL_FOR_AQUACULTURE_POND_HEAT_BALANCE_I_MODEL_DEVELOPMENT)

---

# 13. Agent-Based & Discrete Event Simulation

## 13.1 Agent-Based Models (ABM)

### Individual Fish Agent
Each fish agent has:
- Position (x, y, z)
- Weight, length, age
- Health status (S, E, I, R)
- Behavior rules (feeding, schooling, avoidance)
- Local environment perception

### Farm-Level ABM
Multiple farms as agents in a landscape:
- Disease transmission between farms (distance-dependent)
- Shared water bodies
- Market interactions
- Implemented in NetLogo

### Disease ABM Parameters
- Current speed and direction
- Pathogen lifespan
- Contagiousness
- Fish density
- Distance between facilities

Source: [ResearchGate ABM Fish Disease](https://researchgate.net/publication/303430670)

## 13.2 Discrete Event Simulation (DES)

### RAS as Queuing Network

Culture tanks modeled as queuing systems:
- No queue allowed (no overholding)
- No idle tanks allowed
- Fish batches as "customers"
- Growth stages as "service stations"

### Optimization Variables
- Number/volume of netcages per growth phase
- Fish-batch arrival frequency
- Number of fingerlings per batch
- Days in each culture tank
- Grading criteria along production line

### Tools: ARENA simulation software

Source: [Springer DES Aquaculture](https://link.springer.com/article/10.1007/s10479-011-1048-3)

---

# 14. Dynamic Energy Budget (DEB) Theory

## 14.1 Core State Variables

1. **E** = Reserve energy (J)
2. **V** = Structural volume (cm^3)
3. **E_H** = Maturity level (J)
4. **E_R** = Reproduction buffer (J)

## 14.2 Energy Flow Equations

### Assimilation
```
p_A = {p_Am} * f * V^(2/3) * TC
f = X / (X_K + X)                    # functional response (Holling Type II)
```

### Mobilization (Catabolic power)
```
p_C = [E/V] * ([E_G]*v_dot*V^(2/3) + [p_M]*V) / ([E_G] + kappa*[E/V])
```

### Kappa Rule (Energy Allocation)
```
kappa * p_C  -->  somatic maintenance + growth
(1-kappa) * p_C  -->  maturity maintenance + reproduction
```

### Reserve Dynamics
```
dE/dt = p_A - p_C
```

### Structural Growth
```
dV/dt = (kappa * p_C - [p_M] * V) / [E_G]
```

### Maturity/Reproduction
```
dE_R/dt = (1 - kappa) * p_C - p_J
p_J = min(V, V_p) * [p_M] * (1-kappa)/kappa * TC
```

### Temperature Correction (Arrhenius)
```
TC = exp(T_A/T_1 - T_A/T) * (1 + exp(T_AL/(T-T_L)) + exp(T_AH/(T_H-T)))^(-1)
```

### Length and Weight
```
L = V^(1/3) / delta_M
W_wet = (E/mu_E + kappa_R * E_R/mu_E + V) * rho
```

## 14.3 DEB Parameters (Kuruma Shrimp, Penaeus japonicus)

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Max assimilation | {p_Am} | 1823 | J/cm^2/d |
| Energy conductance | v_dot | 0.0336 | cm/d |
| Structure cost | [E_G] | 4439 | J/cm^3 |
| Max storage density | [E_m] | 13,235 | J/cm^3 |
| Somatic maintenance | [p_M] | 569 | J/cm^3/d |
| Kappa | kappa | 0.98 | - |
| Reproduction efficiency | kappa_R | 0.95 | - |
| Shape coefficient | delta_M | 0.1585 | - |
| Arrhenius temp | T_A | 6200 | K |
| Reference temp | T_1 | 293 | K (20C) |
| Upper tolerance | T_H | 302 | K (29C) |
| Lower tolerance | T_L | 283 | K (10C) |
| Upper Arrhenius | T_AH | 33,800 | K |
| Lower Arrhenius | T_AL | 13,300 | K |

**Model fit:** MRE = 0.048, SMSE = 0.066

## 14.4 DEB Parameters (Mussels: Perna vs Mytilus)

| Parameter | Perna | Mytilus | Unit |
|-----------|-------|---------|------|
| Max ingestion | 15.54 | 9.42 | J/d/cm^2 |
| Half-saturation | 0.50 | 2.10 | ug/L |
| Assimilation eff | 0.69 | 0.80 | - |
| Kappa | 0.82 | 0.47 | - |
| Maintenance | 29.07 | 10.27 | J/d/cm^3 |
| Structure cost | 2800 | 3156 | J/cm^3 |
| Arrhenius T_A | 9826 | 10590 | K |

Source: [PMC DEB Shrimp](https://pmc.ncbi.nlm.nih.gov/articles/PMC9311514/), [PMC DEB Mussels](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219521/)

---

# 15. Unified OpenEnv State-Action-Reward Specification

Based on all the research above, here is a comprehensive specification for an aquaculture OpenEnv environment.

## 15.1 State Variables (Observation Space)

### Fish State
| Variable | Symbol | Range | Unit | Source Model |
|----------|--------|-------|------|-------------|
| Mean fish weight | w | 1-5000 | g | Bioenergetics |
| Fish population | N | 0-100000 | count | Population dynamics |
| Total biomass | B = N*w | 0-500000 | g | Derived |
| Mean fish length | L | 1-100 | cm | VBGF |
| Disease state [S,E,I,R] | S,E,I,R | 0-N each | count | SEIR |
| Growth stage | stage | 0-3 | enum | DEB maturity |

### Water Quality State
| Variable | Symbol | Range | Unit | Source Model |
|----------|--------|-------|------|-------------|
| Dissolved oxygen | DO | 0-15 | mg/L | DO mass balance |
| Water temperature | T | 10-40 | C | Thermal model |
| Total ammonia nitrogen | TAN | 0-10 | mg/L | N mass balance |
| Unionized ammonia | UIA | 0-2 | mg/L | pH/T dependent |
| pH | pH | 6-10 | - | Chemical equil |
| Nitrite | NO2 | 0-5 | mg/L | Nitrification |
| Nitrate | NO3 | 0-200 | mg/L | Nitrification |

### Environmental State
| Variable | Symbol | Range | Unit | Source |
|----------|--------|-------|------|--------|
| Day of year | doy | 1-365 | day | Seasonal |
| Photoperiod | P_h | 8-16 | hours | Latitude |
| Solar radiation | I_solar | 0-1000 | W/m^2 | Weather |
| Air temperature | T_air | -10-45 | C | Weather |
| Wind speed | v_wind | 0-30 | m/s | Weather |

### Economic State
| Variable | Symbol | Range | Unit | Source |
|----------|--------|-------|------|--------|
| Cumulative feed cost | C_feed | 0-inf | currency | Bioeconomic |
| Cumulative revenue | Rev | 0-inf | currency | Market |
| Market fish price | P_mkt | 0-inf | currency/kg | Stochastic |
| Days since stocking | t | 0-365+ | day | Calendar |
| Feed inventory | F_inv | 0-inf | kg | Operations |

## 15.2 Action Variables (Action Space)

| Action | Symbol | Range | Unit | Type |
|--------|--------|-------|------|------|
| Feeding rate | f | 0.0-1.0 | fraction of C_max | Continuous |
| Aeration rate | A | 0.0-1.0 | fraction of max | Continuous |
| Water exchange rate | Q_ex | 0.0-1.0 | fraction of volume/h | Continuous |
| Harvest decision | H | 0 or 1 | binary | Discrete |
| Harvest fraction | H_frac | 0.0-1.0 | fraction of population | Continuous |
| Stocking decision | S_dec | 0-1000 | fish to add | Discrete |
| Treatment/vaccination | V_dec | 0 or 1 | binary | Discrete |

## 15.3 Transition Dynamics

### Fish Growth (dt = 1 day)
```python
# Temperature factor
if T < T_opt:
    tau = exp(-4.6 * ((T_opt - T) / (T_opt - T_min))**4)
else:
    tau = exp(-4.6 * ((T - T_opt) / (T_max - T_opt))**4)

# DO factor
if DO > DO_crit:
    sigma = 1.0
elif DO > DO_min:
    sigma = (DO - DO_min) / (DO_crit - DO_min)
else:
    sigma = 0.0

# UIA factor
if UIA < UIA_crit:
    v = 1.0
elif UIA < UIA_max:
    v = (UIA_max - UIA) / (UIA_max - UIA_crit)
else:
    v = 0.0

# Photoperiod
pi = photoperiod / 12.0

# Anabolism
H = h * pi * f * b * (1-a) * tau * sigma * v
# Catabolism
k = k_min * exp(s * (T - T_min))

# Growth
dw = H * w**m - k * w**n
w_new = w + dw
```

### Water Quality (dt = 1 hour, or discretized daily)
```python
# Fish respiration (oxygen consumption)
FR = (10**(0.40 + 0.016*T - 0.0006*T**2 - 0.016*log(w))) * 1000  # mg O2/kg/h
DO_fish = FR * stocking_density / 1000

# Nitrification oxygen demand
DO_nitrif = 4.57 * TAN_production / V

# Reaeration
K_a = f(wind_speed, depth, T)
DO_reaer = K_a * (DO_sat(T) - DO)

# Photosynthesis (daytime only)
DO_photo = P_max * light_factor * chlorophyll_factor

# DO dynamics
dDO = DO_photo - DO_fish - DO_nitrif - DO_water_resp - DO_sed + DO_reaer + A * aeration_rate
DO_new = DO + dDO * dt

# TAN dynamics
TAN_excretion = feed_amount * protein_frac * 0.16 * N_wasted * 1.2 / V  # mg/L
TAN_nitrif = nitrif_rate * TAN
TAN_exchange = Q_ex * TAN
dTAN = TAN_excretion - TAN_nitrif - TAN_exchange
TAN_new = TAN + dTAN * dt

# UIA calculation
UIA = TAN * F(pH, T)  # lookup table or exponential function
```

### Population Dynamics
```python
# Natural mortality
M_nat = base_mortality * exp(stress_factor * (abs(T - T_opt) + max(0, UIA_crit - UIA)))

# Disease dynamics (if any infected)
if I > 0:
    new_exposed = beta * S * I / N
    new_infected = sigma * E
    new_recovered = gamma * I
    disease_deaths = alpha * I
else:
    new_exposed = new_infected = new_recovered = disease_deaths = 0

# Population update
N_new = N - INT(N * M_nat) - disease_deaths - harvest_amount + stocking_amount
```

### Economic Dynamics
```python
# Feed cost
daily_feed = f * C_max * w**(m) * N / 1000  # kg feed
daily_feed_cost = daily_feed * feed_price_per_kg

# Harvest revenue (if harvesting)
harvest_revenue = harvest_amount * w / 1000 * market_price  # per kg

# Running totals
cumulative_cost += daily_feed_cost + daily_ops_cost
cumulative_revenue += harvest_revenue
```

## 15.4 Reward Functions

### Option A: Growth Tracking (research-aligned)
```python
r = -((w - w_desired) / w_desired)**2 - lambda_f * f**2
```

### Option B: Profit Maximization
```python
r = harvest_revenue - daily_feed_cost - daily_ops_cost - penalty_mortality - penalty_water_quality
```

### Option C: Multi-Objective
```python
r = w_1 * growth_reward + w_2 * survival_reward + w_3 * efficiency_reward + w_4 * water_quality_reward

growth_reward = (w_new - w) / w  # relative growth
survival_reward = -mortality_count / N
efficiency_reward = -f  # minimize feed waste
water_quality_reward = -(max(0, DO_crit - DO) + max(0, UIA - UIA_crit))
```

### Option D: Terminal Reward (harvest optimization)
```python
r_t = -daily_cost_t  (for t < T_harvest)
r_T = market_price * biomass_T - harvest_cost_T  (at harvest)
```

## 15.5 Episode Structure

```
Episode = one production cycle (stocking to harvest)
- Typical duration: 120-365 days
- Time step: 1 day (growth), 1 hour (water quality, substepped)
- Terminal conditions:
  1. Harvest decision taken
  2. Maximum days reached
  3. Population drops below minimum
  4. Water quality catastrophe (DO < 1.0 mg/L for > 6 hours)
```

## 15.6 Default Parameter Set (Nile Tilapia, tropical)

```python
params = {
    # Growth
    'h': 0.4768, 'm': 0.6277, 'n': 0.8373,
    'b': 0.7108, 'a': 0.0559,
    'k_min': 0.0104, 's': 0.0288,
    'T_min': 18.7, 'T_max': 39.7, 'T_opt': 32.4,

    # Water quality
    'DO_crit': 5.0, 'DO_min': 3.0,     # mg/L
    'UIA_crit': 0.025, 'UIA_max': 0.6, # mg/L
    'TAN_max': 5.0,                      # mg/L

    # Population
    'N_initial': 10000,
    'w_initial': 5.0,  # g
    'base_mortality': 0.001, # per day

    # Disease (optional)
    'beta': 0.4, 'sigma': 0.2, 'gamma': 0.1, 'alpha': 0.05,

    # Economic
    'feed_price': 0.5,    # $/kg
    'market_price': 3.0,  # $/kg fish
    'fixed_cost': 10.0,   # $/day
    'discount_rate': 0.0001, # per day (~3.6% annual)

    # Environment
    'pond_volume': 1000,  # m^3
    'pond_depth': 1.5,    # m
    'stocking_density': 50, # fish/m^3
}
```

---

## Key References

- [FAO Fish Growth Model (Annex 3)](https://www.fao.org/4/W5268E/W5268E09.htm)
- [Model-based vs Model-free Feeding Control (arxiv:2306.09915)](https://arxiv.org/html/2306.09915)
- [Feeding Cost Risk in Aquaculture (arxiv:2309.02970)](https://ar5iv.labs.arxiv.org/html/2309.02970)
- [DEB Model for Kuruma Shrimp (PMC9311514)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9311514/)
- [DEB Across Environmental Gradients (PMC6219521)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219521/)
- [Tilapia DO Model (PMC8677810)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8677810/)
- [DO Fishpond Model](https://scialert.net/fulltext/?doi=ijar.2008.83.97)
- [Biofilter Sizing for RAS](https://www.globalseafood.org/advocate/estimating-biofilter-size-for-ras-systems/)
- [Ammonia in Aquatic Systems (UF/IFAS)](https://ask.ifas.ufl.edu/publication/FA031)
- [DO Dynamics in Ponds](https://www.globalseafood.org/advocate/dissolved-oxygen-dynamics/)
- [Epidemic Spread in Aquaculture (PMC10527373)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10527373/)
- [SIR Furunculosis Salmon](https://www.researchgate.net/publication/7679266)
- [Fish Bioenergetics 4.0](https://www.tandfonline.com/doi/full/10.1080/03632415.2017.1377558)
- [Bioeconomic Harvest Tilapia](https://www.scielo.cl/scielo.php?pid=S0718-560X2020000400602&script=sci_arttext)
- [Optimal Feed Ration (von Bertalanffy control)](https://onlinelibrary.wiley.com/doi/10.1155/2024/6512507)
- [DDPG for RAS](https://link.springer.com/article/10.1007/s10499-025-01914-z)
- [Multi-Objective RL RAS](https://link.springer.com/article/10.1007/s10499-025-02320-1)
- [Gym Fishing (GitHub)](https://github.com/boettiger-lab/gym_fishing)
- [EPA AQUATOX](https://www.epa.gov/hydrowq/aquatox)
- [PID DO Control](https://www.sciencedirect.com/science/article/abs/pii/S1537511021001197)
- [Pond Heat Balance Model](https://www.academia.edu/8305074/SIMULATION_MODEL_FOR_AQUACULTURE_POND_HEAT_BALANCE_I_MODEL_DEVELOPMENT)
- [ABM Fish Disease](https://researchgate.net/publication/303430670)
- [DES Aquaculture](https://link.springer.com/article/10.1007/s10479-011-1048-3)
- [Aquaculture Math Modelling Review](https://link.springer.com/article/10.1007/s10499-025-01928-7)
- [Stochastic Bioeconomic Catfish](https://link.springer.com/article/10.1007/s10499-022-00938-z)
