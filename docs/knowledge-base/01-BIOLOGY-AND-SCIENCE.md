# Aquaculture Biology Knowledge Base
## PhD-Level Reference: Growth, Water Chemistry, Feeding, Disease, Welfare & Species Data

---

# Table of Contents

1. [Fish Growth Models & Equations](#1-fish-growth-models--equations)
2. [Dissolved Oxygen Biology](#2-dissolved-oxygen-biology)
3. [Nitrogen Cycle & Ammonia Toxicity](#3-nitrogen-cycle--ammonia-toxicity)
4. [pH, CO2 & Water Buffering](#4-ph-co2--water-buffering)
5. [Feeding Science & FCR](#5-feeding-science--fcr)
6. [Temperature & Metabolism](#6-temperature--metabolism)
7. [Disease Triggers & Environmental Factors](#7-disease-triggers--environmental-factors)
8. [Behavioral Stress Indicators](#8-behavioral-stress-indicators)
9. [Stocking Density & Carrying Capacity](#9-stocking-density--carrying-capacity)
10. [Harvest Timing & Economics](#10-harvest-timing--economics)
11. [Recirculating Aquaculture Systems (RAS)](#11-recirculating-aquaculture-systems-ras)
12. [Species-Specific Biology](#12-species-specific-biology)
13. [Biofloc Technology](#13-biofloc-technology)
14. [Welfare Science & Cortisol](#14-welfare-science--cortisol)
15. [Algae Blooms & Eutrophication](#15-algae-blooms--eutrophication)
16. [Master Water Quality Parameter Table](#16-master-water-quality-parameter-table)

---

# 1. Fish Growth Models & Equations

## 1.1 Specific Growth Rate (SGR)

The most widely used relative growth index in aquaculture.

**Equation:**
```
SGR = [ln(W2) - ln(W1)] / dt * 100
```
- **Units:** % body weight per day (% day^-1)
- **Typical range:** 0.9 - 5.0 % day^-1 (for cultured fish, mean weights 3.5-270 g)

**Instantaneous form (IGR):**
```
IGR = (1/W) * (dW/dt)
```

**Key property:** SGR decreases with body weight following a power law:
```
SGR ~ A * (mean body weight)^(-B),  where A, B > 0
```

## 1.2 Thermal-Unit Growth Coefficient (TGC)

Accounts for temperature accumulation, making it superior to SGR for comparing growth across temperature regimes.

**Classical equation:**
```
TGC = (W2^(1/3) - W1^(1/3)) / SUM(T)
```
where SUM(T) = sum of daily temperatures over the growth period.

**Constant-temperature version:**
```
TGC = (W2^(1/3) - W1^(1/3)) / (T * dt)
```
- **Units:** g^(1/3) * (degree-C * day)^-1
- **Typical range:** 0.1 - 3.2 g^(1/3) (degree-C * day)^-1

**Instantaneous form:**
```
ITGC = (T / W^(2/3)) * (dW/dt)
```

**SGR-TGC relationship:**
```
ITGC = T * W^(1/3) * IGR
TGC ~ (SGR / 100) * (W1^(1/3) / T)     [finite-interval approximation]
```

## 1.3 Von Bertalanffy Growth Function (VBGF)

The most widely used growth model in fisheries biology (5,000+ parameter sets for 1,300+ species in FishBase).

**Length form:**
```
L(t) = L_inf * [1 - exp(-K * (t - t0))]
```

**Weight form (derived via W = a * L^b):**
```
W(t) = W_inf * [1 - exp(-K * (t - t0))]^b
```

**Parameters:**
| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| L_inf | Asymptotic length (theoretical max avg length) | Species-specific (cm) |
| K | Growth coefficient (rate of approach to L_inf) | 0.05 - 2.0 year^-1 |
| t0 | Hypothetical age at zero length (modeling artifact) | Usually negative |
| b | Weight-length exponent | 2.0 - 4.0 (isometric = 3.0) |
| a | Weight-length scaling coefficient | Species-specific |

**Critical note:** L_inf is NOT the maximum length -- it is the asymptotic average. Some individuals exceed L_inf.

## 1.4 Weight-Length Relationship

```
W = a * L^b
```
- **Isometric growth:** b = 3 (weight proportional to cube of length)
- **Positive allometric:** b > 3 (fish gets relatively heavier as it grows)
- **Negative allometric:** b < 3 (fish gets relatively lighter)
- **Typical b range:** 2.0 - 4.0 (usually near 3.0)

**Log-linear form for regression:**
```
log(W) = log(a) + b * log(L)
```

## 1.5 Fulton's Condition Factor

```
K = (W / L^3) * 100     [when W in g, L in cm, scaling factor = 100]
```

**Relative condition factor (for allometric growth):**
```
K' = W / (a * L^b)
```
- K' > 1: fish is in better condition than average
- K' < 1: fish is in worse condition than average

## 1.6 Feed Conversion Ratio (FCR) as Growth Metric

```
FCR = Feed consumed (kg) / Weight gained (kg)
```
Lower FCR = more efficient conversion. See Section 5 for species-specific values.

---

# 2. Dissolved Oxygen Biology

## 2.1 Oxygen Solubility vs. Temperature

| Temperature (deg C) | DO Saturation (mg/L) at 760 mmHg, 0 ppt salinity |
|---------------------|---------------------------------------------------|
| 0 | 14.60 |
| 5 | 12.77 |
| 10 | 11.29 |
| 15 | 10.08 |
| 20 | 9.08 |
| 25 | 8.26 |
| 30 | 7.54 |
| 35 | 6.95 |
| 40 | 6.41 |

**Key principle:** Warm water holds dramatically less oxygen. A pond at 30 deg C holds only 52% of the oxygen that a pond at 0 deg C can hold.

**Depth effect at 20 deg C:**
- Surface (0 m): 9.08 mg/L
- 1.0 m: 9.98 mg/L
- 4.0 m: 12.67 mg/L

## 2.2 Critical DO Thresholds

| Species Category | Optimal (mg/L) | Stress (mg/L) | Lethal (mg/L) |
|------------------|-----------------|----------------|----------------|
| General recommendation | >= 5.0 | 2.0 - 4.0 | < 2.0 |
| Coldwater (salmonids) | >= 6.5 | < 5.0 | < 2.5 - 3.5 |
| Warmwater (catfish) | >= 5.0 | < 3.5 | < 2.0 |
| Tilapia | >= 5.0 | < 3.0 | 1.0 - 2.0 |
| Shrimp | >= 5.0 | < 2.0 | < 1.2 |

**Saturation-based thresholds:**
- Coldwater species: minimum 60% saturation (6.48 mg/L at 15 deg C)
- Warmwater species: minimum 50% saturation (4.13 mg/L at 25 deg C)

## 2.3 Oxygen Consumption Rates

**Average adult fish:** 200 - 500 mg O2 / kg / hour

**Species-specific data:**
| Species / Size | Resting (mg O2/kg/hr) | After feeding | Active max |
|----------------|----------------------|---------------|------------|
| Channel catfish, 10 g | 1,050 | -- | -- |
| Channel catfish, 500 g | 480 | 680 | -- |
| Channel catfish, 500 g (fasted overnight) | 380 | -- | -- |
| Southern catfish, 25 deg C | 160 | -- | -- |
| Southern catfish, 10 deg C | 65 | -- | -- |
| Salmon/trout (active) | -- | -- | ~1,000 (+/- 200) |
| Shrimp | Similar to fish | -- | -- |

**Size effect:** Larger fish consume LESS O2 per kg. 10g catfish use 2.2x more O2/kg than 500g catfish.
**Temperature effect:** O2 consumption roughly doubles per 10 deg C increase (Q10 ~ 2).
**Feeding effect:** Post-feeding O2 consumption is ~1.8x resting rate (specific dynamic action / SDA).

## 2.4 Oxygen Budget in Ponds

**Sources:**
1. Photosynthesis by phytoplankton (most important in ponds -- can produce 2-3x saturation in afternoon)
2. Wind/wave action (diffusion)
3. Artificial aeration

**Sinks (example: shrimp pond, Shigueno 1975):**
- Water column (bacteria, organic matter decomposition): 69.4%
- Bottom sediment: 14.8%
- Target species (Penaeus japonicus): 8.6%
- Fish (bycatch): 6.7%
- Other shrimp: 0.5%

**Critical principle:** In most aquaculture ponds, the fish themselves consume only ~10% of total oxygen. Microbial decomposition in water and sediment dominates.

## 2.5 Oxygen Demand per kg Feed (RAS)

| System Type | O2 consumed per kg feed |
|-------------|------------------------|
| Efficient (non-submerged biofilter, fast solids removal) | ~0.3 kg O2/kg feed |
| Moderate RAS | ~0.5 kg O2/kg feed |
| High-load (submerged biofilter, retained solids) | ~1.0 kg O2/kg feed |

**Feed Oxygen Demand equation:**
```
FOD (kg O2/kg feed) = [(% C in feed/100) - FCE * (% C in fish/100)] * 2.67
                     + [(% N in feed/100) - FCE * (% N in fish/100)] * 4.57
```
where:
- 2.67 = 32/12 = stoichiometric O2 per C oxidized
- 4.57 = 64/14 = stoichiometric O2 per N oxidized (nitrification)
- FCE = feed conversion efficiency

## 2.6 Diel Oxygen Cycle

In tropical ponds, DO follows an extreme daily pattern:
- **Dawn (6-7 AM):** Minimum -- near zero possible (all night respiration, no photosynthesis)
- **Afternoon (2-4 PM):** Maximum -- can reach 200-300% saturation (photosynthetic supersaturation)
- **Difference:** Daily max can be 2-3x saturation level

**Monitoring protocol:** Measure at late afternoon (5-6 PM) and late evening (8-10 PM). Aerate from 10 PM through 7-8 AM.

---

# 3. Nitrogen Cycle & Ammonia Toxicity

## 3.1 The Nitrogen Cycle

```
Feed protein --> Fish excretion --> NH3/NH4+ (TAN)
                                       |
                                       v
                              Nitrosomonas/Nitrosospira (AOB)
                              [ammonia-oxidizing bacteria]
                                       |
                                       v
                                    NO2- (nitrite)
                                       |
                                       v
                              Nitrospira/Nitrobacter (NOB)
                              [nitrite-oxidizing bacteria]
                                       |
                                       v
                                    NO3- (nitrate)
                                    [relatively non-toxic]
```

**Key facts:**
- Both AOB and NOB are autotrophic, aerobic bacteria
- New biofilter requires 6-8 weeks to establish sufficient bacterial colonies
- Nitrification is aerobic and produces CO2 + H+, reducing pH
- Each kg of NH3-N oxidized requires 4.57 kg of molecular oxygen
- Each gram of TAN oxidized consumes 7.14 g of alkalinity (as CaCO3)
- Each gram of TAN oxidized produces ~0.17 g bacterial biomass

## 3.2 Ammonia Chemistry

Total Ammonia Nitrogen (TAN) = NH3 (unionized, toxic) + NH4+ (ionized, ~100x less toxic)

**The critical equation:**
```
Fraction NH3 = 1 / (1 + 10^(pKa - pH))
```

**Temperature-dependent pKa:**
```
pKa = 0.09018 + (2729.92 / T_kelvin)
```
where T_kelvin = temperature in Celsius + 273.15

**Practical examples of NH3 fraction:**
| pH | Temperature | % NH3 of TAN |
|----|-------------|--------------|
| 7.0 | 20 deg C | ~0.4% |
| 7.0 | 25 deg C | ~0.6% |
| 8.0 | 20 deg C | ~4% |
| 8.0 | 25 deg C | ~6% |
| 8.5 | 25 deg C | ~15% |
| 9.0 | 25 deg C | ~36% |
| 9.25 | 25 deg C | ~50% |

**Key rule:** Each 1.0 pH unit increase multiplies unionized NH3 by approximately 10x.

## 3.3 Ammonia Toxicity Thresholds

| UIA-N Level (mg/L) | Effect |
|---------------------|--------|
| 0.00 - 0.02 | Safe zone |
| 0.02 - 0.05 | Chronic stress begins; sublethal effects possible |
| 0.05 - 0.20 | Tissue damage (gills, liver, kidney) |
| 0.20 - 0.50 | Severe damage; growth impairment |
| 0.50 - 1.00 | Acute toxicity; significant mortality risk |
| >= 2.0 | Death for most species |

**Species variation:** Coldwater fish (salmonids) are generally more sensitive than warmwater fish (catfish, tilapia).

## 3.4 Nitrite Toxicity

- **Toxic threshold:** As low as 0.10 mg/L NO2-N for sensitive species
- **Safe level:** < 0.25 mg/L NO2-N general recommendation
- **Mechanism:** Nitrite oxidizes hemoglobin to methemoglobin ("brown blood disease"), reducing oxygen-carrying capacity
- **Chloride protection:** Chloride ions competitively inhibit nitrite uptake at fish gills; maintain Cl:NO2 ratio > 6:1

## 3.5 Nitrate Levels

- **Traditional safe threshold:** Up to 200 mg/L NO3-N
- **Recent research:** May be more detrimental than previously believed
- **Accumulation risk:** In closed RAS systems, nitrate can exceed 250 mg/L without water exchange
- **Management:** Partial water exchange or denitrification reactors

## 3.6 TAN Production from Feed

```
TAN produced (kg/day) = Feed (kg/day) * Protein_fraction * 0.16 * 0.50 * 1.2
```
where:
- 0.16 = g nitrogen per g protein (N content of amino acids)
- 0.50 = fraction of feed nitrogen wasted (not assimilated)
- 1.2 = conversion factor g N to g TAN

**Rule of thumb:** ~4% of feed weight becomes TAN in the system (range: 3-5%).

---

# 4. pH, CO2 & Water Buffering

## 4.1 pH Ranges

| Category | pH Range |
|----------|----------|
| Optimal for most freshwater fish | 6.5 - 8.5 |
| Optimal for Nile tilapia | 6.0 - 8.0 |
| Optimal for marine shrimp | 7.5 - 8.5 |
| Stress threshold (acid) | < 5.0 |
| Stress threshold (alkaline) | > 10.0 |
| Lethal (acid) | ~4.0 |
| Lethal (alkaline) | ~11.0 |
| Optimal for nitrifying bacteria | 7.0 - 8.6 |

## 4.2 CO2 Levels

| CO2 Concentration | Status |
|-------------------|--------|
| < 10 mg/L | Adequate for most aquaculture |
| 10 - 20 mg/L | Monitor closely |
| > 20 mg/L | Requires intervention (aeration/degassing) |

**CO2-pH relationship:** CO2 dissolution in water forms carbonic acid (H2CO3), which dissociates:
```
CO2 + H2O <--> H2CO3 <--> H+ + HCO3- <--> 2H+ + CO3(2-)
```
More CO2 = lower pH. This is why dawn pH is lowest (overnight respiration accumulates CO2) and afternoon pH is highest (photosynthesis removes CO2).

## 4.3 Alkalinity & Buffering

**Alkalinity recommendations:**
| System Type | Minimum Alkalinity (mg CaCO3/L) |
|-------------|----------------------------------|
| Freshwater ponds | 40 |
| Conventional marine ponds | 70 |
| Intensive biofloc/RAS | 100 - 150 |
| Optimal general range | 50 - 300 |

**Alkalinity consumption:**
- Nitrification consumes 7.14 g CaCO3 per g TAN-N oxidized
- Practical rule: ~0.25 kg baking soda supplementation per kg feed
- Liming rate for pH correction: 100-150 kg/ha/day hydrated lime (split doses, applied early morning)

**Hardness:**
- Freshwater: typically < 100 mg CaCO3/L
- Seawater (35 ppt): 6,000 - 7,000 mg CaCO3/L
- Optimal calcium hardness to total alkalinity ratio: ~1:1
- Minimum recommended: > 50 mg CaCO3/L

## 4.4 The Water Buffering System (WBS)

The carbonate equilibrium system buffers against pH swings:
```
Nighttime: CO3(2-) dissolves --> releases OH- --> buffers pH drop from respiration CO2
Daytime:   CO3(2-) precipitates as CaCO3/MgCO3 --> moderates pH rise from photosynthesis
```

The higher the total hardness (TH) and total alkalinity (TA), the stronger the buffering capacity. This is why low-alkalinity ponds experience dangerous pH swings (below 5 at dawn, above 10 at peak photosynthesis).

---

# 5. Feeding Science & FCR

## 5.1 FCR by Species

| Species | Typical FCR | Notes |
|---------|-------------|-------|
| Atlantic salmon | 1.0 - 1.2 | Best-in-class among farmed fish |
| Rainbow trout | 1.0 - 1.5 | |
| Nile tilapia | 1.4 - 2.5 | Edible (fillet) FCR ~4.6 |
| Channel catfish | 1.5 - 2.0 | |
| Common carp | 1.5 - 2.5 | Chinese carp edible FCR ~4.9 |
| Pangasius | 1.5 - 1.8 | |
| Shrimp (P. vannamei) | 1.2 - 2.0 | |
| Sea bass | 1.5 - 2.5 | |

**Key insight:** Feed costs = 30-70% of total production costs. A 0.1 improvement in FCR can represent enormous economic gain.

## 5.2 Feeding Rates (% body weight / day)

**By fish size (Common carp, 20-23 deg C):**
| Fish Weight (g) | Feeding Rate (% BW/day) |
|-----------------|------------------------|
| < 5 | 9% |
| 5 - 20 | 7% |
| 20 - 50 | 6% |
| 50 - 100 | 5% |
| 100 - 300 | 4% |
| 300 - 1,000 | 3% |

**General rules:**
- Juveniles: 5 - 10% body weight / day
- Grow-out: 2 - 5% body weight / day
- Adults near harvest: 1 - 3% body weight / day
- Tilapia: ~3% BW/day
- Catfish: ~4-5% BW/day (season-dependent)

**Each feed event should ideally be ~1% of body weight.** So if feeding 5%/day, deliver 5 separate feedings.

## 5.3 Feeding Frequency

| Species / Life Stage | Feeds per Day |
|---------------------|---------------|
| Salmon/trout fry | 20 - 24 |
| Salmon/trout fingerlings | 6 - 8 |
| Salmon/trout juveniles | 3 - 4 |
| Salmon/trout adults | 1 - 3 |
| Channel catfish fry | 8 - 10 |
| Channel catfish 7.6 cm+ | 3 |
| Channel catfish juvenile-adult | 2 |
| Tilapia fry | 4 - 8 |
| Tilapia fingerlings | 4 - 5 |
| Tilapia adults | 2 - 3 |
| Common carp (optimal at 40 g) | 9 |
| Shrimp (intensive) | 4 - 6 |

## 5.4 Overfeeding Consequences

1. **Uneaten feed decomposes** --> ammonia spikes, oxygen depletion
2. **Increased BOD** (biological oxygen demand) in water column and sediment
3. **Higher FCR** (wasted feed counted, not converted to growth)
4. **Digestive stress** in fish --> reduced immune function
5. **Eutrophication** from excess nitrogen and phosphorus
6. **Disease outbreaks** -- bacterial pathogens thrive on organic matter from decomposing feed

## 5.5 Temperature Effects on Feeding

- Channel catfish: 13-29 deg C = full feeding (6-7 days/week); outside this range = reduce to 4-5 days/week
- Below species-specific lower threshold: feeding stops entirely
- Korean rockfish optimal feeding rates: 3.41% at 16 deg C, 3.75% at 20 deg C, 3.34% at 24 deg C (peak near species optimum)

---

# 6. Temperature & Metabolism

## 6.1 Temperature Classification

| Category | Optimal Range | Species Examples |
|----------|---------------|------------------|
| Coldwater | 10 - 18 deg C | Atlantic salmon, rainbow trout, brown trout |
| Coolwater | 15 - 25 deg C | Walleye, perch, striped bass |
| Warmwater | 25 - 32 deg C | Tilapia, catfish, carp, shrimp |

## 6.2 Species-Specific Temperature Data

### Atlantic Salmon (Salmo salar)
| Parameter | Temperature |
|-----------|-------------|
| Egg incubation optimal | ~10 deg C |
| First-feeding fry optimal | 16 - 20 deg C |
| Parr optimal growth | 18 - 19 deg C |
| Post-smolt optimal growth (70-150 g) | 12.8 deg C |
| Post-smolt optimal growth (150-300 g) | 14.0 deg C |
| Post-smolt optimal FCR | ~11 - 13 deg C |
| Appetite decline onset | > 20 deg C |
| Upper stress threshold | > 19 deg C |
| Critical survival (parr/smolt) | 30 - 33 deg C |
| Practical farming range | 10 - 14 deg C |

### Nile Tilapia (Oreochromis niloticus)
| Parameter | Temperature |
|-----------|-------------|
| Optimal growth | 27 - 32 deg C |
| Best FCR observed | 32 deg C (FCR 2.43) |
| Growth ceases | < 17 deg C |
| Juvenile mortality begins | < 17 deg C and > 35 deg C |
| Lower lethal (adults, wild) | 11 - 12 deg C |
| Upper lethal (adults, wild) | 42 deg C |

### Pacific White Shrimp (Penaeus vannamei)
| Parameter | Temperature |
|-----------|-------------|
| Optimal growth (small, 1 g) | 30 deg C |
| Optimal growth (large, 12-18 g) | 27 deg C |
| General optimal range | 23 - 30 deg C |
| Salinity range | 0.5 - 45 ppt (optimal 10-15 ppt) |

### Channel Catfish
| Parameter | Temperature |
|-----------|-------------|
| Optimal growth | 25 - 30 deg C |
| Feeding stops | < 13 deg C |
| Survival range | near-freezing to ~32 deg C |
| Disease (Edwardsiella) virulence peak | 22 - 28 deg C |

## 6.3 Q10 Temperature Coefficient

The Q10 describes the factor by which a biological rate changes per 10 deg C temperature increase.

**Equation:**
```
Q10 = (R2 / R1) ^ (10 / (T2 - T1))
```
where R1, R2 = rates at temperatures T1, T2.

**Typical Q10 values in fish:**
- Resting metabolic rate, warmwater fish: Q10 < 2
- Resting metabolic rate, coldwater fish (salmonids): Q10 > 2
- Shrimp growth: Q10 ~ 2 (e.g., growth increases from 1.20 to 1.44 g/week with 2 deg C rise from 27 to 29 deg C)
- General oxygen consumption: Q10 ~ 2 (doubles per 10 deg C)

## 6.4 Metabolic Rate Equations

**Standard metabolic rate (SMR):** Minimum aerobic metabolic rate (measured by respirometry). Represents baseline energy cost for survival.

**Active metabolic rate (AMR):** Maximum sustained oxygen consumption during activity. Can be 5-10x SMR.

**Aerobic scope:** AMR - SMR. Represents the energy available for growth, feeding, reproduction, and stress response. Aerobic scope is maximized at the species' optimal temperature and declines toward thermal limits.

**Body mass scaling:**
```
SMR = a * W^b     where b ~ 0.7 - 0.8 (allometric scaling)
```
This is why larger fish have lower mass-specific metabolic rates.

---

# 7. Disease Triggers & Environmental Factors

## 7.1 The Disease Triangle

Disease outbreak = intersection of:
1. **Susceptible host** (stressed, immunocompromised fish)
2. **Virulent pathogen** (sufficient dose/concentration)
3. **Conducive environment** (poor water quality, temperature extremes)

All three must align. Good environmental management prevents outbreaks even when pathogens are present.

## 7.2 Environmental Triggers

| Trigger | Mechanism | Examples |
|---------|-----------|----------|
| High temperature | Pathogen multiplication accelerates; fish immune suppression | Bacterial gill rot optimal 28-35 deg C; Hemorrhagic septicemia peaks > 27 deg C |
| Low DO | Stress-induced immunosuppression; fish crowding at surface | Below 3 mg/L triggers cortisol cascade |
| High ammonia | Gill damage creates pathogen entry points | UIA > 0.05 mg/L causes tissue damage |
| High stocking density | Pathogen transmission rate increases; chronic stress | Overwintering pond disease outbreaks |
| Sudden temperature change | Thermal shock suppresses immune function | Spring/fall transition disease peaks |
| Overfeeding | Organic matter fuels pathogen growth; water quality degradation | Enteritis in grass carp from overfeeding |
| Mechanical injury (handling) | Wounds allow pathogen entry | Saprolegniasis from netting injuries |
| Low pH | Increased ammonia toxicity, stress | < 5.0 causes acute stress |
| Decreased air pressure | Increases pathogen virulence | Enteritis outbreak trigger |

## 7.3 Major Fish Diseases by Environmental Trigger

### Temperature-Triggered Diseases

| Disease | Pathogen | Optimal Temp | Season | Mortality |
|---------|----------|-------------|--------|-----------|
| Hemorrhagic septicemia | Reovirus | > 27 deg C | June-Sept | Mass mortality in fingerlings |
| Enteritis | Aeromonas punctata | 20-25 deg C | May-June, Aug-Sept | 50-90% |
| Bacterial gill rot | Myxococcus piscicolus | 28-35 deg C | Spring/summer | Variable |
| Ichthyophthiriasis ("white spot") | Ichthyophthirius multifilis | 15-25 deg C | Winter/spring | Mass mortality |
| Dactylogyrosis | Dactylogyrus spp. | 20-25 deg C | Late spring | Variable |
| Lernaesis (anchor worm) | Lernaea spp. | 15-33 deg C | April-Oct | High in juveniles |

### Water Quality-Triggered Diseases

| Condition | Resulting Disease Risk |
|-----------|----------------------|
| Low DO + high density | Saprolegniasis (fungal) |
| High organic load | Bacterial gill rot |
| Poor circulation + shallow water | Trichodinasis (protozoan) |
| Post-handling wounds | Erythroderma, saprolegniasis |
| Continuous rain | Trichodinelliasis |
| Contaminated equipment | Secondary bacterial infections |

## 7.4 Disease Prevention Protocol

1. **Pond disinfection** before stocking (quicklime 100 kg/mu or bleaching powder)
2. **Fingerling disinfection** (8 ppm copper sulphate bath, 20-30 min; or 10 ppm bleaching powder)
3. **"Four Fix" feeding** procedure (fixed time, place, quality, quantity)
4. **Daily inspection** especially mornings during epidemic season (May-September)
5. **Feed disinfection** (aquatic grasses: 6 ppm bleaching powder, 20-30 min)
6. **Equipment disinfection** (sunlight exposure 1-2 days, or chemical immersion)
7. **Monthly pond treatment** (bleaching powder 1 ppm, or quicklime 20-25 kg/mu)
8. **Quarantine** of new stock (strict prohibition on transporting diseased fish)
9. **Rotation farming** (prevents host-specific parasites from accumulating)

---

# 8. Behavioral Stress Indicators

## 8.1 Swimming Behavior Changes

| Indicator | Stress Type | Species Documented |
|-----------|------------|-------------------|
| Reduced swimming speed | Hypoxia | Atlantic cod, white sturgeon |
| Reduced tail beat frequency | Hyperoxia | Atlantic salmon |
| Elevated swimming speed | Underfeeding | Multiple species |
| Erratic "tornado" or "hourglass" patterns | Acute stress | Atlantic salmon (cages) |
| Stereotypic circular/triangular loops (10-240 sec) | Chronic confinement | African catfish |
| Vertical swimming loops | High stocking density | Atlantic halibut |
| Sharper turns during feeding | Predictable ration frustration | Atlantic salmon |
| Reduced critical swimming speed (U_crit) | Disease, parasites, pollution | Multiple species |

## 8.2 Feeding Behavior Changes

| Indicator | Meaning |
|-----------|---------|
| Increased latency to start feeding | Acute stress or illness |
| Reduced feeding rate and daily feeding times | Grading stress, handling |
| Reduced self-feeder activation | High stocking density |
| Strong anticipatory response (crowding near feeder before feeding) | Good welfare indicator |
| Loss of feed anticipatory activity | Post-stress state |
| Recovery of feeding behavior SLOWER than cortisol recovery | Important: behavioral recovery lags hormonal recovery |

## 8.3 Ventilation Rate (Opercular Beat Frequency)

- **Response time:** Seconds (fastest behavioral indicator)
- **Baseline:** Acclimated unstressed fish have minimum ventilatory frequency
- **Hypoxia:** Ventilation increases up to a critical low O2 level, then collapses
- **Hyperoxia:** Ventilation decreases when O2 is in excess
- **Stressor-induced increases:** Lighting changes, loud sound, unsuitable temperature, restricted movement, reduced retreat space, handling, air exposure, chemical presence, disease, ammonia/nitrate excess

## 8.4 Other Behavioral Indicators

| Category | Observations |
|----------|-------------|
| **Aggression** | Skin lesion count correlates with aggressive acts; dominance hierarchies form when resources are limited; subordinates show behavioral inhibition, color changes |
| **Body color** | Subordinates show color changes under social stress |
| **Fin condition** | Fixed feeding regimes increase fin damage vs. demand feeding |
| **Vacuum behaviors** | Tilapia: vacuum pit digging in substrate-free tanks |
| **Surface rubbing** | Ichthyophthiriasis: fish rub against objects or jump out of water |
| **Solitary swimming** | Enteritis: fish separates from group, slow solitary movement |
| **Spatial distribution** | Post-acute stress: fish concentrate near bottom of tanks/cages |

## 8.5 Welfare Assessment Framework

**Operational welfare indicators at individual level:**
- Skin condition (lesions, erosion, ulcers)
- Gill health (color, mucus, parasite presence)
- Fin condition (erosion score)
- Eye condition (exophthalmia, cataracts)
- Body condition factor

**Operational welfare indicators at group level:**
- Feeding behavior (anticipatory response, intake rate)
- Swimming patterns (speed, distribution, stereotypies)
- Mortality rate (daily and cumulative)
- Disease prevalence

---

# 9. Stocking Density & Carrying Capacity

## 9.1 Optimal Densities by Species

| Species | System | Optimal Density (kg/m3) | Notes |
|---------|--------|------------------------|-------|
| Atlantic salmon (parr, 0-28 g) | RAS | 1.76 - 14.55 | Size-dependent |
| Atlantic salmon (juvenile, 7-98 g) | RAS | 14.55 - 38.38 | Size-dependent |
| Atlantic salmon (post-smolt) | RAS | < 30 (max recommended) | Higher causes stress |
| Rainbow trout | Raceways | 10 - 40 | Fin erosion above 40 |
| Nile tilapia | Intensive tanks | Up to 136 | With good water management |
| Channel catfish | Ponds | 70 - 200+ | Hybrid catfish can handle higher |
| Common carp | Concrete ponds | 70 - 80 | |
| P. vannamei (shrimp) | Intensive ponds | 70 - 150/m2 stocking | 12-24 tonnes/ha production |
| P. vannamei (shrimp) | Super-intensive tanks | Up to 400/m2 | Requires full RAS |

## 9.2 Density Effects on Biology

**Growth:** Specific growth rate (SGR), final weight, and weight gain decrease significantly at high densities. Effect is body-mass dependent (different optimal densities for different life stages).

**Water quality at high density:**
- DO depleted faster
- Ammonia accumulates faster
- pH swings become more extreme
- Disease transmission increases

**Behavioral effects:**
- Atlantic salmon: increased aggression OR polarized schooling (collision avoidance)
- Rainbow trout: reduced self-feeding
- Sea bass: decreased swimming speed

**Health effects:**
- Reduced immune function
- Higher cortisol levels
- Increased fin erosion
- Greater disease prevalence and mortality

## 9.3 Carrying Capacity Concept

Carrying capacity = maximum biomass a system can sustain without crisis. Determined by:
1. **Oxygen supply rate** (aeration + photosynthesis - microbial demand)
2. **Ammonia removal rate** (nitrification + water exchange)
3. **CO2 removal rate** (degassing + aeration)
4. **Heat exchange capacity** (temperature control)

In RAS: all components must be sized to handle the same feed loading. The weakest link determines carrying capacity.

---

# 10. Harvest Timing & Economics

## 10.1 Growth Curve Shape

Fish growth follows a **sigmoid (S-shaped) curve**:
1. **Lag phase:** Slow initial growth (small fish, developing digestive system)
2. **Exponential phase:** Rapid growth (juveniles, high feeding rate, optimal conditions)
3. **Plateau phase:** Slowing growth as fish approaches asymptotic size (increasing maintenance costs)

## 10.2 Optimal Harvest Principle

**Marginal condition for profit maximization:**
```
Marginal increase in value by delaying harvest
= Sum of (opportunity cost + mortality cost + feed cost + energy cost + maintenance cost)
```

Harvest when the cost of one more day of growth exceeds the additional revenue from that growth.

**Key insight from economic models:**
- A 1% decline in interest rate can induce a 70% increase in optimal harvest weight
- As interest rates decrease, optimal harvest weight and time increase in a stepwise, nonlinear fashion
- Fish markets provide **premiums for larger fish** (often piecewise linear price-weight functions)

## 10.3 Production Cycle Duration

| Species | Seed to Harvest | Market Size |
|---------|----------------|-------------|
| Atlantic salmon | 24-42 months (12-18 mo freshwater + 12-24 mo sea) | 3-6 kg |
| Rainbow trout | 12-18 months | 250g - 3 kg |
| Nile tilapia | 3-5 months (grow-out) | 400g - 1 kg |
| Channel catfish | 6-10 months | 0.5 - 1.5 kg |
| P. vannamei shrimp | ~90 days | 18-25 g |
| Common carp | 12-24 months | 1-3 kg |

## 10.4 Feed Cost Optimization

```
Total feed cost = Daily feed rate * Feed price * Culture duration
Marginal revenue from growth = (dW/dt) * Price per kg at harvest weight
```

**Feed represents 30-70% of total operating costs.** Therefore:
- Optimizing feeding rate (not overfeeding) is the single most important cost lever
- FCR improvement of 0.1 on a 1000-tonne operation saves ~100 tonnes of feed

---

# 11. Recirculating Aquaculture Systems (RAS)

## 11.1 Core Components

Every RAS must address five processes:
1. **Circulation** (pumps, pipe sizing)
2. **Clarification** (mechanical filtration -- drum filters, settling basins)
3. **Biofiltration** (nitrification -- MBBR, trickling filters, fluidized beds)
4. **Aeration** (oxygen supplementation)
5. **CO2 stripping** (degassing columns)

## 11.2 Water Exchange & Flow Rates

| Parameter | Coldwater (salmonids) | Warmwater |
|-----------|-----------------------|-----------|
| Tank turnover rate | <= 30 min | ~60 min |
| Water recirculation rate | 90 - 99% of total volume retained | Same |
| Make-up water | 1 - 10% per day | Same |

**Flow rate estimation (TAN-based):**
```
Flow rate (m3/day) = TAN production (g/day) / (desired TAN conc * biofilter efficiency)
```
- Typical desired tank TAN: 1.5 mg/L (can reach 3 mg/L)
- Biofilter removal efficiency per pass: ~50%
- Passive nitrification (tank walls, pipes): 20-30% of TAN conversion

## 11.3 Biofilter Design

**TAN production estimation:**
```
TAN produced (g/day) = Feed (g/day) * protein% * 0.16 * 0.50 * 1.2
```
Rule of thumb: ~4% of feed weight = TAN.

**Volumetric TAN Conversion Rate (VTR):**
| Biofilter Type | VTR (g TAN/m3/day) | Notes |
|----------------|-------------------|-------|
| Trickling filter (200 m2/m3 SSA) | ~90 | Low but reliable |
| Moving bed biofilm reactor (MBBR) | ~350 | Most popular modern choice |

**Biofilter sizing equation:**
```
Media volume (m3) = TAN production (g/day) / VTR (g TAN/m3/day)
```
Example: 2,300 g TAN/day / 350 VTR = 6.57 m3 MBBR media.

**Nitrification requirements:**
- Dissolved oxygen: > 4 mg/L in biofilter
- pH: 7.0 - 8.6 (optimal for nitrifying bacteria)
- Temperature: most efficient at 27-28 deg C (operates 7-35 deg C)
- Alkalinity: maintain > 100 mg/L CaCO3

**Alkalinity supplementation:**
- 7.14 g CaCO3 consumed per g TAN oxidized
- Practical: ~0.25 kg baking soda per kg feed
- Monitor alkalinity daily in intensive RAS

## 11.4 RAS Water Quality Targets

| Parameter | Target | Critical Limit |
|-----------|--------|---------------|
| DO | > 6 mg/L | > 4 mg/L minimum |
| TAN | < 1.5 mg/L | < 3.0 mg/L |
| UIA-N | < 0.02 mg/L | < 0.05 mg/L |
| Nitrite (NO2-N) | < 0.1 mg/L | < 0.25 mg/L |
| Nitrate (NO3-N) | < 100 mg/L | < 200 mg/L |
| pH | 7.0 - 7.8 | 6.5 - 8.5 |
| CO2 | < 10 mg/L | < 20 mg/L |
| Alkalinity | 100 - 200 mg CaCO3/L | > 50 mg/L |
| TSS | < 15 mg/L | < 25 mg/L |

---

# 12. Species-Specific Biology

## 12.1 Atlantic Salmon (Salmo salar)

| Parameter | Value |
|-----------|-------|
| Market size | 3-6 kg |
| Production cycle | 24-42 months total |
| Freshwater phase | 12-18 months |
| Seawater phase | 12-24 months |
| Optimal growth temp | 10-14 deg C (farming) |
| Upper stress limit | > 20 deg C |
| Critical thermal max | 30-33 deg C |
| FCR | 1.0-1.2 |
| TGC (typical) | 2.0-3.0 |
| SGR (post-smolt) | 0.8-1.5 %/day |
| Optimal DO | > 6.5 mg/L |
| Smoltification | Photoperiod + size triggered |

**Life stages:** Egg --> Alevin --> Fry --> Parr --> Smolt --> Post-smolt --> Adult
**Key biology:** Anadromous (freshwater birth, marine growth). Smoltification involves silver coloration, salt-tolerance development, behavioral changes. Farmed fish are selected for fast growth and late maturation.

## 12.2 Nile Tilapia (Oreochromis niloticus)

| Parameter | Value |
|-----------|-------|
| Market size | 400 g - 1 kg |
| Production cycle (grow-out) | 3-5 months |
| Optimal growth temp | 27-32 deg C |
| Best SGR observed | 2.93 %/day at 32 deg C |
| Lethal low temp (adults) | 11-12 deg C |
| Lethal high temp (adults) | 42 deg C |
| FCR | 1.4-2.5 |
| Optimal pH | 6.0-8.0 |
| Min DO | > 3.0 mg/L (tolerant species) |
| Stocking density (intensive) | Up to 136 kg/m3 |

**Key biology:** Warm freshwater. Extremely hardy and tolerant of poor water quality. Mouth-brooding reproduction. Male monosex culture preferred (XX males or hormonal sex reversal) to prevent uncontrolled reproduction. Can filter-feed on phytoplankton/biofloc.

## 12.3 Pacific White Shrimp (Penaeus vannamei)

| Parameter | Value |
|-----------|-------|
| Market size | 18-25 g (90-day culture) |
| Production cycle | ~90 days |
| Optimal growth temp (small) | 30 deg C |
| Optimal growth temp (large) | 27 deg C |
| Salinity range | 0.5-45 ppt |
| Optimal salinity | 10-15 ppt (isosmotic) |
| FCR | 1.2-2.0 |
| Stocking density (intensive) | 70-150/m2 |
| Stocking density (super-intensive) | Up to 400/m2 |
| Production rate | 12-24 tonnes/ha |
| Survival rate | 70-90% |
| Min DO | > 5.0 mg/L |
| Optimal pH | 8.0-8.5 |

**Key biology:** Euryhaline (wide salinity tolerance). Grows well in low salinity (isosmotic at 10-15 ppt minimizes osmoregulatory energy cost). Molting cycle creates periodic vulnerability. Nocturnal feeder. Susceptible to White Spot Syndrome Virus (WSSV) and Early Mortality Syndrome (EMS/AHPND).

## 12.4 Channel Catfish (Ictalurus punctatus)

| Parameter | Value |
|-----------|-------|
| Market size | 0.5-1.5 kg |
| Production cycle | 6-10 months |
| Optimal growth temp | 25-30 deg C |
| Feeding stops | < 13 deg C |
| Survival range | Near-freezing to ~32 deg C |
| FCR | 1.5-2.0 |
| Feeding rate | 3-5% BW/day |
| DO threshold | > 3.0 mg/L for good growth |
| Stocking density | Highly variable (pond-dependent) |

**Key biology:** Warmwater. Bottom feeder. Can tolerate low DO briefly. Susceptible to Enteric Septicemia (Edwardsiella ictaluri) at 22-28 deg C. Overwinters without feeding. Spawns in cavities.

## 12.5 Rainbow Trout (Oncorhynchus mykiss)

| Parameter | Value |
|-----------|-------|
| Market size | 250 g - 3 kg |
| Production cycle | 12-18 months |
| Optimal growth temp | 12-16 deg C |
| Upper lethal | ~25 deg C |
| FCR | 1.0-1.5 |
| DO requirement | > 6 mg/L |
| Stocking density (raceways) | 10-40 kg/m3 |

**Key biology:** Coldwater. Resident form of steelhead. Sensitive to water quality (ammonia, nitrite). Fin erosion is a major welfare concern at high densities. High-quality flesh commands premium prices.

---

# 13. Biofloc Technology

## 13.1 Principles

Biofloc technology (BFT) is a zero or minimal water exchange system that leverages heterotrophic bacterial communities to convert waste nitrogen into microbial protein.

**Core concept:** By adding a carbon source to maintain a high C:N ratio, heterotrophic bacteria outcompete autotrophic nitrifiers and immobilize ammonia directly into bacterial biomass (biofloc), which the cultured animals can consume as supplemental food.

## 13.2 Carbon-to-Nitrogen Ratio

| C:N Ratio | Dominant Pathway |
|-----------|-----------------|
| < 10:1 | Autotrophic nitrification (NH3 --> NO2 --> NO3) |
| 10-15:1 | Mixed autotrophic + heterotrophic |
| 15-20:1 | Heterotrophic dominance (optimal for BFT) |
| > 20:1 | Carbon excess (not beneficial) |

**Key advantage:** Heterotrophic bacteria grow 10x faster than nitrifying bacteria and produce 10x more biomass per unit substrate. Ammonia immobilization is therefore much more rapid.

## 13.3 Biofloc Composition

- Heterotrophic bacteria (Bacillus, Pseudomonas -- primary floc builders)
- Autotrophic nitrifiers (Nitrosomonas, Nitrobacter)
- Microalgae / phytoplankton
- Zooplankton (rotifers, protozoa)
- Dead organic matter
- Extracellular polymeric substances (EPS -- the "glue")

## 13.4 BFT Requirements

- **Continuous aeration** (24/7 -- keeps floc suspended and aerobic)
- **Carbon source addition** (molasses, sugar, wheat flour, rice bran)
- **Monitoring:** TSS (target 300-500 mg/L floc volume), TAN, DO, pH, alkalinity
- **Best species:** Tilapia (filter-feeds on floc), shrimp (detritivore), catfish

## 13.5 Benefits

1. Reduced or eliminated water exchange
2. Reduced feed costs (biofloc provides ~25-50% of nutritional needs)
3. Improved FCR
4. Improved biosecurity (closed system)
5. Probiotic effect (competitive exclusion of pathogens)

---

# 14. Welfare Science & Cortisol

## 14.1 Cortisol as Stress Biomarker

**Cortisol levels in salmonids (plasma):**
| State | Cortisol (ng/mL) |
|-------|-----------------|
| Unstressed baseline | 0-5 |
| Chronic stress (confinement, crowding) | ~10 (sustained weeks) |
| Acute stress (handling, 1 hr confinement) | 40-200 |
| Recovery to baseline after acute stress | 24-48 hours |

**Rainbow trout response magnitude:** 12-29 fold increase across matrices (plasma, mucus, muscle, fins).

## 14.2 Stress Response Cascade

**Primary response (seconds to minutes):**
- Catecholamine release (adrenaline, noradrenaline)
- Cortisol release via HPA/HPI axis

**Secondary response (minutes to hours):**
- Blood glucose elevation
- Blood lactate increase
- Osmotic disturbance
- Immune cell redistribution

**Tertiary response (hours to weeks):**
- Growth suppression
- Reproductive impairment
- Immune suppression --> increased disease susceptibility
- Behavioral changes (see Section 8)
- Chronic cortisol elevation can downregulate cortisol receptors

## 14.3 Stress Sources in Aquaculture

| Stressor Category | Specific Stressors |
|-------------------|-------------------|
| Physical | Handling, grading, transport, crowding, netting |
| Water quality | Low DO, high ammonia, temperature extremes, pH shifts |
| Social | Hierarchies, aggression, territorial disputes |
| Husbandry | Vaccination, light manipulation, feed deprivation |
| Sensory | Noise, vibration, sudden lighting changes |

## 14.4 Coping Styles

Fish exhibit individual variation in stress responses:
- **Proactive (bold):** Lower cortisol production, higher aggression, active coping strategies
- **Reactive (shy):** Higher cortisol production, lower aggression, passive coping strategies

These are heritable traits with implications for selective breeding in aquaculture.

---

# 15. Algae Blooms & Eutrophication

## 15.1 Harmful Algal Blooms (HABs)

**Causes in aquaculture:**
- Excess nitrogen and phosphorus from uneaten feed and fish waste
- Warm temperatures + high light + nutrient loading
- Reduced grazing pressure (overfishing alters food webs)

**Key organisms:** Cyanobacteria (blue-green algae: Microcystis, Anabaena, Cylindrospermopsis)

## 15.2 Mechanisms of Fish Kill

1. **Oxygen depletion:** Dense bloom dies --> massive BOD from decomposition --> DO crashes to zero --> mass suffocation
2. **Toxin production:** Cyanotoxins (microcystins, cylindrospermopsin, saxitoxin) cause liver damage, neurological effects, gill damage
3. **pH extremes:** Dense blooms drive pH above 10 during peak photosynthesis
4. **Physical gill damage:** Some algae species (Heterosigma, Chattonella) directly damage gill epithelium

## 15.3 Prevention

1. **Reduce nutrient input:** Improve FCR, reduce overfeeding, optimize stocking
2. **Water exchange:** Dilutes nutrients and algal inocula
3. **Clay flocculation:** Sprinkle clay particles during bloom to flocculate and settle algal cells
4. **Maintain alkalinity:** Buffers against pH spikes from photosynthesis
5. **Aeration:** Prevents stratification and nocturnal DO crashes
6. **Monitoring:** Secchi disk transparency (> 30 cm), chlorophyll-a levels, phytoplankton cell counts

## 15.4 Eutrophication Link

Aquaculture-derived nutrients (N, P) have significantly enhanced eutrophication in areas of heavy aquaculture activity. Reducing feed consumption per mass of fish produced and introducing nutrient recycling are the primary mitigation strategies.

---

# 16. Master Water Quality Parameter Table

## Comprehensive Thresholds for Aquaculture Management

| Parameter | Unit | Optimal | Acceptable | Stress | Lethal | Notes |
|-----------|------|---------|------------|--------|--------|-------|
| **Dissolved Oxygen** | mg/L | > 5.0 | 3.5-5.0 | 2.0-3.5 | < 2.0 | Coldwater species need > 6.5 |
| **Temperature** | deg C | Species-specific | +/- 5 of optimal | Near CTmin/CTmax | Beyond CTmin/CTmax | See Section 6.2 |
| **pH** | -- | 6.5-8.5 | 6.0-9.0 | < 5.0 or > 10.0 | < 4.0 or > 11.0 | Affects ammonia toxicity |
| **TAN** | mg/L | < 1.0 | 1.0-3.0 | > 3.0 | Species/pH dependent | Must calculate UIA fraction |
| **UIA-N (NH3)** | mg/L | < 0.02 | 0.02-0.05 | 0.05-0.20 | > 2.0 | pH and temp dependent |
| **Nitrite (NO2-N)** | mg/L | < 0.10 | 0.10-0.25 | 0.25-1.0 | > 1.0 | Use Cl:NO2 > 6:1 protection |
| **Nitrate (NO3-N)** | mg/L | < 50 | 50-150 | 150-250 | > 400 | Accumulates in RAS |
| **CO2** | mg/L | < 10 | 10-20 | 20-40 | > 60 | Suffocation + pH depression |
| **Alkalinity** | mg CaCO3/L | 100-200 | 50-300 | < 40 | < 20 (no buffering) | Consumed by nitrification |
| **Hardness** | mg CaCO3/L | 50-300 | 30-500 | < 20 | -- | Affects osmoregulation |
| **Salinity** | ppt | Species-specific | -- | -- | -- | See Section 12 |
| **TSS** | mg/L | < 15 | 15-25 | > 25 | > 80 | Gill clogging, light reduction |
| **Turbidity** | NTU | < 25 | 25-50 | > 80 | -- | Affects feeding behavior |

## Key Interactions Between Parameters

1. **pH x Temperature --> Ammonia toxicity:** Higher pH + higher temp = exponentially more toxic NH3
2. **Temperature x DO --> Fish stress:** Higher temp = lower DO saturation + higher metabolic demand = double jeopardy
3. **CO2 x pH --> Acidification:** More CO2 = lower pH = more ionized (less toxic) ammonia but direct CO2 toxicity
4. **Alkalinity x Nitrification --> pH stability:** Each gram TAN oxidized consumes 7.14 g alkalinity, potentially crashing pH
5. **Stocking density x All parameters:** Higher density amplifies every water quality challenge
6. **Feeding rate x O2/NH3/CO2:** Each kg feed produces ~0.3-1.0 kg O2 demand, ~40g TAN, and proportional CO2

---

## Sources

- [Frontiers in Marine Science - SGR and TGC Relationships](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1332912/full)
- [UF/IFAS - Dissolved Oxygen for Fish Production](https://ask.ifas.ufl.edu/publication/FA002)
- [UF/IFAS - Ammonia in Aquatic Systems](https://ask.ifas.ufl.edu/publication/FA031)
- [Global Seafood Alliance - DO Concentrations in Pond Aquaculture](https://www.globalseafood.org/advocate/dissolved-oxygen-concentrations-pond-aquaculture/)
- [Global Seafood Alliance - Water Quality Part 2: pH, CO2, Alkalinity](https://www.globalseafood.org/advocate/water-quality-impacts-on-health-and-performance-of-fish-and-shrimp-part-2-ph-carbon-dioxide-alkalinity-hardness-and-the-water-buffering-system/)
- [Global Seafood Alliance - DO Requirements in Aquatic Animal Respiration](https://www.globalseafood.org/advocate/dissolved-oxygen-requirements-in-aquatic-animal-respiration/)
- [Global Seafood Alliance - Understanding Oxygen Demand of Aquafeeds](https://www.globalseafood.org/advocate/understanding-oxygen-demand-aquafeeds/)
- [Global Seafood Alliance - Oxygen Demand of Aquaculture Feed](https://www.globalseafood.org/advocate/oxygen-demand-aquaculture-feed/)
- [Global Seafood Alliance - Estimating Biofilter Size for RAS](https://www.globalseafood.org/advocate/estimating-biofilter-size-for-ras-systems/)
- [Global Seafood Alliance - Flow Rate Estimation for RAS](https://www.globalseafood.org/advocate/flow-rate-estimation-for-ras/)
- [Global Seafood Alliance - Examining Water Temperature in Aquaculture](https://www.globalseafood.org/advocate/examining-water-temperature-aquaculture/)
- [FAO - Dissolved Oxygen in Aquaculture](https://www.fao.org/4/ac175e/AC175E04.htm)
- [FAO - Feeding Rate Tables](https://www.fao.org/4/s4314e/s4314e09.htm)
- [FAO - Main Fish Diseases and Their Control](https://www.fao.org/4/ac264e/ac264e07.htm)
- [PMC - Behavioural Indicators of Welfare in Farmed Fish](https://pmc.ncbi.nlm.nih.gov/articles/PMC3276765/)
- [PMC - Cortisol as Stress Indicator in Fish](https://pmc.ncbi.nlm.nih.gov/articles/PMC10341563/)
- [PMC - Mathematical Model for Predicting O2 in Tilapia Farms](https://pmc.ncbi.nlm.nih.gov/articles/PMC8677810/)
- [PMC - Effects of Temperature on Feeding in Fish](https://pmc.ncbi.nlm.nih.gov/articles/PMC7678922/)
- [PMC - Atlantic Salmon Parr Temperature Effects in RAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12561879/)
- [PMC - Biofloc Technology Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11117240/)
- [PMC - Review of HABs Causing Marine Fish Kills](https://pmc.ncbi.nlm.nih.gov/articles/PMC10871120/)
- [Springer - Stress Responses and Disease Resistance in Salmonids](https://link.springer.com/article/10.1007/BF00004714)
- [Frontiers - Finding the Golden Stocking Density](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2022.930221/full)
- [Atlas Scientific - Water Quality Parameters for Fish Farming](https://atlas-scientific.com/blog/water-quality-parameters-for-fish-farming/)
- [ScienceDirect - Feed Conversion Ratio Overview](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/feed-conversion-ratio)
- [ScienceDirect - Biofloc Technology Overview](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/biofloc-technology)
- [CWEA - How Alkalinity Affects Nitrification](https://www.cwea.org/news/how-alkalinity-affects-nitrification/)
- [FishBase - POPGROWTH Table](https://www.fishbase.se/manual/fishbasethe_popgrowth_table.htm)
- [Aquaculture Research (Wiley) - TGC Model Cautionary Note](https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1365-2109.2003.00859.x)
- [Reviews in Aquaculture (Wiley) - Fish Growth Calculation Review](https://onlinelibrary.wiley.com/doi/abs/10.1111/raq.12071)
