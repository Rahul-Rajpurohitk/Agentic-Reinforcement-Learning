# Commercial Fish Farm Operations -- Deep Research Report
> Compiled April 2026 from 50+ industry sources, academic papers, government data, and trade publications.

---

## TABLE OF CONTENTS
1. [Daily Operations -- What Workers Actually Do](#1-daily-operations)
2. [Monitoring Systems & Sensor Technology](#2-monitoring-systems)
3. [Feeding Systems & Technology](#3-feeding-systems)
4. [Aeration & Dissolved Oxygen Management](#4-aeration--dissolved-oxygen)
5. [Water Quality Parameters -- Thresholds by Species](#5-water-quality-parameters)
6. [Water Exchange & Flow Management](#6-water-exchange)
7. [Economics & Cost Structure](#7-economics)
8. [Disasters & Mass Mortality Events](#8-disasters--mass-mortality)
9. [Norwegian & Scottish Salmon Operations](#9-norway--scotland)
10. [Warm-Water Species (Tilapia & Catfish)](#10-warm-water-species)
11. [Shrimp Farm Operations](#11-shrimp-farming)
12. [Weather & Climate Impact](#12-weather--climate)
13. [Regulations & Environmental Standards](#13-regulations)
14. [Physical Infrastructure -- Pen & Cage Specs](#14-infrastructure)
15. [Mortality Rates by Species](#15-mortality-rates)
16. [Harvest & Grading Operations](#16-harvest-operations)
17. [Industry Scale & Global Statistics](#17-industry-scale)

---

## 1. DAILY OPERATIONS -- WHAT WORKERS ACTUALLY DO

### A Typical Day on a Sea-Cage Salmon Farm (Scotland/Norway)

**No two days are the same.** Plans change hour to hour. But core tasks are non-negotiable:

**Morning (first thing):**
- Environmental checks BEFORE any feeding begins
  - Measure salinity, dissolved oxygen, plankton content, water visibility
  - Flag any abnormalities that might affect feeding or fish behavior
- Check weather conditions and adjust plan for the day

**All-Day Task -- Feeding (consumes one full staff member's entire day):**
- Computer-controlled system delivers pellets through pipes, blowers, and spinners
- TWO cameras per pen: one underwater, one at surface -- monitoring fish behavior and feeding depth
- Staff watch screens and adjust feed amounts in real-time to minimize waste
- Even when the site can't be reached due to weather, feeding is done remotely from the shore base
- **This is the #1 most important task. It happens every single day without fail.**

**Daily -- Mortality Checks:**
- Each pen has a basket sitting at the base of the net to collect dead fish
- Baskets are brought to the surface by capstans
- Workers remove, count, and record every dead fish
- Serves as: (a) cleanliness for the pen environment, (b) early warning system for health problems
- **Mortality spike = first sign of disease outbreak or environmental stress**

**Weekly -- Health Assessments (mandatory in Scotland, publicly reported):**
- Lice counts on 20 sample fish per pen
- Gill assessments scored for amoebic gill disease (AGD) and proliferative gill disease
- Swab samples sent to diagnostic lab
- Two lice species distinguished; life stages documented separately

**Regular -- Net Cleaning:**
- Remote-operated net cleaners (RONC) with camera systems
- High-pressure water jets remove marine growth (biofouling)
- Critical because fouled nets restrict water flow and reduce oxygen exchange

**Every ~2 Weeks -- Dive Teams:**
- Professional divers inspect nets underwater for holes, wear, predator damage
- Repair any damage found

**Ongoing -- General Maintenance:**
- Camera repositioning, winch repairs, engine servicing
- Pen structural checks, barge maintenance, boat upkeep

### Pond Farm Daily Routine (Catfish, Tilapia -- USA)

**Dawn/Early Morning:**
- Check dissolved oxygen -- this is the CRITICAL window (DO is at its daily minimum at dawn)
- If DO is low, activate emergency aerators immediately
- Walk pond banks checking for signs of distress (fish gulping at surface = low oxygen emergency)

**Morning -- Feeding:**
- Feed early morning when DO levels begin rising
- NEVER feed near dark or at night (oxygen demand from digestion + nighttime DO drop = lethal combo)
- Feed once daily for food fish grow-out (twice daily is NOT more effective -- research shows extra feed isn't converted)
- Maximum feeding rates:
  - Regular ponds: 120 lbs/acre/day
  - Intensively aerated ponds: 200 lbs/acre/day
  - Split ponds: 250 lbs/acre/day

**Throughout Day:**
- Monitor water color (should be greenish -- indicates healthy phytoplankton)
- Check water control structures, canal and dike condition
- Pest and predator surveillance (birds are a major issue)
- Equipment checks on aerators, pumps

**Weekly:**
- Water quality testing: temperature, Secchi disc transparency, DO, pH, alkalinity
- Pond bank vegetation control
- Compost pile management

**Monthly/Bi-weekly:**
- Fish sampling: net a sample of 1-2% of population by weight
- Weigh, measure, calculate growth rate and FCR
- Adjust feeding rations based on growth data

### Record-Keeping (Commercial Operations)
Commercial farms maintain MULTIPLE specialized records:
- Stocking data (date, number, size, source)
- Harvesting data (date, number, weight, buyer)
- Periodic sampling results (growth rate, survival)
- Feed distribution (daily amounts per pond)
- Water quality readings
- Accounts and loan tracking
- Mortality logs

---

## 2. MONITORING SYSTEMS & SENSOR TECHNOLOGY

### Parameters Monitored

| Parameter | Why It Matters | Measurement Method |
|-----------|---------------|-------------------|
| Dissolved Oxygen (DO) | **#1 most critical** -- fish die within hours if too low | Optical (luminescent) or electrochemical probes |
| Temperature | Controls metabolism, growth rate, disease susceptibility | Thermistors or RTDs |
| pH | Affects ammonia toxicity, fish stress | Electrochemical glass electrode |
| Salinity | Species tolerance varies widely | Conductivity sensors |
| Ammonia (NH3/NH4+) | Toxic waste product of fish metabolism | Ion-selective electrode or colorimetric |
| Nitrite (NO2-) | Intermediate toxin from biofilter | Colorimetric or ion-selective |
| Turbidity | Indicates plankton density, suspended solids | Optical (nephelometric) |
| Chlorophyll-a | Algal bloom indicator | Fluorescence sensors |

### Sensor Specifications (Industry Standard -- Innovasea aquaMeasure)
- **Temperature range:** -5C to 35C
- **DO range:** 0-150% saturation (optical-based)
- **Data logging:** Up to 12 months continuous
- **Data transfer:** Bluetooth to cloud
- **Other parameters available:** Salinity, depth, chlorophyll, turbidity, blue-green algae, CDOM/FDOM

### Sampling Rates in Practice
| Application | Interval | Notes |
|-------------|----------|-------|
| Intensive RAS / hatchery | 1 sample/second | Factory-level continuous |
| IoT sensor networks | Every 10 seconds | Research/high-tech farms |
| Standard commercial | Every 10 minutes | Most common automated setup |
| Manual spot checks | 1-4x daily | Basic pond farms |
| Validation / calibration | Every 6 hours | Cross-checking sensor accuracy |

### Available Monitoring Platforms
- **YSI (Xylem):** ProSolo (handheld), IQ SensorNet 2020 3G (up to 20 parameters), ODO RTU (continuous)
- **Innovasea:** aquaMeasure wireless sensors, cloud-based dashboards
- **Campbell Scientific:** Automated stations with Modbus TCP/IP, web/mobile alerts
- **IoT/Custom:** Arduino/ESP32 + sensor modules, increasingly common in developing regions

### Alert Systems
Modern systems provide:
- Real-time alerts when DO falls outside optimal range
- SMS/email notifications to farm manager's phone
- Automated aerator activation when DO drops below threshold
- Historical trend analysis for predictive warnings

---

## 3. FEEDING SYSTEMS & TECHNOLOGY

### Feeder Types
1. **Demand feeders:** Fish trigger feed release by bumping a pendulum; no electricity needed; tailored to fish appetite
2. **Timed automatic feeders:** Dispense set amounts at programmed intervals
3. **Computerized/centralized systems:** Cloud-controlled, camera-monitored, adjusts rations based on fish behavior, size, water temperature
4. **Pneumatic/hydraulic blower systems:** Pellets blown through pipes to multiple pens from central hopper (standard in sea-cage operations)

### How Camera-Based Feeding Works (Modern Salmon Farms)
1. Central feed barge holds feed silos (100+ tonnes capacity)
2. Pellets distributed through pneumatic pipes to individual pen positions
3. Each pen has 2 cameras (underwater + surface)
4. Operators watch video feeds and control feeding rate per pen
5. System detects uneaten pellets sinking and stops feeding
6. Some systems now use AI/computer vision for automated pellet detection

### Feed Conversion Ratios (FCR) by Species -- Industry Actuals

| Species | Best Practice FCR | Industry Average FCR | Notes |
|---------|------------------|---------------------|-------|
| Atlantic Salmon | 1.0-1.15 | 1.15-1.3 | Best in aquaculture |
| Rainbow Trout | 0.75-1.2 | 1.0-1.2 | Very efficient |
| Tilapia | 1.5 | 1.5-1.6 | Herbivorous advantage |
| Shrimp (vannamei) | 1.1-1.2 | 1.2-1.6 | Biofloc can improve |
| Channel Catfish | 1.8 | 2.0-2.5 | Large gap between theoretical and farm-level |

### Feeding Rates (% body weight per day)
- Salmon: 0.5-2% depending on temperature and fish size
- Tilapia: 2-5% (higher for fingerlings)
- Catfish: Based on maximum lbs/acre/day rather than body weight
- Shrimp: Multiple small feedings (4x/day), adjusted after sampling uneaten feed 30 min post-feeding

### Feed Composition
- **Salmon feed:** ~40% protein, ~30% fat, marine + plant ingredients
- **Catfish feed:** 28-32% protein, floating extruded pellets
- **Shrimp feed:** ~40% protein, 9% fat, 1.5mm to 2.5mm pellet size progression

### Feed Cost
- **Feed = 50-70% of total operating costs** across ALL species and systems
- In salmon: feed = 44% of total production cost per kg (Norway 2022)
- In shrimp: feed = 70-80% of variable costs
- Global salmon feed market: massive -- Norwegian farms alone consume millions of tonnes annually

---

## 4. AERATION & DISSOLVED OXYGEN MANAGEMENT

### Why DO Is the #1 Killer
- Fish consume oxygen 24/7 through respiration
- Phytoplankton produce oxygen during daylight but CONSUME it at night
- **The danger window is 3-6 AM** -- when DO hits its daily minimum
- Warm water holds LESS dissolved oxygen (20C: 9.07 mg/L max; 30C: 7.54 mg/L max)
- An overnight oxygen crash can kill an entire pond in hours

### DO Thresholds by Species (mg/L)

| Species | Optimal | Stress Begins | Lethal | Notes |
|---------|---------|--------------|--------|-------|
| Atlantic Salmon | >7.0 | <6.0 | <3.0 | Cold-water; very sensitive |
| Rainbow Trout | >7.0 | <5.0 | <1.6 (juvenile) | Need cold, well-oxygenated water |
| Salmon Eggs | >11.0 | <8.0 | -- | Delayed hatching below 11 |
| Channel Catfish | >5.0 | <3.5 | <2.0 | Tolerant; aerated ponds often run <3 |
| Tilapia | >5.0 | <3.0 | <1.0 | Very tolerant; survive <1 ppm briefly |
| Penaeid Shrimp | >4.0 | <3.0 | <1.2 | Deaths immediate below 1.2 |
| General Warmwater | >5.0 | <4.0 | <3.0 | Rule of thumb |
| General Coldwater | >7.0 | <6.0 | <3.0 | Rule of thumb |

### Catfish/Shrimp DO Data (Research)
**Channel catfish response to DO levels:**
- 2.91 mg/L (36% saturation): Reduced feeding and growth
- 3.5 mg/L: Minimum for acceptable survival, production, FCR
- 4.85 mg/L (60% saturation): Good performance

**Shrimp response to DO levels:**
- 2.32 mg/L: 42% survival, 2,976 kg/ha, FCR 2.64
- 2.96 mg/L: 55% survival, 3,631 kg/ha, FCR 2.21
- 3.89 mg/L: 61% survival, 3,975 kg/ha, FCR 1.96
- **Every mg/L of DO improvement = dramatic improvement in survival and FCR**

### Aerator Types & Efficiency

| Aerator Type | SAE (kg O2/kWh) | Best Use Case | Cost |
|-------------|-----------------|---------------|------|
| Paddlewheel (well-designed) | 1.5-2.0 | Pond emergency + routine | $200-1,000/unit |
| Paddlewheel (Asian-style) | 0.5-1.0 | Low-cost pond farms | Cheaper |
| Fine-bubble diffuser | 2.0-2.5 | Deep ponds, continuous use | Blower $500-1,500 + membranes $30-50 each |
| Pure oxygen injection | ~20 | RAS, hatcheries, high-density | Tens of thousands USD |

**Critical real-world factor:** Actual oxygen transfer in the field is only **40-60% of SAE** due to water temperature, salinity, organic loading, and other conditions.

### Oxygen Demand Rule of Thumb
- **1.25 kg O2 consumed per 1 kg feed applied** (Boyd's stoichiometric analysis)
- Design aeration for 30-50% above average demand as safety margin
- Example: 1,000 kg feed over 100 days at 20 h/day aeration = need to deliver 0.625 kg O2/hour minimum

### Aeration Sizing
- Standard: 1 hp per surface acre for emergency aeration
- Intensive shrimp farms (Thailand): 24-36 hp/ha (18-27 kW/ha), running ~20 hours/day
- Aeration cost for shrimp: $0.41-0.53/kg shrimp produced (at $0.10/kWh)

### Salinity Effect on Paddlewheel Efficiency
SAE increases significantly with salinity:
- Freshwater: 1.6-2.1 kg O2/kWh (avg 1.93)
- 11 ppt salinity: 3.1-3.4 (avg 3.22)
- 22 ppt salinity: 3.3-3.7 (avg 3.46)

### Emergency Aeration Protocol
1. Detect low DO (automated sensor alarm or visual observation of fish gulping at surface)
2. Deploy tractor-driven paddlewheels (most common emergency aerator on commercial farms)
3. Run until DO improves -- typically 30 minutes to 2 hours to see improvement
4. Monitor continuously until dawn crisis passes

---

## 5. WATER QUALITY PARAMETERS -- THRESHOLDS

### Master Table of Water Quality Parameters

| Parameter | Optimal Range | Stress Level | Lethal Level | Monitoring Frequency |
|-----------|--------------|-------------|-------------|---------------------|
| **Dissolved Oxygen** | 5-10 mg/L (warmwater) / 7+ (coldwater) | <4 / <6 mg/L | <3 / <3 mg/L | Continuous or 2-4x daily |
| **Temperature** | Species-dependent (see below) | +/- 5C from optimal | Species-dependent | Continuous |
| **pH** | 6.5-8.5 | <6.0 or >9.0 | <4.5 or >10.0 | Daily |
| **Ammonia (un-ionized NH3)** | <0.02 mg/L | >0.05 mg/L | >0.1 mg/L (coldwater) / >0.5 mg/L (warmwater) | Daily to weekly |
| **Total Ammonia Nitrogen (TAN)** | <1.0 mg/L | >2.0 mg/L | >5.0 mg/L | Daily to weekly |
| **Nitrite (NO2-)** | <0.5 mg/L | >1.0 mg/L | >5.0 mg/L | Weekly |
| **Nitrate (NO3-)** | <40 mg/L | >80 mg/L | >200 mg/L | Weekly to biweekly |
| **Alkalinity** | 50-300 mg/L CaCO3 | <20 mg/L | -- | Weekly to biweekly |
| **Hardness** | >50 ppm | <20 ppm | -- | Monthly |
| **Turbidity (Secchi disc)** | 30-45 cm | <20 cm or >60 cm | -- | Weekly |

### Temperature Ranges by Species

| Species | Optimal Growth | Tolerable Range | Lethal |
|---------|---------------|----------------|--------|
| Atlantic Salmon | 12-14C | 4-20C | >23C |
| Rainbow Trout | 13-18C | 4-21C | >25C |
| Channel Catfish | 27-29C (80-85F) | 16-35C | >35C |
| Tilapia | 25-30C | 20-35C | <12C (cold kill) |
| Vannamei Shrimp | 28-32C | 20-35C | <15C or >38C |

### Salinity Tolerances

| Species | Range (g/L) | Notes |
|---------|-------------|-------|
| Catfish, Pangasius, Carp | <5 | Strictly freshwater |
| Atlantic Salmon | 0-35 | Anadromous; starts fresh, moves to sea |
| Tilapia | 0-20 | Euryhaline; tolerates wide range |
| Rainbow Trout | 0-20 | Primarily freshwater |
| Vannamei Shrimp | 2-40 | Very wide tolerance |

---

## 6. WATER EXCHANGE & FLOW MANAGEMENT

### System Types and Exchange Rates

| System | Water Exchange Rate | Recirculation % | Notes |
|--------|-------------------|-----------------|-------|
| Earthen pond | 0-5% daily | None | Relies on rain, evaporation replacement |
| Flow-through (raceway) | 100%+ daily | None | Constant fresh water supply |
| RAS | 5-10% daily | 90-99% | Biofilter handles waste; minimal new water |
| Biofloc | Near zero | N/A | Bacteria process waste in situ |
| Sea cage | Ambient current | N/A | Ocean current provides exchange |

### RAS Specifics
- Daily water exchange: typically 5-10% of total system volume
- Biofilter must receive adequate oxygen (nitrifying bacteria are aerobic)
- pH must stay 5.0-9.0 for freshwater RAS
- Alkalinity supplemented with sodium bicarbonate
- Solids removal is critical -- reduces bacteria growth, oxygen demand, disease
- **24/7 operation required** -- pump failure = fish death within hours
- Skilled technical staff mandatory

### RAS Water Quality Impact (Research)
- High exchange rate (2.6%): Better fish health, fewer lesions, less fin erosion
- Low exchange rate (0.26%): Increased splenic and skin lesions, elevated urea
- Higher flow rates (6 tank volumes/hour): Specific growth rate 2.74%/day vs 2.21%/day at low flow

### Biofloc Water Management
- Near-zero water exchange (the defining feature)
- Heterotrophic bacteria convert ammonia/nitrite into bacterial protein
- C:N ratio maintained at 12-15:1 (feed is naturally 7-10:1, so carbon supplementation needed)
- Carbon source: 0.5-1 kg molasses per 1 kg feed
- Settleable solids target: 10-15 mg/L (>500 mg/L = gill stress)
- Clarifier turnover: every 3-4 days, sized at 1-5% of system volume

---

## 7. ECONOMICS & COST STRUCTURE

### Norwegian Atlantic Salmon (World's Benchmark)

**Production Cost Per Kg (2022 -- Directorate of Fisheries Survey, 992 of 1,170 licenses):**
- **Total: NOK 49.12/kg** (~$4.67 USD) -- up 17.9% year-over-year
- Production costs surged 45% from 2016 to 2022

**Cost Breakdown (2022):**
| Category | Share | NOK/kg (approx) | Notes |
|----------|-------|-----------------|-------|
| Feed | 44% | ~21.6 | Increased 28.8% YoY due to grain prices |
| Sea lice treatment | ~10-12% | ~5-6 | 5 billion NOK total annually |
| Smolt/fry | ~7% | ~3.4 | NOK 16.97 per smolt produced |
| Labor/salaries | ~8-10% | ~4-5 | |
| Depreciation | ~5-7% | ~2.5-3.5 | |
| Other operating costs | ~15-20% | ~7-10 | Insurance, transport, slaughter |

**Revenue (2022):**
- Average sales price: NOK 63.69/kg -- up 31.6% YoY
- Operating margin: 29.1% (up from 18% in 2021)
- Industry EBITDA: NOK 50.2 billion (65.7% surge vs 2021)
- Pre-tax profit: NOK 35.5 billion (doubled from NOK 15.9bn in 2021)

**Sea Lice Costs (Norway, detailed):**
- Total annual: 5-6.6 billion NOK (~$500-660M)
- Cleaner fish: 1.2 NOK/kg salmon (nearly 1 billion NOK total)
  - ~50 million cleaner fish stocked annually
- Louse laser (Stingray): 0.34 NOK/kg
- Bath treatments: 0.5-1 NOK/kg
- Mechanical (Hydrolicer/Thermolicer): 0.55 NOK/kg
- Treatment well-boats: $30-40M to build, up to $100K/day operating cost
- Starvation before treatment: ~700 million NOK annual industry loss
- 300% increase in mechanical treatments from 2016-2020
- ~3,000 treatments performed annually

**International Comparison (2018 data):**
| Country | Production Cost/kg |
|---------|-------------------|
| Chile | NOK 35.40 |
| Norway | NOK 37.85 |
| Faroe Islands | NOK 38.80 |
| Canada | NOK 41.80 |
| Scotland | NOK 48.20 |

### US Catfish
- Average production cost: ~$1.15/lb ($2.53/kg)
- Optimized: $0.95/lb
- Sale price: ~$1.40/lb
- Profit margin: 18-32% depending on efficiency
- Feed: 50-70% of total costs

### Tilapia (USA)
- 147 US farms, total sales $51.2 million (2023)
- Production in ponds or RAS

### Shrimp (Global)
- Production cost varies enormously by country:
  - Ecuador: $2.30-2.40/kg (lowest)
  - India: $3.40-3.80/kg
  - Vietnam: $4.80-5.00/kg (highest, due to high-tech failures)
- Feed: 70-80% of variable costs
- Vietnam survival rate: <40% (vs Ecuador >90%, India >60%, Thailand 55%)

### RAS Economics
- Energy = up to 40% of OPEX for land-based RAS facilities
- 24/7 electricity required for pumps, heaters, chillers, aeration, filtration
- Production cost: ~EUR 4.51/kg for RAS vs EUR 2.68/kg for net cage (same mortality assumption)
- RAS costs ~68% more than cage farming per kg

### General Fish Farming Economics
- Small farm ($2,400-7,600 startup): $2,500-5,000 annual operating
- Medium farm ($25,000-92,000 startup): $16,000-35,000 annual operating
- Large farm ($280,000+ startup): $270,000+ annual operating
- Well-managed US operations: 10-30% profit margins
- Feed dominates ALL cost structures at 50-70%

---

## 8. DISASTERS & MASS MORTALITY EVENTS

### Scale of the Problem
- **865 million farmed salmon died in mass die-offs** across 6 nations in 10 years (2012-2022)
- Cost: **>$15 billion** to the four largest producing countries since 2013
- Events are **increasing in both frequency and magnitude** (statistically significant trends in Norway, Canada, UK)

### Country-Level Expected Maximum Loss (worst 0.1% event/year)
| Country | Maximum Fish Lost |
|---------|------------------|
| Chile | 8.19 million |
| Norway | 5.14 million |
| Canada | 5.05 million |
| New Zealand | 4.39 million |
| Australia | 1.55 million |

### Specific Real-World Incidents

**Chile, 2016 -- Algal Bloom:**
- 39 million farmed salmon killed
- Algae species: *Pseudochattonella*
- Financial loss: **$800 million**
- 4,500 direct jobs lost; 6,000 fishermen affected
- Lost >12% of annual production

**Norway, May 2019 -- Algal Bloom:**
- ~8 million fish killed in ocean net pens
- Algae species: *Chrysochromulina leadbeateri*
- 2.5 million fish were successfully relocated in Nordland region
- Same species caused kills in 1991 and 2008

**Scotland, 2023:**
- 17.4 million farmed salmon died
- Primary cause: warming water
- Tasmania: 68 mass fish death incidences since 2019

**Newfoundland, Canada, 2019:**
- 2.6 million salmon across 10 farms
- Cause: Prolonged warm water; fish suffocated swimming to pen bottoms seeking cooler water
- Compounded by: Chemical parasite treatments had weakened fish

**British Columbia, Canada:**
- Hundreds of tonnes lost at Muchalat North farm
- 23% of farm's stock lost over 10-day period

**San Francisco Bay, 2022:**
- Harmful algal bloom (*Heterosigma akashiwo*)
- Thousands of fish killed: sturgeon, sharks, bass, rays, anchovy

### Root Causes (in order of frequency)
1. **Low dissolved oxygen / hypoxia** -- often from algal bloom die-off at night
2. **High water temperature** -- reduces DO capacity, increases stress
3. **Disease outbreaks** -- spread rapidly in high-density conditions
4. **Harmful algal blooms** -- both toxicity and oxygen depletion
5. **Human error / operational failure** -- equipment failure, delayed response
6. **Treatment mortality** -- fish die during/after lice treatments
7. **Storms / extreme weather** -- pen damage, flooding, debris

### The Overnight Oxygen Crash Mechanism
1. Algae proliferate in warm, nutrient-rich water
2. During daytime: algae produce excess oxygen through photosynthesis
3. At night: ALL organisms (algae + fish) consume oxygen, none produced
4. By 3-6 AM: DO drops below lethal threshold
5. Mass mortality occurs before dawn
6. **Can kill an entire pond/pen population in a single night**

---

## 9. NORWAY & SCOTLAND -- WORLD-LEADING SALMON OPERATIONS

### Norwegian Salmon Production Cycle
1. **Freshwater hatchery:** 6-12 months, fish grow from egg to 100-300g smolt
2. **Sea transfer:** Smolt transported to coastal net pens (spring or autumn)
3. **Sea grow-out:** 12-18 months in sea pens until harvest weight
4. **Harvest weight:** 4-6 kg
5. **Post-harvest fallowing:** Site left empty for minimum 2 months

### Norwegian Pen Specifications
- **Standard circumference:** 160m (gradually becoming norm; largest are 200m)
- **Diameter:** ~50m
- **Net depth:** 20-40m
- **Fish per pen:** Up to 200,000
- **Density regulation:** 97.5% water, 2.5% fish (by volume)
- **Licensing system:** MAB (Maximum Allowable Biomass) per license

### Norwegian Technology
- **Centralized control rooms** monitoring live video feeds from multiple farm sites
- **Camera-based feeding:** Operators watch fish behavior on screens
- **Automated environmental monitoring:** Continuous DO, temperature, salinity
- **Anti-lice technology:**
  - Cleaner fish (wrasse and lumpsucker): ~50 million stocked annually
  - Louse lasers (Stingray): ~200 units in operation
  - Mechanical delousing (Hydrolicer, Thermolicer)
  - Shift away from chemicals: 2012-2015 chemicals were 80% of treatments; by 2017 74% were thermal/mechanical

### Scottish Operations -- Distinctly Hands-On
- Farmers are on the water **every day, all year round**
- **Hand feeding** practiced at seawater sites -- "no substitute for experienced eyes on the water"
- Higher staff-to-fish ratio than industry average
- Operating in extreme conditions: Isle of Muck site is a 2-hour boat journey from nearest port
- Weather challenges: snow, freezing temperatures, gale force winds, torrential rain
- Limited infrastructure: no on-island fuel stations, infrequent winter ferries

### Key Statistics (Norway)
- Production cost: NOK 49.12/kg (2022)
- Sales price: NOK 63.69/kg (2022)
- Operating margin: 29.1%
- Annual salmon deaths in Norway: 52.8 million (2019), up 27.8% from 41.3 million (2015)

---

## 10. WARM-WATER SPECIES -- TILAPIA & CATFISH

### US Catfish (Mississippi Delta -- Heartland of US Aquaculture)

**Production System:**
- >95% produced in earthen ponds
- Pond sizes: now 10-15 acres (down from 20-40 acres -- smaller is easier to manage)
- Optimal temperature: 80-85F (27-29C)
- Some growth at 60F; feeding stops below 50F

**Daily Operations:**
- Feed 7 days/week during growing season (6 days = 3.3% less production; 5 days = 6.9% less)
- Feed early morning as DO rises
- Maximum feed rate: 120 lbs/acre/day (regular) to 250 lbs/acre/day (split ponds)
- Automated aeration systems respond to per-pond DO sensors

**Winter Management:**
- At 55F+: Feed at 1% body weight = 18% weight gain over winter
- Without winter feeding: 9% weight LOSS

**Challenges:**
- Off-flavor (muddy/musty taste from geosmin/MIB produced by blue-green algae)
  - Must submit flavor samples to processing plant BEFORE harvest
  - Off-flavor fish cannot be sold -- entire harvest delayed
- Bird predation (cormorants, egrets)
- Disease: "hamburger gill disease" (protozoan *Aurantiactinomyxon*) can kill entire pond

### Tilapia

**Production:**
- USA: 147 farms, $51.2M total sales (2023)
- Primarily in southern/southwestern states or indoor RAS
- Market size in 10-12 months; but only 7-month outdoor growing season in most southern US
- Must harvest by late fall to avoid cold front losses
- Indoor overwintering of juveniles required in most regions

**Key Needs:**
- Clean water, oxygen, food, light, room to swim
- Temperature: lethal below ~12C (cold kills)
- Very tolerant of low DO (survive <1 ppm briefly)

**Stocking Density:**
- Cage culture: 600-800 fish/m3 for 1/2 lb fish; 200-250/m3 for 1.5 lb fish
- IPRS systems: up to 136 kg/m3
- High-density cages: 100 kg/m3

---

## 11. SHRIMP FARMING

### Biofloc Vannamei Shrimp -- The Dominant System

**Stocking & Production:**
- Stocking density: 250-500 post-larvae/m2
- Production: 3-7 kg/m2 and/or 3-9 kg/m3
- FCR: 1.2:1 to 1.6:1
- Ideal pond size: 500-1,000 m2

**Daily Management Tasks:**
1. Check water quality: pH (daily), alkalinity (2x/week), settleable solids (weekly)
2. Feed 4x/day; adjust based on biomass calculations and sampling
3. Sample for uneaten feed 30 min after each feeding with dip net
4. Carbon supplementation: diluted molasses added continuously through valve
5. Monitor biofloc volume in settling cones

**Water Quality Targets:**
| Parameter | Target | Critical |
|-----------|--------|----------|
| Alkalinity | >150 mg/L | Supplement with NaHCO3 at 0.25 kg per kg feed |
| Settleable solids | 10-15 mg/L | >500 mg/L = gill stress |
| C:N ratio | 12-15:1 | Add 0.5-1 kg carbon per kg feed |

**Feeding:**
- Pellet progression: 1.5mm (early) to 2.5mm (mature)
- Protein: ~40%, Fat: ~9%
- Multiple small feedings (shrimp have tiny intestinal tracts)
- Biofloc itself provides 0.25-0.5 additional growth units per unit of commercial feed

**Solids Management:**
- Clarifier sized at 1-5% of system volume
- Residence time: 20-30 minutes
- Turnover: every 3-4 days

### Shrimp Economics
- Ecuador cost: $2.30-2.40/kg (global benchmark -- lowest)
- India: $3.40-3.80/kg
- Vietnam: $4.80-5.00/kg
- Feed: 70-80% of variable costs
- Aeration cost (Thailand): $0.41-0.53/kg shrimp at 24-36 hp/ha
- **Survival rate is the critical economic variable:**
  - Ecuador: >90%
  - India: >60%
  - Thailand: 55%
  - Vietnam: <40% (explains high cost)

---

## 12. WEATHER & CLIMATE IMPACT

### Temperature Effects
- Growth rate roughly doubles with every 10C increase (within optimal range)
- ~10% growth increase per 1C
- Warm water = less dissolved oxygen capacity:
  - 20C: 9.07 mg/L maximum DO
  - 30C: 7.54 mg/L maximum DO
- **This is why warming is so dangerous -- double hit of faster metabolism + less oxygen**

### Cloud Cover
- Consecutive overcast days reduce phytoplankton photosynthesis
- Lower daytime oxygen production
- Nighttime DO minimum becomes dangerously low
- **2-3 overcast days in a row = potential mortality event in loaded ponds**

### Wind
- **Positive:** Accelerates gas exchange, mixes water, prevents thermal stratification, reduces ammonia/H2S buildup
- **Negative:** Erosion of pond embankments from wave action

### Rainfall
- Modest contribution to DO: 10cm rainfall at 20C delivers only 0.60 mg/L to a 1.5m pond
- Can cause thermal stratification or rapid temperature change
- Flooding can overtop pond embankments, causing fish escapes

### Storms & Hurricanes
- **Cages/net pens:** Physical damage from waves, structural failure, fish escapes
- **Ponds:** Flooding, fish displacement, debris damage, power loss (= no aeration = mass mortality)
- **Storm surge:** Salinity changes in coastal ponds; fish stranding
- **Power loss is the critical threat** -- aerators stop, DO crashes
- Some offshore pens designed to submerge during hurricanes (Ocean Era design -- expensive and complex)

### Ice Cover
- Prevents oxygen diffusion from atmosphere
- Decomposition continues under ice, consuming oxygen
- "Mortality of culture animals is highly probable" under prolonged ice

### Drought
- Reduces pond volume
- Concentrates toxic metabolites and nutrients
- Triggers excessive algal blooms
- In estuaries: salinity may exceed species tolerance

---

## 13. REGULATIONS & ENVIRONMENTAL STANDARDS

### US Regulatory Framework
- **EPA:** NPDES discharge permits for facilities producing >20,000 lbs/year (cold water) or >100,000 lbs/year
- **FDA:** Feed ingredients, medication use, seafood safety
- **USDA:** Disease management, inspection
- **NOAA Fisheries:** Marine aquaculture siting, protected species
- **State/local:** Zoning, building, water use, waste discharge, species certification

### Key Regulatory Thresholds (USA)
- Cold water fish: Discharge permit required if >9,090 kg (20,000 lbs) harvest weight/year AND discharge >30 days/year
- All fish: Effluent guidelines apply if >45,454 kg (100,000 lbs)/year

### Norwegian Regulations
- **MAB system:** Maximum Allowable Biomass per license (not stocking density per se)
- **97.5% water to 2.5% fish** ratio requirement
- **NYTEK/NS 9415** technical standards for cage construction
- **Mandatory fallowing:** At least 2 months empty after harvest
- **Sea lice limits:** Maximum 0.5 adult female lice per fish (reporting mandatory)
- **Traffic light system:** Production areas assigned green/yellow/red based on environmental impact
  - Green: 6% biomass INCREASE allowed
  - Yellow: No change
  - Red: 6% biomass REDUCTION required (costs millions in lost production)

### Scottish Regulations
- Mandatory weekly lice counts on 20 fish per pen (publicly reported)
- Health assessments including gill scoring
- Swab samples to diagnostic laboratories

---

## 14. PHYSICAL INFRASTRUCTURE -- PEN & CAGE SPECIFICATIONS

### Sea Cage Sizes (Modern Commercial)

| Specification | Small/Traditional | Standard (Norway) | Large/Offshore |
|--------------|------------------|-------------------|---------------|
| Circumference | 60-90m | 120-160m | 160-240m |
| Diameter | ~20m | ~40-50m | ~50-75m |
| Net depth | 6-15m | 20-40m | 20-50m+ |
| Surface area | ~300 m2 | ~2,000 m2 | ~4,500 m2 |
| Volume | ~5,000 m3 | ~40,000-80,000 m3 | ~100,000+ m3 |
| Fish capacity | 10,000-50,000 | 100,000-200,000 | 200,000+ |

### Construction
- **Collar:** HDPE (High-Density Polyethylene) -- standard material
- **Net material:** Knotless nylon (reduces fish injury during crowding)
- **Mooring:** Multi-point anchoring to seabed; designed for local wave/current conditions
- **Infrastructure per site:** Multiple pens (typically 6-12), feed barge, accommodation barge, boat landing

### Pond Specifications (Catfish/Tilapia)
- **Size:** 5-15 acres (trend toward smaller for easier management)
- **Depth:** 1.2-2m typical
- **Shape:** Rectangular preferred (easier to seine)
- **Construction:** Earthen embankments; some lined with geomembrane
- **Water supply:** Groundwater wells or canal systems

### RAS Tank Specifications
- **Shape:** Circular (for self-cleaning flow) or raceway (rectangular)
- **Size:** Highly variable -- 10 m3 to 1,000+ m3
- **Density:** 50-100 kg/m3 (flow-through) to 100-400 fish/m3 (full RAS)
- **Fish load with pure oxygen:** Up to 80 kg per 1,000L

---

## 15. MORTALITY RATES BY SPECIES

### Atlantic Salmon (Norway -- Best Data Available)
- **15-20% of fish stocked die** during sea phase (by number)
- **6-9% by weight** (smaller fish die disproportionately)
- Annual deaths in Norway: 52.8 million (2019) -- up 27.8% from 41.3 million (2015)
- **Rising trend is statistically significant** (p < 0.001)
- Fish mortalities cost $15.5 billion over 7 years across top 4 producing countries

### Mortality Cause Breakdown (Salmon)
- Sea lice treatment-related mortality: 5%+ higher in "red zone" production areas
- Production Area 4 (Norway): Mortality rose from 15% to 23% (i.e., 1 in 4 fish don't survive to harvest)
- Treatment itself kills fish -- thermal/mechanical delousing causes stress mortality

### Industry Targets
- Top-tier aquaculture: <10% annual mortality
- Juvenile survival target: >90% post-hatch

### Other Species
- **Shrimp (vannamei):** Survival 40-90% depending on country and management
- **Catfish:** Moderate; disease is primary killer; "hamburger gill disease" can wipe out entire pond
- **Tilapia:** Hardy; cold is the main killer; lose entire crops to unexpected cold fronts

---

## 16. HARVEST & GRADING OPERATIONS

### Salmon Harvest (Sea Cage)
1. **Crowding:** Net within the pen is gradually tightened to concentrate fish
2. **Passive grading:** Panel installed in seine net -- smaller fish swim through, larger retained
3. **Well-boat transfer:** Fish pumped from pen into well-boat for transport to processing
4. **Slaughter:** At processing plant; typically stunning + bleeding
5. **Site fallowing:** Pen site left empty for minimum 2 months post-harvest (Norway regulation)

### Catfish Harvest (Pond -- USA)
1. **Pre-harvest flavor check:** Samples submitted to processing plant; if off-flavor, harvest postponed
2. **Crew:** Minimum 5 people: 2 tractor operators, 2 mudline workers, 1 boat operator
3. **Seine net deployment:**
   - Length: 1.5x widest pond section
   - Depth: 9-12 feet (food fish)
   - Material: Braided polyethylene mesh (knotless nylon preferred -- less fish injury)
4. **Seining:** Tractor pulls seine via hydraulic reel; fish crowded to one end
5. **Efficiency:** A single seine haul catches **at most 2/3 of fish in pond** -- multiple passes needed
6. **Grading:** Fish dipped into box grader -- large fish retained, small swim through
7. **Loading:** Hydraulic loader with boom, lift net, and hanging scale
8. **Target load:** 20,000-25,000 lbs of graded fish per harvest
9. **Transport:** Live hauler trucks with aerated water

### Grading
- Critical for market price -- uniform size commands premium
- Too much crowding in grader = physical damage and poor separation
- Small fish returned to pond for further grow-out

---

## 17. INDUSTRY SCALE & GLOBAL STATISTICS

### Global Production (2022 -- FAO)
- **Total aquaculture:** 130.9 million tonnes
- **Aquatic animals:** 94.4 million tonnes
- **FIRST TIME IN HISTORY:** Aquaculture surpassed capture fisheries as main producer of aquatic animals

### Market Value
- **2025:** $334 billion
- **2033 projection:** $479.5 billion
- **CAGR:** 4.62% (2025-2033)

### Top 10 Producing Countries (89.8% of global production)
1. China (~77 million MT -- dominant)
2. Indonesia
3. India
4. Vietnam
5. Bangladesh
6. Philippines
7. South Korea
8. Norway
9. Egypt
10. Chile

### Production Projections
- 2028: ~148 million MT (1.8% annual growth)
- Growth drivers: rising seafood demand, technology advances, sustainability focus

### Species Production (Major Categories)
- Freshwater fish: ~60% of global aquaculture
- Seaweed: significant but non-animal
- Shrimp: largest by value in many countries
- Salmon: highest value per kg among finfish
- Tilapia: fastest growing tropical species

---

## KEY OPERATIONAL INSIGHTS SUMMARY

### What Fish Farmers Actually Monitor
1. **Dissolved oxygen** (continuous or 2-4x daily) -- the #1 killer
2. **Temperature** (continuous) -- controls everything
3. **Feed consumption** (every feeding) -- first sign of stress
4. **Mortality count** (daily) -- first sign of disease
5. **Fish behavior** (visual, cameras) -- swimming pattern, feeding response
6. **Water color/turbidity** (daily visual) -- algae indicator
7. **pH** (daily) -- affects ammonia toxicity
8. **Ammonia/Nitrite** (daily to weekly) -- waste buildup
9. **Sea lice count** (weekly for salmon) -- regulatory requirement
10. **Weather forecast** (daily) -- prepare for storms, temperature swings

### What Decisions They Make
- **Feed amount:** Adjusted daily based on fish appetite, temperature, DO level
- **Aeration:** Turned on/off based on DO readings (automated or manual)
- **Harvest timing:** Based on size sampling, market price, flavor testing, weather
- **Treatment:** Sea lice counts above regulatory threshold trigger treatment
- **Stocking/destocking:** Based on growth projections, mortality trends, license limits
- **Emergency response:** Low DO alarm, disease outbreak, storm approach

### What Goes Wrong (Most Common Operational Failures)
1. Power failure to aerators during nighttime low-DO period
2. Overfeeding (wastes money, pollutes water, drops DO)
3. Delayed response to disease outbreak
4. Understocking cleaner fish (lice get out of control)
5. Equipment failure during storm
6. Algal bloom -- not detected early enough
7. Treatment mortality -- fish too weak, wrong temperature for chemical use
8. Off-flavor in catfish -- entire harvest delayed

---

## SOURCES

- [FAO Training: Management for Freshwater Fish Culture](https://www.fao.org/fishery/static/FAO_Training/FAO_Training/General/x6709e/x6709e16.htm)
- [The Fish Site: A Day in the Life of a Fish Farm Technician](https://thefishsite.com/articles/a-day-in-the-life-of-a-fish-farm-technician)
- [Worldwide Aquaculture: Fish Farming Management Practices](https://worldwideaquaculture.com/fish-farming-management-practices-to-observe/)
- [DOL: Fish Farm Worker Job Description](https://seasonaljobs.dol.gov/jobs/H-300-24366-578005)
- [IoT-Based Fish Farm Water Quality Monitoring (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460614/)
- [Sightline: Aquaculture Sensors Guide](https://www.sightline.com/how-aquaculture-sensors-measure-key-water-quality-parameters-your-complete-guide-to-dissolved-oxygen-ph-temperature-and-salinity-monitoring/type-of-company/aqua/)
- [Innovasea: Wireless Sensors for Aquaculture](https://www.innovasea.com/aquaculture-intelligence/environmental-monitoring/wireless-sensors/)
- [Wiley: Feed the Fish - Review of Aquaculture Feeders (2025)](https://onlinelibrary.wiley.com/doi/full/10.1111/jwas.70016)
- [Fish Farm Feeder: Types of Automated Feeders](https://www.fishfarmfeeder.com/en/types-automated-feeders-aquaculture/)
- [Fish Farm Feeder: FCR in Aquaculture](https://www.fishfarmfeeder.com/en/fcr-in-aquaculture/)
- [Wikipedia: Feed Conversion Ratio](https://en.wikipedia.org/wiki/Feed_conversion_ratio)
- [Global Seafood Alliance: Dissolved Oxygen in Pond Aquaculture](https://www.globalseafood.org/advocate/dissolved-oxygen-concentrations-pond-aquaculture/)
- [Global Seafood Alliance: Efficiency of Mechanical Aeration](https://www.globalseafood.org/advocate/efficiency-of-mechanical-aeration/)
- [UF/IFAS: The Role of Aeration in Pond Management](https://ask.ifas.ufl.edu/publication/FA021)
- [Beta.co.id: The Aeration Arms Race in Aquaculture](https://beta.co.id/en/blog/the-aeration-arms-race-in-aquaculture-paddlewheels-diffusers-or-pure-oxygen)
- [Atlas Scientific: Water Quality Parameters for Fish Farming](https://atlas-scientific.com/blog/water-quality-parameters-for-fish-farming/)
- [Global Seafood Alliance: Dissolved Oxygen Requirements](https://www.globalseafood.org/advocate/dissolved-oxygen-requirements-in-aquatic-animal-respiration/)
- [FAO: Dissolved Oxygen in Aquaculture](https://www.fao.org/4/ac175e/AC175E04.htm)
- [Fish Farming Expert: Norwegian Salmonid Profits 2022](https://www.fishfarmingexpert.com/norway-salmonid-profits-2022/norwegian-salmonid-farmers-more-than-doubled-pre-tax-profit-in-2022/1593411)
- [SalmonBusiness: Production Costs Surged 45%](https://www.salmonbusiness.com/production-costs-per-kilo-have-surged-45-since-2016-finds-new-report/)
- [The Fish Site: Rising Costs of Salmon Production](https://thefishsite.com/articles/the-rising-costs-of-salmon-production)
- [EY Norway: Norwegian Aquaculture Rising Salmon Prices](https://www.ey.com/en_no/insights/strategy-transactions/how-norwegian-aquaculture-rides-the-wave-of-rising-salmon-prices)
- [BootstrapBee: Is Fish Farming Profitable?](https://bootstrapbee.com/fish/fish-farming-is-profitable)
- [Nature: Mass Mortality Events in Salmon Aquaculture (2024)](https://www.nature.com/articles/s41598-024-54033-9)
- [TIME: Why Massive Numbers of Farmed Salmon Are Dying](https://time.com/6957610/farmed-salmon-dying/)
- [Global Seafood Alliance: HABs and Aquaculture](https://www.globalseafood.org/advocate/killers-at-sea-harmful-algal-blooms-and-their-impact-on-aquaculture/)
- [Mowi Salmon US: Norwegian Sustainable Aquaculture](https://mowisalmon.us/how-farm-raised-salmon-from-norway-is-reshaping-sustainable-aquaculture/)
- [Salmonfacts.com](https://www.salmonfacts.com/)
- [Zayera: How Atlantic Salmon Farming Works in Norway](https://www.zayera.com/how-atlantic-salmon-farming-works-in-norway-hatchery-sea-pens)
- [The Fish Site: Marine Shrimp Biofloc Systems](https://thefishsite.com/articles/marine-shrimp-biofloc-systems-basic-management-practices)
- [MSU Extension: Catfish Nutrition and Feeding](https://extension.msstate.edu/publications/catfish-nutrition-feeding-food-fish)
- [MSU Extension: Harvesting, Loading, and Transport](https://extension.msstate.edu/agriculture/catfish/harvesting-loading-and-transport)
- [NOAA Fisheries: Regulating Aquaculture](https://www.fisheries.noaa.gov/national/aquaculture/regulating-aquaculture)
- [EPA: Aquaculture NPDES Permitting](https://www.epa.gov/npdes/aquaculture-npdes-permitting)
- [Aquasend: Effects of Hurricanes on Aquaculture](https://www.aquasend.com/2020/09/22/effects-of-hurricanes-on-aquaculture/)
- [Global Seafood Alliance: Effects of Weather on Aquaculture](https://www.globalseafood.org/advocate/effects-of-weather-and-climate-on-aquaculture/)
- [FAO: Global Fisheries and Aquaculture Production Record High](https://www.fao.org/newsroom/detail/fao-report-global-fisheries-and-aquaculture-production-reaches-a-new-record-high/en)
- [GlobeNewsWire: Fish Farming Market to Reach $479.5B by 2033](https://www.globenewswire.com/news-release/2026/03/09/3251738/28124/en/Fish-Farming-Industry-Report-2025-Market-to-Reach-479-5-Billion-by-2033)
- [Nature: Baseline Mortality in Norwegian Salmon Farming](https://www.nature.com/articles/s41598-021-93874-6)
- [Manolin: The Cost of Sea Lice](https://blog.manolinaqua.com/en/the-cost-of-sea-lice)
- [The Fish Site: True Cost of Combatting Sea Lice](https://thefishsite.com/articles/counting-the-true-cost-of-combatting-sea-lice)
- [ScienceDirect: Water Exchange Rate Impact on Rainbow Trout](https://www.sciencedirect.com/science/article/pii/S0044848609005006)
- [AquaMaof: RAS Energy Efficiency](https://www.aquamaof.com/ras-blog-post/what-makes-aquamaof-ras-technology-so-energy-efficient/)
