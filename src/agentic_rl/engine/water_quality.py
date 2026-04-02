"""Water quality dynamics engine.

Implements the mass balance equations for:
- Dissolved Oxygen (DO): fish respiration, photosynthesis, aeration, reaeration,
  nitrification oxygen demand, phytoplankton/sediment respiration
- Total Ammonia Nitrogen (TAN): fish excretion, biofilter nitrification, water exchange
- Unionized Ammonia (UIA): pH/temperature-dependent equilibrium (Emerson et al. 1975)
- pH: alkalinity buffer model driven by nitrification acid production + CO2 equilibrium
- Nitrite (NO2): two-stage nitrification intermediate (Nitrosomonas → Nitrobacter)
- Nitrate (NO3): end product of nitrification
- Temperature: thermal exchange with air + heater + water exchange mixing
- Alkalinity: consumed by nitrification, replenished by water exchange

All equations sourced from KB-03 Sections 2.1-2.2, KB-01 Sections 2-4.
Internal sub-stepping at 6-minute intervals for numerical stability.
"""

import math
from ..constants import WATER, SYSTEM, TILAPIA, uia_fraction, do_saturation, ECONOMICS


class WaterQualityEngine:
    """Manages water quality state and dynamics for a single RAS tank.

    The core coupling: feeding → TAN excretion → nitrification consumes O2 and
    alkalinity → pH drops → UIA fraction shifts → fish stress changes.
    This cascade is the heart of the biological realism.
    """

    def __init__(self, volume_m3: float, depth_m: float):
        self.volume_m3 = volume_m3
        self.depth_m = depth_m
        self.surface_area_m2 = volume_m3 / depth_m

        # State variables
        self.DO: float = 7.0           # mg/L dissolved oxygen
        self.temperature: float = 28.0  # °C water temperature
        self.TAN: float = 0.1          # mg/L total ammonia nitrogen
        self.UIA: float = 0.0          # mg/L unionized ammonia (derived)
        self.pH: float = 7.5           # pH units
        self.NO2: float = 0.05         # mg/L nitrite
        self.NO3: float = 5.0          # mg/L nitrate
        self.alkalinity: float = 150.0  # mg/L as CaCO3

        # Algae bloom state — phytoplankton biomass (mg chlorophyll-a / L)
        self.chlorophyll_a: float = 5.0  # baseline moderate algae
        self.algae_bloom_active: bool = False

        # Nighttime DO crash risk tracking (KB-02 Sec 4)
        # The #1 killer in real aquaculture: algae produce O2 by day but consume
        # it at night, and high biomass can crash DO to lethal levels.
        self.nighttime_do_risk: float = 0.0  # 0.0 = safe, 1.0 = imminent crash
        self._do_history: list = []  # rolling DO for trend detection
        self._peak_daytime_do: float = 7.0  # track daytime DO peaks

    def reset(self, temp: float, DO: float, TAN: float, pH: float, NO2: float):
        """Reset water quality to initial conditions."""
        self.temperature = temp
        self.DO = DO
        self.TAN = TAN
        self.pH = pH
        self.NO2 = NO2
        self.NO3 = 5.0
        self.alkalinity = 150.0
        self.chlorophyll_a = 5.0
        self.algae_bloom_active = False
        self.nighttime_do_risk = 0.0
        self._do_history = []
        self._peak_daytime_do = DO
        self._update_uia()

    def _update_uia(self):
        """Recalculate UIA from TAN using the Emerson et al. (1975) equilibrium.

        UIA = TAN / (1 + 10^(pKa - pH))
        pKa = 0.09018 + 2729.92 / T_kelvin
        """
        self.UIA = self.TAN * uia_fraction(self.pH, self.temperature)

    def step(
        self,
        dt_hours: float,
        fish_biomass_kg: float,
        fish_weight_g: float,
        feeding_rate: float,
        aeration_rate: float,
        water_exchange_rate: float,
        is_daytime: bool,
        biofilter_efficiency: float,
        solar_intensity: float = 0.0,
        wind_speed: float = 2.0,
        fish_respiration_rate: float = 0.0,
        humidity: float = 75.0,
    ):
        """Advance water quality by dt_hours using sub-stepping.

        Args:
            dt_hours: Time step (typically 1.0 hour).
            fish_biomass_kg: Total live fish biomass (kg).
            fish_weight_g: Average individual fish weight (g).
            feeding_rate: Normalized feeding rate 0..1.
            aeration_rate: Normalized aeration power 0..1.
            water_exchange_rate: Fraction of tank volume exchanged per hour.
            is_daytime: Whether sun is up (for photosynthesis).
            biofilter_efficiency: Biofilter TAN removal efficiency 0..1.
            solar_intensity: Solar radiation W/m² (for photosynthesis model).
            wind_speed: Wind speed m/s (for reaeration model).
            fish_respiration_rate: Pre-computed respiration rate (mg O2/kg/h).
                If 0.0, falls back to internal allometric model.
            humidity: Relative humidity (%) for evaporation model.
        """
        n_sub = SYSTEM.sub_steps
        sub_dt = dt_hours / n_sub

        for _ in range(n_sub):
            self._sub_step(
                sub_dt, fish_biomass_kg, fish_weight_g, feeding_rate,
                aeration_rate, water_exchange_rate, is_daytime,
                biofilter_efficiency, solar_intensity, wind_speed,
                fish_respiration_rate, humidity,
            )

    def _sub_step(
        self, dt_h: float, biomass_kg: float, fish_w_g: float,
        feed_rate: float, aeration: float, exchange: float,
        daytime: bool, biofilter_eff: float,
        solar_intensity: float, wind_speed: float,
        fish_resp_rate: float = 0.0, humidity: float = 75.0,
    ):
        """Single 6-minute sub-step of water quality dynamics."""
        T = self.temperature
        V_L = self.volume_m3 * 1000  # tank volume in liters
        w = max(fish_w_g, 1.0)

        # ================================================================
        # 1. DISSOLVED OXYGEN DYNAMICS (KB-03 Sec 2.1)
        # dDO/dt = P_photo - R_phyto - R_fish - R_water - R_sed
        #        + K_a*(DO_sat - DO) + A_mech + DO_exchange - DO_nitrif
        # ================================================================

        # 1a. Fish respiration — use pre-computed rate from FishBiologyEngine
        # when available (tilapia-specific polynomial, R²=0.99), otherwise
        # fall back to the general allometric model.
        if fish_resp_rate > 0:
            FR = fish_resp_rate
        else:
            # General allometric: FR = 10^X * 1000 (mg O2/kg/h)
            X = 0.40 + 0.016 * T - 0.0006 * T ** 2 - 0.016 * math.log(max(w, 1.0))
            FR = (10 ** X) * 1000
        DO_fish = FR * biomass_kg / V_L  # mg/L/h consumed

        # 1b. Nitrification oxygen demand
        # 4.57 g O2 consumed per g TAN oxidized to NO3 (KB-01 Sec 3.1)
        nitrif_rate_AOB, nitrif_rate_NOB = self._two_stage_nitrification(biofilter_eff)
        TAN_oxidized_rate = nitrif_rate_AOB * self.TAN  # mg/L/h TAN → NO2
        NO2_oxidized_rate = nitrif_rate_NOB * self.NO2  # mg/L/h NO2 → NO3
        # O2 demand: 3.43 g O2 per g TAN→NO2 (Nitrosomonas)
        #          + 1.14 g O2 per g NO2→NO3 (Nitrobacter)
        #          = 4.57 total per full nitrification
        DO_nitrif = 3.43 * TAN_oxidized_rate + 1.14 * NO2_oxidized_rate

        # 1c. Photosynthesis — solar-dependent (Smith-Talling model, KB-03 Sec 2.1)
        # P_photo scales with solar intensity and chlorophyll-a (phytoplankton)
        if daytime and solar_intensity > 0:
            # Photosynthetic rate: scales with light, phytoplankton, and depth
            # P_max ≈ 0.3 mg O2/L/h per μg chl-a/L at optimal light
            P_max_per_chl = 0.3
            I_k = 200.0  # light saturation parameter (W/m²)
            # Smith function: P = P_max * I / sqrt(I² + I_k²)
            light_factor = solar_intensity / math.sqrt(solar_intensity ** 2 + I_k ** 2)
            # Light attenuation through water column (Beer-Lambert)
            # K_ext includes water + phytoplankton self-shading
            K_ext = 0.4 + 0.02 * self.chlorophyll_a  # m⁻¹
            # Depth-averaged light factor
            if K_ext * self.depth_m > 0.01:
                depth_avg_factor = (1.0 - math.exp(-K_ext * self.depth_m)) / (K_ext * self.depth_m)
            else:
                depth_avg_factor = 1.0
            DO_photo = P_max_per_chl * self.chlorophyll_a * light_factor * depth_avg_factor
        else:
            DO_photo = 0.0

        # 1d. Phytoplankton respiration (KB-03 Sec 2.1)
        # R_phyto = 0.10 * P_max_capacity * 1.08^(T-20), always active
        phyto_resp_base = 0.02 * self.chlorophyll_a  # mg O2/L/h
        DO_phyto_resp = phyto_resp_base * (1.047 ** (T - 20))

        # 1e. Water column microbial respiration (BOD)
        # Background organic matter decomposition, temperature-dependent
        DO_water = 0.08 * (1.047 ** (T - 20))

        # 1f. Sediment oxygen demand
        # Negligible in clean RAS tanks, but accumulates with uneaten feed
        # SOD ≈ 0.02 mg O2/L/h baseline, increases with overfeeding
        sod_base = 0.02
        overfeeding_factor = max(1.0, 1.0 + (feed_rate - 0.7) * 2.0) if feed_rate > 0.7 else 1.0
        DO_sediment = sod_base * overfeeding_factor * (1.047 ** (T - 20))

        # 1g. Reaeration — wind-dependent (KB-03 Sec 2.1)
        # K_a scales with wind speed: K_a = K_a_base * (1 + 0.5 * wind²)
        # At zero wind, still some diffusion. At high wind, vigorous mixing.
        DO_sat = do_saturation(T)
        K_a = WATER.K_a_base * (1.0 + 0.046 * wind_speed ** 2)
        DO_reaer = K_a * (DO_sat - self.DO)

        # 1h. Mechanical aeration
        # Max rate scaled by aeration control signal
        # Equipment failure zeroes aeration_rate upstream
        DO_mech = aeration * SYSTEM.max_aeration_rate

        # 1i. Water exchange DO contribution
        # Fresh water brings in DO at incoming level, dilutes current
        DO_exchange = exchange * (SYSTEM.incoming_water_DO - self.DO)

        # Net DO change
        dDO = (DO_photo + DO_reaer + DO_mech + DO_exchange
               - DO_fish - DO_nitrif - DO_water - DO_phyto_resp - DO_sediment) * dt_h
        self.DO = max(0.0, self.DO + dDO)
        self.DO = min(self.DO, DO_sat * 1.3)  # supersaturation cap

        # ================================================================
        # 2. TAN DYNAMICS (KB-03 Sec 2.2)
        # dTAN/dt = excretion - nitrification - exchange
        # ================================================================

        # 2a. Fish ammonia excretion from feeding
        # TAN (kg/d) = Feed(kg/d) × Protein% × 0.16 × N_wasted% × 1.2
        feed_amount_kg_h = (feed_rate * TILAPIA.max_feeding_pct / 100.0
                            * biomass_kg / 24.0)
        TAN_excretion_kg_h = (feed_amount_kg_h * TILAPIA.protein_fraction
                              * 0.16 * TILAPIA.n_wasted_fraction * 1.2)
        TAN_excretion_mg_L = TAN_excretion_kg_h * 1e6 / V_L

        # 2b. Nitrification removal (TAN → NO2 by Nitrosomonas)
        TAN_nitrif = nitrif_rate_AOB * self.TAN

        # 2c. Water exchange dilution
        TAN_exchange = exchange * (SYSTEM.incoming_water_TAN - self.TAN)

        dTAN = (TAN_excretion_mg_L - TAN_nitrif + TAN_exchange) * dt_h
        self.TAN = max(0.0, self.TAN + dTAN)

        # ================================================================
        # 3. NITRITE DYNAMICS — two-stage nitrification
        # dNO2/dt = (TAN→NO2 by AOB) - (NO2→NO3 by NOB)
        # Nitrosomonas (AOB) are faster starters but NOB lag behind,
        # causing the classic "nitrite spike" in new biofilters.
        # ================================================================
        NO2_produced = nitrif_rate_AOB * self.TAN  # AOB: TAN → NO2
        NO2_consumed = nitrif_rate_NOB * self.NO2  # NOB: NO2 → NO3
        NO2_exchange = exchange * (0.0 - self.NO2)  # fresh water has 0 NO2

        dNO2 = (NO2_produced - NO2_consumed + NO2_exchange) * dt_h
        self.NO2 = max(0.0, self.NO2 + dNO2)

        # ================================================================
        # 4. NITRATE DYNAMICS
        # dNO3/dt = NO2 oxidized - denitrification - exchange dilution
        # NO3 accumulates in RAS — the main reason for water exchange
        # ================================================================
        NO3_produced = nitrif_rate_NOB * self.NO2
        NO3_exchange = exchange * (0.0 - self.NO3)  # fresh water ~ 0 NO3

        # Denitrification: NO3 → N2 (gas) in anoxic microhabitats
        # Occurs in biofilm dead zones and sediment when DO < 2.0 mg/L
        # Rate increases exponentially as DO drops toward 0
        # KB-03 Sec 2.2: denitrifiers are facultative anaerobes
        if self.DO < 2.0:
            # Monod-type inhibition by oxygen: rate = k_denit * (K_O2 / (K_O2 + DO))
            K_O2 = 0.5  # half-saturation for O2 inhibition
            denit_rate = 0.02 * (K_O2 / (K_O2 + max(0.01, self.DO)))
            denit_rate *= (1.047 ** (T - 20))  # temperature dependence
            NO3_denit = denit_rate * self.NO3
        else:
            NO3_denit = 0.0

        dNO3 = (NO3_produced - NO3_denit + NO3_exchange) * dt_h
        self.NO3 = max(0.0, self.NO3 + dNO3)

        # ================================================================
        # 5. ALKALINITY DYNAMICS (KB-01 Sec 3.1)
        # Nitrification consumes 7.14 g CaCO3 per g TAN oxidized
        # Water exchange replenishes alkalinity
        # ================================================================
        alk_consumed_rate = WATER.alkalinity_per_TAN * TAN_oxidized_rate  # mg CaCO3/L/h
        alk_exchange = exchange * (150.0 - self.alkalinity)  # incoming water at 150 mg/L

        dAlk = (-alk_consumed_rate + alk_exchange) * dt_h
        self.alkalinity = max(20.0, min(300.0, self.alkalinity + dAlk))

        # ================================================================
        # 6. pH DYNAMICS — alkalinity buffer model
        # pH is primarily controlled by alkalinity (CaCO3 buffering capacity)
        # Low alkalinity → poor buffering → pH drops from nitrification acid
        # High alkalinity → stable pH near 7.5-8.0
        # CO2 from fish respiration also acidifies, but biofilter strips some CO2
        # ================================================================

        # CO2 production from fish respiration (0.9 mg CO2 per mg O2 consumed)
        CO2_production = DO_fish * 0.9  # mg CO2/L/h
        # CO2 stripping by aeration (degassing)
        CO2_stripping = aeration * 0.5 * CO2_production
        net_CO2_effect = (CO2_production - CO2_stripping) * dt_h

        # Alkalinity-based pH equilibrium
        # At high alkalinity (>120), pH is well-buffered near 7.5-8.0
        # At low alkalinity (<80), pH becomes unstable and drifts down
        # At very low alkalinity (<50), pH can crash
        if self.alkalinity > 120:
            pH_target = 7.5 + 0.005 * (self.alkalinity - 120) / 100
            pH_target = min(pH_target, 8.2)
        elif self.alkalinity > 80:
            pH_target = 7.0 + 0.5 * (self.alkalinity - 80) / 40
        elif self.alkalinity > 40:
            pH_target = 6.5 + 0.5 * (self.alkalinity - 40) / 40
        else:
            pH_target = 6.0 + 0.5 * self.alkalinity / 40

        # pH drifts toward equilibrium, rate depends on buffering capacity
        buffer_strength = min(1.0, self.alkalinity / 150.0)
        drift_rate = 0.02 * (1.0 + (1.0 - buffer_strength) * 2.0)
        dpH = drift_rate * (pH_target - self.pH) * dt_h

        # CO2 acidification (stronger when buffering is weak)
        dpH -= net_CO2_effect * 0.001 * (1.0 - buffer_strength * 0.5)

        self.pH = max(5.5, min(9.5, self.pH + dpH))

        # ================================================================
        # 7. ALGAE DYNAMICS (simplified)
        # Chlorophyll-a grows with nutrients (NO3, light) and temperature
        # Crashes can cause DO swings (bloom → die-off → O2 crash)
        # ================================================================
        if daytime and solar_intensity > 100:
            # Algae growth: nutrient + light + temperature dependent
            nutrient_factor = min(1.0, self.NO3 / 20.0)
            temp_growth = max(0.0, 1.0 - abs(T - 28) / 10.0)
            growth_rate = 0.005 * nutrient_factor * temp_growth * (solar_intensity / 800.0)
        else:
            growth_rate = 0.0
        # Natural die-off
        die_off = 0.003 * self.chlorophyll_a
        dChl = (growth_rate * self.chlorophyll_a - die_off) * dt_h
        self.chlorophyll_a = max(1.0, min(200.0, self.chlorophyll_a + dChl))

        # Detect bloom conditions (>50 μg/L chlorophyll-a)
        self.algae_bloom_active = self.chlorophyll_a > 50.0

        # ================================================================
        # 8. EVAPORATION — concentrates dissolved substances
        # Penman-simplified evaporation: E = f(wind) × (e_s - e_a) × A / V
        # Tropical tanks can lose 5-10 mm/day to evaporation.
        # This concentrates TAN, NO2, NO3 (mass stays, volume shrinks).
        # ================================================================
        if T > 10.0:
            # Saturation vapor pressure (Tetens formula, kPa)
            e_s = 0.6108 * math.exp(17.27 * T / (T + 237.3))
            e_a = e_s * (humidity / 100.0)
            # Evaporation rate: mm/h ≈ f(wind) × (e_s - e_a)
            f_wind = 0.5 + 0.26 * wind_speed  # aerodynamic factor
            evap_mm_h = max(0.0, f_wind * (e_s - e_a) * 0.1)  # scaled for tank
            # Volume fraction lost per sub-step
            evap_volume_L = evap_mm_h * self.surface_area_m2 * dt_h
            if evap_volume_L > 0 and V_L > 100:
                # Concentration factor: solutes stay, water leaves
                conc_factor = V_L / max(V_L - evap_volume_L, V_L * 0.99)
                self.TAN *= conc_factor
                self.NO2 *= conc_factor
                self.NO3 *= conc_factor
                # Track cumulative evaporation for the engine state
                self._evap_mm_day = getattr(self, '_evap_mm_day', 0.0) + evap_mm_h * dt_h

        # ================================================================
        # 9. UPDATE DERIVED VALUES + NIGHTTIME DO CRASH RISK
        # ================================================================
        self._update_uia()

        # Track DO history for trend detection (rolling 6h window)
        self._do_history.append(self.DO)
        if len(self._do_history) > 60:  # 60 sub-steps = 6 hours
            self._do_history = self._do_history[-60:]

        # Track daytime DO peaks (high daytime DO + algae = nighttime crash risk)
        if daytime:
            self._peak_daytime_do = max(self._peak_daytime_do * 0.99, self.DO)
        else:
            # Nighttime DO crash risk assessment (KB-02 Sec 4)
            # Risk factors: high chlorophyll-a, high daytime DO peak (supersaturation),
            # high fish biomass, low aeration
            algae_risk = min(1.0, self.chlorophyll_a / 80.0)  # >80 μg/L = high risk
            # DO swing amplitude: high day peak relative to current night DO
            swing = max(0, self._peak_daytime_do - self.DO)
            swing_risk = min(1.0, swing / 4.0)  # 4 mg/L swing = dangerous
            # Trend: is DO declining over recent sub-steps?
            if len(self._do_history) >= 10:
                recent_trend = self._do_history[-1] - self._do_history[-10]
                trend_risk = min(1.0, max(0, -recent_trend) / 1.0)
            else:
                trend_risk = 0.0
            self.nighttime_do_risk = min(1.0,
                algae_risk * 0.4 + swing_risk * 0.35 + trend_risk * 0.25)

    def _two_stage_nitrification(self, biofilter_eff: float) -> tuple:
        """Two-stage nitrification rates for Nitrosomonas (AOB) and Nitrobacter (NOB).

        Stage 1 (AOB): TAN → NO2, rate K_AOB
        Stage 2 (NOB): NO2 → NO3, rate K_NOB

        NOB are typically slower to establish and more sensitive to low DO,
        which causes the classic "nitrite spike" in immature biofilters.

        Returns:
            (K_AOB, K_NOB): nitrification rate coefficients (1/h)
        """
        T = self.temperature
        # Base nitrification rate: K_NR = 0.11 * 1.08^(T-20) per day (KB-03)
        K_base = 0.11 * (1.08 ** (T - 20)) / 24.0  # convert to per hour

        # AOB (Nitrosomonas): TAN → NO2
        # Active above ~15°C, optimal 25-35°C
        K_AOB = K_base * biofilter_eff

        # NOB (Nitrobacter): NO2 → NO3
        # Slightly slower than AOB, more sensitive to low DO
        # At low DO (<2 mg/L), NOB are inhibited → NO2 accumulates
        DO_factor_NOB = min(1.0, max(0.1, (self.DO - 1.0) / 3.0))
        K_NOB = K_base * biofilter_eff * 0.85 * DO_factor_NOB

        return K_AOB, K_NOB

    def update_temperature(
        self,
        dt_hours: float,
        air_temp: float,
        heater_setting: float,
        volume_m3: float,
        water_exchange_rate: float = 0.0,
    ):
        """Update water temperature from air exchange, heater, and water exchange.

        Thermal model: dT/dt = K_eq*(T_air - T) + Q_heater/(ρ*cp*V) + exchange mixing

        Args:
            dt_hours: Time step (hours).
            air_temp: Air temperature (°C).
            heater_setting: -1.0 (cool) to 1.0 (heat).
            volume_m3: Tank volume (m³).
            water_exchange_rate: Fraction of volume exchanged per hour.
        """
        # Air-water thermal equilibration
        # Rate depends on surface area to volume ratio and wind
        equilibration_rate = 0.01 * (self.surface_area_m2 / volume_m3)
        dT_air = equilibration_rate * (air_temp - self.temperature) * dt_hours

        # Heater/chiller: Q = P * dt / (ρ * cp * V)
        # P = 5 kW, ρ = 1000 kg/m³, cp = 4186 J/kg/°C
        heater_dT_max = (ECONOMICS.heater_power_kw * 3600) / (volume_m3 * 1000 * 4.186)
        dT_heater = heater_setting * heater_dT_max * dt_hours

        # Water exchange temperature mixing
        dT_exchange = water_exchange_rate * (SYSTEM.incoming_water_temp - self.temperature) * dt_hours

        self.temperature += dT_air + dT_heater + dT_exchange
        self.temperature = max(10.0, min(42.0, self.temperature))

    def get_water_quality_score(self) -> float:
        """Composite water quality score [0.0, 1.0].

        Weighted combination:
          DO (35%) + UIA (25%) + Temperature (20%) + pH (10%) + NO2 (10%)

        Each sub-score uses tilapia-specific thresholds from KB-01.
        """
        # DO score: optimal > 5.0, stress < 3.0, lethal < 1.0
        if self.DO >= WATER.DO_optimal:
            do_score = 1.0
        elif self.DO >= WATER.DO_min:
            do_score = (self.DO - WATER.DO_min) / (WATER.DO_optimal - WATER.DO_min)
        else:
            do_score = max(0.0, self.DO / WATER.DO_min * 0.3)

        # UIA score: safe < 0.02, stress > 0.05, lethal > 0.6
        if self.UIA <= WATER.UIA_safe:
            uia_score = 1.0
        elif self.UIA <= WATER.UIA_crit:
            uia_score = 1.0 - (self.UIA - WATER.UIA_safe) / (WATER.UIA_crit - WATER.UIA_safe)
        else:
            uia_score = max(0.0, 1.0 - (self.UIA - WATER.UIA_crit) / (WATER.UIA_lethal - WATER.UIA_crit))

        # pH score: optimal 6.5-8.5
        if WATER.pH_min <= self.pH <= WATER.pH_max:
            ph_score = 1.0
        else:
            deviation = max(WATER.pH_min - self.pH, self.pH - WATER.pH_max, 0)
            ph_score = max(0.0, 1.0 - deviation / 2.0)

        # Temperature score
        from ..constants import temperature_factor
        temp_score = temperature_factor(self.temperature)

        # Nitrite score: safe < 0.1, stress > 0.5, lethal > 5.0
        if self.NO2 <= WATER.NO2_safe:
            no2_score = 1.0
        elif self.NO2 <= WATER.NO2_stress:
            no2_score = 1.0 - 0.5 * (self.NO2 - WATER.NO2_safe) / (WATER.NO2_stress - WATER.NO2_safe)
        else:
            no2_score = max(0.0, 0.5 * (1.0 - (self.NO2 - WATER.NO2_stress) / (WATER.NO2_lethal - WATER.NO2_stress)))

        return (0.35 * do_score + 0.25 * uia_score + 0.20 * temp_score
                + 0.10 * ph_score + 0.10 * no2_score)
