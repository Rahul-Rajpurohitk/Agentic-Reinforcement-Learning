"""Weather and environmental cycle engine.

Models diel temperature cycle, seasonal variation, storms, light/dark,
cloud cover, wind, humidity, and atmospheric pressure.

Source: KB-01 Sec 2.6, KB-02 Sec 12, KB-03 Sec 12.1

The weather engine drives several important simulation dynamics:
- Air temperature → water temperature → fish metabolism
- Solar intensity → photosynthesis → DO production
- Wind speed → reaeration rate → DO exchange
- Storms → temperature drops + cloud cover + power outage risk
- Photoperiod → feeding window and growth rate
"""

import math
from ..constants import SYSTEM, photoperiod_hours


class WeatherEngine:
    """Environmental conditions model for tropical aquaculture.

    Produces realistic diel (24h) and seasonal cycles with stochastic
    perturbations. Storms are modeled as multi-day events with ramping
    severity that affects temperature, light, and wind.
    """

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.base_air_temp: float = 30.0

        # Storm state
        self.storm_active: bool = False
        self.storm_hours_remaining: int = 0
        self.storm_severity: float = 0.0
        self.storm_peak_severity: float = 0.0

        # Cloud cover (0.0 = clear, 1.0 = overcast) — persists between updates
        self.cloud_cover: float = 0.1

        # Humidity (%) — tropical baseline 60-90%
        self.humidity: float = 75.0

        # Random noise state (Gaussian, persists for temporal correlation)
        self._temp_noise: float = 0.0
        self._noise_update_counter: int = 0

    def reset(self, seed: int = 42, base_temp: float = 30.0):
        """Reset weather to initial conditions."""
        import random
        self.rng = random.Random(seed)
        self.base_air_temp = base_temp
        self.storm_active = False
        self.storm_hours_remaining = 0
        self.storm_severity = 0.0
        self.storm_peak_severity = 0.0
        self.cloud_cover = 0.1
        self.humidity = 75.0
        self._temp_noise = 0.0
        self._noise_update_counter = 0

    def get_conditions(self, day: int, hour: int) -> dict:
        """Calculate current environmental conditions.

        Args:
            day: Day of year (1-365).
            hour: Hour of day (0-23).

        Returns:
            Dict with air_temp, is_daytime, solar_intensity, wind_speed,
            cloud_cover, humidity, storm_active, photoperiod_hours.
        """
        # ---- Diel temperature cycle ----
        # Peak at 14:00 (2PM), trough at 06:00 (6AM)
        # Amplitude: ±3°C typical tropical
        hour_angle = (hour - 14) * (2 * math.pi / 24)
        diel_variation = 3.0 * math.cos(hour_angle)

        # ---- Seasonal cycle ----
        # Tropical: mild ±2°C seasonal, peak in boreal summer (day ~172)
        seasonal = 2.0 * math.sin(2 * math.pi * (day - 80) / 365)

        # ---- Storm effects ----
        storm_temp_drop = 0.0
        if self.storm_active:
            # Storm ramps up, peaks, then fades
            # Temperature drops proportional to severity
            storm_temp_drop = self.storm_severity * 8.0

        # ---- Random noise (updated every ~3 hours for temporal correlation) ----
        air_temp = (self.base_air_temp + diel_variation + seasonal
                    - storm_temp_drop + self._temp_noise)

        # ---- Photoperiod and daylight ----
        p_hours = photoperiod_hours(day, SYSTEM.latitude)
        sunrise = 12.0 - p_hours / 2
        sunset = 12.0 + p_hours / 2
        is_daytime = sunrise <= hour < sunset

        # ---- Solar intensity (W/m²) ----
        if is_daytime:
            solar_peak = 800.0
            # Solar angle: peaks at solar noon (12:00)
            solar_angle = (hour - 12.0) * (math.pi / max(p_hours, 1.0))
            solar_raw = solar_peak * max(0, math.cos(solar_angle))
            # Cloud attenuation: 80% reduction at full overcast
            cloud_attenuation = 1.0 - self.cloud_cover * 0.8
            solar_intensity = solar_raw * cloud_attenuation
        else:
            solar_intensity = 0.0

        # ---- Wind speed (m/s) ----
        # Base: 2 m/s with diel variation (windier midday)
        wind_base = 2.0 + 1.0 * math.sin(2 * math.pi * (hour - 6) / 24)
        if self.storm_active:
            wind_base += self.storm_severity * 15.0
        # Slight random variation
        wind_speed = max(0.0, wind_base + self._temp_noise * 0.3)

        # ---- Humidity ----
        # Higher at night, lower during day. Storm increases humidity.
        base_humidity = 75.0
        if is_daytime:
            humidity = base_humidity - 10.0 * (1.0 - self.cloud_cover)
        else:
            humidity = base_humidity + 10.0
        if self.storm_active:
            humidity = min(100.0, humidity + self.storm_severity * 20.0)
        self.humidity = max(30.0, min(100.0, humidity))

        # ---- Cloud cover ----
        # Persistent with slow drift. Storm overrides.
        if self.storm_active:
            effective_cloud = min(1.0, 0.5 + self.storm_severity * 0.5)
        else:
            effective_cloud = self.cloud_cover

        return {
            "air_temp": air_temp,
            "is_daytime": is_daytime,
            "solar_intensity": solar_intensity,
            "wind_speed": wind_speed,
            "cloud_cover": effective_cloud,
            "humidity": self.humidity,
            "storm_active": self.storm_active,
            "photoperiod_hours": p_hours,
        }

    def step(self, hour: int):
        """Advance weather state by one hour.

        Updates storm progression, cloud drift, temperature noise.
        """
        # Storm countdown
        if self.storm_active:
            self.storm_hours_remaining -= 1

            # Storm severity profile: ramp up → peak → taper off
            total_duration = self.storm_hours_remaining + 1  # avoid div by 0
            if self.storm_hours_remaining > total_duration * 0.7:
                # Ramping up
                self.storm_severity = min(
                    self.storm_peak_severity,
                    self.storm_severity + self.storm_peak_severity * 0.1
                )
            elif self.storm_hours_remaining < total_duration * 0.2:
                # Tapering off
                self.storm_severity = max(0.0, self.storm_severity - 0.05)

            if self.storm_hours_remaining <= 0:
                self.storm_active = False
                self.storm_severity = 0.0

        # Cloud cover drift (random walk, bounded)
        if not self.storm_active:
            cloud_drift = self.rng.gauss(0, 0.02)
            self.cloud_cover = max(0.0, min(0.6, self.cloud_cover + cloud_drift))
        else:
            self.cloud_cover = 0.5 + self.storm_severity * 0.5

        # Temperature noise (updated every 3 hours for temporal correlation)
        self._noise_update_counter += 1
        if self._noise_update_counter >= 3:
            self._noise_update_counter = 0
            # Ornstein-Uhlenbeck-like: mean-reverting with drift
            self._temp_noise = 0.7 * self._temp_noise + self.rng.gauss(0, 0.5)
            self._temp_noise = max(-3.0, min(3.0, self._temp_noise))

    def trigger_storm(self, severity: float = 0.5, duration_hours: int = 48):
        """Manually trigger a storm event.

        Args:
            severity: Peak storm severity 0..1 (0.3=mild, 0.5=moderate, 0.8=severe).
            duration_hours: Storm duration in hours.
        """
        self.storm_active = True
        self.storm_peak_severity = min(1.0, severity)
        self.storm_severity = severity * 0.3  # starts at 30% of peak
        self.storm_hours_remaining = duration_hours

    def check_random_storm(self, prob_per_hour: float = 0.0002,
                           day_of_year: int = 180) -> bool:
        """Check for random storm occurrence with seasonal modulation.

        Base probability modulated by tropical wet/dry season:
        - Wet season (May-Oct, days 121-304): 3× base probability
        - Dry season (Nov-Apr): 0.5× base probability
        - Transition months: linear interpolation

        This creates realistic risk profiles: agents must prepare for
        monsoon season storms that threaten DO (cloud cover kills
        photosynthesis) and temperature (rapid cooling stresses fish).

        Returns:
            True if a storm was triggered.
        """
        if self.storm_active:
            return False

        # Seasonal storm probability modulation (tropical monsoon)
        # Peak storm season: day 180 (July), trough: day 0/365 (January)
        seasonal_factor = 1.0 + 2.0 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        seasonal_factor = max(0.3, min(3.0, seasonal_factor))
        adjusted_prob = prob_per_hour * seasonal_factor

        if self.rng.random() < adjusted_prob:
            # Wet season storms tend to be more severe
            base_severity = self.rng.uniform(0.3, 0.8)
            if seasonal_factor > 1.5:
                base_severity = min(1.0, base_severity * 1.2)
            duration = self.rng.randint(12, 72)
            self.trigger_storm(base_severity, duration)
            return True
        return False

    def weather_forecast(self, day: int, hour: int) -> str:
        """Human-readable weather summary for agent observations."""
        conditions = self.get_conditions(day, hour)
        parts = []
        parts.append(f"Air temp: {conditions['air_temp']:.1f}°C")

        if conditions['storm_active']:
            if self.storm_severity > 0.7:
                parts.append(f"SEVERE STORM ({self.storm_hours_remaining}h remaining)")
            elif self.storm_severity > 0.4:
                parts.append(f"Storm active ({self.storm_hours_remaining}h remaining)")
            else:
                parts.append(f"Mild storm ({self.storm_hours_remaining}h remaining)")
        elif conditions['cloud_cover'] < 0.2:
            parts.append("Clear skies")
        elif conditions['cloud_cover'] < 0.5:
            parts.append("Partly cloudy")
        else:
            parts.append("Overcast")

        if conditions['is_daytime']:
            parts.append(f"Daylight ({conditions['solar_intensity']:.0f} W/m²)")
        else:
            parts.append("Night")

        parts.append(f"Wind: {conditions['wind_speed']:.1f} m/s")

        return ". ".join(parts)
