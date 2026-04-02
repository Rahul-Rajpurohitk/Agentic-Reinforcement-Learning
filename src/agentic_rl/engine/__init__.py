"""Fish Farm Simulation Engine.

Modular subsystems:
- water_quality: DO, TAN, UIA, pH, temperature dynamics
- fish_biology: Growth (bioenergetic), feeding response, stress, mortality
- disease: SEIR epidemic model with environmental triggers
- economics: Feed cost, fish value, operating expenses
- weather: Diel cycle, seasonal variation, storm events
- events: Event scheduler (disease, storms, equipment failures, algae blooms)
- simulator: Orchestrator combining all subsystems
"""
