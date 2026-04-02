"""Fish Farm OpenEnv Environment."""

try:
    from .client import FishFarmEnv
    from .models import FarmAction, FarmObservation

    __all__ = [
        "FarmAction",
        "FarmObservation",
        "FishFarmEnv",
    ]
except ImportError:
    # When running outside package context (e.g., pytest from project root),
    # relative imports fail. This is expected — tests import from src/ directly.
    pass
