"""Event system — scheduled and random events that challenge the agent.

Event types:
- disease: Pathogen introduction → triggers SEIR outbreak
- storm: Severe weather → temperature drop + cloud cover + high wind
- equipment_failure: Aerator/heater/biofilter breakdown → subsystem disabled
- algae_bloom: Nutrient-driven bloom → DO swings (daytime surplus → nighttime crash)
- feed_shortage: Feed delivery delayed → inventory runs out
- price_change: Market price shift → affects harvest economics
- power_outage: All equipment disabled → aerator, heater, biofilter off

Events can be:
1. Pre-scheduled (defined in task scenarios with specific trigger hours)
2. Random (stochastic, checked each hour by simulator)

The event system is the primary mechanism for creating multi-crisis scenarios
that test the agent's ability to manage compound failures.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Event:
    """A discrete event that affects the simulation.

    Events have a trigger time, duration, and severity. The simulator
    interprets event types and applies appropriate subsystem effects.
    """
    type: str               # Event category (disease, storm, equipment_failure, etc.)
    trigger_hour: int        # Hour when event activates (absolute sim hour)
    severity: float          # 0.0-1.0, scales the event's impact
    duration_hours: int      # How long the event lasts (0 = instantaneous)
    description: str         # Human-readable description for agent observation
    active: bool = False     # Currently active
    hours_remaining: int = 0 # Countdown
    equipment: str = ""      # For equipment_failure: which equipment
    price_multiplier: float = 1.0  # For price_change events
    triggered: bool = False  # Has this event been triggered (activated at least once)

    def __post_init__(self):
        """Ensure hours_remaining matches duration for new events."""
        if not self.active and self.hours_remaining == 0:
            self.hours_remaining = self.duration_hours


class EventScheduler:
    """Manages scheduled and active events.

    Events are sorted by trigger_hour. Each step(), the scheduler:
    1. Activates any events whose trigger_hour has been reached
    2. Decrements active event timers
    3. Deactivates expired events
    4. Returns list of newly activated events (for simulator to handle)
    """

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.scheduled_events: List[Event] = []
        self.active_events: List[Event] = []
        self.past_events: List[Event] = []
        self.current_hour: int = 0

    def reset(self, seed: int = 42):
        """Reset event scheduler for new episode."""
        import random
        self.rng = random.Random(seed)
        self.scheduled_events = []
        self.active_events = []
        self.past_events = []
        self.current_hour = 0

    def schedule(self, event: Event):
        """Add an event to the schedule.

        Events are kept sorted by trigger_hour for efficient processing.
        """
        self.scheduled_events.append(event)
        self.scheduled_events.sort(key=lambda e: e.trigger_hour)

    def step(self, hour: int) -> List[Event]:
        """Process one hour of event scheduling.

        Args:
            hour: Current absolute simulation hour.

        Returns:
            List of events that were newly activated this step.
        """
        self.current_hour = hour
        newly_activated = []

        # Activate events whose trigger time has been reached
        remaining = []
        for event in self.scheduled_events:
            if event.trigger_hour <= hour:
                event.active = True
                event.triggered = True
                # Instantaneous events (duration=0) get 1 hour minimum
                # to ensure they are processed by the simulator
                if event.duration_hours <= 0:
                    event.hours_remaining = 1
                else:
                    event.hours_remaining = event.duration_hours
                self.active_events.append(event)
                newly_activated.append(event)
            else:
                remaining.append(event)
        self.scheduled_events = remaining

        # Decrement active event timers
        still_active = []
        for event in self.active_events:
            event.hours_remaining -= 1
            if event.hours_remaining <= 0:
                event.active = False
                self.past_events.append(event)
            else:
                still_active.append(event)
        self.active_events = still_active

        return newly_activated

    def has_active(self, event_type: str) -> bool:
        """Check if any event of the given type is currently active."""
        return any(e.type == event_type for e in self.active_events)

    def get_active_severity(self, event_type: str) -> float:
        """Get severity of active event of given type (0.0 if none)."""
        for e in self.active_events:
            if e.type == event_type:
                return e.severity
        return 0.0

    def get_active_event(self, event_type: str) -> Optional[Event]:
        """Get the active Event object of given type, if any."""
        for e in self.active_events:
            if e.type == event_type:
                return e
        return None

    def get_alerts(self) -> List[str]:
        """Get human-readable alert descriptions for all active events."""
        return [e.description for e in self.active_events]

    def equipment_working(self, equipment: str) -> bool:
        """Check if specific equipment is operational.

        Equipment is non-operational if:
        1. An equipment_failure event targets it
        2. A power_outage is active (kills everything)
        """
        # Power outage disables all equipment
        if self.has_active("power_outage"):
            return False

        # Equipment-specific failure
        for e in self.active_events:
            if e.type == "equipment_failure" and e.equipment == equipment:
                return False
        return True

    def get_price_multiplier(self) -> float:
        """Get current market price multiplier from price_change events."""
        for e in self.active_events:
            if e.type == "price_change":
                return e.price_multiplier
        return 1.0

    def get_feed_shortage_severity(self) -> float:
        """Get feed shortage severity (0.0 = normal, 1.0 = no feed available)."""
        return self.get_active_severity("feed_shortage")

    def count_active(self) -> int:
        """Count number of simultaneously active events (crisis complexity)."""
        return len(self.active_events)

    def event_history_summary(self) -> List[dict]:
        """Get summary of all events (past and active) for episode review."""
        all_events = self.past_events + self.active_events
        return [
            {
                "type": e.type,
                "trigger_hour": e.trigger_hour,
                "severity": e.severity,
                "duration": e.duration_hours,
                "description": e.description,
                "status": "active" if e.active else "resolved",
            }
            for e in all_events
        ]
