"""OpenEnv type definitions following the 3-component pattern.

Defines the Action, Observation, and State types that form the contract
between the environment server and clients. Customize these for your
specific problem statement.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action: what the agent sends to the environment each step
# ---------------------------------------------------------------------------
class AgentAction(BaseModel):
    """Action taken by the agent.

    Modify fields here to match your problem statement's action space.
    Examples: a text response, a move direction, a tool call, etc.
    """

    response: str = Field(..., description="The agent's response or action text")
    metadata: Optional[dict] = Field(default=None, description="Optional action metadata")


# ---------------------------------------------------------------------------
# Observation: what the environment returns to the agent after each step
# ---------------------------------------------------------------------------
class EnvObservation(BaseModel):
    """Observation returned by the environment after a step.

    Modify fields here to match what your environment exposes to the agent.
    """

    prompt: str = Field(..., description="Current task prompt or context for the agent")
    feedback: str = Field(default="", description="Feedback from the previous action")
    score: float = Field(default=0.0, description="Current score / partial reward")
    done: bool = Field(default=False, description="Whether the episode is complete")
    turn: int = Field(default=0, description="Current turn number")
    max_turns: int = Field(default=10, description="Maximum turns allowed")
    info: Optional[dict] = Field(default=None, description="Additional info for debugging")


# ---------------------------------------------------------------------------
# State: internal environment state (not visible to the agent)
# ---------------------------------------------------------------------------
class EnvState(BaseModel):
    """Internal environment state.

    This holds the ground truth that the grader uses for evaluation.
    The agent does NOT see this directly.
    """

    task_id: str = Field(default="", description="Identifier for the current task")
    target: str = Field(default="", description="The target/correct answer")
    history: List[str] = Field(default_factory=list, description="History of agent actions")
    current_turn: int = Field(default=0, description="Current turn counter")
    max_turns: int = Field(default=10, description="Maximum turns per episode")
    is_complete: bool = Field(default=False, description="Whether the episode ended")
    final_score: float = Field(default=0.0, description="Final score for the episode")
