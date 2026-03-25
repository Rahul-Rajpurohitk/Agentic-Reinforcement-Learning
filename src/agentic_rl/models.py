"""OpenEnv type definitions for the Code Review environment.

Uses the official openenv-core base classes:
- Action (has: metadata)
- Observation (has: done, reward, metadata)
- State (has: episode_id, step_count)
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class ReviewAction(Action):
    """Action: the agent's code review response.

    The agent analyzes code and reports found issues with severity and fix suggestions.
    """

    issues_found: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "List of issues found. Each dict has keys: "
            "'line' (line number), 'severity' (critical/major/minor), "
            "'category' (bug/security/style/performance/logic), "
            "'description' (what's wrong), 'suggestion' (how to fix)"
        ),
    )
    overall_assessment: str = Field(
        default="comment",
        description="Overall assessment: 'approve', 'request_changes', or 'comment'",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its review (0.0-1.0)",
    )


class ReviewObservation(Observation):
    """Observation: what the agent sees after each step.

    Inherits from Observation which provides: done, reward, metadata
    """

    task_id: str = Field(default="", description="Unique task identifier")
    task_difficulty: str = Field(default="", description="easy, medium, or hard")
    code_snippet: str = Field(default="", description="Code to review")
    language: str = Field(default="python", description="Programming language")
    context: str = Field(default="", description="Additional context about the code")
    feedback: str = Field(default="", description="Feedback on the agent's last action")


class ReviewState(State):
    """Internal environment state (for grading, not visible to agent).

    Inherits from State which provides: episode_id, step_count
    """

    task_id: str = Field(default="", description="Current task ID")
    ground_truth_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The actual issues in the code (ground truth for grading)",
    )
    agent_found_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Issues the agent has found so far",
    )
    max_steps: int = Field(default=3, description="Maximum review attempts")
    is_complete: bool = Field(default=False, description="Episode done flag")
    final_score: float = Field(default=0.0, description="Final grader score")
