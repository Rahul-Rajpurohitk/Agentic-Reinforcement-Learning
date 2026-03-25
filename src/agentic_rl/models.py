"""OpenEnv type definitions for the Code Review environment.

Typed Pydantic models following the OpenEnv spec:
- Action: what the agent sends (code review feedback)
- Observation: what the environment returns (code snippet + reward + done)
- State: internal state for grading (ground-truth bugs)
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReviewAction(BaseModel):
    """Action: the agent's code review response.

    The agent analyzes code and reports found issues with severity and fix suggestions.
    """

    issues_found: List[Dict[str, str]] = Field(
        ...,
        description=(
            "List of issues found. Each dict has keys: "
            "'line' (line number as string), "
            "'severity' (one of: 'critical', 'major', 'minor'), "
            "'category' (one of: 'bug', 'security', 'style', 'performance', 'logic'), "
            "'description' (explanation of the issue), "
            "'suggestion' (how to fix it)"
        ),
    )
    overall_assessment: str = Field(
        ...,
        description="Overall code quality assessment: 'approve', 'request_changes', or 'comment'",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its review (0.0-1.0)",
    )


class ReviewObservation(BaseModel):
    """Observation: what the agent sees after each step.

    Contains the code to review, feedback on the review, and reward signals.
    """

    task_id: str = Field(..., description="Unique task identifier")
    task_difficulty: str = Field(..., description="Task difficulty: easy, medium, or hard")
    code_snippet: str = Field(..., description="The code snippet to review")
    language: str = Field(default="python", description="Programming language of the snippet")
    context: str = Field(default="", description="Additional context about the code's purpose")
    feedback: str = Field(default="", description="Feedback on the agent's last action")
    reward: float = Field(default=0.0, description="Reward signal (0.0 - 1.0)")
    done: bool = Field(default=False, description="Whether the episode is complete")
    info: Dict = Field(default_factory=dict, description="Additional metadata")


class ReviewState(BaseModel):
    """Internal environment state (for grading, not visible to agent)."""

    episode_id: str = Field(default="", description="Unique episode identifier")
    task_id: str = Field(default="", description="Current task ID")
    step_count: int = Field(default=0, description="Steps taken in this episode")
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
