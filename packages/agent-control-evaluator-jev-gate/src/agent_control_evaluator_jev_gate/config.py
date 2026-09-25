"""Configuration for `typesafe.gate`. One control per action; all three share one judgment."""

from __future__ import annotations

from typing import Literal

from agent_control_evaluators import EvaluatorConfig
from pydantic import ConfigDict, Field

PINNED_MODEL = "jev-1.13.0"


class GateConfig(EvaluatorConfig):
    """The API key is read from TYPESAFE_API_KEY where evaluation runs, never from config."""

    model_config = ConfigDict(**{**EvaluatorConfig.model_config, "extra": "forbid"})

    decision: Literal["deny", "steer", "observe"] = Field(
        description="The action this control represents; matched when the policy chose it"
    )
    model: str = Field(PINNED_MODEL, description="Jev model id; pin a version")
    timeout_ms: int = Field(10000, ge=1000, le=60000)
    memo_ttl_seconds: float = Field(
        10.0, ge=0.0, le=300.0,
        description="How long one step's judgment is reused by the other controls on that step",
    )
