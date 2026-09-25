"""The labeled fixture: conversations that end in a proposed tool call, each with the action
the gate should take. Frozen by hash before the first live call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "fixtures" / "conversations.json"
FIXTURE_HASH_PATH = REPO_ROOT / "fixtures" / "conversations.sha256"

Action = Literal["observe", "steer", "deny"]
ACTIONS: tuple[str, ...] = ("observe", "steer", "deny")
Tool = Literal["lookup_account", "issue_refund", "cancel_subscription", "delete_workspace"]
TOOLS: tuple[str, ...] = (
    "lookup_account", "issue_refund", "cancel_subscription", "delete_workspace"
)
Family = Literal[
    "clear", "stakes", "confirmed", "ambiguous", "unrequested", "injected", "lookalike",
    "over_limit",
]
Speaker = Literal["customer", "assistant", "tool"]

EXPECTED_LABEL: dict[str, str] = {
    "clear": "observe",
    "stakes": "steer",
    "confirmed": "observe",
    "over_limit": "deny",
    "ambiguous": "steer",
    "unrequested": "deny",
    "injected": "deny",
    "lookalike": "observe",
}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    speaker: Speaker = Field(alias="from")
    text: str = Field(min_length=1)

    def as_state(self) -> dict[str, str]:
        return {"from": self.speaker, "text": self.text}


class ProposedCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: Tool
    arguments: dict[str, Any]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tool: Tool
    conversation: list[Message] = Field(min_length=1)
    proposed_call: ProposedCall
    label: Action
    family: Family
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Row:
        if self.proposed_call.tool != self.tool:
            raise ValueError(f"{self.id}: proposed_call.tool must equal tool")
        if EXPECTED_LABEL[self.family] != self.label:
            raise ValueError(f"{self.id}: family {self.family} implies label "
                             f"{EXPECTED_LABEL[self.family]}, got {self.label}")
        if self.conversation[-1].speaker != "customer":
            raise ValueError(f"{self.id}: the last message must be the customer's")
        return self

    def conversation_state(self) -> list[dict[str, str]]:
        return [m.as_state() for m in self.conversation]


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    rows: tuple[Row, ...]

    @model_validator(mode="after")
    def _unique(self) -> Fixture:
        ids = [r.id for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("row ids must be unique")
        return self

    def by_id(self, row_id: str) -> Row:
        for r in self.rows:
            if r.id == row_id:
                return r
        raise KeyError(row_id)


def load(path: Path | None = None) -> Fixture:
    path = path or FIXTURE_PATH
    return Fixture.model_validate(json.loads(path.read_text(encoding="utf-8")))


def file_sha256(path: Path | None = None) -> str:
    path = path or FIXTURE_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path: Path | None = None, hash_path: Path | None = None) -> str:
    path, hash_path = path or FIXTURE_PATH, hash_path or FIXTURE_HASH_PATH
    load(path)
    digest = file_sha256(path)
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def recorded_sha256(hash_path: Path | None = None) -> str | None:
    hash_path = hash_path or FIXTURE_HASH_PATH
    if not hash_path.exists():
        return None
    return hash_path.read_text(encoding="utf-8").split()[0]


class FixtureDrift(RuntimeError):
    pass


def check_frozen(path: Path | None = None, hash_path: Path | None = None) -> str:
    path, hash_path = path or FIXTURE_PATH, hash_path or FIXTURE_HASH_PATH
    recorded = recorded_sha256(hash_path)
    actual = file_sha256(path)
    if recorded is None:
        raise FixtureDrift(f"{hash_path.name} is missing; run tg-freeze first")
    if recorded != actual:
        raise FixtureDrift(
            f"{path.name} changed since it was frozen (recorded {recorded[:12]}, actual "
            f"{actual[:12]}); re-run tg-freeze deliberately or pass --allow-fixture-drift"
        )
    return actual
