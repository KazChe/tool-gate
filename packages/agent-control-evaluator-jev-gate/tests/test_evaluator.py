"""The evaluator with a fake Jev client: matching per control, message flow, memo single-flight."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agent_control_evaluator_jev_gate import GateConfig, JevGateEvaluator, clear_memo, judge
from agent_control_evaluator_jev_gate.evaluator import extract_call
from agent_control_evaluators import list_evaluators
from agent_control_models import Step
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

CONV = [{"from": "customer", "text": "Delete workspace ws-sandbox-1, it was a test space."}]


def response(decision="confirm", conf=0.85, authorized=0.9, third=0.05, confirmed=0.1,
             rev=(0.0, 0.0, 0.05, 0.95)) -> SystemOneResponse:
    probs = {"proceed": 0.05, "confirm": 0.05, "refuse": 0.05}
    probs[decision] = 0.9
    return SystemOneResponse(
        model="jev-1.13.0", usage=Usage(input_tokens=1400, output_tokens=0),
        answers={
            "authorized": NoulAnswer(noul=authorized),
            "third_party_instruction": NoulAnswer(noul=third),
            "confirmed": NoulAnswer(noul=confirmed),
            "reversibility": ScoreAnswer(
                score=sum(i * p for i, p in enumerate(rev)), confidence=0.9,
                probabilities={i: p for i, p in enumerate(rev)},
                legend={i: f"level {i}" for i in range(4)},
            ),
            "decision": ChoiceAnswer(choice=decision, confidence=conf, probabilities=probs),
        },
    )


class FakeClient:
    def __init__(self, resp: SystemOneResponse, delay: float = 0.0) -> None:
        self.resp, self.delay, self.calls = resp, delay, 0

    async def system_one(self, state: Any, questions: Any, **kw: Any) -> SystemOneResponse:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.resp


def step(tool="delete_workspace", args=None, conv=CONV) -> Step:
    return Step(type="tool", name=tool, input=args or {"workspace_id": "ws-sandbox-1"},
                context={"conversation": conv})


@pytest.fixture(autouse=True)
def _fresh_memo():
    clear_memo()
    yield
    clear_memo()


def test_registered() -> None:
    assert list_evaluators()["typesafe.gate"] is JevGateEvaluator


def test_extract_call_from_step_and_from_dict() -> None:
    s = step()
    assert extract_call(None, s) == ("delete_workspace", {"workspace_id": "ws-sandbox-1"}, CONV)
    assert extract_call(s.model_dump(mode="json")) == extract_call(None, s)
    with pytest.raises(ValueError):
        extract_call({"name": "x", "input": {}, "context": {}})


async def test_three_controls_one_call_and_only_the_right_one_matches() -> None:
    client = FakeClient(response())  # destructive, authorized, unconfirmed -> steer
    evs = {a: JevGateEvaluator(GateConfig(decision=a), client=client)
           for a in ("deny", "steer", "observe")}
    s = step()
    results = await asyncio.gather(*(e.evaluate_with_context(s.model_dump(mode="json"), s)
                                     for e in evs.values()))
    by = dict(zip(evs, results, strict=True))
    assert client.calls == 1
    assert [a for a, r in by.items() if r.matched] == ["steer"]
    steer = by["steer"]
    assert "cannot be undone" in (steer.message or "")
    assert steer.confidence == pytest.approx(0.85)
    m = steer.metadata or {}
    assert m["action"] == "steer" and m["fired"][0] == "destructive_unconfirmed"
    assert m["message"] == steer.message and m["ablation"]["choice_only"] == "steer"
    assert m["reported_model"] == "jev-1.13.0" and m["input_tokens"] == 1400
    assert sum(1 for r in results if (r.metadata or {})["from_memo"]) == 2


async def test_memo_expires_and_refetches() -> None:
    client = FakeClient(response())
    ev = JevGateEvaluator(GateConfig(decision="steer", memo_ttl_seconds=0.0), client=client)
    s = step()
    await ev.evaluate_with_context(None, s)
    await ev.evaluate_with_context(None, s)
    assert client.calls == 2


async def test_confirmed_row_observes() -> None:
    client = FakeClient(response(decision="proceed", confirmed=0.95))
    ev = JevGateEvaluator(GateConfig(decision="observe"), client=client)
    r = await ev.evaluate_with_context(None, step())
    assert r.matched and (r.metadata or {})["action"] == "observe"


async def test_errors_are_results_not_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    from typesafe_sdk import TypeSafeAPITimeoutError

    class Boom:
        async def system_one(self, *a: Any, **k: Any) -> Any:
            raise TypeSafeAPITimeoutError("slow")

    ev = JevGateEvaluator(GateConfig(decision="deny"), client=Boom())
    r = await ev.evaluate_with_context(None, step())
    assert r.matched is False and r.error and "Timeout" in r.error
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    r = await JevGateEvaluator.from_dict({"decision": "deny"}).evaluate(step().model_dump())
    assert r.error and "TYPESAFE_API_KEY" in r.error


async def test_judge_direct_path() -> None:
    j = await judge(FakeClient(response()), "jev-1.13.0", 10.0, "delete_workspace",
                    {"workspace_id": "ws-sandbox-1"}, CONV)
    assert j.decision.action == "steer" and j.answers.reversibility_score == pytest.approx(2.95)
