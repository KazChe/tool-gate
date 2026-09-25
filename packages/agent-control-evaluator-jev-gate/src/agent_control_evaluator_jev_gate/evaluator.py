"""`typesafe.gate`: one Jev judgment, three controls, deny or steer or observe.

Agent Control calls every control's evaluator separately and never shares a
result, so three controls on one step would cost three Jev calls. The memo
below is single-flight per event loop: the first control to arrive for a step
makes the call, the others await the same task, and the response is kept for a
few seconds. Nothing request-scoped is stored on the instance.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from agent_control_evaluators import Evaluator, EvaluatorMetadata, register_evaluator
from agent_control_models import EvaluatorResult

from agent_control_evaluator_jev_gate.config import GateConfig
from agent_control_evaluator_jev_gate.policy import (
    Answers,
    Decision,
    ablation,
    decide,
    policy_sha256,
)
from agent_control_evaluator_jev_gate.questions import (
    build_questions,
    build_state,
    questions_sha256,
)

API_KEY_ENV = "TYPESAFE_API_KEY"


@dataclass(frozen=True)
class Judgment:
    answers: Answers
    decision: Decision
    reported_model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    request_id: str | None
    from_memo: bool = False


def extract_call(data: Any, step: Any = None) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    """(tool, arguments, conversation) from a Step model or the dict selector '*' produces."""
    src: Any = step if step is not None else data
    name = getattr(src, "name", None) if not isinstance(src, dict) else src.get("name")
    inp = getattr(src, "input", None) if not isinstance(src, dict) else src.get("input")
    ctx = getattr(src, "context", None) if not isinstance(src, dict) else src.get("context")
    if not isinstance(name, str) or not name:
        raise ValueError("step has no tool name")
    if not isinstance(inp, dict):
        raise ValueError("tool step input must be an object of arguments")
    conversation = (ctx or {}).get("conversation") if isinstance(ctx, dict) else None
    if not isinstance(conversation, list):
        raise ValueError("step.context.conversation is required (a list of {from, text})")
    return name, inp, conversation


def parse_answers(resp: Any) -> Answers:
    a = resp.answers
    rev = a["reversibility"]
    dec = a["decision"]
    return Answers(
        authorized=float(a["authorized"].noul),
        third_party_instruction=float(a["third_party_instruction"].noul),
        confirmed=float(a["confirmed"].noul),
        reversibility_score=float(rev.score),
        reversibility_probabilities={int(k): float(v) for k, v in rev.probabilities.items()},
        decision=str(dec.choice),
        decision_confidence=float(dec.confidence),
        decision_probabilities={str(k): float(v) for k, v in dec.probabilities.items()},
    )


async def judge(client: Any, model: str, timeout_s: float, tool: str,
                arguments: dict[str, Any], conversation: list[dict[str, str]]) -> Judgment:
    """One Jev call, five answers, one policy decision. No memo; the runner uses this directly."""
    state = build_state(conversation, tool, arguments)
    started = time.perf_counter()
    resp = await client.system_one(state, build_questions(), model=model, timeout=timeout_s)
    latency_ms = (time.perf_counter() - started) * 1000.0
    answers = parse_answers(resp)
    try:
        request_id = resp.request_id
    except Exception:  # noqa: BLE001
        request_id = None
    return Judgment(
        answers=answers, decision=decide(answers, tool, arguments), reported_model=resp.model,
        input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
        latency_ms=latency_ms, request_id=request_id,
    )


# ---- single-flight memo, keyed by the built state, scoped to the running event loop
_MEMO_LOCK = threading.Lock()
_MEMO: dict[str, tuple[float, Judgment]] = {}
_INFLIGHT: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Task[Judgment]]] = {}


def memo_key(tool: str, arguments: dict[str, Any], conversation: list[dict[str, str]],
             model: str) -> str:
    payload = json.dumps(
        {"m": model, "s": build_state(conversation, tool, arguments)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clear_memo() -> None:
    with _MEMO_LOCK:
        _MEMO.clear()
        _INFLIGHT.clear()


async def judge_memoized(client: Any, model: str, timeout_s: float, ttl: float, tool: str,
                         arguments: dict[str, Any], conversation: list[dict[str, str]]) -> Judgment:
    key = memo_key(tool, arguments, conversation, model)
    loop = asyncio.get_running_loop()
    now = time.monotonic()
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
        if hit and hit[0] > now:
            j = hit[1]
            return Judgment(**{**asdict(j), "answers": j.answers, "decision": j.decision,
                               "from_memo": True})
        inflight = _INFLIGHT.get(key)
        if inflight and inflight[0] is loop and not inflight[1].done():
            task = inflight[1]
            owner = False
        else:
            task = loop.create_task(judge(client, model, timeout_s, tool, arguments, conversation))
            _INFLIGHT[key] = (loop, task)
            owner = True
    try:
        result = await task
    finally:
        if owner:
            with _MEMO_LOCK:
                _INFLIGHT.pop(key, None)
    if owner:
        with _MEMO_LOCK:
            if ttl > 0:
                _MEMO[key] = (time.monotonic() + ttl, result)
        return result
    return Judgment(**{**asdict(result), "answers": result.answers, "decision": result.decision,
                       "from_memo": True})


def _make_client(config: GateConfig) -> tuple[Any, str | None]:
    if not os.environ.get(API_KEY_ENV):
        return None, f"{API_KEY_ENV} is not set"
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError

    try:
        return AsyncTypeSafeClient(model=config.model, timeout=config.timeout_ms / 1000.0), None
    except TypeSafeError as exc:
        return None, f"{type(exc).__name__}: {exc}"


@register_evaluator
class JevGateEvaluator(Evaluator[GateConfig]):
    metadata = EvaluatorMetadata(
        name="typesafe.gate",
        version="0.1.0",
        description=(
            "TypeSafe Jev judges a proposed tool call against the conversation and policy; "
            "matches when the policy's action equals this control's configured decision"
        ),
        requires_api_key=True,
        timeout_ms=10000,
    )
    config_model = GateConfig

    def __init__(self, config: GateConfig, client: Any = None) -> None:
        super().__init__(config)
        self._client = client
        self._client_error: str | None = None
        if client is None:
            self._client, self._client_error = _make_client(config)

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("typesafe_sdk") is not None

    async def evaluate(self, data: Any) -> EvaluatorResult:
        return await self._evaluate(data, None)

    async def evaluate_with_context(self, data: Any, step: Any) -> EvaluatorResult:
        return await self._evaluate(data, step)

    async def _evaluate(self, data: Any, step: Any) -> EvaluatorResult:
        cfg = self.config
        base = {"control_decision": cfg.decision, "requested_model": cfg.model,
                "questions_sha256": questions_sha256(), "policy_sha256": policy_sha256()}
        if self._client is None:
            return EvaluatorResult(matched=False, confidence=0.0,
                                   error=self._client_error or "no client", metadata=base)
        try:
            tool, arguments, conversation = extract_call(data, step)
        except ValueError as exc:
            return EvaluatorResult(matched=False, confidence=0.0, error=str(exc), metadata=base)

        from typesafe_sdk import TypeSafeError

        try:
            j = await judge_memoized(self._client, cfg.model, cfg.timeout_ms / 1000.0,
                                     cfg.memo_ttl_seconds, tool, arguments, conversation)
        except (TypeSafeError, KeyError, ValueError) as exc:
            return EvaluatorResult(matched=False, confidence=0.0,
                                   error=f"{type(exc).__name__}: {exc}", metadata=base)

        matched = j.decision.action == cfg.decision
        meta = {
            **base,
            "action": j.decision.action,
            "fired": list(j.decision.fired),
            "message": j.decision.message,
            "answers": asdict(j.answers),
            "ablation": ablation(j.answers, tool, arguments),
            "reported_model": j.reported_model,
            "input_tokens": j.input_tokens,
            "output_tokens": j.output_tokens,
            "latency_ms": j.latency_ms,
            "request_id": j.request_id,
            "from_memo": j.from_memo,
            "tool": tool,
        }
        return EvaluatorResult(
            matched=matched,
            confidence=j.answers.decision_confidence,
            message=j.decision.message if matched else
            f"policy chose {j.decision.action}; this control represents {cfg.decision}",
            metadata=meta,
        )
