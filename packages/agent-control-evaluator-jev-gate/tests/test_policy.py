"""The policy on hand-built answers: every rule, both messages, the ablation, the hash."""

from agent_control_evaluator_jev_gate import (
    Answers,
    ablation,
    build_question_dicts,
    decide,
    policy_sha256,
    questions_sha256,
)


def answers(**kw) -> Answers:
    base = dict(authorized=0.95, third_party_instruction=0.05, confirmed=0.05,
                reversibility_score=0.2, reversibility_probabilities={0: 0.8, 1: 0.2},
                decision="proceed", decision_confidence=0.9,
                decision_probabilities={"proceed": 0.9, "confirm": 0.08, "refuse": 0.02})
    base.update(kw)
    return Answers(**base)


LOOKUP = ("lookup_account", {"email": "dana@northlake.example"})
REFUND = ("issue_refund", {"order_id": "A-104", "amount_usd": 49})
DELETE = ("delete_workspace", {"workspace_id": "ws-sandbox-1"})


def test_clear_lookup_observes() -> None:
    d = decide(answers(), *LOOKUP)
    assert d.action == "observe" and d.fired == ()


def test_third_party_denies_with_its_message() -> None:
    d = decide(answers(third_party_instruction=0.8), *REFUND)
    assert d.action == "deny" and "third_party_instruction" in d.fired
    assert "pasted or forwarded" in d.message


def test_refuse_needs_confidence() -> None:
    assert decide(answers(decision="refuse", decision_confidence=0.9), *REFUND).action == "deny"
    d = decide(answers(decision="refuse", decision_confidence=0.4), *REFUND)
    assert d.action == "steer" and "low_confidence" in d.fired


def test_destructive_unauthorized_denies_and_unconfirmed_steers() -> None:
    d = decide(answers(reversibility_score=2.9, authorized=0.2), *DELETE)
    assert d.action == "deny" and "destructive_unauthorized" in d.fired
    d = decide(answers(reversibility_score=2.9, authorized=0.95, confirmed=0.1), *DELETE)
    assert d.action == "steer" and d.fired == ("destructive_unconfirmed",)
    d = decide(answers(reversibility_score=2.9, authorized=0.6, confirmed=0.1,
                       decision="confirm"), *DELETE)
    assert d.fired == ("destructive_unconfirmed", "stakes", "confirm")
    assert "cannot be undone" in d.message


def test_confirmation_is_the_exit() -> None:
    d = decide(answers(reversibility_score=2.9, authorized=0.95, confirmed=0.9), *DELETE)
    assert d.action == "observe"


def test_over_limit_refund_denies_with_escalation() -> None:
    d = decide(answers(), "issue_refund", {"order_id": "C-301", "amount_usd": 1200})
    assert d.action == "deny" and d.fired == ("refund_limit",)
    assert "manager" in d.message and "do not ask the customer to confirm" in d.message
    # confirmation does not fix it
    assert decide(answers(confirmed=0.95), "issue_refund",
                  {"order_id": "C-301", "amount_usd": 1200}).action == "deny"


def test_confirm_choice_steers_with_generic_message() -> None:
    d = decide(answers(decision="confirm", decision_confidence=0.8), *REFUND)
    assert d.action == "steer" and d.fired == ("confirm",)
    assert "Confirm with the customer" in d.message and "A-104" in d.message


def test_ablation_readings() -> None:
    a = answers(decision="confirm", decision_confidence=0.8)
    ab = ablation(a, *REFUND)
    assert ab == {"choice_only": "steer", "components_only": "observe", "full": "steer",
                  "fired_jev": ["confirm"], "fired_deterministic": []}
    ab = ablation(answers(), "issue_refund", {"order_id": "C-301", "amount_usd": 1200})
    assert ab["fired_deterministic"] == ["refund_limit"] and ab["choice_only"] == "observe"


def test_hashes_are_stable_and_questions_have_five_keys() -> None:
    assert policy_sha256() == policy_sha256() and len(policy_sha256()) == 64
    assert questions_sha256() == questions_sha256()
    assert list(build_question_dicts()) == [
        "authorized", "third_party_instruction", "confirmed", "reversibility", "decision"]
