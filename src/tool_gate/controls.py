"""The three controls and how they get onto an Agent Control server.

All three point at the same evaluator, `typesafe.gate`, configured with the action the
control represents. `execution: sdk` runs the evaluator in this process, where the
TypeSafe key lives. The steer control carries no static steering_context on purpose, so
the evaluator's message becomes the steering text the agent receives."""

from __future__ import annotations

from typing import Any

AGENT_NAME = "support-desk-agent"
TOOL_STEPS = [
    {"type": "tool", "name": t}
    for t in ("lookup_account", "issue_refund", "cancel_subscription", "delete_workspace")
]
CONTROL_NAMES = {"deny": "gate-deny", "steer": "gate-steer", "observe": "gate-observe"}


def definition(decision: str, model: str = "jev-1.13.0") -> dict[str, Any]:
    return {
        "description": f"Jev tool-call gate; matches when the policy chooses {decision}",
        "enabled": True,
        "execution": "sdk",
        "scope": {"step_types": ["tool"], "stages": ["pre"]},
        "condition": {
            "selector": {"path": "*"},
            "evaluator": {
                "name": "typesafe.gate",
                "config": {"decision": decision, "model": model},
            },
        },
        "action": {"decision": decision},
        "tags": ["tool-gate"],
    }


async def ensure(
    server_url: str, agent_name: str = AGENT_NAME, model: str = "jev-1.13.0"
) -> dict[str, int]:
    """Register the agent, create or update the three controls, attach them. Returns name->id."""
    from agent_control import Agent, AgentControlClient, agents, controls

    ids: dict[str, int] = {}
    async with AgentControlClient(base_url=server_url, timeout=30.0) as client:
        await client.health_check()
        await agents.register_agent(
            client,
            Agent(
                agent_name=agent_name,
                agent_description="Support desk agent with four tools behind a Jev gate",
            ),
            steps=TOOL_STEPS,
        )
        for decision, name in CONTROL_NAMES.items():
            data = definition(decision, model)
            try:
                result = await controls.create_control(client, name=name, data=data)
                control_id = result["control_id"]
            except Exception as exc:  # noqa: BLE001
                if "409" not in str(exc):
                    raise
                existing = (await controls.list_controls(client, name=name, limit=1)).get(
                    "controls", []
                )
                if not existing:
                    raise
                control_id = existing[0]["id"]
                await controls.set_control_data(client, control_id, data)
            try:
                await agents.add_agent_control(client, agent_name, control_id)
            except Exception as exc:  # noqa: BLE001
                if "409" not in str(exc) and "already" not in str(exc).lower():
                    raise
            ids[name] = control_id
    return ids


def plane_action(result: Any) -> tuple[str, list[str], str | None]:
    """(action, matched gate control names, steering message) from an EvaluationResult."""
    matched = [m for m in (result.matches or []) if m.control_name in CONTROL_NAMES.values()]
    names = [m.control_name for m in matched]
    action = "observe"
    message = None
    for m in matched:
        if m.action == "deny":
            action = "deny"
            message = m.result.message
    if action != "deny":
        for m in matched:
            if m.action == "steer":
                action = "steer"
                sc = m.steering_context
                message = sc.message if sc is not None else m.result.message
    return action, names, message
