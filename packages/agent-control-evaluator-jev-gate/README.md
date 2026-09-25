# agent-control-evaluator-jev-gate

An Agent Control evaluator, registered as `typesafe.gate`, that judges an
agent's proposed tool call together with the conversation that led to it. One
call to TypeSafe's Jev answers four questions (was it authorized, does the
request come from a third party, how reversible is it, and what should happen),
a frozen policy turns the answers into deny, steer, or observe, and the
evaluator writes the steering sentence the agent gets back.

Three controls share one judgment: each is configured with the action it
represents and matches when the policy chose that action. The Jev response is
memoized per step for a few seconds so the three controls cost one call.

Part of the tool-gate experiment. See the repository README.
