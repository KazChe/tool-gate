"""Stdout report over an artifact."""

from __future__ import annotations

from typing import Any

from tool_gate import fixture as fx


def _pct(n: int, d: int) -> str:
    return "n/a" if not d else f"{n / d * 100:.0f}%"


def render(artifact: dict[str, Any]) -> str:
    meta, s = artifact["meta"], artifact["summary"] or {}
    rows = artifact["fixture"]
    runs = [r for r in artifact["direct"]["runs"] if r]
    out: list[str] = []
    w = out.append
    w(f"tool-gate eval  status={meta['status']}  dry_run={meta['dry_run']}")
    w(
        f"  fixture {meta['fixture_sha256'][:12]}  rows {meta['fixture_rows']}  questions "
        f"{meta['questions_sha256'][:12]}  policy {meta['policy_sha256'][:12]}  commit "
        f"{(meta['git_commit'] or 'n/a')[:12]}"
    )
    w(
        f"  jev {meta['jev']['requested_model']} -> reported {meta['jev']['reported_model']}  runs "
        f"{meta['jev']['runs_completed']}/{meta['jev']['runs_requested']}"
    )
    if not runs:
        return "\n".join(out) + "\n"
    last = runs[-1]
    plane = artifact.get("plane", {})

    w("")
    hdr = (
        f"{'id':6} {'family':11} {'label':8} {'action':8} {'auth':>5} {'3rd':>5} {'conf':>5} "
        f"{'rev':>5} {'choice':8} {'c.conf':>6} fired / plane"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        c = last.get(r["id"])
        if not c or c.get("error"):
            err = (c or {}).get("error", "")[:60]
            w(f"{r['id']:6} {r['family']:11} {r['label']:8} ERROR {err}")
            continue
        a = c["answers"]
        mark = "" if c["action"] == r["label"] else " <-"
        p = plane.get(r["id"])
        pl = (
            ""
            if not p
            else (
                f" | plane {p['action']}"
                + ("" if p.get("matched_count") == 1 else f" ({p.get('matched_count')} matched)")
                if not p.get("error")
                else " | plane ERROR"
            )
        )
        w(
            f"{r['id']:6} {r['family']:11} {r['label']:8} {c['action']:8} {a['authorized']:5.2f} "
            f"{a['third_party_instruction']:5.2f} {a['confirmed']:5.2f} "
            f"{a['reversibility_score']:5.2f} "
            f"{a['decision']:8} {a['decision_confidence']:6.2f} "
            f"{','.join(c['fired']) or '-'}{pl}{mark}"
        )

    w("")
    w(
        "Agreement with labels per run: "
        + "  ".join(f"{a}/{n}" for a, n in zip(s["agreement_per_run"], s["n_per_run"], strict=True))
    )
    w("Confusion, last run (rows = label, columns = action)")
    cm = s["confusion_last_run"]
    w(f"  {'':9}" + "".join(f"{b:>9}" for b in fx.ACTIONS))
    for a in fx.ACTIONS:
        w(f"  {a:9}" + "".join(f"{cm[a][b]:9}" for b in fx.ACTIONS))
    ek = s["error_kinds_last_run"]
    w(
        f"  deny when steer would do: {ek['deny_when_steer']}   steer when deny was needed: "
        f"{ek['steer_when_deny']}   observe on a deny row: {ek['observe_when_deny']}   "
        f"acted when it should not: {ek['acted_when_should_not']}"
    )
    w("Per family, last run")
    for f, v in s["per_family_last_run"].items():
        w(f"  {f:11} {v['agree']}/{v['n']}")
    ab = s["ablation_agreement_last_run"]
    n = s["n_per_run"][-1]
    w(
        f"Ablation, agreement with labels: Choice alone {ab['choice_only']}/{n}, components alone "
        f"{ab['components_only']}/{n}, full policy {ab['full']}/{n}; rows where a Jev rule fired "
        f"{s['rows_with_jev_rule_fired']}, where a deterministic rule fired "
        f"{s['rows_with_deterministic_rule_fired']}"
    )
    w("Reversibility score per tool (min / mean / max)")
    for t, v in s["reversibility_per_tool"].items():
        w(f"  {t:20} {v['min']:.2f} / {v['mean']:.2f} / {v['max']:.2f}")
    w(
        "third_party_instruction on injected rows: "
        + ", ".join(f"{i} {v:.2f}" for i, v in s["third_party_on_injected"].items())
    )
    w(
        "confirmed on confirmed rows: "
        + ", ".join(f"{i} {v:.2f}" for i, v in s["confirmed_on_confirmed_rows"].items())
    )
    if "repeatability" in s:
        r = s["repeatability"]
        w(
            f"Repeatability over {r['runs']} runs: rows whose action changed "
            f"{len(r['rows_with_action_change'])} {r['rows_with_action_change']}; max authorized "
            f"swing {r['max_authorized_swing']:.2f}"
        )
    if s.get("latency"):
        w(
            f"Latency mean {s['latency']['mean']:,.0f} ms, max {s['latency']['max']:,.0f} ms over "
            f"{s['latency']['n']} calls; cost ${s['cost_per_1000_calls']:.4f} per 1,000 tool calls"
        )
    if s.get("plane"):
        p = s["plane"]
        w("")
        w(
            f"Through Agent Control: {p['rows']} rows; plane action equals direct action on "
            f"{p['agree_with_direct_last_run']}, equals label on {p['agree_with_label']}; exactly "
            f"one gate control matched on {p['exactly_one_match']}; invariant violations "
            f"{p['invariant_violations']}; errors {p['errors']}"
        )
        steers = [(i, x) for i, x in plane.items() if not x.get("error") and x["action"] == "steer"]
        if steers:
            i, x = steers[0]
            w(f"  example steer text ({i}): {x['message']}")
    return "\n".join(out) + "\n"
