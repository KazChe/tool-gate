"""tg-freeze, tg-eval, tg-demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tool_gate import fixture, report
from tool_gate.runner import ModelChanged, RunConfig, run_sync


def load_dotenv(path: Path = fixture.REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def freeze_main(argv: list[str] | None = None) -> int:
    digest = fixture.freeze()
    print(f"froze {fixture.FIXTURE_PATH.name}: sha256 {digest}")
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="tg-eval")
    p.add_argument("--runs", type=int, default=int(os.environ.get("TG_RUNS", "3")))
    p.add_argument("--out", type=Path, default=Path("runs/eval-results.json"))
    p.add_argument("--rows", default=None, help="comma-separated row ids")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="fake Jev, no network")
    p.add_argument("--no-plane", action="store_true", help="skip the Agent Control pass")
    p.add_argument(
        "--server-url", default=os.environ.get("AGENT_CONTROL_URL", "http://localhost:8000")
    )
    p.add_argument("--model", default=os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0"))
    p.add_argument("--allow-fixture-drift", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)
    if args.report_only:
        sys.stdout.write(report.render(json.loads(args.out.read_text(encoding="utf-8"))))
        return 0
    if not args.dry_run and not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is required", file=sys.stderr)
        return 2
    cfg = RunConfig(
        runs=args.runs,
        out=args.out,
        resume=args.resume,
        row_ids=[r.strip() for r in args.rows.split(",")] if args.rows else None,
        dry_run=args.dry_run,
        model=args.model,
        plane=not args.no_plane,
        server_url=args.server_url,
        allow_fixture_drift=args.allow_fixture_drift,
        progress=lambda s: (sys.stderr.write(s), sys.stderr.flush()),
    )
    print(
        f"runs={cfg.runs} model={cfg.model} plane={cfg.plane} server={cfg.server_url} "
        f"out={cfg.out} dry_run={cfg.dry_run}",
        file=sys.stderr,
    )
    try:
        artifact = run_sync(cfg)
    except fixture.FixtureDrift as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    except ModelChanged as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(report.render(artifact))
    errors = sum(1 for r in artifact["direct"]["runs"] for c in r.values() if c.get("error"))
    errors += sum(1 for c in artifact.get("plane", {}).values() if c.get("error"))
    return 1 if errors else 0


def demo_main(argv: list[str] | None = None) -> int:
    """One row per action through Agent Control, printed as a round trip."""
    load_dotenv()
    p = argparse.ArgumentParser(prog="tg-demo")
    p.add_argument(
        "--rows",
        default="dw-01,dw-02,dw-09",
        help="steer, then the confirmed retry, then a denied injection",
    )
    p.add_argument(
        "--server-url", default=os.environ.get("AGENT_CONTROL_URL", "http://localhost:8000")
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    cfg = RunConfig(
        runs=1,
        out=Path("runs/demo.json"),
        row_ids=[r.strip() for r in args.rows.split(",")],
        dry_run=args.dry_run,
        plane=True,
        server_url=args.server_url,
        progress=lambda s: (sys.stderr.write(s), sys.stderr.flush()),
    )
    if not args.dry_run and not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is required", file=sys.stderr)
        return 2
    artifact = run_sync(cfg)
    fx = fixture.load()
    for rid in cfg.row_ids or []:
        row = fx.by_id(rid)
        d = artifact["direct"]["runs"][0][rid]
        pl = artifact["plane"].get(rid, {})
        print(f"\n== {rid} ({row.family}, label {row.label})")
        for m in row.conversation:
            print(f"  {m.speaker:9} {m.text}")
        print(f"  proposed  {row.tool}({json.dumps(row.proposed_call.arguments)})")
        a = d["answers"]
        third = a["third_party_instruction"]
        print(
            f"  jev       authorized {a['authorized']:.2f}  third_party {third:.2f}  "
            f"confirmed {a['confirmed']:.2f}  reversibility {a['reversibility_score']:.2f}  "
            f"decision {a['decision']} ({a['decision_confidence']:.2f})"
        )
        print(f"  policy    {d['action']}  fired {d['fired'] or '-'}")
        print(
            f"  plane     {pl.get('action')}  matched {pl.get('matched')}  "
            f"is_safe {pl.get('is_safe')}"
        )
        if pl.get("message"):
            print(f"  text      {pl['message']}")
    return 0
