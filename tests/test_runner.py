import json
from pathlib import Path

from tool_gate import cli, runner


def test_dry_run_direct_only_writes_summary(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    art = runner.run_sync(runner.RunConfig(runs=2, out=out, dry_run=True, plane=False))
    on_disk = json.loads(out.read_text())
    assert on_disk["meta"]["status"] == "complete"
    assert on_disk["meta"]["jev"]["runs_completed"] == 2
    s = art["summary"]
    assert s["n_per_run"] == [40, 40]
    # The fake client answers by family, so the policy should reproduce every label.
    assert s["agreement_per_run"] == [40, 40], s["misses_last_run"]
    assert s["ablation_agreement_last_run"]["full"] == 40
    assert set(s["reversibility_per_tool"]) == {
        "lookup_account",
        "issue_refund",
        "cancel_subscription",
        "delete_workspace",
    }
    assert s["repeatability"]["rows_with_action_change"] == []
    assert s["cost_per_1000_calls"] > 0


def test_report_and_resume(tmp_path: Path, capsys) -> None:
    out = tmp_path / "r.json"
    rc = cli.eval_main(
        ["--dry-run", "--no-plane", "--runs", "1", "--rows", "rf-01,rf-06,dw-01", "--out", str(out)]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "rf-06" in text and "refund_limit" in text and "Ablation" in text
    art = json.loads(out.read_text())
    del art["direct"]["runs"][0]["rf-01"]
    out.write_text(json.dumps(art))
    art2 = runner.run_sync(
        runner.RunConfig(
            runs=1,
            out=out,
            dry_run=True,
            plane=False,
            row_ids=["rf-01", "rf-06", "dw-01"],
            resume=True,
        )
    )
    assert set(art2["direct"]["runs"][0]) == {"rf-01", "rf-06", "dw-01"}


def test_fixture_drift_refuses(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from tool_gate import fixture

    monkeypatch.setattr(fixture, "FIXTURE_HASH_PATH", tmp_path / "missing")
    with pytest.raises(fixture.FixtureDrift):
        runner.run_sync(
            runner.RunConfig(runs=1, out=tmp_path / "x.json", dry_run=True, plane=False)
        )
