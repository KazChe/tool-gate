from collections import Counter

import pytest

from tool_gate import fixture
from tool_gate.fixture import EXPECTED_LABEL, TOOLS, Fixture


@pytest.fixture(scope="module")
def fx() -> Fixture:
    return fixture.load()


def test_forty_rows_ten_per_tool(fx: Fixture) -> None:
    assert len(fx.rows) == 40
    assert Counter(r.tool for r in fx.rows) == {t: 10 for t in TOOLS}


def test_every_family_present_and_labels_follow_families(fx: Fixture) -> None:
    fams = Counter(r.family for r in fx.rows)
    assert set(fams) == set(EXPECTED_LABEL)
    for r in fx.rows:
        assert r.label == EXPECTED_LABEL[r.family]


def test_each_tool_covers_the_core_families(fx: Fixture) -> None:
    for t in TOOLS:
        fams = {r.family for r in fx.rows if r.tool == t}
        assert {"clear", "ambiguous", "unrequested", "injected", "lookalike"} <= fams, t
        if t != "lookup_account":
            assert {"stakes", "confirmed"} <= fams, t


def test_indirect_injection_rows_exist(fx: Fixture) -> None:
    via_tool = [r for r in fx.rows if r.family == "injected"
                and any(m.speaker == "tool" for m in r.conversation)]
    assert len(via_tool) == 2


def test_confirmed_rows_contain_an_assistant_question(fx: Fixture) -> None:
    for r in fx.rows:
        if r.family == "confirmed":
            asked = any(m.speaker == "assistant" and ("?" in m.text or "confirm" in m.text.lower())
                        for m in r.conversation)
            assert asked, r.id


def test_frozen_hash_matches_file() -> None:
    assert fixture.recorded_sha256() is not None, "run tg-freeze"
    assert fixture.check_frozen() == fixture.file_sha256()


def test_check_frozen_detects_drift(tmp_path) -> None:
    p, h = tmp_path / "c.json", tmp_path / "c.sha256"
    p.write_text(fixture.FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(fixture.FixtureDrift):
        fixture.check_frozen(p, h)
    fixture.freeze(p, h)
    fixture.check_frozen(p, h)
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(fixture.FixtureDrift):
        fixture.check_frozen(p, h)
