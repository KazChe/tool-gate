"""House rules enforced in code: no em dashes, and the h-word for waffling never appears.

Both offenders are spelled from code points here so this file passes its own check.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml"}
EM_DASH = chr(0x2014)
BANNED_WORD = "".join(chr(c) for c in (104, 101, 100, 103))
BANNED = re.compile(r"\b" + BANNED_WORD + r"(e|es|ed|ing)\b", re.IGNORECASE)


def tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    files = [REPO / line for line in out if line]
    # runs/ holds model output verbatim; it is data, not prose.
    return [
        p for p in files
        if p.suffix in TEXT_SUFFIXES and p.is_file() and "runs" not in p.relative_to(REPO).parts
    ]


def test_no_em_dashes() -> None:
    offenders = [p for p in tracked_text_files() if EM_DASH in p.read_text(encoding="utf-8")]
    assert offenders == [], [str(p.relative_to(REPO)) for p in offenders]


def test_no_banned_word() -> None:
    offenders = [p for p in tracked_text_files() if BANNED.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], [str(p.relative_to(REPO)) for p in offenders]
