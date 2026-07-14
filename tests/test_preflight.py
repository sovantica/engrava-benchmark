"""Tests for the $0 dataset pre-flight scan (``scripts/preflight.py``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.preflight import format_report, main, scan_dataset

if TYPE_CHECKING:
    from pathlib import Path


def _item(qid: str, sessions: list[list[dict[str, str]]]) -> dict[str, object]:
    return {
        "question_id": qid,
        "question_type": "single-session-user",
        "question": "?",
        "answer": "x",
        "haystack_dates": ["2026-01-01"] * len(sessions),
        "haystack_session_ids": [f"s{i}" for i in range(len(sessions))],
        "haystack_sessions": sessions,
    }


def test_scan_detects_long_empty_and_duplicate() -> None:
    """scan_dataset flags over-long turns, empty turns, and duplicate sessions."""
    long_text = "alpha " * 20000  # ~20000 cl100k tokens, well over 8192
    dup_session = [{"role": "user", "content": "identical content"}]
    raw = [
        _item("q_long", [[{"role": "user", "content": long_text}]]),
        _item("q_empty", [[{"role": "user", "content": "   "}]]),
        _item("q_dup", [list(dup_session), list(dup_session)]),  # byte-identical dup
    ]

    report = scan_dataset(raw)

    assert report.question_count == 3
    assert [t.question_id for t in report.long_turns] == ["q_long"]
    assert report.long_turns[0].token_count > 8192
    assert [t.question_id for t in report.empty_turns] == ["q_empty"]
    assert [d.question_id for d in report.duplicate_sessions] == ["q_dup"]
    assert report.duplicate_sessions[0].duplicate_count == 1


def test_scan_ignores_assistant_turns() -> None:
    """Only USER turns are scanned (assistant turns are never embedded)."""
    long_text = "beta " * 20000
    raw = [_item("q", [[{"role": "assistant", "content": long_text}]])]

    report = scan_dataset(raw)

    assert report.long_turns == []
    assert report.empty_turns == []


def test_format_report_has_safe_to_run_line() -> None:
    """The rendered summary ends with a clear truncated/skipped safe-to-run line."""
    raw = [
        _item("q_long", [[{"role": "user", "content": "alpha " * 20000}]]),
        _item("q_empty", [[{"role": "user", "content": ""}]]),
    ]
    text = format_report(scan_dataset(raw))
    assert "1 input(s) will be truncated" in text
    assert "1 empty turn(s) skipped" in text
    assert "safe to run" in text


def test_main_scans_file_and_reports(tmp_path: Path, capsys: object) -> None:
    """The CLI reads a dataset file and prints a report (exit 0)."""
    raw = [_item("q_long", [[{"role": "user", "content": "alpha " * 20000}]])]
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    code = main([str(path)])

    assert code == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Pre-flight scan: 1 question(s)." in out
    assert "q_long" in out


def test_main_limit_slices(tmp_path: Path, capsys: object) -> None:
    """``--limit`` scans only the first N questions."""
    raw = [
        _item("q1", [[{"role": "user", "content": "hi"}]]),
        _item("q2", [[{"role": "user", "content": "alpha " * 20000}]]),
    ]
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    code = main([str(path), "--limit", "1"])

    assert code == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "1 question(s)." in out
    assert "q2" not in out  # sliced out
