"""Tests for the leaderboard builder (verified-only + comparability segmenting)."""

from __future__ import annotations

import copy
import json
from typing import Any

import scripts.build_leaderboard as bl


def _write_row(results_dir: Any, row: dict[str, Any]) -> None:
    out = (
        results_dir
        / "longmemeval-s"
        / "longmemeval-official"
        / "engrava"
        / f"{row['result_id']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row), encoding="utf-8")


def test_excludes_unverified_via_loader(tmp_path: Any, valid_sovantica_row: dict[str, Any]) -> None:
    """The publication gate: unverified rows are excluded by load_verified_rows."""

    _write_row(tmp_path, valid_sovantica_row)
    unverified = copy.deepcopy(valid_sovantica_row)
    unverified["result_id"] = "other"
    unverified["verification_status"] = "unverified"
    _write_row(tmp_path, unverified)

    rows = bl.load_verified_rows(tmp_path)
    board = bl.build_leaderboard(rows)
    ids = [r["result_id"] for seg in board["segments"] for r in seg["rows"]]
    assert ids == ["lme-s_engrava_0.4.0_2026-06-20_a1b2c3"]
    assert board["row_count"] == 1


def test_load_verified_rows_filters(tmp_path: Any, valid_sovantica_row: dict[str, Any]) -> None:

    _write_row(tmp_path, valid_sovantica_row)
    drafted = copy.deepcopy(valid_sovantica_row)
    drafted["result_id"] = "drafted"
    drafted["verification_status"] = "pending"
    _write_row(tmp_path, drafted)
    rows = bl.load_verified_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["verification_status"] == "verified"


def test_segments_by_reader_judge(valid_sovantica_row: dict[str, Any]) -> None:
    a = valid_sovantica_row
    b = copy.deepcopy(valid_sovantica_row)
    b["result_id"] = "b"
    b["reader_snapshot"] = "gpt-4o-mini-2024-07-18"  # different reader -> different segment
    board = bl.build_leaderboard([a, b])
    assert len(board["segments"]) == 2


def test_segments_by_harness(valid_sovantica_row: dict[str, Any]) -> None:
    """Cross-harness rows land in different segments and are never co-ranked."""
    a = valid_sovantica_row
    b = copy.deepcopy(valid_sovantica_row)
    b["result_id"] = "b"
    b["harness"] = {
        "name": "external-harness",
        "source": "https://example.test/external-harness",
        "version": "external@1.0",
    }
    board = bl.build_leaderboard([a, b])
    assert len(board["segments"]) == 2
    names = {seg["comparability"]["harness.name"] for seg in board["segments"]}
    assert names == {"longmemeval-official", "external-harness"}


def test_segments_by_harness_version(valid_sovantica_row: dict[str, Any]) -> None:
    """Same harness name but a different version is a different comparability segment."""
    a = valid_sovantica_row
    b = copy.deepcopy(valid_sovantica_row)
    b["result_id"] = "b"
    b["harness"] = dict(a["harness"])
    b["harness"]["version"] = "engrava-benchmark@deadbee"
    board = bl.build_leaderboard([a, b])
    assert len(board["segments"]) == 2
    versions = {seg["comparability"]["harness.version"] for seg in board["segments"]}
    assert versions == {a["harness"]["version"], "engrava-benchmark@deadbee"}


def test_ranks_within_segment_by_overall_micro(valid_sovantica_row: dict[str, Any]) -> None:
    low = valid_sovantica_row
    high = copy.deepcopy(valid_sovantica_row)
    high["result_id"] = "high"
    high["metrics"]["overall_micro"] = 0.9
    board = bl.build_leaderboard([low, high])
    rows = board["segments"][0]["rows"]
    assert [r["result_id"] for r in rows] == ["high", "lme-s_engrava_0.4.0_2026-06-20_a1b2c3"]
