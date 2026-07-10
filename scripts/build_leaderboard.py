"""Aggregate verified result rows into ``leaderboard.json``.

``leaderboard.json`` is a GENERATED artifact (CI rebuilds it on merge; never
hand-edited). engrava.ai pinned-reads it at build time.

Inclusion rule (the publication gate):
    Only ``verification_status == "verified"`` rows enter the leaderboard.
    ``pending`` and ``unverified`` rows are excluded — a fetched/draft row cannot
    reach the public board until a human promotes it to ``verified`` in review.

Rows are grouped into **comparability segments** keyed by the identity tuple
``(benchmark, benchmark_version, dataset_revision, split, reader_snapshot,
reader_endpoint, judge_snapshot, judge_endpoint, scorer_version)`` — rows in
different segments are never co-ranked (the reader/judge comparability guard).
Within a segment, rows are ordered by ``metrics.overall_micro`` descending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = REPO_ROOT / "leaderboard.json"
SCHEMA_VERSION = "1.0"
_SEGMENT_KEYS = (
    "benchmark",
    "benchmark_version",
    "dataset_revision",
    "split",
    "reader_snapshot",
    "reader_endpoint",
    "judge_snapshot",
    "judge_endpoint",
    "scorer_version",
)


def _segment_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return the comparability-tuple key for a row."""
    return tuple(row.get(k) for k in _SEGMENT_KEYS)


def load_verified_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Load every verified row from the partitioned results tree.

    Loads depth-3 result rows only (``results/<benchmark>/<system>/<id>.json``) and
    keeps rows with ``verification_status == "verified"``. Segmentation downstream
    stays keyed off the in-file comparability fields, not directory position.

    Args:
        results_dir: The ``results/`` root.

    Returns:
        The verified rows (unverified/pending excluded).

    """
    import scripts.validate_results as vr  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for path in vr.iter_result_files(results_dir):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("verification_status") == "verified":
            rows.append(row)
    return rows


def build_leaderboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group verified rows into ranked comparability segments.

    Args:
        rows: Verified result rows.

    Returns:
        The leaderboard aggregate object (schema-stamped).

    """
    segments: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        segments.setdefault(_segment_key(row), []).append(row)

    out_segments: list[dict[str, Any]] = []
    for key, seg_rows in segments.items():
        ranked = sorted(
            seg_rows,
            key=lambda r: r.get("metrics", {}).get("overall_micro", 0.0),
            reverse=True,
        )
        out_segments.append(
            {
                "comparability": dict(zip(_SEGMENT_KEYS, key, strict=True)),
                "rows": ranked,
            }
        )
    out_segments.sort(key=lambda s: tuple(str(v) for v in s["comparability"].values()))
    return {
        "schema_version": SCHEMA_VERSION,
        "segments": out_segments,
        "row_count": len(rows),
    }


def main() -> int:
    """Rebuild ``leaderboard.json`` from verified result rows.

    Returns:
        Process exit code (always ``0`` on success).

    """
    rows = load_verified_rows(RESULTS_DIR)
    leaderboard = build_leaderboard(rows)
    OUTPUT_PATH.write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(  # noqa: T201
        f"Wrote {OUTPUT_PATH.name}: {leaderboard['row_count']} verified row(s) "
        f"in {len(leaderboard['segments'])} comparability segment(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
