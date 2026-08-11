"""Tests for the ranked-retrieval-log diff utility (``runners.retrieval_diff``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from runners import retrieval_diff
from runners.retrieval_diff import (
    Classification,
    RetrievalDiffError,
    Verdict,
    classify,
    diff_retrieval_logs,
    load_retrieval_log,
)

if TYPE_CHECKING:
    from pathlib import Path


# --- classify() -------------------------------------------------------------- #
def test_classify_identical() -> None:
    assert classify(["a", "b"], ["a", "b"]) is Classification.IDENTICAL


def test_classify_reordered() -> None:
    assert classify(["a", "b"], ["b", "a"]) is Classification.REORDERED


def test_classify_boundary_single_tail_swap() -> None:
    # Equal length, non-empty, differ only at the last rank -> benign boundary swap.
    assert classify(["a", "b", "c"], ["a", "b", "d"]) is Classification.BOUNDARY


def test_classify_changed_on_length_difference() -> None:
    # A length change is a real membership change, never a boundary swap.
    assert classify(["a", "b"], ["a"]) is Classification.CHANGED


def test_classify_singleton_total_replacement_is_changed() -> None:
    # A single-item list fully replaced (["a"] vs ["b"]) shares no body: it is a real
    # membership change, not a rank-boundary tail swap.
    assert classify(["a"], ["b"]) is Classification.CHANGED


def test_classify_empty_is_not_boundary() -> None:
    # The "boundary requires non-empty" guard: an empty side is a real change.
    assert classify([], ["a"]) is Classification.CHANGED
    assert classify(["a"], []) is Classification.CHANGED


def test_classify_wholesale_change() -> None:
    assert classify(["a", "b", "c"], ["x", "y", "z"]) is Classification.CHANGED


# --- diff_retrieval_logs() --------------------------------------------------- #
def test_diff_all_identical_is_green() -> None:
    log = {"q1": ["a", "b"], "q2": ["c", "d"]}
    result = diff_retrieval_logs(dict(log), dict(log))
    assert result.verdict is Verdict.GREEN
    assert result.counts.identical == 2
    assert result.changed_fraction == 0.0
    assert result.changed_question_ids == []


def test_diff_reordering_stays_green() -> None:
    result = diff_retrieval_logs({"q1": ["a", "b"]}, {"q1": ["b", "a"]})
    assert result.verdict is Verdict.GREEN
    assert result.counts.reordered == 1


def test_diff_small_changed_fraction_is_yellow() -> None:
    # 1 of 10 questions changed membership -> changed_fraction 0.1 (<=0.15) but a
    # genuine change (counts.changed>0), so not GREEN: YELLOW.
    candidate = {f"q{i}": ["a", "b"] for i in range(10)}
    baseline = {f"q{i}": ["a", "b"] for i in range(10)}
    candidate["q0"] = ["x", "y"]
    result = diff_retrieval_logs(candidate, baseline)
    assert result.verdict is Verdict.YELLOW
    assert result.counts.changed == 1
    assert result.changed_question_ids == ["q0"]


def test_diff_large_changed_fraction_is_red() -> None:
    candidate = {f"q{i}": ["x", "y"] for i in range(10)}
    baseline = {f"q{i}": ["a", "b"] for i in range(10)}
    result = diff_retrieval_logs(candidate, baseline)
    assert result.verdict is Verdict.RED


def test_diff_disjoint_question_sets_is_red() -> None:
    # The disjoint-qid-is-material fix: even if the common set is empty, a mismatch
    # in question coverage escalates directly to RED (never a mild verdict).
    result = diff_retrieval_logs({"q1": ["a"]}, {"q2": ["a"]})
    assert result.verdict is Verdict.RED
    assert result.only_in_candidate == ["q1"]
    assert result.only_in_baseline == ["q2"]
    assert result.questions_compared == 0


def test_diff_partial_overlap_missing_qid_is_red() -> None:
    # One shared identical question, one candidate-only question -> missing>0 -> RED.
    result = diff_retrieval_logs({"q1": ["a"], "q2": ["b"]}, {"q1": ["a"]})
    assert result.verdict is Verdict.RED
    assert result.only_in_candidate == ["q2"]


def test_diff_boundary_only_stays_green() -> None:
    # A single benign boundary swap on a modest fraction is still GREEN (material
    # fraction <= 0.10 with zero genuine changes).
    candidate = {f"q{i}": ["a", "b", "c"] for i in range(20)}
    baseline = {f"q{i}": ["a", "b", "c"] for i in range(20)}
    candidate["q0"] = ["a", "b", "d"]
    result = diff_retrieval_logs(candidate, baseline)
    assert result.counts.boundary == 1
    assert result.verdict is Verdict.GREEN


def test_diff_empty_logs_is_green() -> None:
    result = diff_retrieval_logs({}, {})
    assert result.verdict is Verdict.GREEN
    assert result.questions_compared == 0


# --- load_retrieval_log() ---------------------------------------------------- #
def test_load_from_file(tmp_path: Path) -> None:
    path = tmp_path / "retrieval_log.json"
    path.write_text(json.dumps({"q1": ["a", "b"]}), encoding="utf-8")
    assert load_retrieval_log(path) == {"q1": ["a", "b"]}


def test_load_from_directory(tmp_path: Path) -> None:
    (tmp_path / retrieval_diff.RETRIEVAL_LOG_FILENAME).write_text(
        json.dumps({"q1": ["a"]}), encoding="utf-8"
    )
    assert load_retrieval_log(tmp_path) == {"q1": ["a"]}


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RetrievalDiffError, match="not found"):
        load_retrieval_log(tmp_path / "nope.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "retrieval_log.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RetrievalDiffError, match="not valid JSON"):
        load_retrieval_log(path)


def test_load_non_object_raises(tmp_path: Path) -> None:
    path = tmp_path / "retrieval_log.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    with pytest.raises(RetrievalDiffError, match="must be a JSON object"):
        load_retrieval_log(path)


def test_load_non_list_entry_raises(tmp_path: Path) -> None:
    path = tmp_path / "retrieval_log.json"
    path.write_text(json.dumps({"q1": "not-a-list"}), encoding="utf-8")
    with pytest.raises(RetrievalDiffError, match="list of id strings"):
        load_retrieval_log(path)


def test_load_non_string_ids_raises(tmp_path: Path) -> None:
    path = tmp_path / "retrieval_log.json"
    path.write_text(json.dumps({"q1": [1, 2]}), encoding="utf-8")
    with pytest.raises(RetrievalDiffError, match="list of id strings"):
        load_retrieval_log(path)


# --- CLI main() -------------------------------------------------------------- #
def _write_log(path: Path, log: dict[str, list[str]]) -> None:
    path.write_text(json.dumps(log), encoding="utf-8")


def test_cli_green_returns_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cand = tmp_path / "cand.json"
    base = tmp_path / "base.json"
    _write_log(cand, {"q1": ["a"]})
    _write_log(base, {"q1": ["a"]})
    rc = retrieval_diff.main(["--candidate", str(cand), "--baseline", str(base)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "GREEN"


def test_cli_red_returns_one(tmp_path: Path) -> None:
    cand = tmp_path / "cand.json"
    base = tmp_path / "base.json"
    _write_log(cand, {"q1": ["a"]})
    _write_log(base, {"q2": ["a"]})
    rc = retrieval_diff.main(["--candidate", str(cand), "--baseline", str(base)])
    assert rc == 1


def test_cli_changed_out_written(tmp_path: Path) -> None:
    cand = tmp_path / "cand.json"
    base = tmp_path / "base.json"
    changed = tmp_path / "changed.txt"
    _write_log(cand, {"q1": ["x", "y"], "q2": ["a"]})
    _write_log(base, {"q1": ["a", "b"], "q2": ["a"]})
    retrieval_diff.main(
        [
            "--candidate",
            str(cand),
            "--baseline",
            str(base),
            "--changed-out",
            str(changed),
        ]
    )
    assert changed.read_text(encoding="utf-8") == "q1\n"


def test_cli_changed_out_empty_when_no_changes(tmp_path: Path) -> None:
    cand = tmp_path / "cand.json"
    base = tmp_path / "base.json"
    changed = tmp_path / "changed.txt"
    _write_log(cand, {"q1": ["a"]})
    _write_log(base, {"q1": ["a"]})
    retrieval_diff.main(
        [
            "--candidate",
            str(cand),
            "--baseline",
            str(base),
            "--changed-out",
            str(changed),
        ]
    )
    assert changed.read_text(encoding="utf-8") == ""


def test_boundary_does_not_conceal_a_reordered_body() -> None:
    """A boundary swap whose body also reordered is reported, not silently absorbed.

    ``boundary`` is the mildest non-identical class, and the reader this feeds is order-sensitive,
    so a question that swapped an item at the top-k edge *and* reshuffled the ranks above it must
    not read as a plain edge swap. The membership classification stays ``boundary`` — the swap is
    genuinely all that changed about the set — while the reordered body is surfaced separately.
    """
    baseline = {"q": ["a", "b", "c", "d"]}
    candidate = {"q": ["a", "c", "b", "e"]}  # d -> e at the tail, and b/c swapped above it

    result = retrieval_diff.diff_retrieval_logs(candidate, baseline)

    assert result.counts.boundary == 1
    assert result.counts.changed == 0
    assert result.boundary_with_reordered_body == ["q"]


def test_a_clean_boundary_swap_reports_no_reordered_body() -> None:
    """The diagnostic stays empty when only the tail moved."""
    baseline = {"q": ["a", "b", "c", "d"]}
    candidate = {"q": ["a", "b", "c", "e"]}

    result = retrieval_diff.diff_retrieval_logs(candidate, baseline)

    assert result.counts.boundary == 1
    assert result.boundary_with_reordered_body == []


def test_cli_reports_the_reordered_body_diagnostic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The diagnostic reaches the CLI output.

    One that is computed but never printed is not a signal.
    """
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps({"q": ["a", "b", "c", "d"]}), encoding="utf-8")
    candidate.write_text(json.dumps({"q": ["a", "c", "b", "e"]}), encoding="utf-8")

    retrieval_diff.main(["--candidate", str(candidate), "--baseline", str(baseline)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["boundary_with_reordered_body"] == ["q"]
