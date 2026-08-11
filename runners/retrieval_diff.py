r"""Free PRIMARY screen signal: diff two ranked retrieval logs.

Compares the ranked list of official corpus ids retrieved per question between a
candidate run and a stored baseline. Because engrava is a deterministic memory
layer, identical ids in identical order => a byte-identical reader prompt => any
downstream score change on that question is provably reader/judge jitter, not a
retrieval effect. This is the engrava-benchmark analogue of the LongMemEval-S
pre-publish screen (DEC-063): the free way to verify an engrava code change.

The retrieval identity here is the runner's ``RunRecord.ranked_official_ids`` -- a
plain list of official corpus id strings, best first -- so a retrieval log is a
JSON object ``{question_id: [official_corpus_id, ...]}``. No text parsing or
content hashing is needed (unlike the memory-context-blob V2 variant this is
ported from); the ids ARE the identity.

Verdict bands
-------------
    GREEN   retrieval unchanged, or marginal (rank-boundary shuffles / single
            swaps around the top-k boundary on a small fraction) -> benign, no
            reader run needed.
    YELLOW  a non-trivial fraction changed set membership -> inspect; escalate to
            a cheap paid score on the changed-retrieval subset only.
    RED     wholesale/systematic change (large fraction, non-boundary), or the two
            logs cover different question sets -> investigate before trusting the
            change.

Usage:
  python -m runners.retrieval_diff --candidate <run>/retrieval_log.json \\
      --baseline <base>/retrieval_log.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

RETRIEVAL_LOG_FILENAME = "retrieval_log.json"
"""Canonical file name a runner writes its retrieval log under (also the file a
``--baseline`` directory is expected to contain)."""

_GREEN_MATERIAL_FRACTION = 0.10
_YELLOW_CHANGED_FRACTION = 0.15
_BOUNDARY_SYMDIFF_MAX = 2
#: A boundary swap needs a shared body (the first n-1 ranks), so a list of < 2 items
#: cannot be a boundary swap: a singleton total replacement is a real membership change.
_BOUNDARY_MIN_LEN = 2


class RetrievalDiffError(ValueError):
    """Raised when a retrieval log file is missing or structurally malformed."""


class Verdict(StrEnum):
    """The overall retrieval-diff verdict band."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class Classification(StrEnum):
    """How a single question's ranked retrieval changed between two logs."""

    IDENTICAL = "identical"
    REORDERED = "reordered"
    BOUNDARY = "boundary"
    CHANGED = "changed"


@dataclass(frozen=True, slots=True)
class ClassCounts:
    """Per-classification question counts over the common question set.

    Attributes:
        identical: Same ids in the same order.
        reordered: Same id set, different order.
        boundary: A single swap at the top-k boundary — equal length, one id out and
            one in, with the ranks above the last one holding the same ids. Those ranks
            may have been reordered among themselves; see
            :attr:`DiffResult.boundary_with_reordered_body`.
        changed: A genuine change in retrieved set membership.

    """

    identical: int
    reordered: int
    boundary: int
    changed: int


@dataclass(frozen=True, slots=True)
class DiffResult:
    """The outcome of diffing a candidate retrieval log against a baseline.

    Attributes:
        questions_compared: Number of question ids present in BOTH logs.
        only_in_candidate: Sorted question ids present only in the candidate.
        only_in_baseline: Sorted question ids present only in the baseline.
        counts: Per-classification counts over the common question set.
        changed_fraction: Fraction of the UNION that changed materially (genuine
            membership changes plus disjoint-question-set mismatches).
        verdict: The overall GREEN/YELLOW/RED band.
        changed_question_ids: Sorted common question ids classified ``changed``.
        boundary_with_reordered_body: Sorted question ids classified ``boundary`` whose
            ranks above the boundary also changed order. ``boundary`` is the mildest
            non-identical class and the downstream reader is order-sensitive, so a
            reshuffled body must not disappear into it.

    """

    questions_compared: int
    only_in_candidate: list[str]
    only_in_baseline: list[str]
    counts: ClassCounts
    changed_fraction: float
    verdict: Verdict
    changed_question_ids: list[str]
    boundary_with_reordered_body: list[str]


def classify(candidate: list[str], baseline: list[str]) -> Classification:
    """Classify how one question's ranked retrieval changed.

    Args:
        candidate: The candidate run's ranked official ids (best first).
        baseline: The baseline run's ranked official ids (best first).

    Returns:
        The :class:`Classification` for this question.

    """
    if candidate == baseline:
        return Classification.IDENTICAL
    if set(candidate) == set(baseline):
        return Classification.REORDERED
    # A rank-boundary swap keeps the same length (one item out, one in) and shares a
    # non-empty body (the first n-1 ranks hold the same ids). The body's ORDER is not
    # required to match: that is a separate axis, reported alongside rather than folded
    # into the membership classification.
    # A length change, or a singleton total replacement like ["a"] vs ["b"] (empty
    # shared body), is a real membership change -> require equal length >= 2.
    if (
        len(candidate) == len(baseline)
        and len(candidate) >= _BOUNDARY_MIN_LEN
        and len(set(candidate) ^ set(baseline)) <= _BOUNDARY_SYMDIFF_MAX
        and set(candidate[:-1]) == set(baseline[:-1])
    ):
        return Classification.BOUNDARY
    return Classification.CHANGED


def diff_retrieval_logs(
    candidate: dict[str, list[str]], baseline: dict[str, list[str]]
) -> DiffResult:
    """Diff a candidate retrieval log against a baseline and assign a verdict.

    Args:
        candidate: Mapping ``{question_id: [official_corpus_id, ...]}`` for the run
            under test.
        baseline: The same mapping for the stored baseline.

    Returns:
        The :class:`DiffResult` with per-question classification and the overall
        verdict.

    """
    common = sorted(set(candidate) & set(baseline))
    only_candidate = sorted(set(candidate) - set(baseline))
    only_baseline = sorted(set(baseline) - set(candidate))

    tally = dict.fromkeys(Classification, 0)
    changed_ids: list[str] = []
    boundary_reordered: list[str] = []
    for qid in common:
        cls = classify(candidate[qid], baseline[qid])
        tally[cls] += 1
        if cls is Classification.CHANGED:
            changed_ids.append(qid)
        elif cls is Classification.BOUNDARY and candidate[qid][:-1] != baseline[qid][:-1]:
            boundary_reordered.append(qid)
    counts = ClassCounts(
        identical=tally[Classification.IDENTICAL],
        reordered=tally[Classification.REORDERED],
        boundary=tally[Classification.BOUNDARY],
        changed=tally[Classification.CHANGED],
    )

    # A qid present on only one side is a material mismatch (different question
    # sets), counted over the UNION so a disjoint comparison can never look mild.
    # Such qids escalate directly.
    missing = len(only_candidate) + len(only_baseline)
    total = len(common) + missing or 1
    changed_like = counts.changed + missing
    changed_fraction = changed_like / total
    material = counts.boundary + changed_like
    if missing == 0 and counts.changed == 0 and material / total <= _GREEN_MATERIAL_FRACTION:
        verdict = Verdict.GREEN
    elif missing == 0 and changed_fraction <= _YELLOW_CHANGED_FRACTION:
        verdict = Verdict.YELLOW
    else:
        verdict = Verdict.RED

    return DiffResult(
        questions_compared=len(common),
        only_in_candidate=only_candidate,
        only_in_baseline=only_baseline,
        counts=counts,
        changed_fraction=round(changed_fraction, 4),
        verdict=verdict,
        changed_question_ids=changed_ids,
        boundary_with_reordered_body=boundary_reordered,
    )


def resolve_log_path(path: Path) -> Path:
    """Resolve a retrieval-log path, accepting either a file or its directory.

    Args:
        path: A retrieval log file, or a directory containing
            :data:`RETRIEVAL_LOG_FILENAME`.

    Returns:
        The concrete file path to load.

    """
    return path / RETRIEVAL_LOG_FILENAME if path.is_dir() else path


def load_retrieval_log(path: Path) -> dict[str, list[str]]:
    """Load and validate a retrieval log JSON file.

    Args:
        path: A retrieval log file, or a directory containing
            :data:`RETRIEVAL_LOG_FILENAME`.

    Returns:
        The mapping ``{question_id: [official_corpus_id, ...]}``.

    Raises:
        RetrievalDiffError: If the file is missing or not a mapping of string
            question ids to lists of string ids.

    """
    resolved = resolve_log_path(path)
    if not resolved.is_file():
        msg = f"retrieval log not found: {resolved}"
        raise RetrievalDiffError(msg)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"retrieval log is not valid JSON: {resolved}"
        raise RetrievalDiffError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"retrieval log must be a JSON object, got {type(raw).__name__}: {resolved}"
        raise RetrievalDiffError(msg)
    log: dict[str, list[str]] = {}
    for qid, ids in raw.items():
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            msg = f"retrieval log entry {qid!r} must be a list of id strings: {resolved}"
            raise RetrievalDiffError(msg)
        log[str(qid)] = list(ids)
    return log


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diff two ranked retrieval logs and print a GREEN/YELLOW/RED verdict."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--changed-out",
        type=Path,
        default=None,
        help="Write the changed-retrieval question ids here (one per line).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI: diff two retrieval logs, print the report, exit non-zero on RED.

    Args:
        argv: Optional argument vector (for testing).

    Returns:
        ``1`` if the verdict is RED, else ``0``.

    """
    args = _parse_args(argv)
    candidate = load_retrieval_log(args.candidate)
    baseline = load_retrieval_log(args.baseline)
    result = diff_retrieval_logs(candidate, baseline)

    print(  # noqa: T201 - CLI report
        json.dumps(
            {
                "questions_compared": result.questions_compared,
                "only_in_candidate": result.only_in_candidate,
                "only_in_baseline": result.only_in_baseline,
                "counts": {
                    "identical": result.counts.identical,
                    "reordered": result.counts.reordered,
                    "boundary": result.counts.boundary,
                    "changed": result.counts.changed,
                },
                "changed_fraction": result.changed_fraction,
                "verdict": result.verdict.value,
                "changed_question_ids": result.changed_question_ids,
                "boundary_with_reordered_body": result.boundary_with_reordered_body,
            },
            indent=2,
        )
    )
    if args.changed_out is not None:
        suffix = "\n" if result.changed_question_ids else ""
        args.changed_out.write_text(
            "\n".join(result.changed_question_ids) + suffix, encoding="utf-8"
        )
    return 1 if result.verdict is Verdict.RED else 0


if __name__ == "__main__":
    raise SystemExit(main())
