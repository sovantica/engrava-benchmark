"""Official LongMemEval scoring — UNMODIFIED upstream, thin wrapper only.

This module does NOT re-implement scoring. It (a) exposes the official per-
question-type judge prompt used by the upstream ``evaluate_qa.py`` so the runner's
judge asks the exact official question, and (b) aggregates per-question judgments
into the schema's ``metrics`` block using the official metric semantics
(``print_qa_metrics.py``): micro overall (includes abstention items), unweighted
macro over the 6 categories, and the abstention subset as a cross-cutting group.

The unmodified official source is vendored under ``official_scorer/`` and pinned to
a known upstream commit (see ``official_scorer/UPSTREAM.md``); ``scorer_version`` in
an emitted result records that pin. The aggregation here mirrors the official
counting; it introduces no new rubric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from runners.longmemeval.run import Judgment

# The 6 official question_type strings (hyphenated; 1:1 with print_qa_metrics.py).
OFFICIAL_CATEGORIES: tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
)

# Abstention items in LongMemEval are flagged by a question_id suffix ("_abs").
ABSTENTION_QID_SUFFIX = "_abs"


def is_abstention(question_id: str) -> bool:
    """Whether a question is an abstention item (official ``_abs`` qid suffix).

    Args:
        question_id: The question id.

    Returns:
        ``True`` iff the question is an abstention item.

    """
    return question_id.endswith(ABSTENTION_QID_SUFFIX)


def official_judge_prompt(
    *, question: str, gold: str, answer: str, question_type: str, question_id: str
) -> str:
    """Return the official per-question-type judge prompt (verbatim upstream).

    Delegates to the pinned, upstream-verbatim official module so the wording is
    the upstream one; this wrapper does not author a rubric. Abstention items
    (official ``_abs`` qid suffix) select the abstention prompt.

    Args:
        question: The question text.
        gold: The gold answer (rubric/explanation for preference/abstention).
        answer: The model answer to judge.
        question_type: The official question type (selects the template).
        question_id: The question id (detects abstention items).

    Returns:
        The judge prompt string the official scorer would use.

    """
    from runners.longmemeval.official_scorer import evaluate_qa  # noqa: PLC0415

    return evaluate_qa.build_judge_prompt(
        question=question,
        answer=answer,
        gold=gold,
        question_type=question_type,
        abstention=is_abstention(question_id),
    )


class OfficialScorer:
    """Aggregate per-question judgments into the schema ``metrics`` block.

    Implements the runner's ``Scorer`` Protocol. Counting mirrors the official
    ``print_qa_metrics.py``:

    * ``overall_micro`` — correct / total over ALL questions (abstention included).
    * ``per_category`` — correct / total within each of the 6 official categories.
    * ``macro`` — unweighted mean of the per-category accuracies.
    * ``abstention`` — correct / total over the abstention subset (cross-cutting,
      overlaps categories; NOT a 7th category, NOT additive).
    """

    def __init__(self, scorer_version: str) -> None:
        """Initialize the scorer.

        Args:
            scorer_version: The pinned official-scorer identifier recorded into
                the result (e.g. ``"longmemeval@<sha>"``).

        """
        self.scorer_version = scorer_version

    def aggregate(self, judgments: Sequence[Judgment]) -> dict[str, Any]:
        """Aggregate judgments into the official ``metrics`` object.

        Args:
            judgments: Per-question judgments (id, type, correct).

        Returns:
            The ``metrics`` dict matching the results schema shape.

        """
        total = len(judgments)
        overall_correct = sum(1 for j in judgments if j.correct)

        per_category: dict[str, dict[str, Any]] = {}
        for cat in OFFICIAL_CATEGORIES:
            cat_js = [j for j in judgments if j.question_type == cat]
            n = len(cat_js)
            correct = sum(1 for j in cat_js if j.correct)
            per_category[cat] = {
                "accuracy": (correct / n) if n else 0.0,
                "n": n,
            }

        abs_js = [j for j in judgments if is_abstention(j.question_id)]
        abs_n = len(abs_js)
        abs_correct = sum(1 for j in abs_js if j.correct)

        macro = (
            sum(c["accuracy"] for c in per_category.values()) / len(OFFICIAL_CATEGORIES)
            if OFFICIAL_CATEGORIES
            else 0.0
        )

        return {
            "overall_micro": (overall_correct / total) if total else 0.0,
            "macro": macro,
            "abstention": {
                "accuracy": (abs_correct / abs_n) if abs_n else 0.0,
                "n": abs_n,
            },
            "per_category": per_category,
        }
