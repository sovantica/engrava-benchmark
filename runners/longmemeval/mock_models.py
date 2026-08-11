"""Free, offline mock reader + judge for the local smoke path (NO spend).

These implement the runner's ``Reader`` / ``Judge`` Protocol seams without any
network call or paid API. They exist solely to exercise the full pipeline
(ingest -> search -> assemble -> reader -> judge -> score -> emit -> validate)
end-to-end in tests and a local dry run, so the wiring can be verified before the
benchmark maintainers spend money on the canonical run.

They are deliberately trivial and make NO correctness claim — a result produced
with these mock models is a smoke artifact, never a published number.
"""

from __future__ import annotations

import re

GoldAnswer = str | int | float
"""A LongMemEval gold answer: text, or a number for numeric-answer questions.

The dataset stores the gold as JSON, so a counting/duration answer arrives as a
JSON number rather than a string. Offline judges accept both.
"""


class MockReader:
    """Offline reader: echoes a deterministic answer derived from the context.

    Implements the runner's ``Reader`` Protocol. No network, no spend.
    """

    def answer(self, question: str, context: str, *, question_date: str = "") -> str:
        """Return a deterministic pseudo-answer derived from the assembled context.

        Echoes the assembled history verbatim (or ``"I don't know"`` when empty), so
        the offline smoke path is fully deterministic and any answer-bearing content
        present in the retrieved context is reflected — no network, no spend, no
        correctness claim.

        Args:
            question: The question text (unused beyond determinism).
            context: The official assembled history string.
            question_date: The question's date (unused by the mock).

        Returns:
            The assembled context, or ``"I don't know"`` if it is empty.

        """
        _ = question, question_date
        return context.strip() or "I don't know"


class MockJudge:
    """Offline judge: substring-match heuristic (NO correctness claim).

    Implements the runner's ``Judge`` Protocol. Marks an answer "correct" iff the
    gold answer appears (case-insensitively) in the model answer. Deterministic,
    free, offline — for smoke-testing the pipeline only.
    """

    def score(
        self,
        question: str,
        gold: GoldAnswer,
        answer: str,
        *,
        question_type: str,
        question_id: str,
    ) -> bool:
        """Heuristic correctness check for the smoke path.

        A **textual** gold is matched as a substring, unchanged. A **numeric** gold is
        matched on digit boundaries instead: rendering it with :func:`str` and asking
        for a substring would accept any number that merely contains its digits, so a
        gold of ``3`` would be satisfied by ``13``, ``0.3`` and ``2024``. On this
        dataset that is precisely the counting and duration population a numeric gold
        exists for, so the loose rule would report those questions as passing without
        the reader having answered them.

        Args:
            question: The question text (unused).
            gold: The gold answer (text, or a number for numeric-answer questions).
            answer: The model answer.
            question_type: The official question type (unused here).
            question_id: The question id (unused here).

        Returns:
            ``True`` iff ``gold`` is a case-insensitive substring of ``answer``.

        """
        _ = question, question_type, question_id
        if isinstance(gold, str):
            gold_text = gold.strip().lower()
            return bool(gold_text) and gold_text in answer.strip().lower()
        # Digit boundaries, not string boundaries: `\b` sits between a digit and a
        # letter, so `\b3\b` would still match "3rd", and a decimal point is not a
        # word character, so it would match the "3" in "0.3".
        pattern = rf"(?<![\d.]){re.escape(str(gold))}(?![\d.])"
        return re.search(pattern, answer) is not None


class NullJudge:
    """Judge seam for a mode whose reader answer is discarded (no judgment at all).

    Implements the runner's ``Judge`` Protocol without judging: it never inspects
    the gold or the answer and always reports ``False``. The ``retrieval`` mode
    measures the ranked retrieval only and defines the reader output as discarded,
    so it wires this in place of a real (or mock) judge — the run then neither pays
    for nor depends on a verdict it throws away. A run judged this way carries no
    correctness signal whatsoever and can never emit an official row.
    """

    def score(
        self,
        question: str,
        gold: GoldAnswer,
        answer: str,
        *,
        question_type: str,
        question_id: str,
    ) -> bool:
        """Return ``False`` without judging anything.

        Args:
            question: The question text (unused).
            gold: The gold answer (unused — never inspected).
            answer: The model answer (unused — the mode discards it).
            question_type: The official question type (unused).
            question_id: The question id (unused).

        Returns:
            ``False``, always. This is not a correctness claim; the mode that uses
            this judge reports no correctness at all.

        """
        _ = question, gold, answer, question_type, question_id
        return False
