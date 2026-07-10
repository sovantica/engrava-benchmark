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
        gold: str,
        answer: str,
        *,
        question_type: str,
        question_id: str,
    ) -> bool:
        """Heuristic correctness check for the smoke path.

        Args:
            question: The question text (unused).
            gold: The gold answer.
            answer: The model answer.
            question_type: The official question type (unused here).
            question_id: The question id (unused here).

        Returns:
            ``True`` iff ``gold`` is a case-insensitive substring of ``answer``.

        """
        _ = question, question_type, question_id
        if not gold:
            return False
        return gold.strip().lower() in answer.strip().lower()
