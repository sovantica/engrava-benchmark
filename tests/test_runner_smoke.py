"""Free end-to-end smoke test for the LongMemEval runner (NO network, NO spend).

Exercises the FULL pipeline — ingest -> search -> assemble -> reader -> judge ->
score -> emit -> validate — with:

- a tiny in-test memory adapter (no ``engrava`` dependency),
- the offline mock reader + judge (no API, no spend),
- the real official-scorer aggregation + the real result emission + the real
  schema validation.

This proves the wiring works end-to-end before the maintainers spend money on the
canonical OpenAI-direct run. The numbers it produces are smoke artifacts, never a
published result.
"""

from __future__ import annotations

import json
from pathlib import Path

from adapters.base import CorpusTurn, RankedItem, RunContext
from runners.longmemeval import run as runner
from runners.longmemeval.mock_models import MockJudge, MockReader
from runners.longmemeval.official_scorer.evaluate_qa import get_anscheck_prompt
from runners.longmemeval.scorer import OFFICIAL_CATEGORIES, OfficialScorer, is_abstention

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "longmemeval_smoke.json"
CONFIG = Path(__file__).resolve().parents[1] / "runners" / "longmemeval" / "config" / "default.json"


class _KeywordAdapter:
    """Trivial in-test memory adapter (substring scorer). No engrava, no network."""

    def __init__(self) -> None:
        self._corpus: list[CorpusTurn] = []

    def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
        _ = run_ctx
        self._corpus = list(corpus)

    def search(self, query: str, *, top_k: int) -> list[RankedItem]:
        terms = {w.lower() for w in query.split()}
        scored: list[RankedItem] = []
        for turn in self._corpus:
            overlap = sum(1 for w in turn.text.lower().split() if w in terms)
            scored.append(RankedItem(unit_id=turn.unit_id, score=float(overlap)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def test_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    config = runner.load_config(CONFIG)
    questions = runner.load_questions(FIXTURE)
    assert len(questions) == 2

    metrics = runner.run_and_emit(
        config=config,
        questions=questions,
        adapter=_KeywordAdapter(),
        reader=MockReader(),
        judge=MockJudge(),
        result_id="smoke_test_row",
        date="2026-06-29",
        partial=True,
        emit_result=True,
        results_dir=tmp_path,
    )

    # Metrics shape matches the official semantics + schema.
    assert set(metrics) == {"overall_micro", "macro", "abstention", "per_category"}
    assert set(metrics["per_category"]) == set(OFFICIAL_CATEGORIES)
    assert metrics["abstention"]["n"] == 1  # smoke_q2_abs is the one abstention item

    # The emitted file exists at its partitioned path + is schema/layout-valid
    # (write_and_validate raises otherwise).
    emitted = tmp_path / "longmemeval-s" / "engrava" / "smoke_test_row.json"
    assert emitted.exists()
    row = json.loads(emitted.read_text())
    assert row["provenance"] == "sovantica-run"
    assert row["verification_status"] == "unverified"
    assert row["partial"] is True
    assert row["reader_endpoint"] == "api.openai.com"  # D9 default in config
    assert row["judge_snapshot"] == "gpt-4o-2024-08-06"


def test_abstention_detection() -> None:
    assert is_abstention("smoke_q2_abs")
    assert not is_abstention("smoke_q1")


def test_official_judge_prompt_selects_abstention_template() -> None:
    abs_prompt = get_anscheck_prompt(
        "multi-session", "Q?", "explanation", "I don't know", abstention=True
    )
    assert "unanswerable" in abs_prompt
    normal = get_anscheck_prompt("multi-session", "Q?", "gold", "resp", abstention=False)
    assert "Correct Answer" in normal
    assert "unanswerable" not in normal


def test_temporal_prompt_has_offbyone_clause() -> None:
    prompt = get_anscheck_prompt("temporal-reasoning", "Q?", "18", "19", abstention=False)
    assert "off-by-one" in prompt


def test_preference_prompt_uses_rubric() -> None:
    prompt = get_anscheck_prompt(
        "single-session-preference", "Q?", "rubric", "resp", abstention=False
    )
    assert "Rubric" in prompt


def test_unknown_question_type_rejected() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="unknown official question_type"):
        get_anscheck_prompt("not-a-type", "Q?", "g", "r", abstention=False)


def test_mock_reader_abstains_on_empty_context() -> None:
    assert MockReader().answer("q", "") == "I don't know"


def test_mock_judge_handles_missing_gold() -> None:
    judge = MockJudge()
    assert judge.score("q", "", "anything", question_type="t", question_id="i") is False
    assert judge.score("q", "Porto", "I live in Porto", question_type="t", question_id="i")


def test_scorer_empty_judgments() -> None:
    metrics = OfficialScorer(scorer_version="x").aggregate([])
    assert metrics["overall_micro"] == 0.0
    assert metrics["abstention"] == {"accuracy": 0.0, "n": 0}
    assert all(c["n"] == 0 for c in metrics["per_category"].values())
