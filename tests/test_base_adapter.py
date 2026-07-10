"""Tests for the adapter seam + the RunContext leakage guard."""

from __future__ import annotations

import dataclasses
from dataclasses import fields

import pytest

from adapters.base import CorpusTurn, MemoryAdapter, RankedItem, RunContext


def test_corpus_turn_has_no_label_field() -> None:
    """CorpusTurn must not expose any evidence/answer/label field (leakage guard)."""
    names = {f.name for f in fields(CorpusTurn)}
    forbidden = {"has_answer", "answer", "label", "is_evidence", "gold"}
    assert names.isdisjoint(forbidden)
    assert names == {"unit_id", "text", "session_id", "turn_index", "timestamp"}


def test_run_context_is_minimal_and_read_only() -> None:
    """RunContext exposes only run params — never answers/labels/reader/judge/split."""
    names = {f.name for f in fields(RunContext)}
    assert names == {"top_k", "granularity", "embedder_spec"}
    ctx = RunContext(top_k=20, granularity="turn", embedder_spec="local:x")
    # frozen dataclass — assignment must fail.
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.top_k = 5  # type: ignore[misc]


def test_runtime_checkable_protocol() -> None:
    """A class matching ingest+search is recognized as a MemoryAdapter."""

    class Dummy:
        def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
            _ = corpus, run_ctx

        def search(self, query: str, *, top_k: int) -> list[RankedItem]:
            _ = query, top_k
            return []

    assert isinstance(Dummy(), MemoryAdapter)
    assert not isinstance(object(), MemoryAdapter)
