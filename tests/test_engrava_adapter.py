"""Tests for the Engrava adapter that require the optional ``engrava`` package.

Skipped automatically where ``engrava`` is not installed (e.g. the dependency-free
CI leg), so the core suite stays installable with ``[dev]`` only.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("engrava")

from adapters.base import CorpusTurn, RunContext
from adapters.engrava_adapter import (
    AdapterError,
    EngravaAdapter,
    create_embedding_provider,
)


async def _trivial() -> int:
    return 42


def test_run_sync_outside_loop() -> None:
    adapter = EngravaAdapter(object())  # type: ignore[arg-type]
    assert adapter._run_sync(_trivial()) == 42
    adapter.close()


async def test_run_sync_inside_running_loop_raises() -> None:
    # Driving the synchronous adapter API from within a running event loop must
    # raise a clear, typed error rather than the opaque native RuntimeError.
    assert asyncio.get_running_loop() is not None
    adapter = EngravaAdapter(object())  # type: ignore[arg-type]
    with pytest.raises(AdapterError, match="active event loop"):
        adapter._run_sync(_trivial())


def test_unknown_embedding_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        create_embedding_provider("nope:some-model")


def test_search_before_ingest_raises() -> None:
    """search() before ingest() raises a clear error (no real model needed)."""
    adapter = EngravaAdapter(object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="ingest"):
        adapter.search("anything", top_k=1)


def test_close_is_idempotent_and_safe_without_loop() -> None:
    """close() never raises when the loop was never created and is idempotent."""
    adapter = EngravaAdapter(object())  # type: ignore[arg-type]
    adapter.close()
    adapter.close()  # second call is a no-op, must not raise


class _RaisingStore:
    """A store stub whose async close raises, to exercise close() error handling."""

    async def close(self) -> None:
        msg = "store cleanup boom"
        raise RuntimeError(msg)


def test_close_releases_refs_even_when_cleanup_raises() -> None:
    """close() must always release refs (loop, store, conn), then propagate.

    If the awaited store/connection cleanup raises, the reference release still
    runs (via try/finally) and the loop is left closed; the cleanup error then
    surfaces to the caller of the explicit close().
    """
    adapter = EngravaAdapter(object())  # type: ignore[arg-type]
    # Force a live loop, a store whose close() raises, and a non-None conn so the
    # post-close `is None` assertions actually prove each ref was released.
    adapter._loop = asyncio.new_event_loop()
    adapter._store = _RaisingStore()  # type: ignore[assignment]
    adapter._conn = object()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="store cleanup boom"):
        adapter.close()

    # Refs released and loop closed despite the cleanup error.
    assert adapter._loop is None
    assert adapter._store is None
    assert adapter._conn is None


def test_close_does_not_raise_when_loop_is_running() -> None:
    """close() must not RuntimeError when its loop is running (never closes it)."""
    loop = asyncio.new_event_loop()
    adapter = EngravaAdapter(object())  # type: ignore[arg-type]
    adapter._loop = loop
    adapter._conn = object()  # type: ignore[assignment] - non-None so release is proven

    async def _drive_close() -> None:
        # Called on the running loop; close() must skip closing it and only drop
        # references rather than raising "Cannot close a running event loop".
        adapter.close()

    try:
        loop.run_until_complete(_drive_close())
    finally:
        loop.close()

    assert adapter._loop is None
    assert adapter._store is None
    assert adapter._conn is None
    # The running loop was NOT closed by close(); we closed it ourselves above.


class _LoopBoundAsyncProvider:
    """A deterministic async embedding provider that binds to its first loop.

    On the first async call it records the running event loop; every later async
    call asserts it runs on that same loop. This reproduces the failure mode of a
    real async provider (e.g. engrava's OpenAI provider) that caches a loop-bound
    ``httpx.AsyncClient`` on first use: if ``ingest`` and ``search`` run on
    different loops the second call breaks.
    """

    model_name = "stub-async-embedder"

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._bound_loop is None:
            self._bound_loop = loop
        elif loop is not self._bound_loop:
            msg = "async provider used across two different event loops"
            raise RuntimeError(msg)

    def _vector(self, text: str) -> list[float]:
        # Deterministic pseudo-vector derived from the text length + first bytes.
        base = float(len(text) % 7 + 1)
        return [base + i for i in range(self._dim)]

    async def embed(self, text: str) -> list[float]:
        self._check_loop()
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._check_loop()
        return [self._vector(t) for t in texts]


def test_async_provider_survives_ingest_then_search() -> None:
    """Regression: ingest() then search() must run on ONE shared event loop.

    A loop-bound async provider breaks if ingest and search each spin up (and
    close) their own loop. This drives the full sync ingest -> search path and
    fails on the pre-fix per-call ``asyncio.run`` code; it passes once the adapter
    reuses a single persistent loop.
    """
    provider = _LoopBoundAsyncProvider()
    adapter = EngravaAdapter(provider)  # type: ignore[arg-type]
    corpus = [
        CorpusTurn(
            unit_id="u1",
            text="the sky is blue today",
            session_id="s1",
            turn_index=0,
            timestamp="2026-01-01",
        ),
        CorpusTurn(
            unit_id="u2",
            text="i had coffee this morning",
            session_id="s1",
            turn_index=1,
            timestamp="2026-01-01",
        ),
    ]
    run_ctx = RunContext(top_k=2, granularity="turn", embedder_spec="stub")

    adapter.ingest(corpus, run_ctx=run_ctx)
    ranked = adapter.search("what colour is the sky", top_k=2)

    assert all(item.unit_id in {"u1", "u2"} for item in ranked)
    adapter.close()


def test_duplicate_session_haystack_ingests_both_ranks_unique(tmp_path: Path) -> None:
    """A byte-identical duplicate session ingests BOTH copies and ranks unique.

    Drives the full runner path on a haystack with a repeated (byte-identical)
    session: ``load_questions`` ingests BOTH copies (duplicate corpus unit_id by
    design, mirroring the official flat-list), the adapter stores both via its
    engrava-internal ``#dup`` store-id salt, and rank-collapse yields UNIQUE ranked
    unit_ids. ``assemble_context`` then resolves the ranked unit_id to the
    FIRST-occurrence session date (not the dup's later date), offline.
    """
    from runners.longmemeval import run as runner  # noqa: PLC0415

    same_session = [
        {"role": "user", "content": "the meeting is on tuesday afternoon"},
        {"role": "assistant", "content": "noted, tuesday afternoon"},
    ]
    item = {
        "question_id": "q_dup_e2e",
        "question_type": "multi-session",
        "question": "when is the meeting",
        "answer": "tuesday afternoon",
        "haystack_dates": ["2026-01-01", "2026-01-02"],  # different dates
        "haystack_session_ids": ["s_a", "s_b"],  # distinct ids, identical content
        "haystack_sessions": [list(same_session), list(same_session)],
    }
    path = tmp_path / "dup.json"
    path.write_text(json.dumps([item]), encoding="utf-8")

    questions = runner.load_questions(path)
    assert len(questions) == 1
    q = questions[0]
    # Ingest-both: the corpus carries both duplicate copies.
    assert len(q.corpus) == 2
    assert len({t.unit_id for t in q.corpus}) == 1

    provider = _LoopBoundAsyncProvider()
    adapter = EngravaAdapter(provider)  # type: ignore[arg-type]
    run_ctx = RunContext(top_k=5, granularity="turn", embedder_spec="stub")

    adapter.ingest(q.corpus, run_ctx=run_ctx)
    ranked = adapter.search(q.question, top_k=5)
    adapter.close()

    ranked_ids = [item.unit_id for item in ranked]
    assert ranked_ids  # at least one hit
    assert len(ranked_ids) == len(set(ranked_ids))  # rank-collapse: no dup key
    assert all(rid in q.id_map for rid in ranked_ids)

    # assemble_context resolves the ranked unit_id to the FIRST-occurrence date.
    context = runner.assemble_context(ranked_ids, q, top_k=5)
    assert "2026-01-01" in context
    assert "2026-01-02" not in context


class _LimitAssertingProvider(_LoopBoundAsyncProvider):
    """Stub embedder asserting every input is non-empty and within the token cap.

    Reproduces the OpenAI embeddings API contract offline: an empty input or one
    over 8192 cl100k tokens would 400. If the adapter ever passes such an input the
    assertion fails, proving the truncation + empty-skip did NOT happen — no paid
    call required.
    """

    model_name = "stub-limit-embedder"

    def _assert_ok(self, texts: list[str]) -> None:
        import tiktoken  # noqa: PLC0415

        enc = tiktoken.get_encoding("cl100k_base")
        for text in texts:
            assert text.strip(), "empty embed input reached the provider"
            assert len(enc.encode(text)) <= 8192, "embed input exceeds 8192 tokens"

    async def embed(self, text: str) -> list[float]:
        self._assert_ok([text])
        return await super().embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._assert_ok(texts)
        return await super().embed_batch(texts)


def test_long_turn_truncated_and_empty_turn_skipped() -> None:
    """A >8192-token turn is truncated + retrievable; an empty turn is skipped.

    The stub provider asserts every embed input is non-empty and <=8192 cl100k
    tokens, so a pass proves the adapter truncated the long payload and never
    embedded the empty turn. The long turn must still be ingested and retrievable.
    """
    # A user turn far over the 8192-token cap (word repetition → ~1 token/word).
    long_text = "alpha " * 20000
    corpus = [
        CorpusTurn(
            unit_id="u_long",
            text=long_text,
            session_id="s1",
            turn_index=0,
            timestamp="2026-01-01",
        ),
        CorpusTurn(
            unit_id="u_empty",
            text="   ",  # stripped-empty → must be skipped, never embedded
            session_id="s1",
            turn_index=1,
            timestamp="2026-01-01",
        ),
        CorpusTurn(
            unit_id="u_short",
            text="a normal short turn about coffee",
            session_id="s1",
            turn_index=2,
            timestamp="2026-01-01",
        ),
    ]
    provider = _LimitAssertingProvider()
    adapter = EngravaAdapter(provider)  # type: ignore[arg-type]
    run_ctx = RunContext(top_k=5, granularity="turn", embedder_spec="stub")

    adapter.ingest(corpus, run_ctx=run_ctx)  # asserts fire inside the provider
    ranked = adapter.search("alpha", top_k=5)
    adapter.close()

    ranked_ids = {item.unit_id for item in ranked}
    # The long turn was ingested (truncated) and is retrievable; the empty turn is
    # never stored, so it can never be retrieved.
    assert "u_long" in ranked_ids
    assert "u_empty" not in ranked_ids
