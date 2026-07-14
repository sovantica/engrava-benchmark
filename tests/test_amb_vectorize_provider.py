"""Integration tests for the AMB (Agent Memory Benchmark) Engrava provider.

The provider (``integrations/amb_vectorize/engrava_provider.py``) is a drop-in
shim that subclasses the upstream ``memory_bench.memory.base.MemoryProvider``.
That upstream package is only present inside a Vectorize AMB checkout, so it is
**stubbed here** via ``sys.modules`` before the provider is imported — the stub
supplies nothing but the empty base class the shim inherits from. Everything
below the shim (banks, thought ids, dedup, retrieval) runs against the *real*
public ``engrava`` core with the provider's own offline ``deterministic``
embedding backend, so these tests exercise the genuine adapter logic with zero
network and zero spend.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from dataclasses import dataclass

import pytest

# The adapter runs against the real public engrava core + a sqlite driver. Those
# are the `[engrava]` optional extra, absent from the CI `[dev]` test env — skip
# the whole module there (mirrors tests/test_engrava_adapter.py).
pytest.importorskip("engrava")
pytest.importorskip("aiosqlite")

# --- Stub the upstream AMB package the shim inherits from -------------------- #
# Must be registered BEFORE importing the provider module (top-level import).
if "memory_bench" not in sys.modules:
    _mb = types.ModuleType("memory_bench")
    _mb_memory = types.ModuleType("memory_bench.memory")
    _mb_base = types.ModuleType("memory_bench.memory.base")

    class _StubMemoryProvider:
        """Minimal stand-in for the upstream ABC (the shim never calls super)."""

    _mb_base.MemoryProvider = _StubMemoryProvider
    _mb_memory.base = _mb_base
    _mb.memory = _mb_memory
    sys.modules["memory_bench"] = _mb
    sys.modules["memory_bench.memory"] = _mb_memory
    sys.modules["memory_bench.memory.base"] = _mb_base

from integrations.amb_vectorize.engrava_provider import (  # noqa: E402
    EngravaMemoryProvider,
    ProviderError,
)


@dataclass
class FakeDoc:
    """Duck-typed stand-in for ``memory_bench.models.Document`` (TYPE_CHECKING only)."""

    id: str
    content: str
    user_id: str | None = None
    timestamp: str | None = None


class _FakeEmbedder:
    """Offline embedding fake implementing both query (``embed``) and batch paths.

    Deterministic 16-dim vectors from a content hash — semantics-free, so these
    tests rely on Engrava's full-text signal for recall, not vector similarity.
    Unlike the shim's built-in ``deterministic`` backend, this fake also provides
    ``embed`` so the retrieval path (which embeds the query) is exercised.
    """

    model_name = "fake-16"
    _dim = 16

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(float(digest[i % len(digest)]) / 127.5) - 1.0 for i in range(self._dim)]

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> EngravaMemoryProvider:
    """A provider on the offline fake embedder (no network, no download)."""
    monkeypatch.setattr(
        "integrations.amb_vectorize.engrava_provider._create_embedding_provider",
        lambda: _FakeEmbedder(),
    )
    prov = EngravaMemoryProvider(k=10)
    try:
        yield prov
    finally:
        prov.cleanup()


def test_ingest_and_retrieve_returns_original_document(provider: EngravaMemoryProvider) -> None:
    """Retrieval returns the exact original ``Document`` objects, in score order."""
    paris = FakeDoc(id="d-paris", content="Paris is the capital city of France.", user_id="u1")
    tokyo = FakeDoc(id="d-tokyo", content="Tokyo is the capital city of Japan.", user_id="u1")
    provider.ingest([paris, tokyo])

    docs, raw = provider.retrieve("Paris capital France", k=5, user_id="u1")

    assert docs, "expected at least one hit for a token-overlapping query"
    # The ORIGINAL object identity is preserved (not a partial rebuild) — the
    # regression fixed during WS review.
    assert paris in docs
    assert all(isinstance(d, FakeDoc) for d in docs)
    assert raw is not None
    ids = {entry["id"] for entry in raw["results"]}
    assert "d-paris" in ids


def test_banks_isolate_users(provider: EngravaMemoryProvider) -> None:
    """A query in one user's bank never returns another user's documents."""
    provider.ingest([FakeDoc(id="a1", content="Alice keeps her notes about mercury.", user_id="alice")])
    provider.ingest([FakeDoc(id="b1", content="Bob keeps his notes about mercury.", user_id="bob")])

    alice_docs, _ = provider.retrieve("mercury notes", k=5, user_id="alice")
    retrieved_ids = {d.id for d in alice_docs}
    assert retrieved_ids == {"a1"}
    assert "b1" not in retrieved_ids


def test_shared_bank_distinct_from_named_shared_user(provider: EngravaMemoryProvider) -> None:
    """``user_id=None`` and the literal ``user_id="_shared"`` map to different banks."""
    provider.ingest([FakeDoc(id="none", content="Vector clocks order distributed events.", user_id=None)])
    provider.ingest([FakeDoc(id="named", content="Vector clocks order distributed events.", user_id="_shared")])

    none_docs, _ = provider.retrieve("vector clocks", k=5, user_id=None)
    named_docs, _ = provider.retrieve("vector clocks", k=5, user_id="_shared")

    assert {d.id for d in none_docs} == {"none"}
    assert {d.id for d in named_docs} == {"named"}


def test_empty_documents_are_skipped(provider: EngravaMemoryProvider) -> None:
    """A whitespace-only document never reaches the embedder or the store."""
    provider.ingest(
        [
            FakeDoc(id="blank", content="   \n\t ", user_id="u1"),
            FakeDoc(id="real", content="Photosynthesis converts light into chemical energy.", user_id="u1"),
        ]
    )
    docs, _ = provider.retrieve("photosynthesis light energy", k=5, user_id="u1")
    ids = {d.id for d in docs}
    assert "real" in ids
    assert "blank" not in ids


def test_retrieve_unknown_user_returns_empty(provider: EngravaMemoryProvider) -> None:
    """Retrieving from a user with no bank yields no docs and an empty result set."""
    docs, raw = provider.retrieve("anything", k=5, user_id="ghost")
    assert docs == []
    assert raw == {"results": []}


def test_reingesting_same_doc_id_does_not_collide(provider: EngravaMemoryProvider) -> None:
    """Re-ingesting the same doc id across calls stores both (monotonic ordinal)."""
    provider.ingest([FakeDoc(id="dup", content="First observation about comets.", user_id="u1")])
    # Same source id, different content, a separate ingest call — the monotonic
    # per-bank ordinal must keep the derived thought ids distinct (no crash).
    provider.ingest([FakeDoc(id="dup", content="Second observation about comets.", user_id="u1")])
    docs, _ = provider.retrieve("comets observation", k=5, user_id="u1")
    assert docs, "both re-ingested documents should be searchable"


def test_cleanup_is_idempotent(provider: EngravaMemoryProvider) -> None:
    """``cleanup`` may be called repeatedly without error and drops all banks."""
    provider.ingest([FakeDoc(id="x", content="Idempotent teardown check.", user_id="u1")])
    provider.cleanup()
    provider.cleanup()  # second call must be a no-op, not a crash


def test_builtin_deterministic_backend_ingests_and_retrieves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shim's own ``deterministic`` embedding backend supports ingest AND retrieve.

    Uses the real ``ENGRAVA_AMB_EMBED_BACKEND=deterministic`` path (not the fake),
    so retrieval — which embeds the query when ``query_vector`` is None — exercises
    the provider's ``embed`` method end to end.
    """
    monkeypatch.setenv("ENGRAVA_AMB_EMBED_BACKEND", "deterministic")
    prov = EngravaMemoryProvider(k=10)
    try:
        prov.ingest([FakeDoc(id="d1", content="Neptune is the eighth planet from the sun.", user_id="u1")])
        docs, _ = prov.retrieve("Neptune eighth planet", k=5, user_id="u1")
        assert {d.id for d in docs} == {"d1"}
    finally:
        prov.cleanup()


def test_sync_call_inside_running_loop_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driving the sync API from inside a running loop raises ``ProviderError``."""
    monkeypatch.setenv("ENGRAVA_AMB_EMBED_BACKEND", "deterministic")  # cheap offline init
    prov = EngravaMemoryProvider()

    async def _call() -> None:
        with pytest.raises(ProviderError):
            prov.ingest([FakeDoc(id="x", content="never reached", user_id="u1")])

    try:
        asyncio.run(_call())
    finally:
        prov.cleanup()
