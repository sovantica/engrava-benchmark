"""Public-``engrava`` memory adapter for the benchmark.

This adapter plugs the open-source ``engrava`` package into the uniform runner by
implementing :class:`adapters.base.MemoryAdapter` (``ingest`` + ``search``). It
uses **only** the public ``engrava`` pip package — no private
package is imported anywhere in this module — so any number produced through it is
reproducible by a third party who installs ``engrava`` from PyPI.

Retrieval parity
----------------
The ingest + retrieve path here mirrors the parity-proven public emitter that
backs Engrava's published LongMemEval numbers: a plain ``SqliteEngravaCore`` with
``auto_embed=False`` and ``journal_enabled=False`` and **no** ``search_config``
(so ``search_hybrid`` falls back to engrava's default fusion weights), one thought
per user turn, byte-identical embed payloads via :func:`_build_embed_input`, an
explicit ``store_embedding`` per unit, and a single ``search_hybrid`` call with
all fusion weights at their defaults (no ``current_cycle`` → recency inactive,
reflections excluded). The helpers replicated below are pure,
configuration-mirroring utilities, not product internals.

The store is rebuilt fresh per ``ingest`` call (one haystack per question), which
matches the per-question isolation the official LongMemEval flat-index runner
assumes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any, TypeVar

# --- PUBLIC engrava only — no private-package import in this module --
from engrava import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    ThoughtVisibility,
)

from adapters.base import CorpusTurn, RankedItem, RunContext

if TYPE_CHECKING:
    from collections.abc import Coroutine

    import aiosqlite
    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol

logger = logging.getLogger(__name__)

# Pure config-mirror constants (replicated from the parity emitter).
_ESSENCE_LIMIT = 180
_EMBED_BATCH_SIZE = 2048

# The text-embedding-3 family accepts at most 8192 tokens per input (cl100k_base
# encoding). A longer input 400s at the API; the official protocol likewise caps
# to the model's max, so truncating the first 8192 tokens is faithful.
_MAX_EMBED_TOKENS = 8192
_EMBED_ENCODING = "cl100k_base"

_T = TypeVar("_T")


def _truncate_embed_input(text: str) -> str:
    """Truncate an embed payload to at most ``_MAX_EMBED_TOKENS`` cl100k tokens.

    Truncation is by TOKEN (not character) using ``tiktoken``'s ``cl100k_base``
    encoding — the encoding the ``text-embedding-3`` family uses — so the kept
    prefix is exactly what the model would accept. It is deterministic (always the
    first ``_MAX_EMBED_TOKENS`` tokens) → a reproducible vector. Inputs already
    within the limit are returned unchanged (no re-encoding artefacts).

    Args:
        text: The final embed payload (after :func:`_build_embed_input`).

    Returns:
        ``text`` unchanged if within the token limit, else its first
        ``_MAX_EMBED_TOKENS`` tokens decoded back to text.

    """
    import tiktoken  # noqa: PLC0415 - lazy: only needed when ingesting

    enc = tiktoken.get_encoding(_EMBED_ENCODING)
    tokens = enc.encode(text)
    if len(tokens) <= _MAX_EMBED_TOKENS:
        return text
    return enc.decode(tokens[:_MAX_EMBED_TOKENS])


class AdapterError(RuntimeError):
    """Raised on adapter misuse (e.g. driving the sync API inside a running loop)."""


def _build_embed_input(essence: str, content: str) -> str:
    """Build the text payload to embed for a thought, mirroring engrava core.

    Byte-for-byte mirror of engrava core's own (private, ``_``-prefixed) rule:
    when the stripped ``essence`` is a leading prefix of the stripped ``content``
    the essence carries no new information, so ``content`` is embedded alone;
    otherwise the newline-joined payload is used. Reproducing it keeps the stored
    vector identical to an ``auto_embed=True`` store.

    Args:
        essence: The thought's essence (short summary).
        content: The thought's full content.

    Returns:
        The exact text payload that engrava would embed.

    """
    if content.strip().startswith(essence.strip()):
        return content
    return f"{essence}\n{content}"


def _make_id(session_id: str, turn_idx: int) -> str:
    """Derive a deterministic thought id from session + turn index.

    Args:
        session_id: The session identifier.
        turn_idx: The turn index within the session.

    Returns:
        The full hex digest of ``sha256(f"{session_id}:turn:{turn_idx}")``. The id
        is an opaque store key, so the full digest is used — no truncation, no
        birthday-collision concern.

    """
    raw = f"{session_id}:turn:{turn_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _select_local_embedding_device() -> str:
    """Pick the compute device for a local SentenceTransformer model.

    CUDA when available, else CPU; overridable via ``ENGRAVA_BENCH_EMBED_DEVICE``.

    Returns:
        The device string (``"cuda"`` / ``"cpu"`` / the override value).

    """
    override = os.environ.get("ENGRAVA_BENCH_EMBED_DEVICE", "").strip()
    if override:
        return override
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def create_embedding_provider(spec: str) -> EmbeddingProviderProtocol:
    """Create a public ``engrava`` embedding provider from a spec string.

    Every provider class is imported from the PUBLIC ``engrava.embeddings``
    namespace. Supported specs:

    * ``"local:all-MiniLM-L12-v2"``       -> SentenceTransformerProvider ($0)
    * ``"openai:text-embedding-3-small"`` -> OpenAICompatibleProvider
    * ``"ollama:nomic-embed-text"``       -> OllamaProvider

    Args:
        spec: The embedding spec string (``"backend:model"``).

    Returns:
        A public engrava embedding provider.

    Raises:
        ValueError: If the backend is not ``local`` / ``openai`` / ``ollama``.

    """
    backend, _, model = spec.partition(":")
    if not model:
        model = "all-MiniLM-L12-v2"

    if backend == "local":
        from engrava.embeddings.sentence_transformer import (  # noqa: PLC0415
            SentenceTransformerProvider,
        )

        device = _select_local_embedding_device()
        logger.info("Local embedding provider device: %s", device)
        return SentenceTransformerProvider(model_name=model, device=device)

    if backend == "openai":
        from engrava.embeddings.openai_compatible import (  # noqa: PLC0415
            OpenAICompatibleProvider,
        )

        return OpenAICompatibleProvider(
            model_name=model,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    if backend == "ollama":
        from engrava.embeddings.ollama import OllamaProvider  # noqa: PLC0415

        return OllamaProvider(model_name=model)

    msg = f"Unknown embedding backend: {backend!r}. Use local, openai, or ollama."
    raise ValueError(msg)


class EngravaAdapter:
    """Public-``engrava`` adapter implementing :class:`adapters.base.MemoryAdapter`.

    Each :meth:`ingest` builds a fresh in-memory engrava store for one question's
    haystack; :meth:`search` runs a single ``search_hybrid`` with engrava's
    default fusion weights. The retrieval mirrors the parity-proven public path.

    Single-threaded contract: the synchronous ``ingest`` / ``search`` / ``close``
    methods share one persistent event loop and are NOT safe to call concurrently
    from multiple threads; drive one adapter from a single thread.
    """

    adapter_name = "engrava_adapter"

    def __init__(self, embedding_provider: EmbeddingProviderProtocol) -> None:
        """Initialize the adapter with a public engrava embedding provider.

        Args:
            embedding_provider: A provider from :func:`create_embedding_provider`.

        """
        self._provider = embedding_provider
        self._store: SqliteEngravaCore | None = None
        self._conn: aiosqlite.Connection | None = None
        # thought_id -> unit_id, so a ranked hit maps back to the corpus unit.
        self._id_to_unit: dict[str, str] = {}
        # One persistent event loop for this adapter's whole lifetime. Created
        # lazily on the first sync call so ingest, search and close all run on the
        # same loop. Async engrava providers (e.g. the OpenAI provider) cache an
        # ``httpx.AsyncClient`` pinned to the loop of their first call; a fresh
        # ``asyncio.run`` per call would close that loop and invalidate the cached
        # client on the next call. A single shared loop keeps the client valid.
        self._loop: asyncio.AbstractEventLoop | None = None

    def _run_sync(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run a coroutine to completion on the adapter's persistent event loop.

        The ``MemoryAdapter`` interface is synchronous, so this adapter drives its
        async engrava calls on a single, lazily-created event loop held for the
        adapter's whole lifetime (:attr:`_loop`). Reusing one loop keeps any
        loop-bound resource an async provider caches (e.g. an ``httpx.AsyncClient``)
        valid across successive ``ingest``/``search`` calls.

        Running this from inside an already-running event loop is unsupported; we
        detect that and raise a clear, typed error instead of the opaque native
        ``RuntimeError``.

        Args:
            coro: The coroutine to run.

        Returns:
            The coroutine's result.

        Raises:
            AdapterError: If called while an event loop is already running. In that
                case call the underlying ``_ingest_async`` / ``_search_async``
                coroutines directly and await them on the running loop.

        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            msg = (
                "EngravaAdapter's synchronous ingest/search cannot run inside an "
                "active event loop; await _ingest_async/_search_async on the "
                "running loop instead."
            )
            raise AdapterError(msg)
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    # -- MemoryAdapter interface ------------------------------------------------

    def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
        """Ingest one question's haystack into a fresh engrava store.

        Args:
            corpus: The question's haystack turns.
            run_ctx: Read-only run parameters (unused beyond the embedder, which
                is injected at construction; present for interface conformance).

        Raises:
            AdapterError: If called from inside a running event loop.

        """
        _ = run_ctx
        self._run_sync(self._ingest_async(corpus))

    def search(self, query: str, *, top_k: int) -> list[RankedItem]:
        """Retrieve the top-``top_k`` units for ``query`` via ``search_hybrid``.

        Args:
            query: The question text.
            top_k: Maximum number of ranked units to return.

        Returns:
            Up to ``top_k`` :class:`RankedItem` in descending score order.

        Raises:
            AdapterError: If called from inside a running event loop.

        """
        return self._run_sync(self._search_async(query, top_k))

    # -- internals --------------------------------------------------------------

    async def _create_store(self) -> None:
        """Create a fresh in-memory public engrava store matching the parity path.

        Plain Free ``SqliteEngravaCore`` with ``auto_embed=False``,
        ``journal_enabled=False`` and NO ``search_config`` (so ``search_hybrid``
        uses default fusion weights), numpy vector backend, schema ensured.
        """
        import aiosqlite  # noqa: PLC0415

        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA synchronous = NORMAL")

        store = SqliteEngravaCore(
            conn,
            embedding_provider=self._provider,
            auto_embed=False,
            journal_enabled=False,
        )
        await store.ensure_schema()
        self._store, self._conn = store, conn

    async def _ingest_async(self, corpus: list[CorpusTurn]) -> None:
        """Ingest the corpus: one thought per non-empty unit, explicit embeddings.

        Empty (stripped-empty) user turns are skipped — nothing is embedded or
        stored for them (the runner still owns the corpus/id_map slot), so an empty
        input never reaches the embedding API. Each stored payload is truncated to
        the model's token cap via :func:`_truncate_embed_input`.
        """
        await self._close()
        await self._create_store()
        store = self._store
        if store is None:  # pragma: no cover - defensive
            msg = "store not initialized"
            raise RuntimeError(msg)

        model_name = self._provider.model_name
        self._id_to_unit = {}

        plans: list[tuple[ThoughtRecord, str]] = []
        used_ids: set[str] = set()
        for cycle, turn in enumerate(corpus):
            text = turn.text.strip()
            if not text:
                continue
            thought_id = _make_id(turn.session_id, turn.turn_index)
            if thought_id in used_ids:
                thought_id = _make_id(f"{turn.session_id}#dup{cycle}", turn.turn_index)
            used_ids.add(thought_id)

            essence = (
                text[:_ESSENCE_LIMIT].rsplit(" ", 1)[0] if len(text) > _ESSENCE_LIMIT else text
            )
            thought = ThoughtRecord(
                thought_id=thought_id,
                thought_type=ThoughtType.OBSERVATION,
                essence=essence,
                content=text,
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=cycle,
                updated_cycle=cycle,
                source="user-turn",
                source_type=KnowledgeSource.EXPERIENCE,
                confidence=0.6,
                visibility=ThoughtVisibility.PUBLIC,
                access_count=0,
                confirmation_count=0,
                metadata={
                    "session_id": turn.session_id,
                    "turn_index": turn.turn_index,
                    "role": "user",
                    "date": turn.timestamp,
                    "corpus_id": turn.unit_id,
                    "lang": "en",
                    "content_type": "natural_language",
                },
            )
            # Truncate the FINAL embed payload to the model's token cap (empty
            # turns are already skipped above, so no empty input reaches the API).
            embed_payload = _truncate_embed_input(_build_embed_input(essence, text))
            plans.append((thought, embed_payload))
            self._id_to_unit[thought_id] = turn.unit_id

        # Pass 2 — embed distinct payloads in bounded batches.
        distinct: list[str] = []
        seen: set[str] = set()
        for _t, payload in plans:
            if payload not in seen:
                seen.add(payload)
                distinct.append(payload)
        vectors: dict[str, list[float]] = {}
        for start in range(0, len(distinct), _EMBED_BATCH_SIZE):
            batch = distinct[start : start + _EMBED_BATCH_SIZE]
            embedded = await self._provider.embed_batch(batch)
            vectors.update(zip(batch, embedded, strict=True))

        # Pass 3 — create thoughts + store precomputed vectors (deduplicate=False).
        for thought, payload in plans:
            stored = await store.create_thought(thought, deduplicate=False)
            await store.store_embedding(stored.thought_id, vectors[payload], model_name=model_name)

    async def _search_async(self, query: str, top_k: int) -> list[RankedItem]:
        """Run a single default-weighted ``search_hybrid`` and map ids to units."""
        store = self._store
        if store is None:  # pragma: no cover - defensive
            msg = "search() called before ingest()"
            raise RuntimeError(msg)
        result = await store.search_hybrid(
            query_text=query,
            query_vector=None,
            top_k=top_k,
            include_reflections=False,
        )
        ranked: list[RankedItem] = []
        placed: set[str] = set()
        for thought_id, score in result.results:
            unit_id = self._id_to_unit.get(thought_id)
            if unit_id is None or unit_id in placed:
                continue
            ranked.append(RankedItem(unit_id=unit_id, score=float(score)))
            placed.add(unit_id)
        return ranked

    async def _close(self) -> None:
        """Close any open store/connection from a prior ingest."""
        if self._store is not None:
            await self._store.close()
            self._store = None
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # -- teardown ---------------------------------------------------------------

    def close(self) -> None:
        """Release the store, connection and the persistent event loop.

        Idempotent: safe to call more than once and safe when the loop was never
        created. References (loop, store, connection) are ALWAYS released, even if
        the awaited store/connection cleanup raises — the cleanup error then
        propagates after the release. Call this once the adapter is no longer
        needed (the runner does so at the end of a run); :meth:`__del__` is a
        best-effort fallback if it is not called.
        """
        loop = self._loop
        try:
            if loop is not None and not loop.is_closed() and not loop.is_running():
                loop.run_until_complete(self._close())
        finally:
            # Reference release must run even if _close() raised above.
            self._drop(loop)

    def _drop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Close the loop if safe and drop all held references (idempotent).

        A running loop is NEVER closed (that would raise ``RuntimeError``); its
        reference is dropped without closing. Closing an already-closed or
        never-created loop is a no-op.
        """
        if loop is not None and not loop.is_closed() and not loop.is_running():
            loop.close()
        self._loop = None
        # Drop references even if the loop was never created / already closed /
        # still running (sync-only lifetime, or async cleanup could not run).
        self._store = None
        self._conn = None

    def __del__(self) -> None:
        """Best-effort teardown fallback if :meth:`close` was never called.

        Only the synchronous loop close + reference drop is attempted here: the
        async store/connection cleanup is skipped during garbage collection to
        avoid scheduling a coroutine on a loop that may already be finalising.
        """
        # __del__ must never raise; suppress any teardown error during GC.
        with contextlib.suppress(Exception):
            self._drop(self._loop)
