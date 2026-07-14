"""Engrava memory provider for the Agent Memory Benchmark (AMB).

Copy this file into an upstream `agent-memory-benchmark
<https://github.com/vectorize-io/agent-memory-benchmark>`_ checkout as
``src/memory_bench/memory/engrava.py`` and register it in
``src/memory_bench/memory/__init__.py``::

    from .engrava import EngravaMemoryProvider
    ...
    REGISTRY: dict[str, type[MemoryProvider]] = {
        ...
        "engrava": EngravaMemoryProvider,
    }

The provider uses only the public ``engrava`` package and stays in **Group A**:
there is no LLM anywhere in the memory path. Ingest indexes each document as a
single Engrava thought with an explicit embedding; retrieval is one
``search_hybrid`` call (default fusion weights, recency inactive, reflections
excluded), exactly mirroring the parity-proven public retrieval path. Reader,
judge, prompt, scorer, and dataset ownership remain with the upstream AMB harness.

Embeddings are configured from the environment so the memory path stays local
(no API) by default:

* ``ENGRAVA_AMB_EMBED_BACKEND`` — ``local`` (default), ``openai``, or
  ``deterministic`` (offline smoke only; never for publishable results).
* ``ENGRAVA_AMB_EMBED_MODEL`` — embedding model name (backend-specific default).
* ``OPENAI_API_KEY`` / ``ENGRAVA_AMB_EMBED_BASE_URL`` — used by the ``openai``
  backend only.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from typing import TYPE_CHECKING, Protocol, TypeVar

# --- PUBLIC engrava only — no private-package import in this module ----------- #
from engrava import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    ThoughtVisibility,
)

# Upstream AMB package — present when this file is dropped into an AMB checkout.
from memory_bench.memory.base import MemoryProvider

if TYPE_CHECKING:
    from collections.abc import Coroutine

    import aiosqlite
    from memory_bench.models import Document

_ESSENCE_LIMIT = 180
_EMBED_BATCH_SIZE = 32
_MAX_EMBED_TOKENS = 8192
_EMBED_ENCODING = "cl100k_base"
_DEFAULT_TOP_K = 20
_DETERMINISTIC_DIM = 16
_DEFAULT_LOCAL_MODEL = "all-MiniLM-L12-v2"
_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

_T = TypeVar("_T")

# A bank is keyed by ``(user_id is None, user_id)``. The leading boolean tags the
# no-isolation-unit case, so a real ``user_id`` value (including the literal
# ``"_shared"``) can never collide with the shared/no-unit bank.
BankKey = tuple[bool, str | None]


class ProviderError(RuntimeError):
    """Raised on provider misuse (e.g. driving the sync API inside a running loop)."""


class _EmbeddingProvider(Protocol):
    """Minimal async embedding-provider contract used by the provider."""

    model_name: str

    async def embed(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The string to embed.

        Returns:
            The embedding vector for ``text``.

        """

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: The strings to embed.

        Returns:
            One embedding vector per input string, in order.

        """


class _DeterministicEmbeddingProvider:
    """Deterministic, network-free embedding provider for offline smoke runs only.

    Not for publishable results — it encodes only byte statistics, not semantics.
    """

    model_name = "deterministic-smoke-embedding"

    def __init__(self, dimension: int = _DETERMINISTIC_DIM) -> None:
        """Initialize the deterministic provider.

        Args:
            dimension: The embedding dimension (must be positive).

        Raises:
            ValueError: If ``dimension`` is not positive.

        """
        if dimension <= 0:
            msg = "deterministic embedding dimension must be positive"
            raise ValueError(msg)
        self._dimension = dimension

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(float(digest[idx % len(digest)]) / 127.5) - 1.0 for idx in range(self._dimension)]

    async def embed(self, text: str) -> list[float]:
        """Embed one string deterministically.

        Args:
            text: The string to embed.

        Returns:
            The deterministic vector for ``text``.

        """
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings deterministically.

        Args:
            texts: The strings to embed.

        Returns:
            One deterministic vector per input string.

        """
        return [self._vector(text) for text in texts]


def _build_embed_input(essence: str, content: str) -> str:
    """Build the text payload to embed for a thought, mirroring engrava core.

    When the stripped ``essence`` is a leading prefix of the stripped ``content``
    it carries no new information, so ``content`` is embedded alone; otherwise the
    newline-joined payload is used. Reproducing this keeps the stored vector
    identical to an ``auto_embed=True`` store.

    Args:
        essence: The thought's short essence.
        content: The thought's full content.

    Returns:
        The exact text payload engrava would embed.

    """
    if content.strip().startswith(essence.strip()):
        return content
    return f"{essence}\n{content}"


def _essence(text: str) -> str:
    """Return a short essence for ``text`` (first ``_ESSENCE_LIMIT`` chars, word-safe).

    Args:
        text: The full content string.

    Returns:
        ``text`` unchanged when short enough, else a whitespace-trimmed prefix.

    """
    if len(text) <= _ESSENCE_LIMIT:
        return text
    prefix = text[:_ESSENCE_LIMIT]
    return prefix.rsplit(" ", 1)[0] or prefix


def _truncate_embed_input(text: str) -> str:
    """Truncate an embed payload to at most ``_MAX_EMBED_TOKENS`` cl100k tokens.

    Truncation is by token using ``tiktoken``'s ``cl100k_base`` encoding, so the
    kept prefix is exactly what the ``text-embedding-3`` family would accept and is
    deterministic (a reproducible vector). Inputs within the limit are unchanged.

    Args:
        text: The final embed payload.

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


def _thought_id(bank_key: BankKey, doc_id: str, ordinal: int, content: str) -> str:
    """Derive a deterministic, stable-unique thought id for one insertion.

    The ``ordinal`` is a per-bank monotonic counter that advances across every
    :meth:`EngravaMemoryProvider.ingest` call, so re-ingesting the same
    bank/document never regenerates a prior id. A content hash is folded in so
    distinct payloads never share an id even if a counter were ever reused.

    Args:
        bank_key: The per-user bank key.
        doc_id: The source document id.
        ordinal: The bank's monotonic insertion counter (unique within the bank).
        content: The document content (folded in as a stability hash).

    Returns:
        The full hex digest of ``sha256`` over the composed key.

    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    raw = f"{bank_key[0]}:{bank_key[1]}:{doc_id}:{ordinal}:{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _create_embedding_provider() -> _EmbeddingProvider:
    """Create the configured public-engrava embedding provider from the environment.

    The default backend is ``local`` (a network-free SentenceTransformer), keeping
    the memory path Group A. See the module docstring for the recognised variables.

    Returns:
        A public engrava embedding provider (or the deterministic smoke provider).

    Raises:
        ProviderError: If ``ENGRAVA_AMB_EMBED_BACKEND`` is not a recognised backend.

    """
    backend = os.environ.get("ENGRAVA_AMB_EMBED_BACKEND", "local").strip() or "local"
    model = os.environ.get("ENGRAVA_AMB_EMBED_MODEL", "").strip()

    if backend == "deterministic":
        return _DeterministicEmbeddingProvider()

    if backend == "local":
        from engrava.embeddings.sentence_transformer import (  # noqa: PLC0415
            SentenceTransformerProvider,
        )

        device = os.environ.get("ENGRAVA_AMB_EMBED_DEVICE", "cpu").strip() or "cpu"
        return SentenceTransformerProvider(
            model_name=model or _DEFAULT_LOCAL_MODEL,
            device=device,
        )

    if backend == "openai":
        from engrava.embeddings.openai_compatible import (  # noqa: PLC0415
            OpenAICompatibleProvider,
        )

        base_url = os.environ.get("ENGRAVA_AMB_EMBED_BASE_URL", "").strip()
        kwargs: dict[str, str] = {
            "model_name": model or _DEFAULT_OPENAI_MODEL,
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        }
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAICompatibleProvider(**kwargs)

    msg = f"Unknown ENGRAVA_AMB_EMBED_BACKEND: {backend!r}. Use local, openai, or deterministic."
    raise ProviderError(msg)


class _Bank:
    """A per-user Engrava store plus the id map back to the source documents."""

    __slots__ = ("conn", "id_to_doc", "next_ordinal", "store")

    def __init__(self, conn: aiosqlite.Connection, store: SqliteEngravaCore) -> None:
        """Initialize the bank.

        Args:
            conn: The bank's sqlite connection.
            store: The engrava core bound to ``conn``.

        """
        self.conn = conn
        self.store = store
        self.id_to_doc: dict[str, Document] = {}
        # Monotonic insertion counter, advanced across every ingest() call so a
        # re-ingested document never regenerates a previously used thought id.
        self.next_ordinal = 0


class EngravaMemoryProvider(MemoryProvider):
    """Agent Memory Benchmark provider backed by the public Engrava package.

    Each distinct ``user_id`` (isolation unit) gets its own in-memory Engrava
    store, so retrieval never leaks across banks. Ingest indexes one thought per
    document with an explicit embedding; retrieval is a single default-weighted
    ``search_hybrid``. No LLM is used anywhere in the memory path (Group A).

    Single-threaded contract: the synchronous ``ingest`` / ``retrieve`` / ``cleanup``
    methods share one persistent event loop and are NOT safe to call concurrently
    from multiple threads. ``concurrency`` is therefore fixed to 1.
    """

    name = "engrava"
    description = (
        "Local hybrid dense + sparse (BM25) retrieval over the public Engrava "
        "memory core. No LLM in the memory path: one thought per document with an "
        "explicit embedding, a single default-weighted search_hybrid at query time."
    )
    kind = "local"
    provider = "Sovantica"
    variant = "local"
    link = "https://github.com/sovantica/engrava"
    concurrency = 1

    def __init__(self, k: int = _DEFAULT_TOP_K) -> None:
        """Initialize the provider.

        Args:
            k: Default retrieval depth used when the harness passes a non-positive
                ``k`` to :meth:`retrieve`.

        """
        self._k = k
        self._provider = _create_embedding_provider()
        self._banks: dict[BankKey, _Bank] = {}
        # One persistent event loop for this provider's whole lifetime, created
        # lazily on the first sync call so ingest, retrieve, and cleanup all run on
        # the same loop. Async engrava providers cache a loop-bound client; reusing
        # one loop keeps that client valid across calls.
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- MemoryProvider interface ----------------------------------------------- #

    def ingest(self, documents: list[Document]) -> None:
        """Index documents into their per-user Engrava banks.

        Args:
            documents: The documents to ingest. Each document's ``user_id`` selects
                its isolation bank (a shared bank is used when ``user_id`` is None).

        Raises:
            ProviderError: If called from inside a running event loop.

        """
        self._run_sync(self._ingest_async(documents))

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict[str, object] | None]:
        """Retrieve the top-``k`` documents for ``query`` from the user's bank.

        Args:
            query: The query text.
            k: Maximum number of documents to return (falls back to the constructor
                default when non-positive).
            user_id: The isolation unit to search (shared bank when None).
            query_timestamp: Ignored — the Group A path keeps recency inactive to
                mirror the parity-proven default fusion weights.

        Returns:
            A ``(documents, raw_response)`` tuple. ``raw_response`` carries the
            per-hit scores keyed by source document id.

        Raises:
            ProviderError: If called from inside a running event loop.

        """
        _ = query_timestamp
        top_k = k if k > 0 else self._k
        return self._run_sync(self._retrieve_async(query, top_k, user_id))

    def cleanup(self) -> None:
        """Release every bank's store/connection and the persistent event loop.

        Idempotent: safe to call more than once and when the loop was never created.
        When the loop is idle (the normal case) every bank store and connection is
        closed on it *before* references and the loop are dropped, so nothing is
        leaked. References and the loop are then ALWAYS dropped, even if a close
        raises. The provider's own loop only runs during a synchronous
        ingest/retrieve call, so under the single-threaded contract it is idle here;
        if ``cleanup`` is somehow reached while that loop is still running, an async
        close cannot be driven synchronously and the caller must close the provider
        from outside that loop instead.
        """
        loop = self._loop
        try:
            if loop is not None and not loop.is_closed() and not loop.is_running():
                loop.run_until_complete(self._cleanup_async())
        finally:
            self._drop(loop)

    # -- internals -------------------------------------------------------------- #

    def _run_sync(self, coro: Coroutine[object, object, _T]) -> _T:
        """Run a coroutine to completion on the provider's persistent event loop.

        Args:
            coro: The coroutine to run.

        Returns:
            The coroutine's result.

        Raises:
            ProviderError: If called while an event loop is already running.

        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            msg = (
                "EngravaMemoryProvider's synchronous ingest/retrieve cannot run "
                "inside an active event loop; the AMB harness offloads sync "
                "providers to a worker thread by default."
            )
            raise ProviderError(msg)
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    @staticmethod
    def _bank_key(user_id: str | None) -> BankKey:
        """Return the collision-proof bank key for a ``user_id``.

        The leading boolean tags the no-isolation-unit case (``user_id is None``),
        so ``user_id="_shared"`` and ``user_id=None`` map to different banks.

        Args:
            user_id: The isolation unit, or None when the dataset declares none.

        Returns:
            The tagged bank key.

        """
        return (user_id is None, user_id)

    async def _ensure_bank(self, bank_key: BankKey) -> _Bank:
        """Return the bank for ``bank_key``, creating a fresh in-memory store if new.

        Args:
            bank_key: The per-user bank key.

        Returns:
            The existing or newly created bank.

        """
        existing = self._banks.get(bank_key)
        if existing is not None:
            return existing

        import aiosqlite  # noqa: PLC0415

        conn = await aiosqlite.connect(":memory:")
        store: SqliteEngravaCore | None = None
        try:
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
        except BaseException:
            # Any failure before the bank is registered must not leak the freshly
            # opened store/connection; close what we opened, then re-raise.
            if store is not None:
                with contextlib.suppress(Exception):
                    await store.close()
            with contextlib.suppress(Exception):
                await conn.close()
            raise
        bank = _Bank(conn, store)
        self._banks[bank_key] = bank
        return bank

    def _plan_thought(
        self, bank_key: BankKey, doc: Document, ordinal: int
    ) -> tuple[ThoughtRecord, str]:
        """Build the thought record + embed payload for one document.

        Args:
            bank_key: The bank key the document belongs to.
            doc: The source document.
            ordinal: The bank's monotonic insertion counter for this document.

        Returns:
            A ``(thought, embed_payload)`` pair.

        """
        content = doc.content
        essence = _essence(content)
        thought = ThoughtRecord(
            thought_id=_thought_id(bank_key, doc.id, ordinal, content),
            thought_type=ThoughtType.OBSERVATION,
            essence=essence,
            content=content,
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=ordinal,
            updated_cycle=ordinal,
            source="amb-document",
            source_type=KnowledgeSource.EXPERIENCE,
            confidence=0.6,
            visibility=ThoughtVisibility.PUBLIC,
            access_count=0,
            confirmation_count=0,
            metadata={
                "benchmark": "agent-memory-benchmark",
                "doc_id": doc.id,
                "user_id": doc.user_id or "",
                "timestamp": doc.timestamp or "",
            },
        )
        payload = _truncate_embed_input(_build_embed_input(essence, content))
        return thought, payload

    async def _ingest_async(self, documents: list[Document]) -> None:
        """Index documents into their banks with explicit, deduplicated embeddings.

        Empty (stripped-empty) documents are skipped, so no empty input reaches the
        embedding backend. Distinct payloads are embedded in bounded batches.

        Args:
            documents: The documents to ingest.

        """
        plans: list[tuple[_Bank, ThoughtRecord, Document, str]] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            bank_key = self._bank_key(doc.user_id)
            bank = await self._ensure_bank(bank_key)
            ordinal = bank.next_ordinal
            bank.next_ordinal += 1
            thought, payload = self._plan_thought(bank_key, doc, ordinal)
            plans.append((bank, thought, doc, payload))

        distinct: list[str] = []
        seen: set[str] = set()
        for _bank, _thought, _doc, payload in plans:
            if payload not in seen:
                seen.add(payload)
                distinct.append(payload)

        vectors: dict[str, list[float]] = {}
        for start in range(0, len(distinct), _EMBED_BATCH_SIZE):
            batch = distinct[start : start + _EMBED_BATCH_SIZE]
            embedded = await self._provider.embed_batch(batch)
            vectors.update(zip(batch, embedded, strict=True))

        model_name = self._provider.model_name
        for bank, thought, doc, payload in plans:
            stored = await bank.store.create_thought(thought, deduplicate=False)
            await bank.store.store_embedding(
                stored.thought_id,
                vectors[payload],
                model_name=model_name,
            )
            bank.id_to_doc[stored.thought_id] = doc

    async def _retrieve_async(
        self,
        query: str,
        top_k: int,
        user_id: str | None,
    ) -> tuple[list[Document], dict[str, object] | None]:
        """Run one default-weighted ``search_hybrid`` on the user's bank.

        Args:
            query: The query text.
            top_k: Maximum number of documents to return.
            user_id: The isolation unit to search.

        Returns:
            A ``(documents, raw_response)`` tuple; the raw response records the
            per-hit scores keyed by source document id.

        """
        bank = self._banks.get(self._bank_key(user_id))
        if bank is None:
            return [], {"results": []}

        result = await bank.store.search_hybrid(
            query_text=query,
            query_vector=None,
            top_k=top_k,
            include_reflections=False,
        )
        docs: list[Document] = []
        raw: list[dict[str, object]] = []
        placed: set[str] = set()
        for thought_id, score in result.results:
            doc = bank.id_to_doc.get(thought_id)
            if doc is None or doc.id in placed:
                continue
            # Return the ORIGINAL document (in score order) so the reader keeps
            # every field (messages/timestamp/context), not a partial rebuild.
            docs.append(doc)
            raw.append({"id": doc.id, "score": float(score)})
            placed.add(doc.id)
        return docs, {"results": raw}

    async def _cleanup_async(self) -> None:
        """Close every bank's store and connection, robust to a partial failure.

        Every store and connection gets a close attempt even if an earlier one
        raises; the first error (if any) is re-raised after all are attempted, so
        no connection is leaked by an early exit.

        Raises:
            BaseException: The first error raised while closing a store/connection,
                re-raised only after every bank has been closed.

        """
        banks = list(self._banks.values())
        self._banks = {}
        first_error: BaseException | None = None
        for bank in banks:
            for closer in (bank.store.close, bank.conn.close):
                try:
                    await closer()
                except Exception as exc:  # noqa: BLE001 - close every bank; surface first
                    first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def _close_banks_sync(self) -> None:
        """Best-effort synchronous close of every bank store/connection on the loop.

        Used by :meth:`__del__` so a dropped provider does not leak connections.
        Runs the async cleanup on the persistent loop when it is safe to do so.
        """
        loop = self._loop
        if loop is None or loop.is_closed() or loop.is_running() or not self._banks:
            return
        with contextlib.suppress(Exception):
            loop.run_until_complete(self._cleanup_async())

    def _drop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Close the loop if safe and drop all held references (idempotent).

        Args:
            loop: The loop captured before cleanup (never closed while running).

        """
        try:
            if loop is not None and not loop.is_closed() and not loop.is_running():
                loop.close()
        finally:
            # References are dropped even if loop.close() raises.
            self._loop = None
            self._banks = {}

    def __del__(self) -> None:
        """Best-effort teardown fallback if :meth:`cleanup` was never called.

        Closes any still-open bank stores/connections before dropping the loop, so
        a garbage-collected provider does not leak sqlite connections. The store
        close and the loop/reference drop are guarded independently, so a failure
        in the former never skips the latter and no exception escapes ``__del__``.
        ``BaseException`` is suppressed here (finalizer only) so nothing at all can
        propagate out of garbage collection.
        """
        with contextlib.suppress(BaseException):
            self._close_banks_sync()
        with contextlib.suppress(BaseException):
            self._drop(self._loop)
