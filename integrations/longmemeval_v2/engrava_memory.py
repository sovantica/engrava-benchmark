"""Engrava memory backend for the LongMemEval-V2 harness.

Copy this file into an upstream LongMemEval-V2 checkout as
``memory_modules/engrava_memory.py`` and register it from
``memory_modules/memory.py``:

    from .engrava_memory import EngravaMemory  # noqa: E402,F401

The backend uses only the public ``engrava`` package. Reader, judge, prompt,
scorer, runtime input materialization, context truncation, and leaderboard
packaging remain owned by the upstream LongMemEval-V2 harness.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import aiosqlite
from engrava import (
    KnowledgeSource,
    LifecycleStatus,
    OpenAICompatibleProvider,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    ThoughtVisibility,
)
from memory_modules.memory import Memory, MemoryContextItem, register_memory, require

if TYPE_CHECKING:
    from collections.abc import Coroutine

_ESSENCE_LIMIT = 180
_DEFAULT_EMBED_TOKENS = 4096
_DEFAULT_EMBED_BATCH_SIZE = 32
_DEFAULT_TOP_K = 6
_DEFAULT_MAX_CONTEXT_CHARS = 20_000
_EMBED_ENCODING = "cl100k_base"

_T = TypeVar("_T")


class _EmbeddingProvider(Protocol):
    """Minimal async embedding provider contract used by the adapter."""

    model_name: str

    async def embed(self, text: str) -> list[float]:
        """Embed one text string."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""


class _DeterministicEmbeddingProvider:
    """Small deterministic embedding provider for smoke tests only."""

    model_name = "deterministic-smoke-embedding"

    def __init__(self, dimension: int = 16) -> None:
        require(dimension > 0, "deterministic embedding dimension must be positive")
        self._dimension = dimension

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for idx in range(self._dimension):
            byte = digest[idx % len(digest)]
            values.append((float(byte) / 127.5) - 1.0)
        return values

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


def _dict_param(params: dict[str, object], key: str) -> dict[str, object]:
    value = params.get(key, {})
    require(isinstance(value, dict), f"{key} must be an object")
    return dict(value)


def _str_param(params: dict[str, object], key: str, default: str) -> str:
    value = params.get(key, default)
    require(isinstance(value, str), f"{key} must be a string")
    return value.strip() or default


def _optional_str_param(params: dict[str, object], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    require(isinstance(value, str), f"{key} must be null or a string")
    stripped = value.strip()
    return stripped or None


def _int_param(params: dict[str, object], key: str, default: int) -> int:
    value = params.get(key, default)
    require(isinstance(value, int) and not isinstance(value, bool), f"{key} must be an integer")
    require(value > 0, f"{key} must be positive")
    return value


def _optional_int_param(params: dict[str, object], key: str) -> int | None:
    value = params.get(key)
    if value is None:
        return None
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{key} must be null or an integer",
    )
    require(value > 0, f"{key} must be positive when provided")
    return value


def _float_param(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{key} must be a number",
    )
    require(value >= 0, f"{key} must be non-negative")
    return float(value)


def _read_api_key(api_key_env: str, api_key_file: str | None, base_url: str) -> str:
    env_value = os.getenv(api_key_env, "").strip()
    if env_value:
        return env_value
    if api_key_file is not None:
        path = Path(api_key_file)
        require(path.exists(), f"missing api_key_file: {path}")
        value = path.read_text(encoding="utf-8").strip()
        require(value, f"empty api_key_file: {path}")
        return value
    if "api.openai.com" not in base_url:
        return "EMPTY"
    msg = f"Missing API key via {api_key_env}"
    raise RuntimeError(msg)


def _create_embedding_provider(params: dict[str, object]) -> _EmbeddingProvider:
    backend = _str_param(params, "backend", "openai-compatible")
    if backend == "deterministic":
        return _DeterministicEmbeddingProvider(
            dimension=_optional_int_param(params, "dimension") or 16,
        )
    require(
        backend in {"openai", "openai-compatible"},
        "embedding_params.backend must be openai-compatible or deterministic",
    )
    model = _str_param(params, "model", "Qwen/Qwen3-Embedding-8B")
    base_url = _str_param(params, "base_url", "http://localhost:8114/v1")
    api_key_env = _str_param(params, "api_key_env", "OPENAI_API_KEY")
    api_key_file = _optional_str_param(params, "api_key_file")
    dimension = _optional_int_param(params, "dimension")
    # Retry knobs (public engrava provider params). Operational-only: they change how
    # transient rate-limit (HTTP 429) responses are ridden out, never the embedding
    # output — a re-tried batch is byte-identical, so the result is unaffected. Bounded
    # backoff long enough to outlast OpenAI's per-minute embedding TPM window on a low
    # usage tier (the default 3 attempts / 1 s gives up in ~3 s, well short of ~60 s).
    max_attempts = _int_param(params, "max_attempts", 12)
    base_retry_delay_s = _float_param(params, "base_retry_delay_s", 5.0)
    return OpenAICompatibleProvider(
        model_name=model,
        base_url=base_url,
        api_key=_read_api_key(api_key_env, api_key_file, base_url),
        dimension=dimension,
        max_attempts=max_attempts,
        base_retry_delay_s=base_retry_delay_s,
    )


def _truncate_embed_input(text: str, max_tokens: int) -> str:
    import tiktoken  # noqa: PLC0415

    enc = tiktoken.get_encoding(_EMBED_ENCODING)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _build_embed_input(essence: str, content: str) -> str:
    if content.strip().startswith(essence.strip()):
        return content
    return f"{essence}\n{content}"


def _text_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ""


def _goal_text(trajectory: dict[str, object]) -> str:
    direct = _text_value(trajectory.get("goal"))
    if direct:
        return direct
    metadata = trajectory.get("metadata")
    if isinstance(metadata, dict):
        original_goal = _text_value(metadata.get("original_goal"))
        if original_goal:
            return original_goal
    return "<goal not found>"


def _state_rows(trajectory: dict[str, object]) -> list[dict[str, object]]:
    public_states = trajectory.get("states")
    if isinstance(public_states, list) and public_states:
        return [state for state in public_states if isinstance(state, dict)]
    content = trajectory.get("content")
    if isinstance(content, list) and content:
        return [state for state in content if isinstance(state, dict)]
    return []


def _state_observation_text(state: dict[str, object]) -> str:
    direct = _text_value(state.get("accessibility_tree")) or _text_value(state.get("text"))
    if direct:
        return direct
    observation = state.get("observation")
    if isinstance(observation, dict):
        return _text_value(observation.get("text"))
    return ""


def _state_action(state: dict[str, object]) -> str:
    return _text_value(state.get("action"))


def _state_thoughts(state: dict[str, object]) -> str:
    return _text_value(state.get("thought")) or _text_value(state.get("thoughts"))


def _state_index(state: dict[str, object], fallback: int) -> int:
    value = state.get("state_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _state_step(state: dict[str, object], fallback: int) -> int:
    value = state.get("step")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _state_url(state: dict[str, object]) -> str:
    return _text_value(state.get("url")) or "<url not found>"


def _essence(text: str) -> str:
    if len(text) <= _ESSENCE_LIMIT:
        return text
    prefix = text[:_ESSENCE_LIMIT]
    return prefix.rsplit(" ", 1)[0] or prefix


def _context_limit(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[truncated]"


@dataclass(frozen=True)
class _StateContext:
    """Normalized trajectory state fields used to build memory context."""

    trajectory_id: str
    goal: str
    outcome: str
    state_index: int
    step: int
    url: str
    action: str
    thoughts: str
    observation: str


def _format_state_context(state: _StateContext) -> str:
    lines = [
        f"Trajectory ID: {state.trajectory_id}",
        f"Goal: {state.goal}",
    ]
    if state.outcome:
        lines.append(f"Outcome: {state.outcome}")
    lines.extend(
        [
            f"State index: {state.state_index}",
            f"Step: {state.step}",
            f"URL: {state.url}",
        ]
    )
    if state.action:
        lines.append(f"Action: {state.action}")
    if state.thoughts:
        lines.append(f"Thoughts: {state.thoughts}")
    if state.observation:
        lines.extend(["Observation:", state.observation])
    return "\n".join(lines).strip()


def _thought_id(trajectory_id: str, state_index: int, step: int, ordinal: int) -> str:
    raw = f"{trajectory_id}:state:{state_index}:step:{step}:ordinal:{ordinal}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@register_memory
class EngravaMemory(Memory):
    """LongMemEval-V2 memory backend powered by public Engrava."""

    memory_type = "engrava"

    def __init__(self, memory_params: dict[str, object]) -> None:
        """Initialize the Engrava backend from LongMemEval-V2 memory params."""
        super().__init__(memory_params)
        embedding_params = _dict_param(memory_params, "embedding_params")
        retrieval_params = _dict_param(memory_params, "retrieval_params")
        self._provider = _create_embedding_provider(embedding_params)
        self._max_embed_tokens = _int_param(
            embedding_params,
            "max_input_tokens",
            _DEFAULT_EMBED_TOKENS,
        )
        self._embed_batch_size = _int_param(
            embedding_params,
            "batch_size",
            _DEFAULT_EMBED_BATCH_SIZE,
        )
        self._top_k = _int_param(retrieval_params, "top_k", _DEFAULT_TOP_K)
        self._max_context_chars = _int_param(
            retrieval_params,
            "max_context_chars",
            _DEFAULT_MAX_CONTEXT_CHARS,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._store: SqliteEngravaCore | None = None
        self._conn: aiosqlite.Connection | None = None
        self._id_to_context: dict[str, str] = {}
        self._cycle = 0

    def insert(self, trajectory: dict[str, object]) -> None:
        """Index one LongMemEval-V2 trajectory into Engrava."""
        self._run_sync(self._insert_async(trajectory))

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        """Return Engrava search results as LongMemEval-V2 text context items."""
        _ = query_image
        return self._run_sync(self._query_async(query))

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[MemoryContextItem],
    ) -> dict[str, object] | None:
        """Return lightweight query metadata for run auditability."""
        _ = query, query_image
        return {
            "system": "engrava",
            "returned_items": len(memory_context),
            "text_only": True,
        }

    def _run_sync(self, coro: Coroutine[object, object, _T]) -> _T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            msg = "EngravaMemory cannot be called from an active event loop"
            raise RuntimeError(msg)
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    async def _ensure_store(self) -> SqliteEngravaCore:
        if self._store is not None:
            return self._store
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
        self._conn = conn
        self._store = store
        return store

    async def _insert_async(self, trajectory: dict[str, object]) -> None:
        store = await self._ensure_store()
        trajectory_id = _text_value(trajectory.get("id"))
        require(trajectory_id, "trajectory id must be a non-empty string")
        goal = _goal_text(trajectory)
        outcome = _text_value(trajectory.get("outcome"))
        states = _state_rows(trajectory)
        require(states, f"trajectory {trajectory_id} has no states/content rows")

        plans: list[tuple[ThoughtRecord, str, str]] = []
        for ordinal, state in enumerate(states):
            state_index = _state_index(state, ordinal)
            step = _state_step(state, ordinal)
            context = _format_state_context(
                _StateContext(
                    trajectory_id=trajectory_id,
                    goal=goal,
                    outcome=outcome,
                    state_index=state_index,
                    step=step,
                    url=_state_url(state),
                    action=_state_action(state),
                    thoughts=_state_thoughts(state),
                    observation=_state_observation_text(state),
                )
            )
            if not context:
                continue
            essence = _essence(context)
            thought_id = _thought_id(trajectory_id, state_index, step, ordinal)
            thought = ThoughtRecord(
                thought_id=thought_id,
                thought_type=ThoughtType.OBSERVATION,
                essence=essence,
                content=context,
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=self._cycle,
                updated_cycle=self._cycle,
                source="longmemeval-v2-state",
                source_type=KnowledgeSource.EXPERIENCE,
                confidence=0.6,
                visibility=ThoughtVisibility.PUBLIC,
                access_count=0,
                confirmation_count=0,
                metadata={
                    "benchmark": "longmemeval-v2",
                    "trajectory_id": trajectory_id,
                    "state_index": state_index,
                    "step": step,
                },
            )
            embed_input = _truncate_embed_input(
                _build_embed_input(essence, context),
                self._max_embed_tokens,
            )
            plans.append((thought, embed_input, context))
            self._cycle += 1

        distinct_payloads: list[str] = []
        seen_payloads: set[str] = set()
        for _thought, payload, _context in plans:
            if payload in seen_payloads:
                continue
            seen_payloads.add(payload)
            distinct_payloads.append(payload)

        vectors: dict[str, list[float]] = {}
        for start in range(0, len(distinct_payloads), self._embed_batch_size):
            batch = distinct_payloads[start : start + self._embed_batch_size]
            embedded = await self._provider.embed_batch(batch)
            vectors.update(zip(batch, embedded, strict=True))

        for thought, payload, context in plans:
            stored = await store.create_thought(thought, deduplicate=False)
            await store.store_embedding(
                stored.thought_id,
                vectors[payload],
                model_name=self._provider.model_name,
            )
            self._id_to_context[stored.thought_id] = _context_limit(
                context,
                self._max_context_chars,
            )

    async def _query_async(self, query: str) -> list[MemoryContextItem]:
        store = await self._ensure_store()
        result = await store.search_hybrid(
            query_text=query,
            query_vector=None,
            top_k=self._top_k,
            include_reflections=False,
        )
        context_items: list[MemoryContextItem] = []
        seen_text: set[str] = set()
        for thought_id, _score in result.results:
            text = self._id_to_context.get(thought_id)
            if not text or text in seen_text:
                continue
            context_items.append({"type": "text", "value": text})
            seen_text.add(text)
        return context_items

    async def _close_async(self) -> None:
        if self._store is not None:
            await self._store.close()
            self._store = None
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def close(self) -> None:
        """Release the in-memory store and event loop."""
        loop = self._loop
        try:
            if loop is not None and not loop.is_closed() and not loop.is_running():
                loop.run_until_complete(self._close_async())
        finally:
            if loop is not None and not loop.is_closed() and not loop.is_running():
                loop.close()
            self._loop = None
            self._store = None
            self._conn = None

    def __del__(self) -> None:
        """Best-effort cleanup if the harness drops the object without close()."""
        with contextlib.suppress(Exception):
            self.close()
