"""Integration tests for the LongMemEval-V2 Engrava memory backend.

The backend (``integrations/longmemeval_v2/engrava_memory.py``) plugs into the
upstream ``memory_modules.memory`` harness API, which only exists inside a
LongMemEval-V2 checkout. That module is **stubbed here** via ``sys.modules``
before import — the stub supplies the ``Memory`` base, the ``register_memory``
decorator, the ``require`` assertion helper, and the ``MemoryContextItem`` alias
the backend depends on. Everything below (state normalization, thought building,
ingest, retrieval) runs against the *real* public ``engrava`` core with the
backend's own ``deterministic`` embedding backend — no network, no spend.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

# The backend runs against the real public engrava core + a sqlite driver — the
# `[engrava]` optional extra, absent from the CI `[dev]` test env. Skip the whole
# module there (mirrors tests/test_engrava_adapter.py).
pytest.importorskip("engrava")
pytest.importorskip("aiosqlite")

# --- Stub the upstream LongMemEval-V2 harness API --------------------------- #
# Must be registered BEFORE importing the backend module (top-level import).
if "memory_modules" not in sys.modules:
    _mm = types.ModuleType("memory_modules")
    _mm_memory = types.ModuleType("memory_modules.memory")

    class _StubMemory:
        """Minimal stand-in for the upstream ``Memory`` base class."""

        def __init__(self, memory_params: dict[str, object]) -> None:
            self.memory_params = memory_params

    def _register_memory(cls: type) -> type:
        """Identity decorator standing in for the upstream registry hook."""
        return cls

    def _require(condition: object, message: str = "") -> None:
        """Raise ``ValueError`` when ``condition`` is falsy (upstream contract)."""
        if not condition:
            raise ValueError(message)

    _mm_memory.Memory = _StubMemory
    _mm_memory.MemoryContextItem = dict
    _mm_memory.register_memory = _register_memory
    _mm_memory.require = _require
    _mm.memory = _mm_memory
    sys.modules["memory_modules"] = _mm
    sys.modules["memory_modules.memory"] = _mm_memory

from integrations.longmemeval_v2.engrava_memory import EngravaMemory  # noqa: E402


def _memory_params() -> dict[str, object]:
    """Offline params: deterministic embeddings, small retrieval depth."""
    return {
        "embedding_params": {"backend": "deterministic"},
        "retrieval_params": {"top_k": 6},
    }


def _trajectory() -> dict[str, object]:
    """A minimal LongMemEval-V2 trajectory with one observation-bearing state."""
    return {
        "id": "traj-1",
        "goal": "Find the pricing page and record the Pro tier cost",
        "outcome": "success",
        "states": [
            {
                "state_index": 0,
                "step": 0,
                "url": "https://example.com/pricing",
                "action": "click the pricing link",
                "thought": "look for the Pro tier price",
                "accessibility_tree": "Pricing page: the Pro tier costs 49 dollars per month.",
            }
        ],
    }


@pytest.fixture
def memory() -> EngravaMemory:
    """A backend on the deterministic embedding backend, closed after the test."""
    mem = EngravaMemory(_memory_params())
    try:
        yield mem
    finally:
        mem.close()


def test_insert_and_query_returns_text_context_items(memory: EngravaMemory) -> None:
    """Query returns text context items carrying the inserted state's content."""
    memory.insert(_trajectory())

    items = memory.query("Pro tier price 49 dollars pricing")

    assert items, "expected at least one context item for a token-overlapping query"
    assert all(item["type"] == "text" for item in items)
    joined = "\n".join(item["value"] for item in items)
    assert "Pro tier costs 49 dollars" in joined
    assert "Trajectory ID: traj-1" in joined


def test_query_with_no_matching_memory_returns_empty(memory: EngravaMemory) -> None:
    """A query sharing no tokens with any state returns no context items."""
    memory.insert(_trajectory())
    items = memory.query("xyzzy plugh quux nonmatching token")
    assert items == []


def test_insert_requires_states(memory: EngravaMemory) -> None:
    """A trajectory with no states/content rows is rejected by the upstream contract."""
    with pytest.raises(ValueError):
        memory.insert({"id": "empty", "goal": "g", "states": []})


def test_post_query_hook_reports_metadata(memory: EngravaMemory) -> None:
    """The post-query hook summarises the returned context for auditability."""
    memory.insert(_trajectory())
    items = memory.query("Pro tier pricing")
    meta = memory.post_query_hook(query="Pro tier pricing", query_image=None, memory_context=items)
    assert meta == {"system": "engrava", "returned_items": len(items), "text_only": True}


def test_close_is_idempotent(memory: EngravaMemory) -> None:
    """``close`` may be called more than once without error."""
    memory.insert(_trajectory())
    memory.close()
    memory.close()  # second call must be a no-op


def test_call_inside_running_loop_raises() -> None:
    """Driving the sync API from inside a running loop raises ``RuntimeError``."""
    mem = EngravaMemory(_memory_params())

    async def _call() -> None:
        with pytest.raises(RuntimeError):
            mem.insert(_trajectory())

    try:
        asyncio.run(_call())
    finally:
        mem.close()
