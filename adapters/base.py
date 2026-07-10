"""The pluggable memory-system adapter seam.

A memory system plugs into the benchmark by implementing exactly one small
interface: :class:`MemoryAdapter` (``ingest`` + ``search``). Everything that is
*not* the memory layer — dataset loading, context assembly, the reader LLM, the
reader prompt, the judge LLM, and the official scorer — is owned by the runner
and is identical for every system. The only thing that varies between systems is
the memory layer, so no system can win in the reader or the prompt: the memory
layer is the sole independent variable. (This is equal-footing by construction.)

A third party benchmarks their own database by dropping a single file
``adapters/<their_db>.py`` that implements :class:`MemoryAdapter`, running the
same runner, and submitting the result. See ``adapters/README.md``.

Equal-footing enforcement boundary
----------------------------------
:class:`RunContext` is **read-only and minimal**. It exposes ONLY the corpus an
adapter must ingest and the declared run parameters. It deliberately does NOT
expose gold answers, judge/oracle labels, hidden split metadata, reader outputs,
judge outputs, or any ranking/leaderboard hint. An adapter that could read the
answers could "retrieve" the needle by cheating; the type makes that impossible
by construction, and PR review backstops it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CorpusTurn:
    """One ingestible unit of the haystack: a single conversation turn.

    Only the fields an adapter legitimately needs to store and later retrieve a
    unit are exposed. Notably there is **no** ``has_answer`` / label / gold flag:
    whether a turn is evidence for the question is exactly the thing an adapter
    must not see.

    Attributes:
        unit_id: Stable identifier for this retrievable unit (the benchmark's
            canonical corpus id). Returned by :meth:`MemoryAdapter.search` so the
            runner can map a ranked hit back to its text for context assembly.
        text: The turn's natural-language content to index/embed.
        session_id: Identifier of the conversation session this turn belongs to.
        turn_index: Zero-based index of the turn within its session.
        timestamp: The session/turn date string, as provided by the dataset
            (may be empty if the dataset has none).

    """

    unit_id: str
    text: str
    session_id: str
    turn_index: int
    timestamp: str


@dataclass(frozen=True, slots=True)
class RankedItem:
    """One retrieval result: a unit id and its retrieval score.

    Attributes:
        unit_id: The :attr:`CorpusTurn.unit_id` of the retrieved unit.
        score: Retrieval score (higher is better). Used only for ordering; the
            runner re-ranks by descending score defensively.

    """

    unit_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RunContext:
    """Read-only, minimal run context handed to an adapter (leakage guard).

    This is the equal-footing enforcement boundary (ADR system architecture §1).
    It carries ONLY what an adapter needs to do honest ingestion + retrieval:

    Attributes:
        top_k: Retrieval breadth the runner will ask for. Declared up front so an
            adapter may size internal structures; the runner still passes
            ``top_k`` explicitly to :meth:`MemoryAdapter.search`.
        granularity: The retrieval granularity for this run (e.g. ``"turn"``).
        embedder_spec: Opaque embedder spec string (e.g.
            ``"local:all-MiniLM-L12-v2"`` or ``"openai:text-embedding-3-small"``)
            an adapter MAY honor to embed its units. It is a handle/spec, not a
            channel to any answer or label.

    Intentionally absent (and never to be added): gold answers, evidence/label
    flags, judge or reader outputs, split/headline metadata, leaderboard hints.

    """

    top_k: int
    granularity: str
    embedder_spec: str


@runtime_checkable
class MemoryAdapter(Protocol):
    """The one interface a memory system implements to enter the benchmark.

    An adapter owns ONLY the memory layer. It MUST NOT override the reader, the
    reader prompt, context assembly, the judge, or the scorer; it MUST NOT access
    benchmark answers/labels; and it MUST NOT carry any private/internal data on
    the public surface. A counted lever lives behind ``ingest``/``search`` only.
    """

    def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
        """Ingest the haystack for a single question into the memory system.

        Called once per question with that question's full haystack of turns.
        The adapter stores/indexes the units however it likes (embed, write-time
        transforms, etc.). Any write-time generative-LLM use must be disclosed in
        the result's ``system_config.memory_pipeline_llms`` (which decides the
        Group A/B axis) — it is not configured here.

        Args:
            corpus: The question's haystack as a list of :class:`CorpusTurn`.
            run_ctx: Read-only run parameters (:class:`RunContext`).

        """
        ...

    def search(self, query: str, *, top_k: int) -> list[RankedItem]:
        """Retrieve the top-``top_k`` units for ``query``, best first.

        Args:
            query: The question text to retrieve against.
            top_k: Maximum number of ranked units to return.

        Returns:
            Up to ``top_k`` :class:`RankedItem` in descending score order. Each
            ``unit_id`` MUST be a :attr:`CorpusTurn.unit_id` from the corpus that
            was ingested for this question.

        """
        ...
