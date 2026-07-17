"""Uniform LongMemEval runner.

One pipeline, system-independent except for the memory layer:

    dataset -> adapter.ingest -> adapter.search (retrieval)
            -> context assembly (runner) -> reader LLM (runner)
            -> judge LLM (runner) -> official scorer (unmodified) -> result.json

The **runner** owns context assembly, the reader, the reader prompt, the judge,
and the official scorer; the **adapter** owns only ingest + retrieve. This is the
equal-footing contract: the memory layer is the only independent variable.

Scope (Phase 1)
---------------
This module is the runner *skeleton* / glue. It wires the dataset, the adapter
seam, context assembly, and the result-record assembly, and exposes pluggable
``reader`` / ``judge`` / ``scorer`` seams. The published full-500 reader/judge/
scorer *content and methodology* are produced and ratified by the benchmark
maintainers (the parity-proven official runner); this file is the public glue
that calls them. It deliberately does not embed paid-LLM reader/judge calls or
fabricated numbers — those are supplied by the maintainers' reproduction run.

Every parameter that affects the number is read from ``config/`` and written into
the result JSON — nothing that moves the number is hard-coded silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from adapters.base import CorpusTurn, RunContext
from runners.longmemeval import official_reader

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from adapters.base import MemoryAdapter

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET_ENV = "ENGRAVA_BENCH_LONGMEMEVAL_S"
DEFAULT_DATASET_PATH = HERE / "_cache" / "longmemeval_s_cleaned.json"
SMOKE_DATASET = REPO_ROOT / "tests" / "fixtures" / "longmemeval_smoke.json"
SMOKE_EMBEDDER_SPEC = "local:all-MiniLM-L12-v2"


# --------------------------------------------------------------------------- #
# Pluggable reader / judge / scorer seams (owned by the runner, not adapters)
# --------------------------------------------------------------------------- #
class Reader(Protocol):
    """Answers a question from the runner-assembled context (runner-owned)."""

    def answer(self, question: str, context: str, *, question_date: str = "") -> str:
        """Produce an answer string for ``question`` given ``context``.

        ``context`` is the official assembled history string; ``question_date`` is
        the question's date (the official reader prompt includes ``Current Date``).
        """
        ...


class Judge(Protocol):
    """Scores a model answer against the gold answer (runner-owned)."""

    def score(
        self,
        question: str,
        gold: str,
        answer: str,
        *,
        question_type: str,
        question_id: str,
    ) -> bool:
        """Return ``True`` iff ``answer`` is judged correct for ``question``.

        ``question_id`` is needed to detect abstention items (official ``_abs``
        suffix), which select the abstention judge prompt.
        """
        ...


class Scorer(Protocol):
    """Aggregates per-question judgments into the official metrics block."""

    def aggregate(self, judgments: Sequence[Judgment]) -> dict[str, Any]:
        """Aggregate judgments into the ``metrics`` object (see results schema)."""
        ...


@dataclass(frozen=True, slots=True)
class SessionContent:
    """Runner-internal full session content for context assembly.

    Carries the session's date and its ordered turns (``role`` + ``content`` only;
    the evidence ``has_answer`` flag is popped exactly as the official reader does).
    This is RUNNER-INTERNAL — it lives on :class:`Question`, never on a
    :class:`CorpusTurn`, so an adapter cannot reach it. It is what the official
    round-expansion needs (the assistant replies + the session date), which the
    user-only adapter corpus does not carry.
    """

    date: str
    turns: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class Question:
    """One LongMemEval question with its haystack.

    ``answer`` (the gold), ``id_map`` (neutral unit id -> official corpus id) and
    ``sessions`` (full session content for assembly) are **runner-internal**: they
    are never passed to an adapter. The adapter only ever sees :attr:`corpus`,
    whose :class:`CorpusTurn` ids are neutral and opaque.
    """

    question_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    corpus: list[CorpusTurn]
    # Neutral adapter-facing unit id -> the official `answer`/`noans` corpus id.
    # Runner-internal only (used solely if an official retrieval-log artifact is
    # emitted); the evidence-encoding official id never reaches an adapter.
    id_map: dict[str, str]
    # Neutral session id -> full session content (date + role/content turns).
    # Runner-internal: the official reader's round-expansion needs the assistant
    # replies + the session date, which the user-only adapter corpus omits.
    sessions: dict[str, SessionContent]


@dataclass(frozen=True, slots=True)
class Judgment:
    """One scored question."""

    question_id: str
    question_type: str
    correct: bool


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Per-question record captured during a run, for the reproduction artifact.

    Carries exactly what the reproduction artifact needs and nothing that leaks
    gold: the hypothesis (the reader's answer), the judge verdict, and the ranked
    retrieval mapped to official corpus ids (via ``Question.id_map``). The gold
    answer is NOT stored here.
    """

    question_id: str
    question_type: str
    hypothesis: str
    correct: bool
    # The adapter's ranked retrieval, mapped to official corpus ids (best first).
    ranked_official_ids: list[str]


# --------------------------------------------------------------------------- #
# Dataset loading + corpus construction (system-independent)
# --------------------------------------------------------------------------- #
class DatasetError(ValueError):
    """Raised when a dataset item is structurally malformed."""


def _official_corpus_id(session_id: str, i_turn: int, *, has_answer: bool) -> str:
    """Build the official turn-granularity corpus_id for a user turn.

    Mirrors the official flat-index runner: base id ``f"{session_id}_{i_turn+1}"``
    over the FULL turn list; in an ``answer_*`` session a non-evidence user turn
    rewrites ``answer`` -> ``noans``. The result **encodes evidence status**, so it
    is RUNNER-INTERNAL only — kept in :attr:`Question.id_map` and used solely if an
    official retrieval-log artifact is emitted. It is NEVER passed to an adapter.

    Args:
        session_id: The session identifier.
        i_turn: Zero-based index in the full session turn list.
        has_answer: Whether this turn is evidence (id-construction only).

    Returns:
        The official corpus id string.

    """
    base = f"{session_id}_{i_turn + 1}"
    if "answer" not in session_id:
        return base
    if has_answer:
        return base
    return base.replace("answer", "noans")


def _neutral_session_id(session: list[dict[str, Any]]) -> str:
    """Derive a stable, opaque, non-positional session id from session CONTENT.

    The adapter-facing session id is a hash of the session's ordered turns
    (role + content per turn) — NOT of its id string and NOT of its haystack
    position. Deriving it from content gives every property the equal-footing
    guard needs, with no marker parsing at all:

    * **Marker-independent / evidence-free.** It never inspects the session id or
      any ``has_answer`` flag, so no ``answer``/``noans`` token can survive and the
      same turns hash identically whether the session is the evidence session or a
      plain distractor (content does not encode the evidence ROLE).
    * **Stable.** The same session content always yields the same id.
    * **Distinct.** Different session content yields a different id; there is no
      substring normalisation that could over-collapse or collide ids. The FULL
      sha256 digest is used (the id is an opaque key, so length is free) — no
      truncation, so no birthday-collision concern.
    * **Position-independent.** Content does not depend on haystack ordinal.

    Args:
        session: The session's ordered turns (each a ``{"role", "content", ...}``
            mapping). Only ``role`` and ``content`` are read.

    Returns:
        A deterministic, evidence-free, position-independent session id.

    """
    hasher = hashlib.sha256()
    for turn in session:
        role = str(turn.get("role", ""))
        content = str(turn.get("content") or "")
        # Length-prefix each field so concatenation is unambiguous (no delimiter
        # injection can make two distinct sessions serialise identically).
        hasher.update(f"{len(role)}:{role}{len(content)}:{content}".encode())
    return f"sess-{hasher.hexdigest()}"


def _neutral_unit_id(neutral_session_id: str, i_turn: int) -> str:
    """Build the neutral, opaque adapter-facing unit id for a user turn.

    Encodes the session's own (neutralised) identity plus the within-session turn
    index — no haystack ordinal, no evidence/answer status. This is the only id an
    adapter ever sees, so it can infer neither which turns are gold nor where the
    session sits in the haystack. It is purely a key: the system ranks by the
    embedded turn *text*, never by id, so neutralising it cannot change ranking.

    Args:
        neutral_session_id: The session's neutral id (:func:`_neutral_session_id`).
        i_turn: Zero-based index of the turn within its session.

    Returns:
        A deterministic, evidence-free, position-independent unit id
        (``"<neutral_session_id>#<i_turn>"``).

    """
    return f"{neutral_session_id}#{i_turn}"


def load_questions(dataset_path: Path, *, limit: int | None = None) -> list[Question]:
    """Load LongMemEval questions and build the per-question corpus.

    The corpus exposes only what an adapter may legitimately see: a NEUTRAL,
    evidence-free ``unit_id`` plus the turn text. Evidence flags and the official
    (evidence-encoding) corpus id are consumed here, kept runner-internal in
    :attr:`Question.id_map`, and are NOT placed on :class:`CorpusTurn`.

    Args:
        dataset_path: Path to the (public) LongMemEval JSON dataset.
        limit: Optional head-slice to the first N questions (partial runs). ``0`` is
            a valid limit (zero questions); ``None`` means no limit.

    Returns:
        The loaded questions with their corpora.

    Raises:
        DatasetError: If an item's ``haystack_sessions`` and ``haystack_session_ids``
            have mismatched lengths.

    Note:
        A byte-identical duplicate session in one haystack (a legitimate
        LongMemEval-S case) is INGESTED, not rejected: its turns are appended to
        the corpus (mirroring the official flat-list, so FTS corpus statistics
        match the ingest-both protocol), while ``id_map`` and the assembly session
        content stay first-occurrence. The adapter collapses the duplicate
        content-derived ``unit_id`` to a single unique entry at rank time.

    """
    raw: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    if limit is not None:
        raw = raw[:limit]

    questions: list[Question] = []
    for item in raw:
        session_ids: list[str] = list(item.get("haystack_session_ids", []))
        sessions: list[list[dict[str, Any]]] = list(item.get("haystack_sessions", []))
        dates: list[str] = list(item.get("haystack_dates", []))
        qid = item.get("question_id", "<unknown>")
        if len(sessions) != len(session_ids):
            msg = (
                f"question {qid!r}: haystack_sessions ({len(sessions)}) and "
                f"haystack_session_ids ({len(session_ids)}) length mismatch"
            )
            raise DatasetError(msg)

        corpus: list[CorpusTurn] = []
        id_map: dict[str, str] = {}
        session_content: dict[str, SessionContent] = {}
        for s_idx, (session, s_id) in enumerate(zip(sessions, session_ids, strict=True)):
            date_str = dates[s_idx] if s_idx < len(dates) else ""
            # Stable, opaque, non-positional session id derived from the session's
            # CONTENT (its ordered turns) — no haystack ordinal, no id-string marker.
            neutral_session_id = _neutral_session_id(session)
            # Runner-internal full session content for assembly: role + content only
            # (the evidence `has_answer` flag is popped, exactly as upstream does).
            # FIRST-occurrence wins: a byte-identical duplicate session may carry a
            # different haystack date; keep the first so the assembled date stays
            # consistent with the first-occurrence corpus/id_map entries.
            if neutral_session_id not in session_content:
                session_content[neutral_session_id] = SessionContent(
                    date=date_str,
                    turns=[
                        {"role": str(t.get("role", "")), "content": str(t.get("content") or "")}
                        for t in session
                    ],
                )
            for i_turn, turn in enumerate(session):
                if turn.get("role") != "user":
                    continue
                text = (turn.get("content") or "").strip()
                neutral_id = _neutral_unit_id(neutral_session_id, i_turn)
                # A repeated neutral_id means two byte-identical sessions occur in
                # this haystack — a LEGITIMATE LongMemEval-S case (~13 questions).
                # INGEST BOTH: append the duplicate CorpusTurn so the corpus mirrors
                # the official flat-list by construction. This matters because
                # engrava's `search_hybrid` fuses FTS (corpus-statistics-sensitive)
                # with vector search, so ingest-both vs skip-dup could diverge on the
                # dup questions; the adapter stores both copies via its engrava-
                # internal `#dup{position}` store-id salt and collapses to a unique
                # unit_id at rank time. The adapter-facing unit_id stays the content-
                # derived neutral id (no ordinal salt) — the leakage guard holds.
                # id_map + session_content stay FIRST-occurrence (guarded); a ranked
                # duplicate unit_id resolves deterministically to its first turn.
                if neutral_id not in id_map:
                    # Official (evidence-encoding) id stays runner-internal in id_map.
                    id_map[neutral_id] = _official_corpus_id(
                        s_id, i_turn, has_answer=bool(turn.get("has_answer", False))
                    )
                corpus.append(
                    CorpusTurn(
                        unit_id=neutral_id,
                        text=text,
                        session_id=neutral_session_id,
                        turn_index=i_turn,
                        timestamp=date_str,
                    )
                )
        questions.append(
            Question(
                question_id=item.get("question_id", ""),
                question_type=item.get("question_type", ""),
                question=item.get("question", ""),
                question_date=str(item.get("question_date", "")),
                answer=item.get("answer", ""),
                corpus=corpus,
                id_map=id_map,
                sessions=session_content,
            )
        )
    return questions


def _split_unit_id(unit_id: str) -> tuple[str, int] | None:
    """Split a neutral ``<session_id>#<turn_index>`` unit id into its parts.

    Args:
        unit_id: The adapter-facing neutral unit id.

    Returns:
        ``(neutral_session_id, turn_index)`` or ``None`` if it is malformed.

    """
    session_part, sep, turn_part = unit_id.rpartition("#")
    if not sep or not turn_part.isdigit():
        return None
    return session_part, int(turn_part)


def assemble_context(ranked_unit_ids: list[str], question: Question, *, top_k: int) -> str:
    """Assemble the official reader history from the ranked units (runner-owned).

    Replicates the official ``flat-turn`` assembly (run_generation.py at the pinned
    commit): for each ranked user turn, **round-expand** to ``[turn, next_turn]``
    (the user turn + its assistant reply, capped at the session boundary) using the
    runner-internal session content; collect ``(session_date, round_turns)`` pairs;
    then keep top-K by rank, chronologically re-sort, JSON-format, and tiktoken-
    truncate via :mod:`official_reader`.

    This is runner-owned and reads NO gold/answer/evidence: ``question.sessions``
    carries only role + content (``has_answer`` popped at load).

    Args:
        ranked_unit_ids: Unit ids from the adapter, best first.
        question: The question (for its runner-internal ``sessions`` content).
        top_k: The official ``topk_context``.

    Returns:
        The assembled, truncated history string for the reader.

    """
    rounds: list[tuple[str, list[dict[str, str]]]] = []
    for unit_id in ranked_unit_ids:
        parsed = _split_unit_id(unit_id)
        if parsed is None:
            continue
        neutral_session_id, turn_index = parsed
        session = question.sessions.get(neutral_session_id)
        if session is None or turn_index >= len(session.turns):
            continue
        # Round-expansion: the retrieved turn + the immediately following turn
        # (capped at the session boundary).
        round_turns = [session.turns[turn_index]]
        if turn_index + 1 < len(session.turns):
            round_turns.append(session.turns[turn_index + 1])
        rounds.append((session.date, round_turns))

    return official_reader.assemble_history(rounds, top_k=top_k)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(
    questions: list[Question],
    adapter: MemoryAdapter,
    reader: Reader,
    judge: Judge,
    scorer: Scorer,
    *,
    run_ctx: RunContext,
    top_k: int,
) -> tuple[dict[str, Any], list[RunRecord]]:
    """Run the uniform pipeline over all questions and aggregate metrics.

    Args:
        questions: The loaded questions.
        adapter: The plugged-in memory adapter (ingest + search).
        reader: The runner-owned reader.
        judge: The runner-owned judge.
        scorer: The runner-owned official scorer aggregator.
        run_ctx: The read-only run context handed to the adapter.
        top_k: Retrieval breadth.

    Returns:
        A ``(metrics, records)`` pair: the ``metrics`` object (results-schema shape)
        and the per-question :class:`RunRecord` list for the reproduction artifact.

    """
    judgments: list[Judgment] = []
    records: list[RunRecord] = []
    for q in questions:
        adapter.ingest(q.corpus, run_ctx=run_ctx)
        ranked = adapter.search(q.question, top_k=top_k)
        ranked_unit_ids = [r.unit_id for r in ranked]
        context = assemble_context(ranked_unit_ids, q, top_k=top_k)
        answer = reader.answer(q.question, context, question_date=q.question_date)
        correct = judge.score(
            q.question,
            q.answer,
            answer,
            question_type=q.question_type,
            question_id=q.question_id,
        )
        judgments.append(
            Judgment(
                question_id=q.question_id,
                question_type=q.question_type,
                correct=correct,
            )
        )
        # Map the ranked neutral ids to official corpus ids via id_map (the one
        # consumer of id_map); used only for the runner-side reproduction artifact.
        ranked_official = [q.id_map[u] for u in ranked_unit_ids if u in q.id_map]
        records.append(
            RunRecord(
                question_id=q.question_id,
                question_type=q.question_type,
                hypothesis=answer,
                correct=correct,
                ranked_official_ids=ranked_official,
            )
        )
    return scorer.aggregate(judgments), records


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a runner config JSON.

    Args:
        config_path: Path to a config file under ``config/``.

    Returns:
        The parsed config dict.

    """
    parsed: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return parsed


def apply_model_config(
    config: dict[str, Any],
    *,
    models: str,
    endpoint: str | None = None,
    reader_endpoint: str | None = None,
    judge_endpoint: str | None = None,
    reader_model: str | None = None,
    judge_model: str | None = None,
    reader_max_tokens: int | None = None,
    reader_api_key_env: str | None = None,
) -> None:
    """Resolve the effective reader/judge config in place for the chosen backend.

    For ``--models ollama`` the local ``ollama`` config block is promoted into the
    effective ``reader``/``judge`` blocks, so both the client factory and the
    emitted result read the ACTUAL local model + endpoint (the row then lands in a
    non-canonical segment, never the canonical gpt-4o headline). CLI overrides are
    then applied. ``mock`` and ``openai`` are unchanged unless an explicit override
    is given; the canonical default is never silently altered.

    Override precedence: ``endpoint`` sets BOTH the reader and judge endpoint (the
    symmetric ``--models ollama`` case); the finer ``reader_endpoint`` /
    ``judge_endpoint`` are applied AFTER it, so they win when both are given. This
    lets a bring-your-own-model run point the reader at one endpoint (e.g. a cheap
    hosted reader) while the judge keeps the canonical one.

    Args:
        config: The loaded config (mutated in place).
        models: The selected backend (``openai``/``ollama``/``mock``).
        endpoint: Optional reader+judge endpoint override (applies to both).
        reader_endpoint: Optional reader-only endpoint override (wins over
            ``endpoint`` for the reader).
        judge_endpoint: Optional judge-only endpoint override (wins over
            ``endpoint`` for the judge).
        reader_model: Optional reader model id override.
        judge_model: Optional judge model id override.
        reader_max_tokens: Optional reader generation length override, written to
            ``config["reader"]["sampling"]["max_tokens"]``. When ``None`` the reader
            keeps the official cot generation length (800).
        reader_api_key_env: Optional name of the env var holding the reader's API key
            (written to ``config["reader"]["api_key_env"]``). Lets a third-party
            reader endpoint use a different key env than the canonical judge. The key
            value is never stored — only the env-var name.

    """
    if models == "ollama":
        ollama_cfg = config.get(
            "ollama",
            {
                "reader": {
                    "model": "gemma3:4b",
                    "snapshot": "gemma3:4b",
                    "endpoint": "http://localhost:11434",
                    "sampling": {"temperature": 0.0},
                },
                "judge": {
                    "model": "gemma3:4b",
                    "snapshot": "gemma3:4b",
                    "endpoint": "http://localhost:11434",
                },
            },
        )
        config["ollama"] = ollama_cfg
        # Promote so build_result records the actual local model/endpoint.
        config["reader"] = dict(ollama_cfg["reader"])
        config["judge"] = dict(ollama_cfg["judge"])

    if endpoint is not None:
        config["reader"]["endpoint"] = endpoint
        config["judge"]["endpoint"] = endpoint
    if reader_endpoint is not None:
        config["reader"]["endpoint"] = reader_endpoint
    if judge_endpoint is not None:
        config["judge"]["endpoint"] = judge_endpoint
    if reader_model is not None:
        config["reader"]["model"] = reader_model
        config["reader"]["snapshot"] = reader_model
    if judge_model is not None:
        config["judge"]["model"] = judge_model
        config["judge"]["snapshot"] = judge_model
    if reader_max_tokens is not None:
        sampling = dict(config["reader"].get("sampling", {"temperature": 0.0}))
        sampling["max_tokens"] = reader_max_tokens
        config["reader"]["sampling"] = sampling
    if reader_api_key_env is not None:
        config["reader"]["api_key_env"] = reader_api_key_env


# --------------------------------------------------------------------------- #
# Model + adapter factories (config-driven; equal footing preserved)
# --------------------------------------------------------------------------- #
def _build_openai_compatible_reader_judge(
    reader_cfg: Mapping[str, Any], judge_cfg: Mapping[str, Any]
) -> tuple[Reader, Judge]:
    """Build OpenAI-compatible reader + judge from their config blocks.

    Shared by the canonical ``openai`` path and the local ``ollama`` path — both use
    the OpenAI Chat Completions API; only the endpoint/model differ.

    Each block may declare its own ``api_key_env`` (default ``OPENAI_API_KEY``), so a
    reader on a third-party endpoint can read a different key env than the judge
    (e.g. an OpenRouter reader keyed by ``OPENROUTER_API_KEY`` judged by an OpenAI
    ``gpt-4o`` keyed by ``OPENAI_API_KEY``). The key value itself is never in config —
    only the env-var name.

    Args:
        reader_cfg: The reader block (``snapshot``, ``endpoint``, optional
            ``sampling`` and ``api_key_env``).
        judge_cfg: The judge block (``snapshot``, ``endpoint``, optional
            ``api_key_env``).

    Returns:
        A ``(reader, judge)`` pair.

    """
    from runners.longmemeval.openai_models import OpenAIJudge, OpenAIReader  # noqa: PLC0415

    reader = OpenAIReader(
        model_snapshot=reader_cfg["snapshot"],
        endpoint=reader_cfg["endpoint"],
        sampling=reader_cfg.get("sampling", {"temperature": 0.0}),
        api_key_env=reader_cfg.get("api_key_env", "OPENAI_API_KEY"),
    )
    judge = OpenAIJudge(
        model_snapshot=judge_cfg["snapshot"],
        endpoint=judge_cfg["endpoint"],
        api_key_env=judge_cfg.get("api_key_env", "OPENAI_API_KEY"),
    )
    return reader, judge


def build_reader_judge(config: Mapping[str, Any], *, models: str) -> tuple[Reader, Judge]:
    """Build the reader + judge from config.

    Args:
        config: The runner config (declares reader/judge model+endpoint).
        models: ``"openai"`` (canonical, paid, OpenAI-direct), ``"ollama"`` (a local
            OpenAI-compatible server, free), or ``"mock"`` (free, offline smoke).

    Returns:
        A ``(reader, judge)`` pair.

    Raises:
        ValueError: If ``models`` is not ``"openai"``, ``"ollama"``, or ``"mock"``.

    """
    if models == "mock":
        from runners.longmemeval.mock_models import MockJudge, MockReader  # noqa: PLC0415

        return MockReader(), MockJudge()
    if models in ("openai", "ollama"):
        # Both use the OpenAI Chat Completions API. The effective reader/judge blocks
        # are resolved by apply_model_config (ollama promotes its local block +
        # overrides into config["reader"]/["judge"]), so the same blocks the result
        # records are the blocks the clients use.
        return _build_openai_compatible_reader_judge(config["reader"], config["judge"])
    msg = f"unknown models mode: {models!r} (use 'openai', 'ollama', or 'mock')"
    raise ValueError(msg)


def build_engrava_adapter(config: Mapping[str, Any]) -> MemoryAdapter:
    """Build the public-engrava adapter from config (requires ``engrava``).

    Args:
        config: The runner config (declares the embedder spec).

    Returns:
        The Engrava memory adapter.

    """
    from adapters.engrava_adapter import (  # noqa: PLC0415 - optional engrava dep
        EngravaAdapter,
        create_embedding_provider,
    )

    provider = create_embedding_provider(config["embedder_spec"])
    return EngravaAdapter(provider)


def run_and_emit(
    *,
    config: Mapping[str, Any],
    questions: list[Question],
    adapter: MemoryAdapter,
    reader: Reader,
    judge: Judge,
    result_id: str,
    date: str,
    partial: bool,
    emit_result: bool,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the pipeline, aggregate official metrics, optionally emit + validate.

    Args:
        config: The runner config.
        questions: Loaded questions.
        adapter: The memory adapter.
        reader: The reader (real or mock).
        judge: The judge (real or mock).
        result_id: Stable result id.
        date: Run date (``YYYY-MM-DD``).
        partial: Whether this is a head-sliced (non-headline) run.
        emit_result: If ``True``, write + validate ``results/<result_id>.json``.
        results_dir: Optional override for the results directory (tests).

    Returns:
        The computed ``metrics`` object.

    """
    from runners.longmemeval import artifact as artifact_mod  # noqa: PLC0415
    from runners.longmemeval import emit as emit_mod  # noqa: PLC0415
    from runners.longmemeval.scorer import OfficialScorer  # noqa: PLC0415

    scorer = OfficialScorer(scorer_version=config["scorer_version"])
    run_ctx = RunContext(
        top_k=config["top_k"],
        granularity=config["granularity"],
        embedder_spec=config["embedder_spec"],
    )
    try:
        metrics, records = run(
            questions, adapter, reader, judge, scorer, run_ctx=run_ctx, top_k=config["top_k"]
        )
    finally:
        # Release adapter-held resources (persistent event loop, DB connection).
        # The MemoryAdapter protocol does not mandate close(); call it when present.
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    # The reproduction artifact is part of every emitted result: the row records the
    # checksum, and emission writes the bundle beside the row under results/.
    bundle = artifact_mod.build_artifact(config=config, records=records, result_id=result_id)
    checksum = artifact_mod.artifact_checksum(bundle)

    if emit_result:
        row = emit_mod.build_result(
            config=config,
            metrics=metrics,
            n=len(questions),
            result_id=result_id,
            date=date,
            verification_status="unverified",
            partial=partial,
            artifact_checksum=checksum,
        )
        target_dir = results_dir if results_dir is not None else emit_mod.RESULTS_DIR
        emit_mod.write_artifact_and_validate(
            row,
            bundle,
            results_dir=target_dir,
        )
        print(  # noqa: T201
            f"Wrote reproduction artifact: {emit_mod.artifact_reference(row)} ({checksum})"
        )
        print(f"Emitted + validated: {emit_mod.result_reference(row)}")  # noqa: T201
    return metrics


@dataclass(frozen=True)
class DatasetSelection:
    """Resolved dataset path plus a non-sensitive source label for logs."""

    path: Path
    source: str


def _resolve_dataset(explicit: Path | None) -> DatasetSelection | None:
    """Resolve the dataset from CLI, env var, or repo-local cache."""
    if explicit is not None:
        return DatasetSelection(explicit, "--dataset")
    env_value = os.environ.get(DEFAULT_DATASET_ENV, "").strip()
    if env_value:
        return DatasetSelection(Path(env_value), DEFAULT_DATASET_ENV)
    if DEFAULT_DATASET_PATH.is_file():
        return DatasetSelection(DEFAULT_DATASET_PATH, "repo-local cache")
    return None


def _default_dataset_hint() -> str:
    """Return a non-sensitive description of the default dataset cache path."""
    try:
        return DEFAULT_DATASET_PATH.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "the repo-local cache path"


def _default_result_id(date: str) -> str:
    """Return the default result id for an official Engrava run."""
    from runners.longmemeval import emit as emit_mod  # noqa: PLC0415

    return f"lme-s_engrava_{emit_mod.engrava_version()}_{date}"


def _suppress_known_dependency_warnings() -> None:
    """Keep CLI output free of dependency warning file paths for known benign noise."""
    warnings.filterwarnings(
        "ignore",
        message=r".*get_sentence_embedding_dimension.*",
        category=FutureWarning,
        module=r"engrava\.embeddings\.sentence_transformer",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongMemEval uniform runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config" / "default.json",
        help="Runner config JSON (default: config/default.json).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=False,
        help=(
            "Path to the public LongMemEval dataset JSON. Defaults to the "
            f"{DEFAULT_DATASET_ENV} env var, then {_default_dataset_hint()}."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Head-slice to the first N questions (partial run; never a headline).",
    )
    parser.add_argument(
        "--models",
        choices=["openai", "ollama", "mock"],
        default="openai",
        help=(
            "Reader/judge backend: 'openai' (canonical, paid), 'ollama' (a local "
            "OpenAI-compatible server, free), or 'mock' (free offline smoke)."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Override the reader+judge endpoint (e.g. 'http://<host>:11434' for a "
            "local Ollama server). Applies to the active --models backend; recorded "
            "in the result so a local run lands in a non-canonical segment."
        ),
    )
    parser.add_argument(
        "--reader-endpoint",
        default=None,
        help=(
            "Override ONLY the reader endpoint (e.g. 'https://openrouter.ai/api/v1' "
            "for a hosted bring-your-own reader), leaving the judge endpoint alone. "
            "Wins over --endpoint for the reader; recorded in the result."
        ),
    )
    parser.add_argument(
        "--judge-endpoint",
        default=None,
        help=(
            "Override ONLY the judge endpoint, leaving the reader endpoint alone. "
            "Wins over --endpoint for the judge; recorded in the result."
        ),
    )
    parser.add_argument(
        "--reader-model",
        default=None,
        help="Override the reader model id (e.g. 'gemma3:4b' for --models ollama).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the judge model id (e.g. 'gemma3:4b' for --models ollama).",
    )
    parser.add_argument(
        "--reader-max-tokens",
        type=int,
        default=None,
        help=(
            "Override the reader generation length (config reader.sampling.max_tokens). "
            "Default: the official cot generation length (800). Raise it for a reader "
            "that needs more room (e.g. a reasoning model)."
        ),
    )
    parser.add_argument(
        "--reader-api-key-env",
        default=None,
        help=(
            "Name of the env var holding the reader's API key (e.g. "
            "'OPENROUTER_API_KEY' for a reader on a third-party endpoint), so the "
            "reader can authenticate against a different provider than the judge. "
            "Default: OPENAI_API_KEY. Only the env-var NAME is passed, never the key."
        ),
    )
    parser.add_argument(
        "--result-id",
        default=None,
        help="Stable result id (default: derived from split + engrava version + date).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Run date YYYY-MM-DD (default: today, UTC).",
    )
    parser.add_argument(
        "--emit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write + validate the result row and sibling artifact bundle after the "
            "run (default: true; use --no-emit for exploratory runs)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run the free offline smoke path: fixture dataset, mock reader/judge, "
            "local MiniLM embedder, limit=2, and no result emission."
        ),
    )
    parser.add_argument(
        "--embedder-spec",
        default=None,
        help=(
            "Override the config embedder spec (e.g. 'local:all-MiniLM-L12-v2' for a "
            "free, offline embedder). The override is written into the result so "
            "provenance stays honest. Default: the config value."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory to emit the result row + sibling artifact bundle into "
            "(default: the repo 'results/' tree). Point it elsewhere to capture a "
            "run's output without touching the canonical results tree — e.g. an "
            "isolated smoke run. Passing it also makes --smoke emit (it otherwise "
            "emits nothing)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load config + dataset, run the pipeline, optionally emit.

    With ``--models openai`` this makes PAID reader+judge calls (the maintainers'
    canonical run). With ``--models mock`` it runs the full pipeline offline with
    no spend (the local smoke path). Either way nothing is fabricated.

    Args:
        argv: Optional argument vector (for testing).

    Returns:
        Process exit code.

    """
    from datetime import datetime  # noqa: PLC0415

    args = _parse_args(argv)
    _suppress_known_dependency_warnings()
    config = load_config(args.config)
    if args.smoke:
        args.dataset = SMOKE_DATASET
        args.models = "mock"
        args.limit = 2
        args.embedder_spec = SMOKE_EMBEDDER_SPEC
        # The smoke path is a free wiring check that emits nothing by default; an
        # explicit --results-dir opts in to capturing its bundle (into an isolated tree).
        if args.results_dir is None:
            args.emit = False
    if args.embedder_spec:
        # Honest override: change the spec AND the embedder label written to the row.
        config["embedder_spec"] = args.embedder_spec
        config["embedder"] = args.embedder_spec.partition(":")[2] or args.embedder_spec
        config["embedder_endpoint"] = (
            "local" if args.embedder_spec.startswith("local:") else config["embedder_endpoint"]
        )
    # Resolve the effective reader/judge config for the chosen backend (ollama
    # promotes its local block; CLI overrides apply). The emitted row records these.
    apply_model_config(
        config,
        models=args.models,
        endpoint=args.endpoint,
        reader_endpoint=args.reader_endpoint,
        judge_endpoint=args.judge_endpoint,
        reader_model=args.reader_model,
        judge_model=args.judge_model,
        reader_max_tokens=args.reader_max_tokens,
        reader_api_key_env=args.reader_api_key_env,
    )
    print(  # noqa: T201 - CLI user feedback
        "Loaded runner config: "
        f"benchmark={config['benchmark']} split={config['split']} "
        f"top_k={config['top_k']} embedder={config['embedder']} models={args.models} "
        f"reader={config['reader']['model']}@{config['reader']['endpoint']}"
    )
    dataset = _resolve_dataset(args.dataset)
    if dataset is None:
        print(  # noqa: T201
            "No dataset found. Pass --dataset, set "
            f"{DEFAULT_DATASET_ENV}, or place the public LongMemEval-S file at "
            f"{_default_dataset_hint()}."
        )
        return 2
    if not dataset.path.is_file():
        print(f"Dataset path from {dataset.source} does not exist or is not a file.")  # noqa: T201
        return 2

    questions = load_questions(dataset.path, limit=args.limit)
    print(f"Loaded {len(questions)} questions from dataset source: {dataset.source}.")  # noqa: T201

    date = args.date or datetime.now(tz=UTC).strftime("%Y-%m-%d")
    result_id = args.result_id or _default_result_id(date)
    reader, judge = build_reader_judge(config, models=args.models)
    adapter = build_engrava_adapter(config)

    metrics = run_and_emit(
        config=config,
        questions=questions,
        adapter=adapter,
        reader=reader,
        judge=judge,
        result_id=result_id,
        date=date,
        partial=args.limit is not None,
        emit_result=args.emit,
        results_dir=args.results_dir.resolve() if args.results_dir is not None else None,
    )
    print(  # noqa: T201
        f"overall_micro={metrics['overall_micro']:.4f} macro={metrics['macro']:.4f} "
        f"abstention={metrics['abstention']['accuracy']:.4f} (n={metrics['abstention']['n']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
