"""$0 dataset pre-flight scan for the LongMemEval runner.

Scans every question's USER turns and REPORTS (never crashes) the dataset shapes
that affect a paid run, so a maintainer sees them before spending:

* **Over-long turns** — user turns whose embed payload exceeds the embedding
  model's ``8192`` cl100k-token cap. These are TRUNCATED at ingest (deterministic
  first-8192-token prefix), matching the official protocol — reported, not fatal.
* **Empty turns** — stripped-empty user turns. These are SKIPPED at ingest (never
  embedded), so they never reach the API — reported, not fatal.
* **Duplicate sessions** — byte-identical repeated sessions in one haystack (a
  legitimate LongMemEval-S case, already handled by the runner via ingest-both +
  first-occurrence collapse) — purely informational.

The scan uses the SAME cl100k_base encoding and 8192 cap as the adapter's
truncation, so its "will be truncated" count matches what ingest actually does.

Usage:
    python scripts/preflight.py <dataset.json> [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Mirror of the adapter's embedding limits (adapters/engrava_adapter.py).
_MAX_EMBED_TOKENS = 8192
_EMBED_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class LongTurn:
    """A user turn whose content exceeds the embedding token cap.

    Attributes:
        question_id: The question the turn belongs to.
        session_index: Zero-based haystack index of the turn's session.
        turn_index: Zero-based turn index within its session.
        token_count: The turn's cl100k token count (before truncation).

    """

    question_id: str
    session_index: int
    turn_index: int
    token_count: int


@dataclass(frozen=True)
class EmptyTurn:
    """A stripped-empty user turn (skipped at ingest).

    Attributes:
        question_id: The question the turn belongs to.
        session_index: Zero-based haystack index of the turn's session.
        turn_index: Zero-based turn index within its session.

    """

    question_id: str
    session_index: int
    turn_index: int


@dataclass(frozen=True)
class DuplicateSession:
    """A byte-identical duplicate session within one question's haystack.

    Attributes:
        question_id: The question the duplicate belongs to.
        duplicate_count: Number of extra (byte-identical) session copies beyond
            the first occurrence.

    """

    question_id: str
    duplicate_count: int


@dataclass
class PreflightReport:
    """The aggregate result of a dataset pre-flight scan.

    Attributes:
        question_count: Number of questions scanned.
        long_turns: User turns over the token cap (will be truncated at ingest).
        empty_turns: Stripped-empty user turns (skipped at ingest).
        duplicate_sessions: Questions with a byte-identical duplicate session.

    """

    question_count: int = 0
    long_turns: list[LongTurn] = field(default_factory=list)
    empty_turns: list[EmptyTurn] = field(default_factory=list)
    duplicate_sessions: list[DuplicateSession] = field(default_factory=list)


def _session_signature(session: list[dict[str, Any]]) -> str:
    """Return a content signature for a session (role + content per turn).

    Args:
        session: The session's ordered turns.

    Returns:
        A hex digest that is identical for two byte-identical sessions.

    """
    hasher = hashlib.sha256()
    for turn in session:
        role = str(turn.get("role", ""))
        content = str(turn.get("content") or "")
        hasher.update(f"{len(role)}:{role}{len(content)}:{content}".encode())
    return hasher.hexdigest()


def scan_dataset(raw: list[dict[str, Any]]) -> PreflightReport:
    """Scan loaded dataset items for embedding-limit and duplicate shapes.

    Pure function (no I/O): takes the parsed dataset list and returns a typed
    :class:`PreflightReport`. Only USER turns are considered (the runner indexes
    user turns only), matching the adapter's ingest granularity.

    Args:
        raw: The parsed LongMemEval dataset (a list of question items).

    Returns:
        A :class:`PreflightReport` with per-issue detail.

    """
    import tiktoken  # noqa: PLC0415 - lazy: only needed when scanning

    enc = tiktoken.get_encoding(_EMBED_ENCODING)
    report = PreflightReport(question_count=len(raw))

    for item in raw:
        qid = str(item.get("question_id", "<unknown>"))
        sessions: list[list[dict[str, Any]]] = list(item.get("haystack_sessions", []))

        signatures: set[str] = set()
        dup_count = 0
        for s_idx, session in enumerate(sessions):
            signature = _session_signature(session)
            if signature in signatures:
                dup_count += 1
            else:
                signatures.add(signature)

            for i_turn, turn in enumerate(session):
                if turn.get("role") != "user":
                    continue
                text = (turn.get("content") or "").strip()
                if not text:
                    report.empty_turns.append(EmptyTurn(qid, s_idx, i_turn))
                    continue
                token_count = len(enc.encode(text))
                if token_count > _MAX_EMBED_TOKENS:
                    report.long_turns.append(LongTurn(qid, s_idx, i_turn, token_count))

        if dup_count:
            report.duplicate_sessions.append(DuplicateSession(qid, dup_count))

    return report


def format_report(report: PreflightReport) -> str:
    """Render a human-readable pre-flight summary.

    Args:
        report: The scan result.

    Returns:
        A multi-line summary ending in a clear safe-to-run line.

    """
    lines: list[str] = []
    lines.append(f"Pre-flight scan: {report.question_count} question(s).")

    long_qids = sorted({t.question_id for t in report.long_turns})
    lines.append(
        f"Over-long user turns (>{_MAX_EMBED_TOKENS} cl100k tokens, "
        f"truncated at ingest): {len(report.long_turns)} "
        f"in {len(long_qids)} question(s)."
    )
    lines.extend(
        f"  - {turn.question_id}: session {turn.session_index}, "
        f"turn {turn.turn_index} = {turn.token_count} tokens"
        for turn in report.long_turns
    )

    empty_qids = sorted({t.question_id for t in report.empty_turns})
    lines.append(
        f"Empty user turns (skipped at ingest): {len(report.empty_turns)} "
        f"in {len(empty_qids)} question(s)."
    )
    lines.extend(
        f"  - {turn.question_id}: session {turn.session_index}, turn {turn.turn_index}"
        for turn in report.empty_turns
    )

    lines.append(
        f"Byte-identical duplicate sessions (handled: ingest-both + collapse): "
        f"{len(report.duplicate_sessions)} question(s)."
    )

    lines.append(
        f">> {len(report.long_turns)} input(s) will be truncated / "
        f"{len(report.empty_turns)} empty turn(s) skipped — safe to run."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: scan a dataset file and print the pre-flight summary.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` — pre-flight only reports, never fails a run).

    """
    parser = argparse.ArgumentParser(description="$0 LongMemEval dataset pre-flight scan.")
    parser.add_argument("dataset", type=Path, help="Path to the LongMemEval JSON dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Scan only the first N questions.")
    args = parser.parse_args(argv)

    raw: list[dict[str, Any]] = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit is not None:
        raw = raw[: args.limit]

    report = scan_dataset(raw)
    print(format_report(report))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
