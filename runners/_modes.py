"""Shared, benchmark-agnostic run-mode surface for engrava-benchmark runners.

Every benchmark runner exposes the same four run modes with identical core
parametrization, so a code change or a benchmark test is verified the same way
everywhere. This module is the single source of truth for the mode taxonomy and
the shared argparse parent; a per-benchmark runner supplies only what genuinely
differs (dataset path, canonical reader/judge identity, ``top_k``).

The four modes
--------------
========= ================= ================ ============ ========================
mode      embedder          reader / judge   spend        purpose
========= ================= ================ ============ ========================
smoke     local (offline)   mock             $0 offline   tiny CI wiring check
plumbing  local (offline)   mock             $0 offline   full real-shape pipeline
retrieval real              mock, no judge   embeddings   ranked retrieval log + diff
score     real              real             paid         the official number
========= ================= ================ ============ ========================

Only ``score`` yields an official (publishable) number; the other three modes are
non-official by construction (they mock or discard the reader/judge), so they can
never contaminate a canonical row.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


class RunMode(StrEnum):
    """The uniform run mode selected by ``--mode`` across every benchmark."""

    SMOKE = "smoke"
    PLUMBING = "plumbing"
    RETRIEVAL = "retrieval"
    SCORE = "score"


class EmbedderKind(StrEnum):
    """Which embedder a mode resolves to."""

    LOCAL = "local"
    """A deterministic, offline, $0 local embedder."""
    REAL = "real"
    """The benchmark's real (config-declared) embedder."""


class ModelKind(StrEnum):
    """Which reader/judge backend a mode resolves to."""

    MOCK = "mock"
    """The offline, $0, non-official mock reader/judge."""
    REAL = "real"
    """The canonical (paid) reader/judge."""


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """The benchmark-agnostic resolution of a single run mode.

    Attributes:
        mode: The run mode this spec describes.
        embedder: Whether the mode uses the local (offline) or the real embedder.
        reader_judge: Whether the reader/judge is mocked or the real paid backend.
        runs_reader: Whether a REAL reader is exercised (only ``score``). The
            non-score modes either mock the reader (``smoke``/``plumbing``) or
            discard its output (``retrieval``), so its answer is never meaningful.
        runs_judge: Whether the mode judges the reader's answer at all. ``score``
            judges with the canonical judge and ``smoke``/``plumbing`` exercise the
            judge seam with the mock judge as part of their wiring check;
            ``retrieval`` defines the reader output as discarded, so it judges
            nothing — a retrieval run must never depend on (or pay for) a verdict
            it throws away.
        emits_official_row: Whether the mode may emit a canonical/publishable
            result row. Only ``score`` may; the other three are non-official.
        emits_retrieval_log: Whether the mode emits a ranked retrieval log for a
            deterministic retrieval-diff (only ``retrieval``).

    Raises:
        ValueError: When the fields contradict each other. The combinations below are
            not merely unused — they are unpublishable, and a stored boolean that
            downstream code faithfully honours is exactly how an unpublishable run
            would acquire a publishable row.

    """

    mode: RunMode
    embedder: EmbedderKind
    reader_judge: ModelKind
    runs_reader: bool
    runs_judge: bool
    emits_official_row: bool
    emits_retrieval_log: bool

    def __post_init__(self) -> None:
        """Reject internally contradictory specifications at construction.

        Raises:
            ValueError: When officialness, reader use or judging disagree.

        """
        if self.emits_official_row and (
            self.reader_judge is not ModelKind.REAL
            or self.embedder is not EmbedderKind.REAL
            or not self.runs_reader
            or not self.runs_judge
        ):
            msg = (
                f"{self.mode}: a mode may be official only when the embedder, the reader and "
                "the judge are all real and exercised — a mocked or discarded component cannot "
                "produce a publishable row"
            )
            raise ValueError(msg)
        if self.runs_reader and not self.runs_judge:
            msg = (
                f"{self.mode}: a mode that exercises a real reader must judge its answer; an "
                "unjudged reader answer is paid for and then discarded"
            )
            raise ValueError(msg)


_MODE_TABLE: dict[RunMode, ModeSpec] = {
    RunMode.SMOKE: ModeSpec(
        mode=RunMode.SMOKE,
        embedder=EmbedderKind.LOCAL,
        reader_judge=ModelKind.MOCK,
        runs_reader=False,
        runs_judge=True,
        emits_official_row=False,
        emits_retrieval_log=False,
    ),
    RunMode.PLUMBING: ModeSpec(
        mode=RunMode.PLUMBING,
        embedder=EmbedderKind.LOCAL,
        reader_judge=ModelKind.MOCK,
        runs_reader=False,
        runs_judge=True,
        emits_official_row=False,
        emits_retrieval_log=False,
    ),
    RunMode.RETRIEVAL: ModeSpec(
        mode=RunMode.RETRIEVAL,
        embedder=EmbedderKind.REAL,
        reader_judge=ModelKind.MOCK,
        runs_reader=False,
        runs_judge=False,
        emits_official_row=False,
        emits_retrieval_log=True,
    ),
    RunMode.SCORE: ModeSpec(
        mode=RunMode.SCORE,
        embedder=EmbedderKind.REAL,
        reader_judge=ModelKind.REAL,
        runs_reader=True,
        runs_judge=True,
        emits_official_row=True,
        emits_retrieval_log=False,
    ),
}


def resolve_mode(mode: RunMode) -> ModeSpec:
    """Resolve a run mode to its benchmark-agnostic spec.

    Args:
        mode: The selected run mode.

    Returns:
        The :class:`ModeSpec` describing what the mode resolves to.

    """
    return _MODE_TABLE[mode]


def add_mode_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared run-mode CLI surface every benchmark runner composes.

    Contributes only the COMMON flags shared by every benchmark: ``--mode`` and
    ``--baseline``. A runner keeps ownership of its own flags (``--limit``,
    ``--dataset``, ``--results-dir`` ...) and simply calls this to add the shared
    surface; the flags are intentionally minimal so nothing a runner already owns
    is duplicated.

    Args:
        parser: The runner's argument parser to extend.

    Returns:
        The same parser, for chaining.

    """
    parser.add_argument(
        "--mode",
        choices=[m.value for m in RunMode],
        default=None,
        help=(
            "Unified run mode: 'smoke' (tiny offline wiring check), 'plumbing' "
            "(full offline real-shape pipeline), 'retrieval' (real embedder, mock "
            "reader, emit a ranked retrieval log for a deterministic diff), or "
            "'score' (the canonical paid run). Omit to keep the runner's own "
            "explicit flags exactly as before."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Directory (or file) holding a baseline retrieval log to diff against "
            "in 'retrieval' mode. A RED verdict exits non-zero."
        ),
    )
    return parser
