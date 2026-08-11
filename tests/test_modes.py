"""Tests for the shared, benchmark-agnostic run-mode surface (``runners._modes``)."""

from __future__ import annotations

import argparse

import pytest

from runners import _modes
from runners._modes import EmbedderKind, ModelKind, RunMode


def test_run_mode_values() -> None:
    assert [m.value for m in RunMode] == ["smoke", "plumbing", "retrieval", "score"]


def test_every_mode_resolves() -> None:
    for mode in RunMode:
        spec = _modes.resolve_mode(mode)
        assert spec.mode is mode


def test_only_score_is_official_and_runs_reader() -> None:
    for mode in RunMode:
        spec = _modes.resolve_mode(mode)
        is_score = mode is RunMode.SCORE
        assert spec.emits_official_row is is_score
        assert spec.runs_reader is is_score
        assert (spec.reader_judge is ModelKind.REAL) is is_score


def test_only_retrieval_judges_nothing() -> None:
    # retrieval discards the reader answer, so it must not depend on a judgment;
    # every other mode exercises its judge (mock for the offline modes).
    for mode in RunMode:
        spec = _modes.resolve_mode(mode)
        assert spec.runs_judge is (mode is not RunMode.RETRIEVAL)


def test_only_retrieval_emits_a_retrieval_log() -> None:
    for mode in RunMode:
        spec = _modes.resolve_mode(mode)
        assert spec.emits_retrieval_log is (mode is RunMode.RETRIEVAL)


def test_offline_modes_use_local_embedder() -> None:
    for mode in (RunMode.SMOKE, RunMode.PLUMBING):
        assert _modes.resolve_mode(mode).embedder is EmbedderKind.LOCAL
    for mode in (RunMode.RETRIEVAL, RunMode.SCORE):
        assert _modes.resolve_mode(mode).embedder is EmbedderKind.REAL


def test_offline_modes_mock_the_reader_judge() -> None:
    for mode in (RunMode.SMOKE, RunMode.PLUMBING, RunMode.RETRIEVAL):
        assert _modes.resolve_mode(mode).reader_judge is ModelKind.MOCK


def test_mode_spec_is_frozen() -> None:
    spec = _modes.resolve_mode(RunMode.SCORE)
    with pytest.raises(AttributeError):
        spec.mode = RunMode.SMOKE  # type: ignore[misc]


def test_add_mode_arguments_contributes_shared_surface() -> None:
    parser = argparse.ArgumentParser()
    returned = _modes.add_mode_arguments(parser)
    assert returned is parser
    args = parser.parse_args(["--mode", "retrieval", "--baseline", "some/dir"])
    assert args.mode == "retrieval"
    assert str(args.baseline) == "some/dir"


def test_add_mode_arguments_defaults_are_none() -> None:
    parser = argparse.ArgumentParser()
    _modes.add_mode_arguments(parser)
    args = parser.parse_args([])
    assert args.mode is None
    assert args.baseline is None


def test_add_mode_arguments_rejects_unknown_mode() -> None:
    parser = argparse.ArgumentParser()
    _modes.add_mode_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "bogus"])


def test_a_contradictory_spec_cannot_be_constructed() -> None:
    """The officialness invariant holds by construction, not by convention.

    ``emits_official_row`` decides whether a run may write a publishable row. If it is merely a
    field someone fills in, a future table edit can mark a mocked mode official and every
    downstream check would faithfully honour it. The invariant belongs in the type.
    """
    with pytest.raises(ValueError, match="official"):
        _modes.ModeSpec(
            mode=_modes.RunMode.SMOKE,
            embedder=_modes.EmbedderKind.LOCAL,
            reader_judge=_modes.ModelKind.MOCK,
            runs_reader=False,
            runs_judge=True,
            emits_official_row=True,
            emits_retrieval_log=False,
        )


def test_a_spec_that_judges_nothing_cannot_claim_to_run_a_reader() -> None:
    """A mode whose answer is discarded cannot also be exercising a real reader."""
    with pytest.raises(ValueError, match="reader"):
        _modes.ModeSpec(
            mode=_modes.RunMode.RETRIEVAL,
            embedder=_modes.EmbedderKind.REAL,
            reader_judge=_modes.ModelKind.MOCK,
            runs_reader=True,
            runs_judge=False,
            emits_official_row=False,
            emits_retrieval_log=True,
        )


def test_the_shipped_table_satisfies_the_invariants() -> None:
    """Every shipped spec passes the same construction check."""
    for mode in _modes.RunMode:
        assert _modes.resolve_mode(mode).mode is mode
