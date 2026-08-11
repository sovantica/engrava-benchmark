"""Tests for the LongMemEval runner ``--mode`` convenience dial.

The hard invariant under test: omitting ``--mode`` reproduces the exact prior
behaviour, while ``--mode`` only establishes DEFAULTS that explicit flags override.
All offline / no spend: the adapter and reader/judge are in-test fakes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from adapters.base import CorpusTurn, RankedItem, RunContext
from runners import _modes
from runners._modes import RunMode
from runners.longmemeval import emit
from runners.longmemeval import run as runner
from runners.longmemeval.mock_models import MockJudge, MockReader

if TYPE_CHECKING:
    import pytest

CONFIG = Path(__file__).resolve().parents[1] / "runners" / "longmemeval" / "config" / "default.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "longmemeval_smoke.json"
# A dataset whose first gold answer is a JSON number, as ~6% of LongMemEval-S is.
NUMERIC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "longmemeval_numeric_gold.json"


class _RecordingJudge:
    """Judge stand-in that records every scoring call (offline, no spend)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def score(
        self,
        question: str,
        gold: object,
        answer: str,
        *,
        question_type: str,
        question_id: str,
    ) -> bool:
        """Record the call and report incorrect."""
        _ = question, gold, answer, question_type
        self.calls.append(question_id)
        return False


def _fake_reader_judge(_config: Any, *, models: str) -> tuple[MockReader, MockJudge]:
    """Offline reader/judge stand-in used where the resolved ``models`` is not asserted."""
    _ = models
    return MockReader(), MockJudge()


class _NeutralAdapter:
    """In-test memory adapter (no engrava): records corpus + spec, ranks all."""

    def __init__(self) -> None:
        self._corpus: list[CorpusTurn] = []
        self.last_spec: str | None = None

    def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
        self._corpus = list(corpus)
        self.last_spec = run_ctx.embedder_spec

    def search(self, query: str, *, top_k: int) -> list[RankedItem]:
        _ = query
        return [RankedItem(unit_id=t.unit_id, score=1.0) for t in self._corpus][:top_k]


def _namespace(**overrides: Any) -> argparse.Namespace:
    """Build a parsed-args namespace with the mode-relevant sentinel defaults."""
    base: dict[str, Any] = {
        "mode": None,
        "models": None,
        "emit": None,
        "embedder_spec": None,
        "smoke": False,
        "baseline": None,
        "results_dir": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# --- _apply_run_mode: pure resolution --------------------------------------- #
def test_no_mode_preserves_historical_defaults() -> None:
    args = _namespace()
    spec = runner._apply_run_mode(args)
    assert spec is None
    assert args.models == "openai"
    assert args.emit is True
    assert args.embedder_spec is None  # config's real embedder is used
    assert args.smoke is False


def test_mode_smoke_delegates_to_smoke_path() -> None:
    args = _namespace(mode="smoke")
    spec = runner._apply_run_mode(args)
    assert spec is not None
    assert spec.mode is RunMode.SMOKE
    assert args.smoke is True  # the existing --smoke block does the rest


def test_mode_plumbing_defaults() -> None:
    args = _namespace(mode="plumbing")
    runner._apply_run_mode(args)
    assert args.models == "mock"
    assert args.embedder_spec == runner.LOCAL_EMBEDDER_SPEC
    assert args.emit is False


def test_mode_retrieval_defaults() -> None:
    args = _namespace(mode="retrieval")
    spec = runner._apply_run_mode(args)
    assert spec is not None
    assert spec.emits_retrieval_log is True
    assert args.models == "mock"  # free reader/judge, discarded
    assert args.embedder_spec is None  # REAL embedder (config default)
    assert args.emit is False  # non-official


def test_mode_score_resolves_to_canonical_openai_path() -> None:
    args = _namespace(mode="score")
    spec = runner._apply_run_mode(args)
    assert spec is not None
    assert spec.emits_official_row is True
    assert args.models == "openai"
    assert args.embedder_spec is None  # REAL embedder (config default)
    assert args.emit is True


def test_explicit_models_wins_over_mode_default() -> None:
    args = _namespace(mode="plumbing", models="openai")
    runner._apply_run_mode(args)
    assert args.models == "openai"  # explicit flag not overridden


def test_explicit_emit_ignored_in_non_official_mode() -> None:
    # A non-official mode (retrieval/plumbing) can NEVER write a canonical row, even
    # with --emit: its reader/judge are mock, so an emitted row would be dishonest.
    args = _namespace(mode="retrieval", emit=True)
    runner._apply_run_mode(args)
    assert args.emit is False


def test_explicit_emit_honoured_in_score_mode() -> None:
    # score is the only official mode; explicit --emit / --no-emit win there.
    emit_args = _namespace(mode="score", emit=True)
    runner._apply_run_mode(emit_args)
    assert emit_args.emit is True

    no_emit_args = _namespace(mode="score", emit=False)
    runner._apply_run_mode(no_emit_args)
    assert no_emit_args.emit is False


def test_explicit_embedder_wins_over_mode_default() -> None:
    args = _namespace(mode="plumbing", embedder_spec="openai:text-embedding-3-small")
    runner._apply_run_mode(args)
    assert args.embedder_spec == "openai:text-embedding-3-small"


# --- main(): end-to-end mode wiring (offline) ------------------------------- #
def _patch_offline(monkeypatch: pytest.MonkeyPatch, adapter: _NeutralAdapter) -> None:
    monkeypatch.setattr(runner, "build_engrava_adapter", lambda _config: adapter)


def test_main_mode_score_uses_openai_and_emits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    _patch_offline(monkeypatch, _NeutralAdapter())

    def _reader_judge(config: dict[str, Any], *, models: str) -> tuple[MockReader, MockJudge]:
        assert models == "openai"  # score -> canonical openai path
        assert config["reader"]["endpoint"] == "api.openai.com"
        return MockReader(), MockJudge()

    monkeypatch.setattr(runner, "build_reader_judge", _reader_judge)

    rc = runner.main(["--config", str(CONFIG), "--dataset", str(FIXTURE), "--mode", "score"])
    assert rc == 0
    rows = list((results_dir / "longmemeval-s" / "longmemeval-official" / "engrava").glob("*.json"))
    assert len(rows) == 1  # score emits an official row


def test_main_mode_retrieval_writes_log_and_uses_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    _patch_offline(monkeypatch, _NeutralAdapter())

    def _reader_judge(config: dict[str, Any], *, models: str) -> tuple[MockReader, MockJudge]:
        _ = config
        assert models == "mock"  # retrieval uses the free mock reader/judge
        return MockReader(), MockJudge()

    monkeypatch.setattr(runner, "build_reader_judge", _reader_judge)

    out_dir = tmp_path / "retrieval-out"
    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--mode",
            "retrieval",
            "--results-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    log_path = out_dir / _modes_retrieval_filename()
    assert log_path.is_file()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    # Real ranked ids per question (the neutral adapter ranks every corpus turn,
    # which the runner maps to official ids for the two fixture questions).
    assert set(log) == {"smoke_q1", "smoke_q2_abs"}
    # retrieval mode is non-official: no result row emitted.
    assert not (results_dir).exists()


def test_main_mode_retrieval_runs_on_a_numeric_gold_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cheap retrieval gate must survive a numeric gold answer end-to-end and
    # still emit a well-formed retrieval log. Reader/judge come from the real
    # factory (mock backend), so nothing about the numeric path is stubbed out.
    _patch_offline(monkeypatch, _NeutralAdapter())

    out_dir = tmp_path / "retrieval-out"
    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(NUMERIC_FIXTURE),
            "--mode",
            "retrieval",
            "--results-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    log = json.loads((out_dir / _modes_retrieval_filename()).read_text(encoding="utf-8"))
    assert set(log) == {"numeric_q1", "numeric_q2"}
    assert all(isinstance(i, str) for ids in log.values() for i in ids)
    assert log["numeric_q1"]  # the ranked official ids of the numeric-gold question


def test_main_models_mock_scores_a_numeric_gold_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    # The offline judge path itself (no --mode, --models mock) must handle a
    # numeric gold rather than dying on the first one.
    _patch_offline(monkeypatch, _NeutralAdapter())

    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(NUMERIC_FIXTURE),
            "--models",
            "mock",
            "--no-emit",
        ]
    )
    assert rc == 0


def test_main_mode_retrieval_never_invokes_the_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _RecordingJudge()
    _patch_offline(monkeypatch, _NeutralAdapter())
    monkeypatch.setattr(
        runner, "build_reader_judge", lambda _config, *, models: (MockReader(), judge)
    )

    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--mode",
            "retrieval",
            "--results-dir",
            str(tmp_path / "retrieval-out"),
        ]
    )
    assert rc == 0
    assert judge.calls == []  # the mode discards the answer; nothing is judged


def test_main_mode_score_still_invokes_the_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guard on the paid canonical path: score judges every question, unchanged.
    judge = _RecordingJudge()
    _patch_offline(monkeypatch, _NeutralAdapter())
    monkeypatch.setattr(
        runner, "build_reader_judge", lambda _config, *, models: (MockReader(), judge)
    )

    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--mode",
            "score",
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )
    assert rc == 0
    assert judge.calls == ["smoke_q1", "smoke_q2_abs"]


def test_resolve_judge_only_replaces_a_non_judging_mode() -> None:
    judge = _RecordingJudge()
    for mode in (RunMode.SMOKE, RunMode.PLUMBING, RunMode.SCORE):
        assert runner.resolve_judge(judge, _modes.resolve_mode(mode)) is judge
    assert runner.resolve_judge(judge, None) is judge  # the historical no-mode path
    assert runner.resolve_judge(judge, _modes.resolve_mode(RunMode.RETRIEVAL)) is not judge


def test_main_mode_retrieval_baseline_green_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_offline(monkeypatch, _NeutralAdapter())
    monkeypatch.setattr(runner, "build_reader_judge", _fake_reader_judge)

    out_dir = tmp_path / "run"
    baseline_dir = tmp_path / "baseline"

    common = ["--config", str(CONFIG), "--dataset", str(FIXTURE), "--mode", "retrieval"]
    # First run establishes the baseline log.
    assert runner.main([*common, "--results-dir", str(baseline_dir)]) == 0
    # Second run against the identical baseline -> GREEN -> exit 0.
    rc = runner.main([*common, "--results-dir", str(out_dir), "--baseline", str(baseline_dir)])
    assert rc == 0


def test_main_mode_retrieval_baseline_red_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_offline(monkeypatch, _NeutralAdapter())
    monkeypatch.setattr(runner, "build_reader_judge", _fake_reader_judge)

    out_dir = tmp_path / "run"
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    # A disjoint baseline (different question ids) is a material change -> RED.
    (baseline_dir / "retrieval_log.json").write_text(
        json.dumps({"other_q": ["x"]}), encoding="utf-8"
    )

    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--mode",
            "retrieval",
            "--results-dir",
            str(out_dir),
            "--baseline",
            str(baseline_dir),
        ]
    )
    assert rc == 1  # RED verdict propagates as a non-zero exit


def test_main_mode_plumbing_uses_local_embedder_no_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    adapter = _NeutralAdapter()
    _patch_offline(monkeypatch, adapter)
    monkeypatch.setattr(runner, "build_reader_judge", _fake_reader_judge)

    rc = runner.main(["--config", str(CONFIG), "--dataset", str(FIXTURE), "--mode", "plumbing"])
    assert rc == 0
    assert adapter.last_spec == runner.LOCAL_EMBEDDER_SPEC
    assert not results_dir.exists()  # plumbing is non-official


def test_main_mode_smoke_equivalent_to_smoke_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    adapter = _NeutralAdapter()
    _patch_offline(monkeypatch, adapter)

    rc = runner.main(["--mode", "smoke"])
    assert rc == 0
    assert adapter.last_spec == runner.SMOKE_EMBEDDER_SPEC
    assert not results_dir.exists()


def _modes_retrieval_filename() -> str:
    from runners import retrieval_diff  # noqa: PLC0415

    return retrieval_diff.RETRIEVAL_LOG_FILENAME


def test_smoke_mode_never_emits_even_with_a_results_dir(tmp_path: Path) -> None:
    """``--mode smoke`` cannot reach the emission path, whatever else is passed.

    The non-official guard promises that a mocked mode can never write a result row, but the
    smoke branch returned before reaching it: with ``--results-dir`` the legacy fast path left
    ``emit`` unresolved and the final sentinel turned it into ``True``, handing a two-question,
    mock-model, local-embedder run to the emission path.

    The legacy ``--smoke`` flag keeps its own behaviour — capturing a bundle into an isolated tree
    is what it is for. What must hold is that the *mode surface* means what it says.
    """
    args = _namespace(mode="smoke", emit=None, results_dir=str(tmp_path))

    spec = runner._apply_run_mode(args)

    assert spec is not None
    assert spec.emits_official_row is False
    assert args.emit is False


def test_smoke_mode_refuses_an_explicit_emit(tmp_path: Path) -> None:
    """An explicit ``--emit`` does not buy an exception either."""
    args = _namespace(mode="smoke", emit=True, results_dir=str(tmp_path))

    runner._apply_run_mode(args)

    assert args.emit is False
