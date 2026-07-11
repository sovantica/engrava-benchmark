"""Coverage for CLI entry points, runner factories, and emit provenance helpers.

All offline / no spend: the OpenAI path is only constructed (never called), and the
dataset/result IO uses tmp paths.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

import scripts.build_leaderboard as bl
import scripts.validate_results as vr
from adapters.base import CorpusTurn, RankedItem, RunContext
from runners.longmemeval import emit
from runners.longmemeval import run as runner
from runners.longmemeval.mock_models import MockJudge, MockReader
from runners.longmemeval.openai_models import OpenAIJudge, OpenAIReader


class _NeutralAdapter:
    """A minimal in-test memory adapter (no engrava): records corpus, ranks all."""

    def __init__(self) -> None:
        self._corpus: list[CorpusTurn] = []
        self.last_spec: str | None = None

    def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
        self._corpus = list(corpus)
        self.last_spec = run_ctx.embedder_spec

    def search(self, query: str, *, top_k: int) -> list[RankedItem]:
        _ = query
        return [RankedItem(unit_id=t.unit_id, score=1.0) for t in self._corpus][:top_k]


CONFIG = Path(__file__).resolve().parents[1] / "runners" / "longmemeval" / "config" / "default.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "longmemeval_smoke.json"


# --- runner factories -------------------------------------------------------- #
def test_build_reader_judge_mock() -> None:
    config = runner.load_config(CONFIG)
    reader, judge = runner.build_reader_judge(config, models="mock")
    assert isinstance(reader, MockReader)
    assert isinstance(judge, MockJudge)


def test_build_reader_judge_openai_constructs_without_calling() -> None:
    pytest.importorskip("openai")
    config = runner.load_config(CONFIG)
    reader, judge = runner.build_reader_judge(config, models="openai")
    # Constructed but never invoked (no network, no key needed at construction).
    assert reader is not None
    assert judge is not None


def test_build_reader_judge_rejects_unknown_mode() -> None:
    config = runner.load_config(CONFIG)
    with pytest.raises(ValueError, match="unknown models mode"):
        runner.build_reader_judge(config, models="bogus")


def test_clients_build_with_dummy_key_when_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    # No env key set: the client still initialises (a local server ignores auth) via
    # a dummy non-empty placeholder. Construction makes no network call.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reader = OpenAIReader(
        model_snapshot="gemma3:4b",
        endpoint="http://localhost:11434",
        sampling={"temperature": 0.0},
    )
    judge = OpenAIJudge(model_snapshot="gemma3:4b", endpoint="http://localhost:11434")
    rclient = reader._ensure_client()
    jclient = judge._ensure_client()
    assert str(rclient.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert str(jclient.base_url).rstrip("/") == "http://localhost:11434/v1"
    # The dummy placeholder is never a real credential.
    assert rclient.api_key == "ollama"


# --- emit provenance helpers ------------------------------------------------- #
def test_engrava_helpers_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> Any:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(emit.metadata, "version", _raise)
    monkeypatch.setattr(emit.metadata, "distribution", _raise)
    assert emit.engrava_version() == "unknown"
    assert emit.engrava_dist_hash().startswith("sha256:")


def test_file_checksum(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert emit.file_checksum(f).startswith("sha256:")


def test_runner_commit_shape() -> None:
    assert emit.runner_commit().startswith("engrava-benchmark@")


# --- CLI main() -------------------------------------------------------------- #
def test_main_no_dataset_fails_before_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "must not be called before a dataset is resolved"
        raise AssertionError(msg)

    monkeypatch.delenv(runner.DEFAULT_DATASET_ENV, raising=False)
    monkeypatch.setattr(runner, "DEFAULT_DATASET_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(runner, "load_questions", _boom)
    monkeypatch.setattr(runner, "build_reader_judge", _boom)
    monkeypatch.setattr(runner, "build_engrava_adapter", _boom)

    rc = runner.main(["--config", str(CONFIG)])
    assert rc == 2
    assert "No dataset found" in capsys.readouterr().out


def test_main_default_uses_env_dataset_and_emits_in_repo_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    monkeypatch.setenv(runner.DEFAULT_DATASET_ENV, str(FIXTURE))
    monkeypatch.setattr(runner, "build_engrava_adapter", lambda _config: _NeutralAdapter())

    def _reader_judge(config: dict[str, Any], *, models: str) -> tuple[MockReader, MockJudge]:
        assert models == "openai"
        assert config["reader"]["endpoint"] == "api.openai.com"
        return MockReader(), MockJudge()

    monkeypatch.setattr(runner, "build_reader_judge", _reader_judge)

    rc = runner.main([])
    assert rc == 0
    rows = list((results_dir / "longmemeval-s" / "longmemeval-official" / "engrava").glob("*.json"))
    assert len(rows) == 1
    row = json.loads(rows[0].read_text())
    artifact_dir = rows[0].with_suffix("")
    assert artifact_dir.is_dir()
    assert (
        row["reproduction_artifact_url"]
        == f"results/longmemeval-s/longmemeval-official/engrava/{row['result_id']}/"
    )


def test_main_mock_run_emits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point emit at a tmp results dir and run the mock pipeline end-to-end via main().
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(runner, "build_engrava_adapter", lambda _config: _NeutralAdapter())

    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--models",
            "mock",
            "--limit",
            "2",
            "--emit",
            "--result-id",
            "cli_cov_row",
            "--date",
            "2026-06-29",
        ]
    )
    assert rc == 0
    part = results_dir / "longmemeval-s" / "longmemeval-official" / "engrava"
    assert (part / "cli_cov_row.json").exists()
    assert (part / "cli_cov_row").is_dir()


def test_main_smoke_forces_free_no_emit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    adapter = _NeutralAdapter()
    monkeypatch.setattr(runner, "build_engrava_adapter", lambda _config: adapter)

    rc = runner.main(["--smoke"])

    assert rc == 0
    assert adapter.last_spec == runner.SMOKE_EMBEDDER_SPEC
    assert not results_dir.exists()


def test_main_byo_reader_override_recorded_in_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bring-your-own reader run (cheap hosted reader + canonical gpt-4o judge)
    # records the ACTUAL overridden reader model/endpoint/max_tokens in the row, and
    # stays verification_status=unverified so it is never mistaken for the canonical
    # headline. The judge endpoint/snapshot remain canonical (so the row validates).
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(runner, "build_engrava_adapter", lambda _config: _NeutralAdapter())

    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--models",
            "mock",
            "--limit",
            "2",
            "--emit",
            "--reader-endpoint",
            "https://openrouter.ai/api/v1",
            "--reader-model",
            "openai/gpt-oss-120b",
            "--reader-max-tokens",
            "1500",
            "--result-id",
            "byo_reader_row",
            "--date",
            "2026-06-29",
        ]
    )
    assert rc == 0
    row = json.loads(
        (
            results_dir
            / "longmemeval-s"
            / "longmemeval-official"
            / "engrava"
            / "byo_reader_row.json"
        ).read_text()
    )
    assert row["reader_endpoint"] == "https://openrouter.ai/api/v1"
    assert row["reader_model"] == "openai/gpt-oss-120b"
    assert row["reader_sampling"]["max_tokens"] == 1500
    assert row["verification_status"] == "unverified"  # never a verified headline
    # The judge stays canonical, so the row is schema-valid and lands correctly.
    assert row["judge_endpoint"] == "api.openai.com"
    assert row["judge_snapshot"] == "gpt-4o-2024-08-06"


def test_main_byo_reader_with_key_env_stays_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Provenance guarantee: a cross-provider reader run — routed to a separate
    # endpoint AND authenticated via a separate key env (--reader-api-key-env) —
    # emits successfully through the full main()->emit path and is *always*
    # verification_status=unverified, regardless of the result-id / date. A screen
    # run can never masquerade as the canonical headline. This exercises the key-env
    # flag through a full emitted row and pins the "cannot be verified" invariant on
    # a different id/date combination than test_main_byo_reader_override_recorded_in_row.
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter-key")
    results_dir = tmp_path / "results"
    monkeypatch.setattr(emit, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(runner, "build_engrava_adapter", lambda _config: _NeutralAdapter())

    rc = runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--models",
            "mock",
            "--limit",
            "2",
            "--emit",
            "--reader-endpoint",
            "https://openrouter.ai/api/v1",
            "--reader-model",
            "openai/gpt-oss-120b",
            "--reader-max-tokens",
            "1500",
            "--reader-api-key-env",
            "OPENROUTER_API_KEY",
            "--result-id",
            "byo_reader_keyenv_row",
            "--date",
            "2026-07-01",
        ]
    )
    assert rc == 0
    row = json.loads(
        (
            results_dir
            / "longmemeval-s"
            / "longmemeval-official"
            / "engrava"
            / "byo_reader_keyenv_row.json"
        ).read_text()
    )
    # The overridden reader is recorded; the row can never be a verified headline.
    assert row["reader_model"] == "openai/gpt-oss-120b"
    assert row["reader_endpoint"] == "https://openrouter.ai/api/v1"
    assert row["verification_status"] == "unverified"
    # Judge stays canonical so the row still validates.
    assert row["judge_snapshot"] == "gpt-4o-2024-08-06"


def test_main_embedder_override_updates_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(emit, "RESULTS_DIR", tmp_path)
    adapter = _NeutralAdapter()
    monkeypatch.setattr(runner, "build_engrava_adapter", lambda _config: adapter)
    runner.main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE),
            "--models",
            "mock",
            "--no-emit",
            "--embedder-spec",
            "local:all-MiniLM-L12-v2",
        ]
    )
    assert adapter.last_spec == "local:all-MiniLM-L12-v2"


# --- script main() ----------------------------------------------------------- #
def test_validate_main_empty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(vr, "RESULTS_DIR", tmp_path)
    assert vr.main() == 0
    assert "no rows yet" in capsys.readouterr().out


def test_validate_main_on_tree(
    tmp_path: Path,
    valid_sovantica_row: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    write_valid_artifact,
) -> None:
    write_valid_artifact(tmp_path, valid_sovantica_row)
    out = (
        tmp_path
        / "longmemeval-s"
        / "longmemeval-official"
        / "engrava"
        / f"{valid_sovantica_row['result_id']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(valid_sovantica_row))
    monkeypatch.setattr(vr, "RESULTS_DIR", tmp_path)
    assert vr.main([str(out)]) == 0


def test_build_leaderboard_main(
    tmp_path: Path,
    valid_sovantica_row: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    write_valid_artifact,
) -> None:
    write_valid_artifact(tmp_path, valid_sovantica_row)
    out = (
        tmp_path
        / "longmemeval-s"
        / "longmemeval-official"
        / "engrava"
        / f"{valid_sovantica_row['result_id']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(valid_sovantica_row))
    monkeypatch.setattr(bl, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(bl, "OUTPUT_PATH", tmp_path / "leaderboard.json")
    assert bl.main() == 0
    board = json.loads((tmp_path / "leaderboard.json").read_text())
    assert board["row_count"] == 1
