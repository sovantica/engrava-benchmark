"""Tests for the local OpenAI-compatible (Ollama) reader/judge path.

All offline: the OpenAI SDK is exercised only for client construction (no network),
and the model-config resolution + CLI wiring are pure functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from adapters.base import RankedItem
from runners.longmemeval import run as runner
from runners.longmemeval.mock_models import MockJudge, MockReader
from runners.longmemeval.openai_models import OpenAIJudge, OpenAIReader, resolve_base_url

if TYPE_CHECKING:
    from adapters.base import CorpusTurn, RunContext

CONFIG = Path(__file__).resolve().parents[1] / "runners" / "longmemeval" / "config" / "default.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "longmemeval_smoke.json"


# --- base URL resolution ----------------------------------------------------- #
def test_resolve_base_url_canonical_bare_host() -> None:
    # The canonical bare host keeps the https default + /v1 — unchanged behaviour.
    assert resolve_base_url("api.openai.com") == "https://api.openai.com/v1"


def test_resolve_base_url_local_http() -> None:
    assert resolve_base_url("http://msi.example.net:11434") == "http://msi.example.net:11434/v1"


def test_resolve_base_url_https_scheme() -> None:
    assert resolve_base_url("https://proxy.example.net") == "https://proxy.example.net/v1"


def test_resolve_base_url_keeps_existing_v1() -> None:
    # An endpoint already ending in /v1 is not doubled.
    assert resolve_base_url("http://host:11434/v1") == "http://host:11434/v1"
    assert resolve_base_url("http://host:11434/v1/") == "http://host:11434/v1"


# --- apply_model_config: promotion + overrides ------------------------------- #
def test_apply_model_config_openai_unchanged() -> None:
    config = runner.load_config(CONFIG)
    runner.apply_model_config(config, models="openai")
    # Canonical default untouched.
    assert config["reader"]["endpoint"] == "api.openai.com"
    assert config["reader"]["snapshot"] == "gpt-4o-2024-08-06"
    assert config["judge"]["endpoint"] == "api.openai.com"


def test_apply_model_config_ollama_promotes_block() -> None:
    config = runner.load_config(CONFIG)
    runner.apply_model_config(config, models="ollama")
    # The local ollama block is promoted into the effective reader/judge.
    assert config["reader"]["endpoint"] == "http://localhost:11434"
    assert config["reader"]["model"] == "gemma3:4b"
    assert config["judge"]["endpoint"] == "http://localhost:11434"


def test_apply_model_config_cli_overrides() -> None:
    config = runner.load_config(CONFIG)
    runner.apply_model_config(
        config,
        models="ollama",
        endpoint="http://msi.example.net:11434",
        reader_model="gemma3:4b",
        judge_model="llama3:8b",
    )
    assert config["reader"]["endpoint"] == "http://msi.example.net:11434"
    assert config["reader"]["model"] == "gemma3:4b"
    assert config["reader"]["snapshot"] == "gemma3:4b"
    assert config["judge"]["model"] == "llama3:8b"
    assert config["judge"]["endpoint"] == "http://msi.example.net:11434"


def test_apply_model_config_default_when_block_absent() -> None:
    # A config without an `ollama` block still resolves to a sane local default.
    config = {"reader": {}, "judge": {}}
    runner.apply_model_config(config, models="ollama")
    assert config["reader"]["endpoint"] == "http://localhost:11434"
    assert "ollama" in config


# --- configurable reader seam: DEFAULTS ALWAYS OFFICIAL (hard invariant) ------ #
def test_apply_model_config_no_overrides_is_byte_identical_to_canonical() -> None:
    # With NO new flags the effective config must be exactly the canonical one: same
    # reader/judge model+snapshot+endpoint, same reader sampling (no max_tokens), same
    # embedder. This is the published-path invariant — assert byte-for-byte.
    canonical = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = runner.load_config(CONFIG)
    runner.apply_model_config(config, models="openai")
    assert config == canonical
    # The reader declares no max_tokens, so the reader falls back to the official 800.
    assert "max_tokens" not in config["reader"].get("sampling", {})


def test_reader_endpoint_override_leaves_judge_endpoint_canonical() -> None:
    # --reader-endpoint retargets ONLY the reader; the judge keeps api.openai.com.
    config = runner.load_config(CONFIG)
    runner.apply_model_config(
        config,
        models="openai",
        reader_endpoint="https://openrouter.ai/api/v1",
        reader_model="openai/gpt-oss-120b",
    )
    assert config["reader"]["endpoint"] == "https://openrouter.ai/api/v1"
    assert config["reader"]["model"] == "openai/gpt-oss-120b"
    assert config["judge"]["endpoint"] == "api.openai.com"  # canonical judge untouched
    assert config["judge"]["snapshot"] == "gpt-4o-2024-08-06"


def test_endpoint_then_reader_endpoint_precedence() -> None:
    # --endpoint sets both; --reader-endpoint / --judge-endpoint refine per side.
    config = runner.load_config(CONFIG)
    runner.apply_model_config(
        config,
        models="openai",
        endpoint="http://shared:11434",
        reader_endpoint="https://openrouter.ai/api/v1",
        judge_endpoint="api.openai.com",
    )
    assert config["reader"]["endpoint"] == "https://openrouter.ai/api/v1"
    assert config["judge"]["endpoint"] == "api.openai.com"


def test_reader_max_tokens_override_written_to_sampling() -> None:
    config = runner.load_config(CONFIG)
    runner.apply_model_config(config, models="openai", reader_max_tokens=1500)
    assert config["reader"]["sampling"]["max_tokens"] == 1500
    # Existing sampling params are preserved, not clobbered.
    assert config["reader"]["sampling"]["temperature"] == 0.0
    # The judge is not given a max_tokens sampling override.
    assert "sampling" not in config["judge"] or "max_tokens" not in config["judge"]["sampling"]


def test_reader_api_key_env_override_and_builder_threading() -> None:
    # The reader can read a different key env than the judge (cross-provider run).
    pytest.importorskip("openai")
    config = runner.load_config(CONFIG)
    runner.apply_model_config(
        config,
        models="openai",
        reader_endpoint="https://openrouter.ai/api/v1",
        reader_model="openai/gpt-oss-120b",
        reader_api_key_env="OPENROUTER_API_KEY",
    )
    assert config["reader"]["api_key_env"] == "OPENROUTER_API_KEY"
    # The judge keeps the canonical default (no api_key_env key => OPENAI_API_KEY).
    assert config["judge"].get("api_key_env", "OPENAI_API_KEY") == "OPENAI_API_KEY"
    reader, judge = runner.build_reader_judge(config, models="openai")
    assert isinstance(reader, OpenAIReader)
    assert isinstance(judge, OpenAIJudge)
    assert reader._api_key_env == "OPENROUTER_API_KEY"
    assert judge._api_key_env == "OPENAI_API_KEY"


# --- build_reader_judge selects the local clients ---------------------------- #
def test_build_reader_judge_ollama_selects_openai_compatible_clients() -> None:
    pytest.importorskip("openai")
    config = runner.load_config(CONFIG)
    runner.apply_model_config(config, models="ollama", endpoint="http://host:11434")
    reader, judge = runner.build_reader_judge(config, models="ollama")
    assert isinstance(reader, OpenAIReader)
    assert isinstance(judge, OpenAIJudge)
    # They target the local endpoint, not api.openai.com.
    assert reader._endpoint == "http://host:11434"
    assert judge._endpoint == "http://host:11434"


# --- the emitted row records the ACTUAL local endpoint/model ----------------- #
class _NeutralAdapter:
    def __init__(self) -> None:
        self._c: list[CorpusTurn] = []

    def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
        self._c = list(corpus)
        _ = run_ctx

    def search(self, query: str, *, top_k: int) -> list[RankedItem]:
        _ = query
        return [RankedItem(unit_id=t.unit_id, score=1.0) for t in self._c][:top_k]


def test_local_reader_canonical_judge_row_records_actual_reader(tmp_path: Path) -> None:
    # A local Ollama READER with the canonical OpenAI JUDGE is a legitimate
    # non-canonical-reader `sovantica-run` row: it validates, and records the ACTUAL
    # local reader endpoint/model (so it lands in a non-canonical reader segment,
    # never mislabeled as the gpt-4o reader headline).
    config = runner.load_config(CONFIG)
    config["reader"] = {
        "model": "gemma3:4b",
        "snapshot": "gemma3:4b",
        "endpoint": "http://msi.example.net:11434",
        "sampling": {"temperature": 0.0},
    }
    # judge stays canonical (config default) so the sovantica-run schema rule holds.
    questions = runner.load_questions(FIXTURE)
    runner.run_and_emit(
        config=config,
        questions=questions,
        adapter=_NeutralAdapter(),
        reader=MockReader(),
        judge=MockJudge(),
        result_id="local_reader_row",
        date="2026-06-30",
        partial=True,
        emit_result=True,
        results_dir=tmp_path,
    )
    row = json.loads(
        (
            tmp_path
            / "longmemeval-s"
            / "longmemeval-official"
            / "engrava"
            / "local_reader_row.json"
        ).read_text()
    )
    assert row["reader_endpoint"] == "http://msi.example.net:11434"
    assert row["reader_model"] == "gemma3:4b"
    assert row["reader_snapshot"] == "gemma3:4b"  # NOT the canonical gpt-4o snapshot
    assert row["judge_endpoint"] == "api.openai.com"  # canonical judge preserved


def test_fully_local_run_row_is_rejected_not_mislabeled(tmp_path: Path) -> None:
    # A fully-local run (Ollama JUDGE too) is exploratory, not a publishable
    # sovantica-run: emitting it is correctly REJECTED by the schema's canonical-judge
    # rule — the pipeline never silently mislabels a local run as the canonical headline.
    config = runner.load_config(CONFIG)
    runner.apply_model_config(config, models="ollama", endpoint="http://msi.example.net:11434")
    questions = runner.load_questions(FIXTURE)
    with pytest.raises(ValueError, match="failed validation"):
        runner.run_and_emit(
            config=config,
            questions=questions,
            adapter=_NeutralAdapter(),
            reader=MockReader(),
            judge=MockJudge(),
            result_id="fully_local_row",
            date="2026-06-30",
            partial=True,
            emit_result=True,
            results_dir=tmp_path,
        )


def test_fully_local_run_without_emit_succeeds(tmp_path: Path) -> None:
    # The free end-to-end value: a fully-local run computes metrics without emitting
    # a schema-invalid sovantica-run row.
    config = runner.load_config(CONFIG)
    runner.apply_model_config(config, models="ollama", endpoint="http://msi.example.net:11434")
    questions = runner.load_questions(FIXTURE)
    metrics = runner.run_and_emit(
        config=config,
        questions=questions,
        adapter=_NeutralAdapter(),
        reader=MockReader(),
        judge=MockJudge(),
        result_id="local_metrics",
        date="2026-06-30",
        partial=True,
        emit_result=False,
    )
    assert set(metrics) == {"overall_micro", "macro", "abstention", "per_category"}
    assert list(tmp_path.iterdir()) == []


# --- OpenAIReader.answer: config-driven max_tokens + empty-content guard ------ #
class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _RecordingCompletions:
    """Records the create() kwargs and returns a canned content (no network)."""

    def __init__(self, content: str | None) -> None:
        self._content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions: _RecordingCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, content: str | None) -> None:
        self.chat = _FakeChat(_RecordingCompletions(content))


def _reader_with_fake(
    content: str | None, sampling: dict[str, object]
) -> tuple[OpenAIReader, _RecordingCompletions]:
    reader = OpenAIReader(
        model_snapshot="openai/gpt-oss-120b",
        endpoint="https://openrouter.ai/api/v1",
        sampling=sampling,
    )
    client = _FakeClient(content)
    reader._client = client  # inject the fake so _ensure_client() makes no network call
    return reader, client.chat.completions


def test_answer_uses_official_max_tokens_by_default() -> None:
    reader, completions = _reader_with_fake("Porto", {"temperature": 0.0})
    out = reader.answer("Where?", "some context", question_date="2026-01-01")
    assert out == "Porto"
    assert completions.calls[0]["max_tokens"] == 800  # official default
    assert completions.calls[0]["temperature"] == 0.0


def test_answer_honors_sampling_max_tokens_override() -> None:
    reader, completions = _reader_with_fake("Porto", {"temperature": 0.0, "max_tokens": 1500})
    reader.answer("Where?", "some context")
    call = completions.calls[0]
    assert call["max_tokens"] == 1500  # override wins
    assert call["temperature"] == 0.0
    # max_tokens must NOT be passed twice (once explicit, once via **sampling).
    assert list(call.keys()).count("max_tokens") == 1


def test_answer_empty_content_guard_returns_empty_string() -> None:
    # A reasoning model may leave message.content empty/None (CoT elsewhere): no crash.
    reader_none, _ = _reader_with_fake(None, {"temperature": 0.0})
    assert reader_none.answer("q", "ctx") == ""
    reader_empty, _ = _reader_with_fake("   ", {"temperature": 0.0})
    assert reader_empty.answer("q", "ctx") == ""


def test_main_ollama_dry_run_no_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    # --models ollama with no dataset exits before network/backend construction.
    rc = runner.main(["--config", str(CONFIG), "--models", "ollama"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "models=ollama" in out
    assert "localhost:11434" in out
    assert "No dataset found" in out
