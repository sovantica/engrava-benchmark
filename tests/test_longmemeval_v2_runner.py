"""Tests for the LongMemEval-V2 mode wrapper over the upstream harness.

All offline / no spend: the upstream harness is never present here — every
``subprocess.run`` is mocked, and the retrieval-log extractor is exercised against a
small ``prompt_rows.jsonl`` fixture.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from runners import _modes, retrieval_diff
from runners._modes import RunMode
from runners.longmemeval_v2 import run
from runners.longmemeval_v2.retrieval_log import (
    RetrievalLogError,
    extract_retrieval_log,
    item_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURES = Path(__file__).resolve().parent / "fixtures"
QUESTIONS = FIXTURES / "longmemeval_v2_questions.json"
HAYSTACK = FIXTURES / "longmemeval_v2_haystack.json"
PROMPT_ROWS = FIXTURES / "longmemeval_v2_prompt_rows.jsonl"

_NO_READER = run.ReaderJudgeArgs(
    model=None, base_url=None, api_key_env=None, evaluator_model=None, evaluator_api_key_env=None
)
_FULL_READER = run.ReaderJudgeArgs(
    model="qwen3.5-9b",
    base_url="https://example.test/v1",
    api_key_env="READER_KEY",
    evaluator_model="gpt-5.2",
    evaluator_api_key_env="JUDGE_KEY",
)


def _command_for(mode: RunMode, reader_judge: run.ReaderJudgeArgs) -> list[str]:
    """Build the harness command for a mode with fixed, path-only inputs."""
    return run.build_harness_command(
        harness_dir=Path("/harness"),
        spec=_modes.resolve_mode(mode),
        domain="web",
        questions_path=Path("/q.json"),
        haystack_path=Path("/h.json"),
        output_dir=Path("/out"),
        memory_config_path=Path("/cfg.json"),
        memory_context_max_tokens=200_000,
        trajectories_path=Path("/data/trajectories.jsonl"),
        reader_judge=reader_judge,
    )


# --------------------------------------------------------------------------- #
# build_harness_command: mode -> flags
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", [RunMode.SMOKE, RunMode.PLUMBING, RunMode.RETRIEVAL])
def test_non_score_modes_skip_reader(mode: RunMode) -> None:
    command = _command_for(mode, _NO_READER)
    assert "--skip-reader" in command
    assert "--model" not in command
    assert "--evaluator-model" not in command
    # The common, always-present flags.
    assert "--domain" in command
    assert "--memory-config-path" in command
    assert "--trajectories-path" in command


def test_score_mode_forwards_reader_judge_and_no_skip_reader() -> None:
    command = _command_for(RunMode.SCORE, _FULL_READER)
    assert "--skip-reader" not in command
    assert command[command.index("--model") + 1] == "qwen3.5-9b"
    assert command[command.index("--base-url") + 1] == "https://example.test/v1"
    assert command[command.index("--api-key-env") + 1] == "READER_KEY"
    assert command[command.index("--evaluator-model") + 1] == "gpt-5.2"
    assert command[command.index("--evaluator-api-key-env") + 1] == "JUDGE_KEY"


def test_score_mode_omits_unset_reader_flags() -> None:
    command = _command_for(RunMode.SCORE, _NO_READER)
    assert "--skip-reader" not in command
    assert "--model" not in command
    assert "--api-key-env" not in command


def test_command_starts_with_interpreter_and_harness_entrypoint() -> None:
    command = _command_for(RunMode.PLUMBING, _NO_READER)
    assert command[0] == run.sys.executable
    assert command[1].endswith(str(Path("evaluation") / "harness.py"))


def test_trajectories_omitted_when_none() -> None:
    command = run.build_harness_command(
        harness_dir=Path("/harness"),
        spec=_modes.resolve_mode(RunMode.PLUMBING),
        domain="web",
        questions_path=Path("/q.json"),
        haystack_path=Path("/h.json"),
        output_dir=Path("/out"),
        memory_config_path=Path("/cfg.json"),
        memory_context_max_tokens=1,
        trajectories_path=None,
        reader_judge=_NO_READER,
    )
    assert "--trajectories-path" not in command


# --------------------------------------------------------------------------- #
# _resolve_memory_config: per-mode default config selection
# --------------------------------------------------------------------------- #
def _config_namespace(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "memory_config_path": None,
        "embedding_model": "text-embedding-3-small",
        "embedding_base_url": "https://api.openai.com/v1",
        "embedding_api_key_env": "OPENAI_API_KEY",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_local_modes_write_deterministic_config(tmp_path: Path) -> None:
    spec = _modes.resolve_mode(RunMode.PLUMBING)
    path = run._resolve_memory_config(_config_namespace(), spec, tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["memory_params"]["embedding_params"]["backend"] == "deterministic"


def test_real_modes_write_openai_compatible_config(tmp_path: Path) -> None:
    spec = _modes.resolve_mode(RunMode.RETRIEVAL)
    path = run._resolve_memory_config(_config_namespace(), spec, tmp_path)
    params = json.loads(path.read_text(encoding="utf-8"))["memory_params"]["embedding_params"]
    assert params["backend"] == "openai-compatible"
    assert params["model"] == "text-embedding-3-small"
    assert params["api_key_env"] == "OPENAI_API_KEY"


def _write_config(path: Path, backend: str) -> Path:
    path.write_text(
        json.dumps({"memory_params": {"embedding_params": {"backend": backend}}}),
        encoding="utf-8",
    )
    return path


def test_explicit_memory_config_matching_mode_is_returned_unchanged(tmp_path: Path) -> None:
    # A real-backend config in a real mode (score) passes the contract and is used as-is.
    explicit = _write_config(tmp_path / "given.json", "openai-compatible")
    ns = _config_namespace(memory_config_path=explicit)
    spec = _modes.resolve_mode(RunMode.SCORE)
    assert run._resolve_memory_config(ns, spec, tmp_path) == explicit
    assert not (tmp_path / run.MEMORY_CONFIG_FILENAME).exists()


def test_explicit_real_config_rejected_in_offline_mode(tmp_path: Path) -> None:
    # A real embedder smuggled into an offline ($0) mode breaks the mode's cost guarantee.
    explicit = _write_config(tmp_path / "given.json", "openai-compatible")
    ns = _config_namespace(memory_config_path=explicit)
    spec = _modes.resolve_mode(RunMode.PLUMBING)
    with pytest.raises(ValueError, match="deterministic"):
        run._resolve_memory_config(ns, spec, tmp_path)


def test_explicit_deterministic_config_rejected_in_real_mode(tmp_path: Path) -> None:
    # A deterministic embedder in retrieval mode makes the retrieval-diff meaningless.
    explicit = _write_config(tmp_path / "given.json", "deterministic")
    ns = _config_namespace(memory_config_path=explicit)
    spec = _modes.resolve_mode(RunMode.RETRIEVAL)
    with pytest.raises(ValueError, match="real embedder"):
        run._resolve_memory_config(ns, spec, tmp_path)


# --------------------------------------------------------------------------- #
# _resolve_trajectories_path
# --------------------------------------------------------------------------- #
def test_trajectories_explicit_wins() -> None:
    ns = argparse.Namespace(trajectories_path=Path("/x/traj.jsonl"), data_root=Path("/d"))
    assert run._resolve_trajectories_path(ns) == Path("/x/traj.jsonl")


def test_trajectories_derived_from_data_root() -> None:
    ns = argparse.Namespace(trajectories_path=None, data_root=Path("/d"))
    assert run._resolve_trajectories_path(ns) == Path("/d") / run.TRAJECTORIES_FILENAME


def test_trajectories_none_when_unset() -> None:
    ns = argparse.Namespace(trajectories_path=None, data_root=None)
    assert run._resolve_trajectories_path(ns) is None


# --------------------------------------------------------------------------- #
# _write_smoke_subset
# --------------------------------------------------------------------------- #
def test_smoke_subset_head_slices_and_filters_haystack(tmp_path: Path) -> None:
    q_path, h_path = run._write_smoke_subset(QUESTIONS, HAYSTACK, tmp_path, limit=2)
    subset = json.loads(q_path.read_text(encoding="utf-8"))
    haystack = json.loads(h_path.read_text(encoding="utf-8"))
    assert [q["id"] for q in subset] == ["web_q1", "web_q2"]
    assert set(haystack) == {"web_q1", "web_q2"}


def test_smoke_subset_rejects_non_list_questions(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(run.SmokeInputError):
        run._write_smoke_subset(bad, HAYSTACK, tmp_path, limit=2)


def test_smoke_subset_rejects_non_object_haystack(tmp_path: Path) -> None:
    bad = tmp_path / "bad_haystack.json"
    bad.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(run.SmokeInputError):
        run._write_smoke_subset(QUESTIONS, bad, tmp_path, limit=2)


# --------------------------------------------------------------------------- #
# extract_retrieval_log + item_key
# --------------------------------------------------------------------------- #
def test_extract_retrieval_log_parses_headers_and_hashes_fallback() -> None:
    log = extract_retrieval_log(PROMPT_ROWS)
    assert log["web_q1"] == ["traj_a#3", "traj_b#0"]
    first, second = log["web_q2"]
    assert first.startswith("h:")  # no structured headers -> content hash
    assert second == "traj_c#1"


def test_item_key_hashes_non_dict_and_non_string() -> None:
    assert item_key("plain text").startswith("h:")
    assert item_key({"value": {"nested": 1}}).startswith("h:")


def test_extract_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RetrievalLogError, match="not found"):
        extract_retrieval_log(tmp_path / "absent.jsonl")


def test_extract_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(RetrievalLogError, match="not valid JSON"):
        extract_retrieval_log(path)


def test_extract_missing_question_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"memory_context": []}) + "\n", encoding="utf-8")
    with pytest.raises(RetrievalLogError, match="question_id"):
        extract_retrieval_log(path)


def test_extract_duplicate_question_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    row = json.dumps({"question_id": "q", "memory_context": []})
    path.write_text(f"{row}\n\n{row}\n", encoding="utf-8")  # blank line skipped
    with pytest.raises(RetrievalLogError, match="duplicate"):
        extract_retrieval_log(path)


def test_extract_non_list_memory_context_raises(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"question_id": "q", "memory_context": {}}) + "\n", encoding="utf-8")
    with pytest.raises(RetrievalLogError, match="must be a list"):
        extract_retrieval_log(path)


# --------------------------------------------------------------------------- #
# main(): end-to-end with a mocked harness subprocess
# --------------------------------------------------------------------------- #
def _fake_run(
    returncode: int, prompt_rows_src: str | None = None
) -> tuple[Callable[..., subprocess.CompletedProcess[bytes]], list[list[str]]]:
    """Build a subprocess.run stand-in that records the command and optionally writes rows."""
    calls: list[list[str]] = []

    def _run(command: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[bytes]:
        _ = (cwd, check)
        calls.append(command)
        if prompt_rows_src is not None:
            out = Path(command[command.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / run.PROMPT_ROWS_FILENAME).write_text(prompt_rows_src, encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode)

    return _run, calls


def _common_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--harness-dir",
        str(tmp_path / "harness"),
        "--questions-path",
        str(QUESTIONS),
        "--haystack-path",
        str(HAYSTACK),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def test_main_requires_mode(tmp_path: Path) -> None:
    assert run.main(_common_args(tmp_path)) == 2


def test_main_requires_harness_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(run.HARNESS_ENV, raising=False)
    rc = run.main(
        [
            "--questions-path",
            str(QUESTIONS),
            "--haystack-path",
            str(HAYSTACK),
            "--output-dir",
            str(tmp_path / "out"),
            "--mode",
            "plumbing",
        ]
    )
    assert rc == 2


def test_main_score_requires_explicit_reader_and_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An official (score) run must not fall through to an implicit upstream reader/judge.
    fake, calls = _fake_run(returncode=0)
    monkeypatch.setattr(run.subprocess, "run", fake)
    # Harness dir is set by _common_args, so the only failing precondition is the model.
    assert run.main(_common_args(tmp_path, "--mode", "score")) == 2
    assert run.main(_common_args(tmp_path, "--mode", "score", "--model", "m")) == 2
    assert calls == []  # never shelled the harness


def test_main_plumbing_shells_harness_with_skip_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, calls = _fake_run(returncode=0)
    monkeypatch.setattr(run.subprocess, "run", fake)
    rc = run.main(_common_args(tmp_path, "--mode", "plumbing"))
    assert rc == 0
    assert len(calls) == 1
    assert "--skip-reader" in calls[0]
    # A deterministic per-mode config was written to the output dir.
    config = json.loads((tmp_path / "out" / run.MEMORY_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert config["memory_params"]["embedding_params"]["backend"] == "deterministic"


def test_main_score_forwards_reader_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = _fake_run(returncode=0)
    monkeypatch.setattr(run.subprocess, "run", fake)
    rc = run.main(
        _common_args(
            tmp_path,
            "--mode",
            "score",
            "--model",
            "qwen3.5-9b",
            "--evaluator-model",
            "gpt-5.2",
        )
    )
    assert rc == 0
    assert "--skip-reader" not in calls[0]
    assert calls[0][calls[0].index("--model") + 1] == "qwen3.5-9b"


def test_main_harness_nonzero_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _ = _fake_run(returncode=7)
    monkeypatch.setattr(run.subprocess, "run", fake)
    assert run.main(_common_args(tmp_path, "--mode", "plumbing")) == 7


def test_main_smoke_subsets_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = _fake_run(returncode=0)
    monkeypatch.setattr(run.subprocess, "run", fake)
    rc = run.main(_common_args(tmp_path, "--mode", "smoke", "--smoke-limit", "1"))
    assert rc == 0
    q_arg = Path(calls[0][calls[0].index("--questions-path") + 1])
    assert q_arg.name == run.SMOKE_QUESTIONS_FILENAME
    assert [q["id"] for q in json.loads(q_arg.read_text(encoding="utf-8"))] == ["web_q1"]


def test_main_retrieval_no_baseline_writes_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _ = _fake_run(returncode=0, prompt_rows_src=PROMPT_ROWS.read_text(encoding="utf-8"))
    monkeypatch.setattr(run.subprocess, "run", fake)
    rc = run.main(_common_args(tmp_path, "--mode", "retrieval"))
    assert rc == 0
    log = json.loads((tmp_path / "out" / "retrieval_log.json").read_text(encoding="utf-8"))
    assert log["web_q1"] == ["traj_a#3", "traj_b#0"]


def test_main_retrieval_green_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _ = _fake_run(returncode=0, prompt_rows_src=PROMPT_ROWS.read_text(encoding="utf-8"))
    monkeypatch.setattr(run.subprocess, "run", fake)
    # Establish a baseline from a first (identical) run.
    baseline_dir = tmp_path / "baseline"
    assert run.main(_common_args(tmp_path, "--mode", "retrieval")) == 0
    (baseline_dir).mkdir()
    (baseline_dir / "retrieval_log.json").write_text(
        (tmp_path / "out" / "retrieval_log.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    rc = run.main(_common_args(tmp_path, "--mode", "retrieval", "--baseline", str(baseline_dir)))
    assert rc == 0


def test_main_retrieval_red_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _ = _fake_run(returncode=0, prompt_rows_src=PROMPT_ROWS.read_text(encoding="utf-8"))
    monkeypatch.setattr(run.subprocess, "run", fake)
    baseline = tmp_path / "baseline.json"
    # A disjoint baseline (different question ids) is a material change -> RED.
    baseline.write_text(json.dumps({"other_q": ["x"]}), encoding="utf-8")
    rc = run.main(_common_args(tmp_path, "--mode", "retrieval", "--baseline", str(baseline)))
    assert rc == 1


def test_a_row_without_memory_context_is_rejected(tmp_path: Path) -> None:
    """A missing retrieval field must fail, not read as "retrieved nothing".

    An absent ``memory_context`` and an empty one are different facts: the first says the harness
    did not report what it retrieved, the second says it retrieved nothing. Treating them alike
    turns a corrupt row into a plausible retrieval result, and the diff that consumes this log
    would score it as a total membership change rather than as an error.
    """
    rows = tmp_path / "prompt_rows.jsonl"
    rows.write_text(json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")

    with pytest.raises(RetrievalLogError, match="memory_context"):
        extract_retrieval_log(rows)


def test_an_empty_memory_context_is_still_accepted(tmp_path: Path) -> None:
    """Retrieving nothing is a legitimate outcome and stays legitimate."""
    rows = tmp_path / "prompt_rows.jsonl"
    row = json.dumps({"question_id": "q1", "memory_context": []})
    rows.write_text(row + "\n", encoding="utf-8")

    assert extract_retrieval_log(rows) == {"q1": []}


@pytest.mark.parametrize(
    "question_id", [None, True, 12, ["a"]], ids=["null", "bool", "int", "list"]
)
def test_a_non_string_question_id_is_rejected(tmp_path: Path, question_id: object) -> None:
    """Question ids are not coerced.

    ``str(None)`` is ``"None"`` and ``str(True)`` is ``"True"``: coercion invents ids that look
    like real ones, and two upstream rows of different types can collide into one key or raise a
    misleading duplicate error.
    """
    rows = tmp_path / "prompt_rows.jsonl"
    rows.write_text(
        json.dumps({"question_id": question_id, "memory_context": []}) + "\n", encoding="utf-8"
    )

    with pytest.raises(RetrievalLogError, match="question_id"):
        extract_retrieval_log(rows)


def test_a_stale_prompt_rows_file_is_not_extracted_as_fresh(tmp_path: Path) -> None:
    """A prompt-rows file the run did not write must not be mistaken for its output.

    The upstream harness writes straight into the output directory, and a zero exit does not
    prove it rewrote the file: a run that reused a directory, or that exited cleanly without
    producing rows, would leave a previous run's file in place. Extracting it would screen the
    wrong build's retrieval and read as a normal result.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    stale = output_dir / run.PROMPT_ROWS_FILENAME
    stale.write_text(
        json.dumps({"question_id": "q1", "memory_context": []}) + "\n", encoding="utf-8"
    )
    started_after = stale.stat().st_mtime + 1

    with pytest.raises(run.HarnessOutputError, match="not written by this run"):
        run._handle_retrieval(output_dir, baseline=None, started_at=started_after)


def test_the_retrieval_log_is_written_atomically(tmp_path: Path) -> None:
    """The log appears whole or not at all, and no temporary file is left behind."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    rows = output_dir / run.PROMPT_ROWS_FILENAME
    rows.write_text(
        json.dumps({"question_id": "q1", "memory_context": []}) + "\n", encoding="utf-8"
    )

    assert run._handle_retrieval(output_dir, baseline=None, started_at=0.0) == 0

    written = {p.name for p in output_dir.iterdir()}
    assert written == {run.PROMPT_ROWS_FILENAME, retrieval_diff.RETRIEVAL_LOG_FILENAME}
