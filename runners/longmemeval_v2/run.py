"""Uniform LongMemEval-V2 runner — a thin mode wrapper over the upstream harness.

V2's reader, judge, and scorer live in the external upstream harness (checked out
separately; path via ``--harness-dir`` or the ``LME_V2_HARNESS_DIR`` env var). This
runner therefore does NOT reimplement scoring — it composes the shared run-mode
surface (:mod:`runners._modes`) with mode-appropriate harness flags, shells the
harness via :mod:`subprocess`, and reuses :mod:`runners.retrieval_diff` for the free
retrieval-regression signal.

Mode → harness invocation
-------------------------
========= ============================ =========== ==============================
mode      embedding config             reader      artifact
========= ============================ =========== ==============================
smoke     deterministic (offline)      --skip-reader  none (tiny subset, $0)
plumbing  deterministic (offline)      --skip-reader  none (full domain, $0)
retrieval real (text-embedding-3-small) --skip-reader  retrieval log + optional diff
score     real (text-embedding-3-small) reader/judge   the official harness artifacts
========= ============================ =========== ==============================

Only ``score`` runs the real reader/judge and may produce official artifacts; the
other three pass ``--skip-reader`` and are non-official by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runners import _modes, retrieval_diff
from runners._modes import EmbedderKind, ModeSpec, RunMode
from runners.longmemeval_v2.retrieval_log import extract_retrieval_log

HARNESS_ENV = "LME_V2_HARNESS_DIR"
HARNESS_ENTRYPOINT = ("evaluation", "harness.py")
PROMPT_ROWS_FILENAME = "prompt_rows.jsonl"
MEMORY_CONFIG_FILENAME = "memory_config.json"
SMOKE_QUESTIONS_FILENAME = "smoke_questions.json"
SMOKE_HAYSTACK_FILENAME = "smoke_haystack.json"
TRAJECTORIES_FILENAME = "trajectories.jsonl"

DEFAULT_DOMAIN = "web"
DEFAULT_SMOKE_LIMIT = 2
DEFAULT_MEMORY_CONTEXT_MAX_TOKENS = 200_000
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_API_KEY_ENV = "OPENAI_API_KEY"

# Canonical retrieval config, shared by every mode (== the V2 screen).
_TOP_K = 20
_MAX_CONTEXT_CHARS = 24_000
# Deterministic (offline) embedding dimension for the $0 plumbing/smoke modes.
_DETERMINISTIC_DIMENSION = 64
# Real (text-embedding-3-small) embedding knobs, mirroring the canonical screen config.
_EMBEDDING_MAX_INPUT_TOKENS = 4096
_EMBEDDING_BATCH_SIZE = 32
_EMBEDDING_MAX_ATTEMPTS = 12
_EMBEDDING_BASE_RETRY_DELAY_S = 5.0


class SmokeInputError(ValueError):
    """Raised when the smoke subset cannot be built from the given input files."""


class HarnessOutputError(RuntimeError):
    """Raised when the harness output cannot be trusted as this run's own product.

    A zero exit status says the harness finished, not that it wrote what this run is about
    to read. Treating a leftover artifact as fresh would screen a different build and report
    a perfectly ordinary result, so the mismatch is raised rather than worked around.
    """


@dataclass(frozen=True, slots=True)
class ReaderJudgeArgs:
    """Reader/judge pass-through flags for ``score`` mode (owned by the harness).

    Every field is optional: only the flags the caller actually set are forwarded, so
    nothing about the canonical reader/judge is silently hard-coded here. Only the
    env-var NAMES are passed, never key values.

    Attributes:
        model: The reader model id (``--model``).
        base_url: The reader OpenAI-compatible base URL (``--base-url``).
        api_key_env: Name of the env var holding the reader's API key (``--api-key-env``).
        evaluator_model: The judge model id (``--evaluator-model``).
        evaluator_api_key_env: Name of the env var holding the judge's API key
            (``--evaluator-api-key-env``).

    """

    model: str | None
    base_url: str | None
    api_key_env: str | None
    evaluator_model: str | None
    evaluator_api_key_env: str | None


def _deterministic_memory_config() -> dict[str, Any]:
    """Build the offline, $0 deterministic embedding config (smoke/plumbing).

    Returns:
        A memory config with a deterministic (hash-vector) embedder. Its retrieval is
        meaningless, so it is for plumbing ONLY — never a retrieval judgement or score.

    """
    return {
        "memory_type": "engrava",
        "memory_params": {
            "embedding_params": {
                "backend": "deterministic",
                "dimension": _DETERMINISTIC_DIMENSION,
            },
            "retrieval_params": {"top_k": _TOP_K, "max_context_chars": _MAX_CONTEXT_CHARS},
        },
    }


def _real_memory_config(*, model: str, base_url: str, api_key_env: str) -> dict[str, Any]:
    """Build the real ``text-embedding-3-small`` embedding config (retrieval/score).

    Args:
        model: The embedding model id.
        base_url: The OpenAI-compatible embeddings base URL.
        api_key_env: Name of the env var holding the embeddings API key (never the key).

    Returns:
        The memory config the harness passes to the engrava adapter.

    """
    return {
        "memory_type": "engrava",
        "memory_params": {
            "embedding_params": {
                "backend": "openai-compatible",
                "model": model,
                "base_url": base_url,
                "api_key_env": api_key_env,
                "max_input_tokens": _EMBEDDING_MAX_INPUT_TOKENS,
                "batch_size": _EMBEDDING_BATCH_SIZE,
                "max_attempts": _EMBEDDING_MAX_ATTEMPTS,
                "base_retry_delay_s": _EMBEDDING_BASE_RETRY_DELAY_S,
            },
            "retrieval_params": {"top_k": _TOP_K, "max_context_chars": _MAX_CONTEXT_CHARS},
        },
    }


def _reader_judge_flags(reader_judge: ReaderJudgeArgs) -> list[str]:
    """Turn the set reader/judge fields into harness flags (unset fields are skipped).

    Args:
        reader_judge: The reader/judge pass-through arguments.

    Returns:
        The ``--model``/``--base-url``/... flag list for the harness (only set fields).

    """
    flags: list[str] = []
    if reader_judge.model is not None:
        flags += ["--model", reader_judge.model]
    if reader_judge.base_url is not None:
        flags += ["--base-url", reader_judge.base_url]
    if reader_judge.api_key_env is not None:
        flags += ["--api-key-env", reader_judge.api_key_env]
    if reader_judge.evaluator_model is not None:
        flags += ["--evaluator-model", reader_judge.evaluator_model]
    if reader_judge.evaluator_api_key_env is not None:
        flags += ["--evaluator-api-key-env", reader_judge.evaluator_api_key_env]
    return flags


def build_harness_command(
    *,
    harness_dir: Path,
    spec: ModeSpec,
    domain: str,
    questions_path: Path,
    haystack_path: Path,
    output_dir: Path,
    memory_config_path: Path,
    memory_context_max_tokens: int,
    trajectories_path: Path | None,
    reader_judge: ReaderJudgeArgs,
) -> list[str]:
    """Build the upstream-harness argv for a resolved run mode.

    Non-``score`` modes (``smoke``/``plumbing``/``retrieval``) append ``--skip-reader``
    and pass no reader/judge flags, so they can never run the paid reader. Only
    ``score`` (``spec.runs_reader``) forwards the reader/judge pass-through flags.

    Args:
        harness_dir: The upstream LongMemEval-V2 checkout root.
        spec: The resolved run-mode spec.
        domain: The V2 domain (``web`` or ``enterprise``).
        questions_path: The questions file (already subset for ``smoke``).
        haystack_path: The matching haystack file (already subset for ``smoke``).
        output_dir: Where the harness writes its artifacts (incl. ``prompt_rows.jsonl``).
        memory_config_path: The engrava memory config the harness loads.
        memory_context_max_tokens: The assembled-context token cap for the harness.
        trajectories_path: The trajectories file, or ``None`` to rely on the harness default.
        reader_judge: The reader/judge pass-through flags (used only in ``score`` mode).

    Returns:
        The full command list, starting with the current interpreter and the harness
        entry point, ready for :func:`subprocess.run`.

    """
    command: list[str] = [
        sys.executable,
        str(harness_dir.joinpath(*HARNESS_ENTRYPOINT)),
        "--domain",
        domain,
        "--questions-path",
        str(questions_path),
        "--haystack-path",
        str(haystack_path),
        "--output-dir",
        str(output_dir),
        "--memory-config-path",
        str(memory_config_path),
        "--memory-context-max-tokens",
        str(memory_context_max_tokens),
    ]
    if trajectories_path is not None:
        command += ["--trajectories-path", str(trajectories_path)]
    if spec.runs_reader:
        command += _reader_judge_flags(reader_judge)
    else:
        command.append("--skip-reader")
    return command


def _write_smoke_subset(
    questions_path: Path, haystack_path: Path, output_dir: Path, limit: int
) -> tuple[Path, Path]:
    """Write a tiny questions/haystack subset for the offline smoke mode.

    The V2 harness has no ``--limit`` flag, so the smoke subset is materialised as its
    own two files: the first ``limit`` questions plus their haystack entries.

    Args:
        questions_path: The full domain questions file (a JSON list).
        haystack_path: The full domain haystack file (a JSON object ``{qid: [...]}``).
        output_dir: Directory to write the subset files into.
        limit: How many questions to keep (head slice).

    Returns:
        The ``(subset_questions_path, subset_haystack_path)`` pair.

    Raises:
        SmokeInputError: If the questions file is not a JSON list or the haystack file
            is not a JSON object.

    """
    questions_raw: Any = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions_raw, list):
        msg = f"smoke questions file must be a JSON list: {questions_path}"
        raise SmokeInputError(msg)
    subset = questions_raw[:limit]
    ids = {str(q["id"]) for q in subset if isinstance(q, dict) and "id" in q}

    haystack_raw: Any = json.loads(haystack_path.read_text(encoding="utf-8"))
    if not isinstance(haystack_raw, dict):
        msg = f"smoke haystack file must be a JSON object: {haystack_path}"
        raise SmokeInputError(msg)
    subset_haystack = {qid: value for qid, value in haystack_raw.items() if qid in ids}

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_questions_path = output_dir / SMOKE_QUESTIONS_FILENAME
    subset_haystack_path = output_dir / SMOKE_HAYSTACK_FILENAME
    subset_questions_path.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    subset_haystack_path.write_text(json.dumps(subset_haystack, indent=2) + "\n", encoding="utf-8")
    return subset_questions_path, subset_haystack_path


def _precondition_error(args: argparse.Namespace, mode: RunMode) -> str | None:
    """Return a precondition error message for the resolved mode, or None if OK.

    Args:
        args: The parsed CLI namespace.
        mode: The resolved run mode.

    Returns:
        A human-readable error message, or ``None`` when the preconditions hold.

    """
    if mode is RunMode.SCORE and (args.model is None or args.evaluator_model is None):
        return (
            "--mode score requires --model and --evaluator-model so the official run's "
            "reader/judge identity is explicit, never an implicit upstream default."
        )
    if args.harness_dir is None:
        return f"pass --harness-dir or set {HARNESS_ENV} to the LongMemEval-V2 checkout."
    return None


def _require_config_matches_mode(config_path: Path, spec: ModeSpec) -> None:
    """Ensure a caller-supplied memory config honours the mode's embedder contract.

    A mode's embedder kind is a cost/fidelity guarantee: offline modes (smoke/plumbing)
    must use the deterministic embedder, and retrieval/score must use a real one. An
    explicit ``--memory-config-path`` must not silently break that (e.g. a real embedder
    smuggled into a "$0 offline" plumbing run, or a deterministic embedder making a
    retrieval-diff meaningless).

    Args:
        config_path: The caller-supplied memory config.
        spec: The resolved run-mode spec.

    Raises:
        ValueError: If the config's embedding backend contradicts the mode.

    """
    data = json.loads(config_path.read_text(encoding="utf-8"))
    embedding_params = data.get("memory_params", {}).get("embedding_params", {})
    is_deterministic = embedding_params.get("backend") == "deterministic"
    backend = embedding_params.get("backend")
    if spec.embedder is EmbedderKind.LOCAL and not is_deterministic:
        msg = (
            f"mode '{spec.mode.value}' is offline and requires a deterministic embedding "
            f"backend, but {config_path} uses backend={backend!r}. Omit --memory-config-path "
            f"or supply a deterministic config."
        )
        raise ValueError(msg)
    if spec.embedder is not EmbedderKind.LOCAL and is_deterministic:
        msg = (
            f"mode '{spec.mode.value}' needs a real embedder, but {config_path} uses the "
            f"deterministic backend, whose retrieval is meaningless. Supply a real config."
        )
        raise ValueError(msg)


def _resolve_memory_config(args: argparse.Namespace, spec: ModeSpec, output_dir: Path) -> Path:
    """Resolve the memory-config path, writing a per-mode default when none is given.

    Args:
        args: The parsed CLI namespace.
        spec: The resolved run-mode spec (selects deterministic vs real embeddings).
        output_dir: Directory to write a generated config into.

    Returns:
        The path to the memory config the harness should load.

    """
    if args.memory_config_path is not None:
        path: Path = args.memory_config_path
        _require_config_matches_mode(path, spec)
        return path
    if spec.embedder is EmbedderKind.LOCAL:
        config = _deterministic_memory_config()
    else:
        config = _real_memory_config(
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            api_key_env=args.embedding_api_key_env,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / MEMORY_CONFIG_FILENAME
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def _resolve_trajectories_path(args: argparse.Namespace) -> Path | None:
    """Resolve the trajectories file from an explicit flag or ``--data-root``.

    Args:
        args: The parsed CLI namespace.

    Returns:
        An explicit ``--trajectories-path``; else ``<data-root>/trajectories.jsonl``
        when ``--data-root`` is given; else ``None`` (rely on the harness default).

    """
    if args.trajectories_path is not None:
        explicit: Path = args.trajectories_path
        return explicit
    if args.data_root is not None:
        root: Path = args.data_root
        return root / TRAJECTORIES_FILENAME
    return None


def _handle_retrieval(output_dir: Path, baseline: Path | None, *, started_at: float) -> int:
    """Extract the retrieval log after a ``retrieval`` run and optionally diff it.

    Args:
        output_dir: The harness output dir holding ``prompt_rows.jsonl``.
        baseline: A baseline retrieval log (file or directory) to diff against, or
            ``None`` to only write the candidate log.
        started_at: When the harness was launched, as a ``time.time()`` stamp. The
            prompt-rows file must be newer: the harness writes straight into the output
            directory and a zero exit does not prove it rewrote anything, so a reused
            directory would otherwise have a previous run's rows extracted as this run's
            result — screening the wrong build while looking entirely normal.

    Returns:
        ``1`` if a baseline is given and the verdict is RED, else ``0``.

    Raises:
        HarnessOutputError: When the prompt-rows file predates the run.

    """
    prompt_rows = output_dir / PROMPT_ROWS_FILENAME
    if prompt_rows.is_file() and prompt_rows.stat().st_mtime < started_at:
        msg = (
            f"{prompt_rows} was not written by this run (last modified before the harness "
            "started); refusing to extract a previous run's retrieval"
        )
        raise HarnessOutputError(msg)
    log = extract_retrieval_log(prompt_rows)
    log_path = output_dir / retrieval_diff.RETRIEVAL_LOG_FILENAME
    # Written via a temporary file in the same directory and moved into place, so a
    # reader never observes a half-written log and a crash mid-write leaves the previous
    # one intact rather than a truncated file that parses.
    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(log_path)
    print(f"Wrote retrieval log: {log_path} ({len(log)} questions)")  # noqa: T201
    if baseline is None:
        return 0
    base = retrieval_diff.load_retrieval_log(baseline)
    result = retrieval_diff.diff_retrieval_logs(log, base)
    print(  # noqa: T201
        f"retrieval-diff verdict: {result.verdict} "
        f"(changed_fraction={result.changed_fraction:.4f}, compared={result.questions_compared})"
    )
    return 1 if result.verdict is retrieval_diff.Verdict.RED else 0


def _default_harness_dir() -> Path | None:
    """Return the harness dir from ``LME_V2_HARNESS_DIR``, or ``None`` if unset."""
    value = os.environ.get(HARNESS_ENV, "").strip()
    return Path(value) if value else None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongMemEval-V2 runner: a mode wrapper over the upstream harness."
    )
    parser.add_argument(
        "--harness-dir",
        type=Path,
        default=_default_harness_dir(),
        help=f"The upstream LongMemEval-V2 checkout (default: ${HARNESS_ENV}).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Dataset root; {TRAJECTORIES_FILENAME} is read from here unless --trajectories-path.",
    )
    parser.add_argument(
        "--trajectories-path",
        type=Path,
        default=None,
        help="Explicit trajectories file (wins over --data-root).",
    )
    parser.add_argument(
        "--domain",
        choices=["web", "enterprise"],
        default=DEFAULT_DOMAIN,
        help="The V2 domain to run (default: web).",
    )
    parser.add_argument("--questions-path", type=Path, required=True, help="Questions file.")
    parser.add_argument("--haystack-path", type=Path, required=True, help="Haystack file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Harness output directory.")
    parser.add_argument(
        "--memory-config-path",
        type=Path,
        default=None,
        help="Memory config to load; if omitted, a per-mode config is written to --output-dir.",
    )
    parser.add_argument(
        "--memory-context-max-tokens",
        type=int,
        default=DEFAULT_MEMORY_CONTEXT_MAX_TOKENS,
        help="Assembled-context token cap passed to the harness.",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=DEFAULT_SMOKE_LIMIT,
        help="Number of questions in the smoke subset (default: 2).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Real embedding model id for retrieval/score modes.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=DEFAULT_EMBEDDING_BASE_URL,
        help="Real embedding OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default=DEFAULT_EMBEDDING_API_KEY_ENV,
        help="Name of the env var holding the embeddings API key (never the key value).",
    )
    parser.add_argument(
        "--model", default=None, help="Reader model id (score mode; forwarded to the harness)."
    )
    parser.add_argument("--base-url", default=None, help="Reader base URL (score mode).")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Name of the env var holding the reader's API key (score mode; name only).",
    )
    parser.add_argument("--evaluator-model", default=None, help="Judge model id (score mode).")
    parser.add_argument(
        "--evaluator-api-key-env",
        default=None,
        help="Name of the env var holding the judge's API key (score mode; name only).",
    )
    # The shared, benchmark-agnostic run-mode surface (--mode, --baseline).
    _modes.add_mode_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI: resolve the mode, shell the upstream harness, screen retrieval if asked.

    Only ``--mode score`` runs the real reader/judge and produces official artifacts;
    ``smoke``/``plumbing``/``retrieval`` pass ``--skip-reader`` and are non-official.

    Args:
        argv: Optional argument vector (for testing).

    Returns:
        The harness exit code, or — in ``retrieval`` mode with a ``--baseline`` — the
        retrieval-diff exit code (``1`` on a RED verdict).

    """
    args = _parse_args(argv)

    if args.mode is None:
        print("error: --mode is required (smoke|plumbing|retrieval|score).")  # noqa: T201
        return 2
    mode = RunMode(args.mode)
    spec = _modes.resolve_mode(mode)

    precondition = _precondition_error(args, mode)
    if precondition is not None:
        print(f"error: {precondition}")  # noqa: T201
        return 2
    harness_dir: Path = args.harness_dir

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_path: Path = args.questions_path
    haystack_path: Path = args.haystack_path
    if mode is RunMode.SMOKE:
        questions_path, haystack_path = _write_smoke_subset(
            questions_path, haystack_path, output_dir, args.smoke_limit
        )

    try:
        memory_config_path = _resolve_memory_config(args, spec, output_dir)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}")  # noqa: T201
        return 2
    reader_judge = ReaderJudgeArgs(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        evaluator_model=args.evaluator_model,
        evaluator_api_key_env=args.evaluator_api_key_env,
    )
    command = build_harness_command(
        harness_dir=harness_dir,
        spec=spec,
        domain=args.domain,
        questions_path=questions_path,
        haystack_path=haystack_path,
        output_dir=output_dir,
        memory_config_path=memory_config_path,
        memory_context_max_tokens=args.memory_context_max_tokens,
        trajectories_path=_resolve_trajectories_path(args),
        reader_judge=reader_judge,
    )

    print(  # noqa: T201
        f"Resolved --mode {mode}: embedder={spec.embedder} reader/judge={spec.reader_judge} "
        f"official_row={spec.emits_official_row} retrieval_log={spec.emits_retrieval_log}"
    )
    print(f"harness command: {' '.join(command)}")  # noqa: T201

    started_at = time.time()
    completed = subprocess.run(command, cwd=harness_dir, check=False)  # noqa: S603
    if completed.returncode != 0:
        print(f"harness exited with code {completed.returncode}")  # noqa: T201
        return completed.returncode

    if spec.emits_retrieval_log:
        return _handle_retrieval(output_dir, args.baseline, started_at=started_at)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
