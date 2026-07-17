"""Execute the published "Quick reproduce" example on the free offline smoke path.

The reproducibility promise is the whole value of this repo: the README's
copy-paste reproduce recipe (clone -> ``make install`` -> ``pip install
engrava==<pinned>`` -> ``python runners/longmemeval/run.py`` -> ``make validate``
-> ``make leaderboard``) must actually work. A silently broken command (a renamed
make target, a changed runner flag, a moved entry point, a stale engrava pin) would
publicly falsify that promise. This module makes the promise self-checking.

The full recipe is *paid* (OpenAI-direct reader + judge + embedder), so it cannot
run in CI. Instead the paid bare ``python runners/longmemeval/run.py`` line is
executed as its free offline form ``python runners/longmemeval/run.py --smoke
--results-dir results`` (fixture dataset + a local MiniLM embedder + mock
reader/judge), with ``OPENAI_API_KEY`` and the dataset env var unset — zero API
spend. That command is what emits the bundle; ``make validate`` and ``make
leaderboard`` then run against **that** command's own output. The test asserts a
fresh row appeared, so a runner that stops emitting (yet still exits 0) fails here
instead of being masked by a pre-seeded bundle.

Isolation
---------
The runner, validator, and leaderboard builder resolve their results directory from
their own module location (an editable install, so those modules resolve to the
real checkout regardless of CWD). To guarantee the test never dirties the working
tree, the whole recipe runs inside a **temporary copy of the repository**, and the
smoke run emits into that copy via the runner's explicit ``--results-dir`` seam
(bare ``--smoke`` emits nothing, so the paid line's default emission is mapped to an
isolated one). The test asserts the real ``results/`` tree and ``leaderboard.json``
are byte-for-byte unchanged afterwards.

Doc binding (no markers in the published Markdown)
--------------------------------------------------
The blocks are selected by anchor substrings that live in this module — there is no
test-only fence syntax or marker in ``README.md`` (the engrava rule), so the public
Markdown and its engrava.ai mirror stay clean. Editing a documented command so it no
longer runs, renaming a make target, or bumping the engrava pin out of step with the
installed package makes this test fail.

The engrava.ai benchmark blog post mirrors this reproduce block by hand; it lives in
another repo and cannot be tested from here — this module covers the in-repo
canonical source (``README.md``) only.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.docs._md_blocks import REPO_ROOT, ShellBlock, block_with_anchor

if TYPE_CHECKING:
    from collections.abc import Iterable

README = REPO_ROOT / "README.md"
MAKEFILE = REPO_ROOT / "Makefile"

# --- Anchors / allowlist (live here, NOT in the published Markdown) --------------
# The reproduce recipe is the block that ends by rebuilding the leaderboard; the free
# wiring-check sibling is the block that runs the runner with ``--smoke``.
REPRODUCE_ANCHOR = "make leaderboard"
SMOKE_ANCHOR = "run.py --smoke"

# The bare canonical (paid, emitting) run line the offline mapping rewrites.
BARE_RUN_CMD = "python runners/longmemeval/run.py"
# The documented free wiring-check sibling (emits nothing on its own).
DOCUMENTED_SMOKE_CMD = "python runners/longmemeval/run.py --smoke"
# The offline-safe form the test actually runs: the free smoke path, but emitting
# into an isolated results dir inside the throwaway copy (so the documented command
# produces the very bundle `make validate` / `make leaderboard` then act on).
OFFLINE_RUN_CMD = f"{DOCUMENTED_SMOKE_CMD} --results-dir results"

# The engrava pin in the reproduce block: ``pip install "engrava==X.Y.Z"``.
ENGRAVA_PIN_RE = re.compile(r"engrava==([0-9][0-9A-Za-z.\-]*)")

# Make targets the recipe depends on; a rename must fail CI with a precise message.
REQUIRED_MAKE_TARGETS = ("install", "validate", "leaderboard")

# Env vars the reproduce recipe exports for a paid run; they MUST be unset here.
FORBIDDEN_ENV = ("OPENAI_API_KEY", "ENGRAVA_BENCH_LONGMEMEVAL_S")

# The recipe's exact setup lines that the offline mapping deliberately drops: the
# checkout under test IS the repo, the environment is already installed, and the
# paid env vars must stay unset. This is an ALLOWLIST — any documented line NOT
# matched here (nor rewritten, nor a kept make target) fails the test as unmapped,
# so a new meaningful command cannot be silently ignored.
_DROPPABLE_EXACT = frozenset(
    {
        "cd engrava-benchmark",
        "python -m venv .venv && source .venv/bin/activate",
        "make install",
    }
)
_DROPPABLE_PREFIXES = (
    "git clone ",  # clone step (URL may change)
    "export OPENAI_API_KEY=",  # paid reader/judge key — must stay unset
    "export ENGRAVA_BENCH_LONGMEMEVAL_S=",  # paid dataset pointer — must stay unset
)
# Shell metacharacters that could introduce a SECOND command on a line: chaining
# (`&&`/`||`/`;`/`|`), backgrounding (`&`), and command/process substitution (`$(...)`,
# backticks, `>(...)` — all caught via `$`, backtick, and the parens). A prefix-matched
# droppable line containing any of these (e.g. `export OPENAI_API_KEY=x && make prepare`,
# `export OPENAI_API_KEY=$(make prepare)`) must NOT be dropped whole — the appended
# command would be silently lost — so it is flagged unmapped instead. Only a "plain"
# line with none of these can be prefix-droppable. `<`/`>` are deliberately excluded:
# the real recipe's dataset line uses `<...>` placeholder brackets, and a bare
# redirection introduces no new command (process substitution is caught by the parens).
# The exact-match droppables are whole recognised commands, so the legitimate `&&` in
# the venv-activate line is unaffected (it is matched before this check).
_METACHAR_RE = re.compile(r"[;&|$()`]")

# Generous bound for the offline subprocess: a one-time MiniLM load on a constrained
# CPU dominates; validate + leaderboard are fast.
_RUN_TIMEOUT_S = 300


# --------------------------------------------------------------------------- #
# Lightweight drift guards (no engrava needed — run in every CI leg)
# --------------------------------------------------------------------------- #
def _reproduce_block() -> ShellBlock:
    return block_with_anchor(README, REPRODUCE_ANCHOR)


def test_reproduce_block_documents_the_expected_commands() -> None:
    """The README reproduce recipe still contains each command the test maps."""
    commands = _reproduce_block().command_lines
    joined = "\n".join(commands)
    for expected in ("make install", "make validate", "make leaderboard", BARE_RUN_CMD):
        assert expected in commands, (
            f"reproduce recipe in README.md no longer contains {expected!r} "
            f"(found commands: {commands}); update the recipe and this test together"
        )
    assert ENGRAVA_PIN_RE.search(joined), (
        "reproduce recipe in README.md no longer pins the engrava version "
        f'(expected a `pip install "engrava==X.Y.Z"` line); found: {commands}'
    )


def test_reproduce_block_pins_a_well_formed_engrava_version() -> None:
    """The documented engrava pin parses to a non-empty version string."""
    version = _pinned_engrava_version()
    assert re.fullmatch(r"[0-9][0-9A-Za-z.\-]*", version), (
        f"engrava pin {version!r} in the README reproduce recipe is not a well-formed version"
    )


def test_smoke_sibling_is_documented() -> None:
    """The README documents the exact free ``--smoke`` sibling the test executes."""
    smoke_block = block_with_anchor(README, SMOKE_ANCHOR)
    assert DOCUMENTED_SMOKE_CMD in smoke_block.command_lines, (
        f"README no longer documents the free sibling {DOCUMENTED_SMOKE_CMD!r} "
        f"(found: {smoke_block.command_lines}); the offline mapping relies on it"
    )


def test_required_make_targets_exist() -> None:
    """The Makefile defines every target the reproduce recipe invokes."""
    makefile = MAKEFILE.read_text(encoding="utf-8")
    for target in REQUIRED_MAKE_TARGETS:
        pattern = re.compile(rf"^{re.escape(target)}\s*:", re.MULTILINE)
        assert pattern.search(makefile), (
            f"Makefile target {target!r} is missing or renamed; the documented "
            f"`make {target}` step in README.md would break"
        )


def test_runner_exposes_the_smoke_flag() -> None:
    """``run.py --help`` advertises ``--smoke`` (a rename breaks CI precisely)."""
    result = subprocess.run(  # noqa: S603 — repo-authored entry point
        [sys.executable, str(REPO_ROOT / "runners" / "longmemeval" / "run.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=_RUN_TIMEOUT_S,
        env=_offline_env(),
    )
    assert result.returncode == 0, f"run.py --help failed:\n{result.stderr}"
    assert "--smoke" in result.stdout, (
        "run.py no longer exposes the documented --smoke flag; the reproduce "
        "recipe's free offline path is gone"
    )


def test_offline_mapping_of_the_real_recipe() -> None:
    """The current recipe maps to the free runner command plus the two make steps."""
    assert _offline_commands(_reproduce_block()) == [
        OFFLINE_RUN_CMD,
        "make validate",
        "make leaderboard",
    ]


@pytest.mark.parametrize(
    "line",
    [
        "export OPENAI_API_KEY=x && make prepare",  # chained onto a paid-env export
        "git clone https://example.com/x.git && ./setup.sh",  # chained onto clone
        'pip install "engrava==9.9.9" ; ./do_extra.sh',  # chained onto the engrava pin
        "export OPENAI_API_KEY=x & make prepare",  # backgrounded second command
        "export OPENAI_API_KEY=$(make prepare)",  # command substitution
        "git clone `./setup.sh`",  # backtick command substitution
        "export OPENAI_API_KEY=x || make prepare",  # OR-chained second command
        "export SOME_REQUIRED=1",  # a brand-new, unrecognised documented command
    ],
)
def test_offline_mapping_flags_chained_or_unknown_commands(line: str) -> None:
    """A metacharacter-bearing droppable line (or a new command) fails as unmapped."""
    block = ShellBlock(path=README, rel="README.md", start_line=1, body=f"{line}\nmake validate")
    with pytest.raises(pytest.fail.Exception, match="unmapped command"):
        _offline_commands(block)


# --------------------------------------------------------------------------- #
# End-to-end execution (needs the engrava extra; skipped on the [dev]-only leg)
# --------------------------------------------------------------------------- #
def test_reproduce_example_executes_offline(tmp_path: Path) -> None:
    """Run the reproduce recipe's offline-safe form end-to-end; assert a clean exit.

    Executes, in a throwaway copy of the repo and with zero API spend, the free
    offline runner command (``--smoke --results-dir results``), then ``make
    validate`` and ``make leaderboard``. The runner command itself emits the bundle
    the make steps act on, and the test asserts a fresh row appeared — so a runner
    that stops emitting fails here rather than passing on a stale bundle. The real
    working tree is asserted untouched.
    """
    pytest.importorskip("engrava")
    pytest.importorskip("aiosqlite")

    # The engrava pin in the doc must match the package the recipe actually runs with.
    installed = metadata.version("engrava")
    pinned = _pinned_engrava_version()
    assert pinned == installed, (
        f"README pins engrava=={pinned} but the installed engrava is {installed}; "
        "the reproduce recipe would not reproduce with the documented version"
    )

    real_before = _tree_snapshot()

    repo = tmp_path / "engrava-benchmark"
    shutil.copytree(REPO_ROOT, repo, ignore=_COPY_IGNORE)
    copy_results = repo / "results"
    rows_before = _result_rows(copy_results)

    # Build the offline command list from the README recipe (not hard-coded) and run
    # it as one script, from the copy, with the paid env vars unset. The runner step
    # emits into the copy's results/ (via --results-dir); make validate/leaderboard
    # then act on THAT freshly emitted bundle.
    commands = _offline_commands(_reproduce_block())
    script = "set -euo pipefail\n" + "\n".join(commands) + "\n"
    completed = subprocess.run(  # noqa: S603 — repo-authored recipe, fixed argv
        ["bash", "-c", script],  # noqa: S607 — bash resolved from PATH by design
        check=False,
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=_RUN_TIMEOUT_S,
        env=_offline_env(),
    )
    assert completed.returncode == 0, (
        "the offline reproduce recipe exited "
        f"{completed.returncode}.\n--- script ---\n{script}"
        f"\n--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )

    # The documented runner command must have emitted its OWN bundle: a fresh row plus
    # its sibling artifact directory. A runner that regressed to emit nothing (yet
    # exited 0) fails HERE — the make steps alone would not catch it.
    new_rows = _result_rows(copy_results) - rows_before
    assert len(new_rows) == 1, (
        "the reproduce recipe's runner step did not emit exactly one new result row "
        f"(new rows: {sorted(new_rows)}); a broken --smoke emission would be masked "
        f"otherwise.\n--- stdout ---\n{completed.stdout}"
    )
    new_row = copy_results / next(iter(new_rows))
    assert new_row.with_suffix("").is_dir(), (
        f"emitted row {new_row.name} has no sibling artifact bundle directory"
    )

    # make leaderboard rewrote the copy's leaderboard; the smoke row stays unverified
    # so it is correctly NOT promoted onto the board.
    assert (repo / "leaderboard.json").is_file()

    # The real tree must be byte-for-byte unchanged (no results/ or leaderboard drift).
    assert _tree_snapshot() == real_before, (
        "running the reproduce example dirtied the real working tree; emission or "
        "leaderboard rebuild escaped the temporary copy"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# Heavy/regenerable directories are excluded from the throwaway copy: the venv and
# caches are large and irrelevant; without ``.venv`` the copy's Makefile falls back
# to the ``python`` on PATH (the active interpreter, forced via ``_offline_env``).
_COPY_IGNORE = shutil.ignore_patterns(
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    "*.egg-info",
)


def _pinned_engrava_version() -> str:
    """Return the engrava version pinned in the README reproduce recipe."""
    match = ENGRAVA_PIN_RE.search(_reproduce_block().body)
    if match is None:
        pytest.fail("README reproduce recipe does not pin an engrava version")
    return match.group(1)


def _is_droppable(command: str) -> bool:
    """Return whether ``command`` is one of the recipe's known-droppable setup lines.

    Matches only the specific clone / cd / venv-activate / ``make install`` /
    ``pip install "engrava==..."`` / paid-env-export lines actually present in the
    current recipe. A new documented line (e.g. a future ``export SOME_SETTING=x``)
    does NOT match and is therefore surfaced as unmapped rather than silently dropped.

    Prefix-matched (variable) droppables are dropped only when they introduce no
    further command: a line like ``export OPENAI_API_KEY=x && make prepare`` (or
    ``... & make prepare``, or ``export OPENAI_API_KEY=$(make prepare)``) matches the
    export prefix but also runs another command, so it is NOT droppable — it falls
    through to the unmapped-command failure instead of being swallowed whole. The
    exact-match droppables are whole recognised commands, so the legitimate ``&&`` in
    the venv-activate line is unaffected.

    Args:
        command: A single recipe command line.

    Returns:
        ``True`` iff the line is a recognised, safely droppable setup step.

    """
    if command in _DROPPABLE_EXACT:
        return True
    if _METACHAR_RE.search(command):
        # A prefix match plus a command-introducing metacharacter would hide the
        # chained/backgrounded/substituted command; flag it unmapped instead.
        return False
    if command.startswith(_DROPPABLE_PREFIXES):
        return True
    return command.startswith("pip install") and "engrava==" in command


def _offline_commands(block: ShellBlock) -> list[str]:
    """Map the reproduce recipe to its offline-safe, zero-spend command list.

    Known-droppable setup lines (see :func:`_is_droppable`) are removed; the bare paid
    runner line becomes its free, emitting, isolated form (:data:`OFFLINE_RUN_CMD`);
    the ``make validate`` / ``make leaderboard`` lines are kept verbatim. Any other
    documented command fails the test, forcing the doc and this mapping to stay in
    sync so no meaningful command can be silently ignored.

    Args:
        block: The extracted reproduce recipe block.

    Returns:
        The offline-safe commands to execute, in document order.

    """
    executable: list[str] = []
    for command in block.command_lines:
        if _is_droppable(command):
            continue
        if command == BARE_RUN_CMD:
            executable.append(OFFLINE_RUN_CMD)
            continue
        if command in ("make validate", "make leaderboard"):
            executable.append(command)
            continue
        pytest.fail(
            f"reproduce recipe contains an unmapped command {command!r}; extend the "
            "offline mapping in this test so the documented example stays executable"
        )
    return executable


def _offline_env() -> dict[str, str]:
    """Return a deterministic, offline, zero-spend environment for the recipe.

    The paid env vars are removed so no API call can be made; the native thread pools
    are pinned so a MiniLM load does not contend on a constrained CPU; the active
    interpreter's directory leads ``PATH`` and is passed as ``PYTHON`` so both the
    ``python`` line and the ``make`` targets use this venv (the copy has no ``.venv``).

    Returns:
        A copy of ``os.environ`` with the offline overrides applied.

    """
    env = dict(os.environ)
    for name in FORBIDDEN_ENV:
        env.pop(name, None)
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHON": sys.executable,
        }
    )
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return env


def _result_rows(results_dir: Path) -> set[str]:
    """Return the partitioned result-row files under ``results_dir``, as posix strings.

    Only depth-4 rows (``<benchmark>/<harness>/<system>/<id>.json``) are counted, so
    the generated ``leaderboard.json`` and bundle components are ignored. Used to
    detect the row a runner step freshly emits.

    Args:
        results_dir: A ``results/`` root (inside the throwaway copy).

    Returns:
        The set of row paths relative to ``results_dir``.

    """
    return {
        p.relative_to(results_dir).as_posix()
        for p in results_dir.glob("*/*/*/*.json")
        if p.is_file()
    }


def _tree_snapshot() -> dict[str, str]:
    """Return a content hash of the real ``results/`` tree and ``leaderboard.json``.

    Used to prove the test leaves the working tree byte-for-byte unchanged. This is
    defense-in-depth: the real isolation is that the recipe runs entirely inside a
    throwaway copy and emits via ``--results-dir`` into it. The snapshot covers the
    regular files under ``results/`` plus ``leaderboard.json`` (the only artifacts a
    stray emission or rebuild could touch).

    Returns:
        A mapping of repo-relative path -> sha256 hex digest.

    """
    targets: Iterable[Path] = (
        *sorted((REPO_ROOT / "results").rglob("*")),
        REPO_ROOT / "leaderboard.json",
    )
    snapshot: dict[str, str] = {}
    for path in targets:
        if path.is_file():
            rel = path.relative_to(REPO_ROOT).as_posix()
            snapshot[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot
