"""Assemble + write a schema-valid result JSON from a runner run.

This module turns a completed run (config + computed ``metrics`` + provenance
inputs) into a ``results/<result_id>.json`` row that validates against
``results/schema/results.schema.json``, then validates it via
``scripts.validate_results``. It records the reproducibility pins (engrava version
+ dist hash, runner commit, scorer version, artifact path + checksum) so the row is
a real reproduction key, not an aspiration.

Phase 1 emits only ``provenance: sovantica-run`` rows (``citation: null``). The
schema is forward-compatible for vendor/community rows added in a later phase.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def engrava_dist_hash() -> str:
    """Return a stable identity hash of the installed ``engrava`` distribution.

    Hashes the installed package's RECORD (file list + hashes) so the value
    pins the exact installed wheel. Falls back to the version string if the
    RECORD is unavailable.

    Returns:
        ``"sha256:<hex>"`` identifying the installed engrava distribution.

    """
    try:
        dist = metadata.distribution("engrava")
        record = dist.read_text("RECORD") or dist.version
    except metadata.PackageNotFoundError:
        record = "engrava-not-installed"
    digest = hashlib.sha256(record.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def engrava_version() -> str:
    """Return the installed ``engrava`` version (or ``"unknown"`` if absent)."""
    try:
        return metadata.version("engrava")
    except metadata.PackageNotFoundError:
        return "unknown"


def runner_commit() -> str:
    """Return ``engrava-benchmark@<short-sha>`` for the current runner checkout.

    Returns:
        The runner-commit provenance string; ``@unknown`` if git is unavailable.

    """
    try:
        sha = subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        sha = "unknown"
    return f"engrava-benchmark@{sha}"


def file_checksum(path: Path) -> str:
    """Return ``"sha256:<hex>"`` for a file's contents.

    Args:
        path: The file to checksum.

    Returns:
        The sha256 checksum string.

    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def build_result(
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    n: int,
    result_id: str,
    date: str,
    verification_status: str = "unverified",
    partial: bool = False,
    reproduction_artifact_url: str = "",
    artifact_checksum: str = "",
    artifact_license: str = "MIT",
    notes: str = "",
) -> dict[str, Any]:
    """Build a ``sovantica-run`` result row from a completed run.

    Args:
        config: The runner config used (provides every number-affecting param).
        metrics: The computed ``metrics`` block (official semantics).
        n: Number of questions scored.
        result_id: Stable, immutable render key.
        date: Run date (``YYYY-MM-DD``).
        verification_status: ``verified`` | ``pending`` | ``unverified``. A
            ``sovantica-run`` row only becomes ``verified`` after maintainer review.
        partial: ``True`` for a head-sliced run (never a headline).
        reproduction_artifact_url: Repo-relative path of the reproduction artifact
            directory. If omitted, it is derived from the row's canonical result path.
        artifact_checksum: ``sha256:`` checksum of the artifact.
        artifact_license: License of the artifact contents.
        notes: Free-form notes.

    Returns:
        A dict matching the results schema (provenance ``sovantica-run``).

    """
    reader = config["reader"]
    judge = config["judge"]
    eng_version = engrava_version()
    memory_pipeline_llms: list[dict[str, Any]] = []  # stock engrava => Group A
    harness_cfg = config["harness"]
    # The harness fixes reader+judge+prompt+context-format+scorer, so it is the coarse
    # identity axis above the finer version pins. For the native runner the version is
    # the current runner commit unless config overrides it.
    harness = {
        "name": harness_cfg["name"],
        "source": harness_cfg["source"],
        "version": harness_cfg.get("version") or runner_commit(),
    }
    row = {
        "schema_version": "1.0",
        "result_id": result_id,
        "system": "Engrava",
        "system_version": eng_version,
        "tier": "engrava",
        "provenance": "sovantica-run",
        "verification_status": verification_status,
        "benchmark": config["benchmark"],
        "benchmark_version": config["benchmark_version"],
        "dataset_revision": config["dataset_revision"],
        "split": config["split"],
        "partial": partial,
        "date": date,
        "engrava_version": eng_version,
        "engrava_dist_hash": engrava_dist_hash(),
        "runner_commit": runner_commit(),
        "harness": harness,
        "system_config": {
            "adapter": "engrava_adapter",
            "embedder": config["embedder"],
            "embedder_endpoint": config["embedder_endpoint"],
            "memory_pipeline_llms": memory_pipeline_llms,
            "params": {"embedder_spec": config["embedder_spec"]},
        },
        "group": "A" if not memory_pipeline_llms else "B",
        "reader_model": reader["model"],
        "reader_snapshot": reader["snapshot"],
        "reader_endpoint": reader["endpoint"],
        "reader_sampling": reader.get("sampling", {"temperature": 0.0}),
        "judge_model": judge["model"],
        "judge_snapshot": judge["snapshot"],
        "judge_endpoint": judge["endpoint"],
        "scorer_version": config["scorer_version"],
        "reader_version": config["reader_version"],
        "retriever": config["retriever"],
        "granularity": config["granularity"],
        "top_k": config["top_k"],
        "ingestion_regime": config["ingestion_regime"],
        "metrics": dict(metrics),
        "n": n,
        "reproduction_artifact_url": reproduction_artifact_url,
        "artifact_checksum": artifact_checksum,
        "artifact_license": artifact_license,
        "citation": None,
        "notes": notes,
    }
    if not reproduction_artifact_url:
        row["reproduction_artifact_url"] = artifact_reference(row)
    return row


def result_path(row: Mapping[str, Any], *, results_dir: Path = RESULTS_DIR) -> Path:
    """Return the partitioned path: ``<benchmark>/<harness>/<system>/<result_id>.json``.

    The three directory segments are the row's own canonical benchmark, harness, and
    system slugs, so the path is a faithful projection of the content.

    Args:
        row: The result row.
        results_dir: The ``results/`` root (default: the repo ``results/``).

    Returns:
        The partitioned path for this row.

    """
    from scripts import canonical_slugs as cs  # noqa: PLC0415

    benchmark = cs.benchmark_slug(dict(row))
    harness = cs.harness_slug(dict(row))
    system = cs.system_slug(dict(row))
    return results_dir / benchmark / harness / system / f"{row['result_id']}.json"


def artifact_path(row: Mapping[str, Any], *, results_dir: Path = RESULTS_DIR) -> Path:
    """Return the sibling reproduction-artifact directory for ``row``.

    Args:
        row: The result row.
        results_dir: The ``results/`` root (default: the repo ``results/``).

    Returns:
        ``results/<benchmark>/<harness>/<system>/<result_id>/`` as a filesystem path.

    """
    return result_path(row, results_dir=results_dir).with_suffix("")


def artifact_reference(row: Mapping[str, Any]) -> str:
    """Return the repo-relative artifact directory recorded in a row.

    The value is intentionally independent of the caller's local checkout path, so
    emitted rows never reveal maintainer-specific filesystem locations.

    Args:
        row: The result row.

    Returns:
        A POSIX repo-relative path ending with ``/``.

    """
    from scripts import canonical_slugs as cs  # noqa: PLC0415

    return (
        f"results/{cs.benchmark_slug(dict(row))}/{cs.harness_slug(dict(row))}/"
        f"{cs.system_slug(dict(row))}/{row['result_id']}/"
    )


def result_reference(row: Mapping[str, Any]) -> str:
    """Return the repo-relative result row path recorded by official runs."""
    return artifact_reference(row).rstrip("/") + ".json"


def _check_unique_result_id(row: Mapping[str, Any], *, out: Path, results_dir: Path) -> None:
    """Raise if ``row`` would duplicate a result id at another path."""
    import scripts.validate_results as vr  # noqa: PLC0415

    result_id = row["result_id"]
    for existing in vr.iter_result_files(results_dir):
        if existing.resolve() == out.resolve():
            continue
        try:
            other = json.loads(existing.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if other.get("result_id") == result_id:
            rel = existing.resolve().relative_to(results_dir.resolve())
            msg = (
                f"duplicate result_id {result_id!r}: already present at "
                f"{rel.as_posix()}; result_id must be globally unique"
            )
            raise ValueError(msg)


def write_artifact_and_validate(
    row: Mapping[str, Any],
    artifact_bundle: Mapping[str, str],
    *,
    results_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    """Write a row and its sibling artifact bundle, then validate both.

    Args:
        row: The result row.
        artifact_bundle: ``{filename: contents}`` from
            :func:`runners.longmemeval.artifact.build_artifact`.
        results_dir: The ``results/`` root (default: the repo ``results/``).

    Returns:
        ``(row_path, artifact_dir)``.

    Raises:
        ValueError: If the row or sibling artifact bundle fails validation.

    """
    from runners.longmemeval import artifact as artifact_mod  # noqa: PLC0415

    out = result_path(row, results_dir=results_dir)
    artifact_dir = artifact_path(row, results_dir=results_dir)
    _check_unique_result_id(row, out=out, results_dir=results_dir)

    artifact_mod.write_artifact(artifact_bundle, artifact_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import scripts.validate_results as vr  # noqa: PLC0415

    errors = vr.validate_file(out, vr.build_validator(), results_dir=results_dir)
    if errors:
        out.unlink(missing_ok=True)
        shutil.rmtree(artifact_dir, ignore_errors=True)
        msg = "emitted result failed validation:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)
    return out, artifact_dir
