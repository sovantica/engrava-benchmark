"""Validate the partitioned results store against the schema + the rules.

Results live as ``results/<benchmark>/<system>/<result_id>.json``. Validation has
several layers:

1. JSON Schema (``results/schema/results.schema.json``) — structure, enums,
   required fields, the provenance/judge/citation conditionals.
2. Cross-field rules the schema cannot express ergonomically:
   - ``group`` is **recomputed**, never trusted: ``A`` iff
     ``system_config.memory_pipeline_llms`` is empty, else ``B``; a mismatch
     against the submitted value is rejected.
   - **headline eligibility:** a headline row (``partial == false`` AND
     ``verification_status == "verified"``) must report all 6 ``per_category``
     keys, ``abstention``, and ``n > 0``.
3. Layout rules (partitioned store):
   - **slug rule:** each path segment matches ``^[a-z0-9][a-z0-9-]*$`` and is a
     registered canonical slug (``scripts/canonical_slugs.py``) — aliases and
     case-drift are rejected;
   - **path/content agreement:** the ``<benchmark>``/``<system>`` directory
     segments equal the canonical slug of the in-file ``benchmark``/``system``;
     the filename equals ``<result_id>.json``;
   - **global ``result_id`` uniqueness** across the whole ``results/`` tree.

Usage:
    python scripts/validate_results.py [path ...]
    # default: validate every results/<benchmark>/<system>/*.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - dependency guard
    print(  # noqa: T201
        "error: jsonschema is required. Install dev deps: pip install -e '.[dev]'",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

# Allow running both as a module (`python -m scripts.validate_results`) and as a
# script (`python scripts/validate_results.py`): the latter needs the repo root on
# sys.path so the `scripts` package resolves.
if __package__ in (None, ""):  # pragma: no cover - script-invocation bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import canonical_slugs as cs

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "results" / "schema" / "results.schema.json"
RESULTS_DIR = REPO_ROOT / "results"

_OFFICIAL_CATEGORIES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
)

# A well-formed artifact checksum: sha256 + 64 lowercase hex chars.
_CHECKSUM_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def build_validator() -> Draft202012Validator:
    """Return the compiled result-row JSON Schema validator."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _artifact_reference(row: dict[str, Any]) -> str:
    """Return the expected repo-relative artifact directory for ``row``."""
    return f"results/{cs.benchmark_slug(row)}/{cs.system_slug(row)}/{row.get('result_id')}/"


def _artifact_dir(row: dict[str, Any], results_dir: Path) -> Path:
    """Return the expected in-repo artifact directory for ``row``."""
    return results_dir / cs.benchmark_slug(row) / cs.system_slug(row) / str(row.get("result_id"))


def _artifact_path_errors(row: dict[str, Any]) -> list[str]:
    """Return errors for the row's repo-relative artifact reference."""
    url = row.get("reproduction_artifact_url")
    if not isinstance(url, str) or not url.strip():
        return ["sovantica-run row must carry a reproduction_artifact_url"]
    if url.startswith("/") or "://" in url:
        return ["reproduction_artifact_url must be a repo-relative artifact directory path"]
    try:
        expected_url = _artifact_reference(row)
    except KeyError as exc:
        return [str(exc.args[0]) if exc.args else "unregistered artifact path"]
    if url != expected_url:
        return [
            f"reproduction_artifact_url {url!r} != expected repo-relative path {expected_url!r}"
        ]
    return []


def _artifact_bundle_errors(
    row: dict[str, Any],
    *,
    results_dir: Path,
    checksum: str,
) -> list[str]:
    """Return errors for the row's sibling artifact bundle on disk."""
    try:
        bundle_dir = _artifact_dir(row, results_dir)
    except KeyError as exc:
        return [str(exc.args[0]) if exc.args else "unregistered artifact path"]
    if not bundle_dir.is_dir():
        rel = bundle_dir.relative_to(results_dir)
        return [f"artifact directory is missing: {rel}"]
    actual = _bundle_checksum(bundle_dir)
    if actual != checksum:
        return [f"artifact checksum mismatch: row says {checksum!r}, bundle is {actual!r}"]
    return []


def _artifact_errors(row: dict[str, Any], *, results_dir: Path | None = None) -> list[str]:
    """Return reproduction-artifact violations for a ``sovantica-run`` row.

    A ``sovantica-run`` row must carry a repo-relative artifact directory path and a
    well-formed ``artifact_checksum`` (``sha256:<64hex>``). When ``results_dir`` is
    supplied, the sibling artifact directory must exist and match the checksum.

    Args:
        row: The parsed result row.
        results_dir: Optional ``results/`` root used to verify the in-repo bundle.

    Returns:
        A list of artifact error strings (empty = OK).

    """
    if row.get("provenance") != "sovantica-run":
        return []
    errors: list[str] = []
    errors.extend(_artifact_path_errors(row))
    checksum = row.get("artifact_checksum")
    if not isinstance(checksum, str) or not _CHECKSUM_RE.match(checksum):
        errors.append(
            "sovantica-run row must carry a well-formed artifact_checksum (sha256:<64 hex chars>)"
        )
    elif results_dir is not None:
        errors.extend(_artifact_bundle_errors(row, results_dir=results_dir, checksum=checksum))
    return errors


def _expected_group(row: dict[str, Any]) -> str:
    """Return the group implied by the memory-pipeline LLM list."""
    llms = row.get("system_config", {}).get("memory_pipeline_llms", [])
    return "A" if not llms else "B"


def _cross_field_errors(row: dict[str, Any]) -> list[str]:
    """Return cross-field rule violations not expressible in JSON Schema."""
    errors: list[str] = []

    expected = _expected_group(row)
    if row.get("group") != expected:
        errors.append(
            f"group mismatch: submitted {row.get('group')!r} but "
            f"memory_pipeline_llms implies {expected!r}"
        )

    is_headline = row.get("partial") is False and (row.get("verification_status") == "verified")
    if is_headline:
        metrics = row.get("metrics", {})
        per_cat = metrics.get("per_category", {})
        missing = [c for c in _OFFICIAL_CATEGORIES if c not in per_cat]
        if missing:
            errors.append(f"headline row missing per_category keys: {missing}")
        if "abstention" not in metrics:
            errors.append("headline row missing abstention metric")
        if not isinstance(row.get("n"), int) or row.get("n", 0) <= 0:
            errors.append("headline row must report n > 0")

    return errors


def _segment_errors(segments: tuple[str, ...]) -> list[str]:
    """Return slug-shape violations for the two path segments."""
    return [
        f"path segment {seg!r} is not a valid slug "
        "(^[a-z0-9][a-z0-9-]*$ — lowercase, leading alphanumeric, hyphenated)"
        for seg in segments
        if not cs.SLUG_RE.match(seg)
    ]


def _layout_errors(path: Path, row: dict[str, Any], results_dir: Path) -> list[str]:
    """Return path/slug/agreement violations for a result file's location.

    Args:
        path: The result file path.
        row: The parsed row.
        results_dir: The ``results/`` root the path is relative to.

    Returns:
        A list of layout error strings (empty = OK).

    """
    errors: list[str] = []
    try:
        rel = path.resolve().relative_to(results_dir.resolve())
    except ValueError:
        # A path outside results_dir (e.g. an explicit ad-hoc file) skips layout checks.
        return errors

    parts = rel.parts
    expected_depth = 3  # <benchmark>/<system>/<file>.json
    if len(parts) != expected_depth:
        return [
            f"result must live at results/<benchmark>/<system>/<result_id>.json; "
            f"found depth {len(parts)} at {rel.as_posix()!r}"
        ]

    benchmark_seg, system_seg, filename = parts
    errors.extend(_segment_errors((benchmark_seg, system_seg)))

    if benchmark_seg not in cs.REGISTERED_BENCHMARK_SLUGS:
        errors.append(
            f"benchmark segment {benchmark_seg!r} is not a registered canonical slug "
            f"{sorted(cs.REGISTERED_BENCHMARK_SLUGS)}"
        )
    if system_seg not in cs.REGISTERED_SYSTEM_SLUGS:
        errors.append(
            f"system segment {system_seg!r} is not a registered canonical slug "
            f"{sorted(cs.REGISTERED_SYSTEM_SLUGS)}"
        )

    # path <-> content agreement: segments must equal the row's own canonical slugs.
    try:
        if benchmark_seg != cs.benchmark_slug(row):
            errors.append(
                f"benchmark path segment {benchmark_seg!r} != canonical slug of "
                f"in-file benchmark/split ({cs.benchmark_slug(row)!r})"
            )
    except KeyError as exc:
        errors.append(str(exc.args[0]) if exc.args else "unregistered benchmark/split")
    try:
        if system_seg != cs.system_slug(row):
            errors.append(
                f"system path segment {system_seg!r} != canonical slug of in-file "
                f"system ({cs.system_slug(row)!r})"
            )
    except KeyError as exc:
        errors.append(str(exc.args[0]) if exc.args else "unregistered system")

    expected_name = f"{row.get('result_id', '')}.json"
    if filename != expected_name:
        errors.append(f"filename {filename!r} != <result_id>.json ({expected_name!r})")

    return errors


def _bundle_checksum(artifact_dir: Path) -> str:
    r"""Recompute the ``sha256:`` checksum of an on-disk artifact bundle.

    Mirrors ``runners.longmemeval.artifact.artifact_checksum`` without importing the
    runner: hash each component's ``name\\0contents\\0`` in sorted-name order.

    Args:
        artifact_dir: The directory holding the artifact components.

    Returns:
        ``"sha256:<hex>"`` over the bundle.

    """
    hasher = hashlib.sha256()
    for file in sorted(p for p in artifact_dir.iterdir() if p.is_file()):
        hasher.update(file.name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(file.read_text(encoding="utf-8").encode("utf-8"))
        hasher.update(b"\x00")
    return f"sha256:{hasher.hexdigest()}"


def validate_file(
    path: Path,
    validator: Draft202012Validator,
    *,
    results_dir: Path = RESULTS_DIR,
) -> list[str]:
    """Validate a single result file; return a list of error strings (empty=OK).

    Args:
        path: The result file.
        validator: The compiled JSON Schema validator.
        results_dir: The ``results/`` root, used for the layout/path checks.

    Returns:
        All schema, cross-field, layout, and artifact errors for this file.

    """
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]
    errors = [
        f"schema: {e.message} (at {'/'.join(str(p) for p in e.path)})"
        for e in validator.iter_errors(row)
    ]
    errors.extend(_cross_field_errors(row))
    errors.extend(_layout_errors(path, row, results_dir))
    errors.extend(_artifact_errors(row, results_dir=results_dir))
    return errors


def iter_result_files(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Return every result JSON in the partitioned tree (skips ``schema/``).

    Args:
        results_dir: The ``results/`` root.

    Returns:
        Sorted result file paths under ``results/<benchmark>/<system>/``.

    """
    return sorted(p for p in results_dir.glob("*/*/*.json") if p.is_file())


def _rel_to_results(path: Path, results_dir: Path) -> str:
    """Return ``path`` relative to ``results_dir`` (posix), or its name as a fallback."""
    try:
        return path.resolve().relative_to(results_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _duplicate_id_errors(
    paths: list[Path], *, results_dir: Path = RESULTS_DIR
) -> dict[Path, list[str]]:
    """Return per-file errors for any ``result_id`` duplicated across the tree.

    Args:
        paths: All result files.
        results_dir: The ``results/`` root, for reporting paths unambiguously
            relative to it (a bare filename is ambiguous under partitioning).

    Returns:
        A mapping of file -> error strings for files sharing a ``result_id``.

    """
    by_id: dict[str, list[Path]] = {}
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rid = row.get("result_id")
        if isinstance(rid, str) and rid:
            by_id.setdefault(rid, []).append(path)

    errors: dict[Path, list[str]] = {}
    for rid, group in by_id.items():
        if len(group) > 1:
            rels = sorted(_rel_to_results(p, results_dir) for p in group)
            others = ", ".join(rels)
            for path in group:
                errors.setdefault(path.resolve(), []).append(
                    f"duplicate result_id {rid!r} (in: {others}); result_id must be globally unique"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate the partitioned results store (or the given files).

    Args:
        argv: Optional list of explicit paths; defaults to every result file under
            ``results/<benchmark>/<system>/``. Global ``result_id`` uniqueness is
            always checked across the WHOLE tree, regardless of ``argv``.

    Returns:
        ``0`` if all valid, ``1`` if any file fails.

    """
    validator = build_validator()

    # Read the module global at call time so tests can repoint RESULTS_DIR.
    tree = iter_result_files(RESULTS_DIR)
    paths = [Path(a) for a in argv] if argv else tree
    if not paths:
        print("No result files to validate (the results tree holds no rows yet).")  # noqa: T201
        return 0

    # Global uniqueness spans the entire tree, not just the files under review.
    dup_errors = _duplicate_id_errors(tree, results_dir=RESULTS_DIR)

    failed = 0
    for path in paths:
        errors = validate_file(path, validator, results_dir=RESULTS_DIR)
        errors.extend(dup_errors.get(path.resolve(), []))
        if errors:
            failed += 1
            print(f"FAIL {path.name}")  # noqa: T201
            for err in errors:
                print(f"  - {err}")  # noqa: T201
        else:
            print(f"OK   {path.name}")  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
