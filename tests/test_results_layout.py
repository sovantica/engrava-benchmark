"""Tests for the partitioned results store + its layout-validation rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import scripts.validate_results as vr
from runners.longmemeval import artifact, emit
from scripts import canonical_slugs as cs

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "results/schema/results.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA)


def _canonical_path(results_dir: Path, row: dict[str, Any]) -> Path:
    """Write a row at its canonical partitioned path and return the file path."""
    out = results_dir / cs.benchmark_slug(row) / cs.system_slug(row) / f"{row['result_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row), encoding="utf-8")
    return out


# --- canonical slugs --------------------------------------------------------- #
def test_canonical_slugs(valid_sovantica_row: dict[str, Any]) -> None:
    assert cs.benchmark_slug(valid_sovantica_row) == "longmemeval-s"
    assert cs.system_slug(valid_sovantica_row) == "engrava"


def test_unregistered_benchmark_raises() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(KeyError, match="unregistered benchmark"):
        cs.benchmark_slug({"benchmark": "nope", "split": "s_full_500"})


def test_unregistered_system_raises() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(KeyError, match="unregistered system"):
        cs.system_slug({"system": "NotARegisteredSystem"})


def test_layout_errors_on_unregistered_content(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    # A row whose in-file benchmark is unregistered: the path can't match a
    # canonical slug, so validation reports the unregistered benchmark/system.
    row = copy.deepcopy(valid_sovantica_row)
    row["benchmark"] = "nope"
    bad = tmp_path / "longmemeval-s" / "engrava" / f"{row['result_id']}.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps(row))
    errors = vr.validate_file(bad, VALIDATOR, results_dir=tmp_path)
    assert any("unregistered benchmark" in e for e in errors)


def test_emit_result_path_is_partitioned(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    path = emit.result_path(valid_sovantica_row, results_dir=tmp_path)
    assert path == tmp_path / "longmemeval-s" / "engrava" / (
        f"{valid_sovantica_row['result_id']}.json"
    )


# --- path <-> content agreement --------------------------------------------- #
def test_valid_partitioned_row_passes(
    tmp_path: Path, valid_sovantica_row: dict[str, Any], write_valid_artifact
) -> None:
    write_valid_artifact(tmp_path, valid_sovantica_row)
    path = _canonical_path(tmp_path, valid_sovantica_row)
    assert vr.validate_file(path, VALIDATOR, results_dir=tmp_path) == []


def test_benchmark_path_must_match_content(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    rid = valid_sovantica_row["result_id"]
    wrong = tmp_path / "wrong-benchmark" / "engrava" / f"{rid}.json"
    wrong.parent.mkdir(parents=True)
    wrong.write_text(json.dumps(valid_sovantica_row))
    errors = vr.validate_file(wrong, VALIDATOR, results_dir=tmp_path)
    assert any("benchmark segment" in e or "benchmark path segment" in e for e in errors)


def test_system_path_must_match_content(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    rid = valid_sovantica_row["result_id"]
    wrong = tmp_path / "longmemeval-s" / "some-other-system" / f"{rid}.json"
    wrong.parent.mkdir(parents=True)
    wrong.write_text(json.dumps(valid_sovantica_row))
    errors = vr.validate_file(wrong, VALIDATOR, results_dir=tmp_path)
    assert any("system segment" in e or "system path segment" in e for e in errors)


def test_filename_must_be_result_id(tmp_path: Path, valid_sovantica_row: dict[str, Any]) -> None:
    wrong = tmp_path / "longmemeval-s" / "engrava" / "not-the-id.json"
    wrong.parent.mkdir(parents=True)
    wrong.write_text(json.dumps(valid_sovantica_row))
    errors = vr.validate_file(wrong, VALIDATOR, results_dir=tmp_path)
    assert any("filename" in e for e in errors)


# --- slug rule --------------------------------------------------------------- #
def test_no_flat_dump_rejected(tmp_path: Path, valid_sovantica_row: dict[str, Any]) -> None:
    rid = valid_sovantica_row["result_id"]
    flat = tmp_path / f"{rid}.json"
    flat.write_text(json.dumps(valid_sovantica_row))
    errors = vr.validate_file(flat, VALIDATOR, results_dir=tmp_path)
    assert any("results/<benchmark>/<system>" in e for e in errors)


def test_uppercase_segment_rejected(tmp_path: Path, valid_sovantica_row: dict[str, Any]) -> None:
    rid = valid_sovantica_row["result_id"]
    bad = tmp_path / "LongMemEval-S" / "engrava" / f"{rid}.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps(valid_sovantica_row))
    errors = vr.validate_file(bad, VALIDATOR, results_dir=tmp_path)
    assert any("not a valid slug" in e or "not a registered canonical slug" in e for e in errors)


def test_unregistered_system_alias_rejected(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    bad = tmp_path / "longmemeval-s" / "engrava-oss" / (f"{valid_sovantica_row['result_id']}.json")
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps(valid_sovantica_row))
    errors = vr.validate_file(bad, VALIDATOR, results_dir=tmp_path)
    assert any("not a registered canonical slug" in e for e in errors)


# --- global result_id uniqueness -------------------------------------------- #
def test_global_result_id_uniqueness(tmp_path: Path, valid_sovantica_row: dict[str, Any]) -> None:
    # Two files sharing one result_id anywhere in the tree must fail.
    a = _canonical_path(tmp_path, valid_sovantica_row)

    twin = copy.deepcopy(valid_sovantica_row)
    twin["reader_snapshot"] = "gpt-4o-mini-2024-07-18"  # different config, SAME id
    b = a.parent / "duplicate.json"
    b.write_text(json.dumps(twin))  # filename differs but result_id is identical

    dup_errors = vr._duplicate_id_errors(vr.iter_result_files(tmp_path))
    assert a.resolve() in dup_errors
    assert b.resolve() in dup_errors
    assert any("duplicate result_id" in e for e in dup_errors[a.resolve()])


def test_iter_result_files_skips_schema(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    _canonical_path(tmp_path, valid_sovantica_row)
    artifact_dir = tmp_path / "longmemeval-s" / "engrava" / valid_sovantica_row["result_id"]
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}")
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()
    (schema_dir / "results.schema.json").write_text("{}")
    files = vr.iter_result_files(tmp_path)
    assert all("schema" not in p.relative_to(tmp_path).parts for p in files)
    assert len(files) == 1


def test_duplicate_error_reports_relative_path(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    a = _canonical_path(tmp_path, valid_sovantica_row)
    b = a.parent / "duplicate.json"
    b.write_text(json.dumps(valid_sovantica_row))
    dup = vr._duplicate_id_errors(vr.iter_result_files(tmp_path), results_dir=tmp_path)
    # Reported by path-under-results (posix, with the benchmark/system prefix), not
    # the ambiguous bare filename.
    assert any("longmemeval-s/engrava/" in e for e in dup[a.resolve()])


# --- emit enforces global uniqueness BEFORE writing ------------------------- #
def test_emit_rejects_duplicate_result_id_at_different_path(
    tmp_path: Path, valid_sovantica_row: dict[str, Any], valid_artifact_bundle: dict[str, str]
) -> None:
    import pytest  # noqa: PLC0415

    # A file carrying this result_id already exists elsewhere in the tree (here under
    # a different filename in the same partition). Emit must refuse BEFORE writing,
    # not defer the collision to CI.
    canonical = emit.result_path(valid_sovantica_row, results_dir=tmp_path)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    pre_existing = canonical.parent / "previously-placed.json"
    pre_existing.write_text(json.dumps(valid_sovantica_row), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate result_id"):
        emit.write_artifact_and_validate(
            valid_sovantica_row,
            valid_artifact_bundle,
            results_dir=tmp_path,
        )
    # Emit did not create its own file while a colliding one exists.
    assert not canonical.exists()


def test_emit_overwrites_same_path_same_id(
    tmp_path: Path,
    valid_sovantica_row: dict[str, Any],
    valid_artifact_bundle: dict[str, str],
) -> None:
    # Re-emitting the SAME row to its own canonical path is an idempotent overwrite,
    # not a duplicate (the only file with that id stays at its one canonical path).
    valid_sovantica_row["artifact_checksum"] = artifact.artifact_checksum(valid_artifact_bundle)
    valid_sovantica_row["reproduction_artifact_url"] = emit.artifact_reference(valid_sovantica_row)
    first, _first_artifact = emit.write_artifact_and_validate(
        valid_sovantica_row,
        valid_artifact_bundle,
        results_dir=tmp_path,
    )
    # Reuse the row that was made valid by the first write's bundle checksum.
    row = json.loads(first.read_text(encoding="utf-8"))
    again, _again_artifact = emit.write_artifact_and_validate(
        row,
        valid_artifact_bundle,
        results_dir=tmp_path,
    )
    assert again.exists()
