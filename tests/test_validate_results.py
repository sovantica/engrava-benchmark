"""Tests for the result schema + validator cross-field rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import scripts.validate_results as vr

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "results/schema/results.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA)


def _schema_errors(row: dict[str, Any]) -> list[str]:
    return [e.message for e in VALIDATOR.iter_errors(row)]


def test_valid_row_passes_schema_and_rules(valid_sovantica_row: dict[str, Any]) -> None:
    assert _schema_errors(valid_sovantica_row) == []
    assert vr._cross_field_errors(valid_sovantica_row) == []


def test_hyphenated_category_keys_required(valid_sovantica_row: dict[str, Any]) -> None:
    """A non-official (underscored) category key is rejected by the schema."""
    row = valid_sovantica_row
    per = row["metrics"]["per_category"]
    per["single_session_user"] = per.pop("single-session-user")
    assert _schema_errors(row) != []


def test_metric_cell_shape(valid_sovantica_row: dict[str, Any]) -> None:
    """Each per-category cell must be {accuracy, n} — a bare float is invalid."""
    row = valid_sovantica_row
    row["metrics"]["per_category"]["multi-session"] = 0.85
    assert _schema_errors(row) != []


def test_sovantica_run_requires_official_judge(valid_sovantica_row: dict[str, Any]) -> None:
    """A sovantica-run row with a non-official judge endpoint/snapshot fails."""
    row = valid_sovantica_row
    row["judge_endpoint"] = "https://example.test"
    assert _schema_errors(row) != []


def test_sovantica_run_requires_official_judge_model(
    valid_sovantica_row: dict[str, Any],
) -> None:
    """A sovantica-run row with a non-official judge_model is rejected."""
    row = valid_sovantica_row
    row["judge_model"] = "gpt-4o-mini"
    assert _schema_errors(row) != []


def test_conditionals_require_provenance_present() -> None:
    """The provenance conditionals must not fire on a row missing provenance.

    A row without ``provenance`` should fail on the missing required field, but the
    sovantica-run-only constraints (citation null, official judge) must NOT be
    applied to it (the ``if`` guards include ``required: ["provenance"]``).
    """
    row: dict[str, Any] = {
        "judge_model": "some-other-model",
        "judge_endpoint": "https://example.test",
        "judge_snapshot": "some-snapshot",
        "citation": {
            "source_title": "x",
            "source_url": "https://x",
            "retrieved_date": "2026-06-29",
            "as_reported_config": "y",
        },
    }
    messages = _schema_errors(row)
    # It fails (provenance + many fields missing) ...
    assert messages
    # ... but NOT because the sovantica-run judge/citation constraints were applied.
    assert not any("gpt-4o-2024-08-06" in m for m in messages)
    assert not any("api.openai.com" in m for m in messages)


def test_sovantica_run_forbids_citation(valid_sovantica_row: dict[str, Any]) -> None:
    """A sovantica-run row must carry citation: null."""
    row = valid_sovantica_row
    row["citation"] = {
        "source_title": "x",
        "source_url": "https://x",
        "retrieved_date": "2026-06-29",
        "as_reported_config": "y",
    }
    assert _schema_errors(row) != []


def test_vendor_reported_requires_citation(valid_sovantica_row: dict[str, Any]) -> None:
    """A non-sovantica-run row with null citation fails the conditional."""
    row = valid_sovantica_row
    row["provenance"] = "vendor-reported"
    row["citation"] = None
    assert _schema_errors(row) != []


def test_group_recompute_mismatch(valid_sovantica_row: dict[str, Any]) -> None:
    """group is recomputed: a pipeline LLM with group A is rejected."""
    row = valid_sovantica_row
    row["system_config"]["memory_pipeline_llms"] = [
        {
            "role": "write_time_digest",
            "model": "gpt-4o-mini",
            "snapshot": "gpt-4o-mini-2024-07-18",
            "endpoint": "api.openai.com",
            "prompt_ref": "prompts/digest.md@x",
            "sampling": {"temperature": 0.0},
        }
    ]
    # group still says "A" -> mismatch
    errs = vr._cross_field_errors(row)
    assert any("group mismatch" in e for e in errs)


def test_headline_requires_all_categories(valid_sovantica_row: dict[str, Any]) -> None:
    """A verified, non-partial row missing a category is not headline-eligible."""
    row = valid_sovantica_row
    del row["metrics"]["per_category"]["multi-session"]
    # schema also catches this, but the cross-field rule must too
    errs = vr._cross_field_errors(row)
    assert any("missing per_category" in e for e in errs)


def test_validate_file_roundtrip(
    tmp_path: Path, valid_sovantica_row: dict[str, Any], write_valid_artifact
) -> None:
    write_valid_artifact(tmp_path, valid_sovantica_row)
    path = tmp_path / "longmemeval-s" / "engrava" / f"{valid_sovantica_row['result_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(valid_sovantica_row), encoding="utf-8")
    assert vr.validate_file(path, VALIDATOR, results_dir=tmp_path) == []
