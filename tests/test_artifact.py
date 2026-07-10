"""Tests for the reproduction-artifact bundle + its validation rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from jsonschema import Draft202012Validator

import scripts.validate_results as vr
from runners.longmemeval import artifact
from runners.longmemeval.run import RunRecord

if TYPE_CHECKING:
    from typing import Any

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "results/schema/results.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA)

_RECORDS = [
    RunRecord(
        question_id="q1",
        question_type="single-session-user",
        hypothesis="the user adopted a beagle",
        correct=True,
        ranked_official_ids=["s1_2", "s1_1", "s2_1"],
    ),
    RunRecord(
        question_id="q2_abs",
        question_type="multi-session",
        hypothesis="I don't know",
        correct=False,
        ranked_official_ids=["s3_1"],
    ),
]

_CONFIG: dict[str, Any] = {
    "benchmark": "longmemeval",
    "benchmark_version": "v1",
    "dataset_revision": "longmemeval_s@rev1",
    "split": "s_full_500",
    "retriever": "search_hybrid",
    "granularity": "turn",
    "top_k": 20,
    "ingestion_regime": "one-thought-per-user-turn",
    "embedder": "text-embedding-3-small",
    "embedder_spec": "openai:text-embedding-3-small",
    "embedder_endpoint": "api.openai.com",
    "reader": {
        "model": "gpt-4o",
        "snapshot": "gpt-4o-2024-08-06",
        "endpoint": "api.openai.com",
        # A NESTED secret that must NOT leak.
        "api_key": "fake-reader-nested-secret",
    },
    "judge": {
        "model": "gpt-4o",
        "snapshot": "gpt-4o-2024-08-06",
        "endpoint": "api.openai.com",
        # A doubly-nested secret under a `headers` dict.
        "headers": {"authorization": "Bearer judge-token-secret"},
    },
    "reader_version": "longmemeval@abc",
    "scorer_version": "longmemeval@abc",
    # A top-level secret-looking key that must NOT leak into the bundle.
    "api_key": "fake-should-never-appear",
}


# --- bundle contents --------------------------------------------------------- #
def test_hypotheses_are_two_field_no_gold() -> None:
    hyps = artifact.build_hypotheses(_RECORDS)
    for h in hyps:
        assert set(h) == {"question_id", "hypothesis"}
    # The gold answer never appears as a field; only the reader's hypothesis text.
    assert hyps[0]["hypothesis"] == "the user adopted a beagle"


def test_judge_labels_carry_verdict_only() -> None:
    labels = artifact.build_judge_labels(_RECORDS)
    assert labels[0] == {
        "question_id": "q1",
        "question_type": "single-session-user",
        "correct": True,
    }
    assert all("answer" not in label and "gold" not in label for label in labels)


def test_retrieval_log_uses_official_ids() -> None:
    log = artifact.build_retrieval_log(_RECORDS)
    assert log[0]["question_id"] == "q1"
    items = log[0]["retrieval_results"]["ranked_items"]
    # Ranked official corpus ids, best first, no gold/evidence fields.
    assert items == [{"corpus_id": "s1_2"}, {"corpus_id": "s1_1"}, {"corpus_id": "s2_1"}]
    for item in items:
        assert set(item) == {"corpus_id"}


def test_bundle_has_no_gold_or_secret() -> None:
    bundle = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r1")
    blob = "\n".join(bundle.values())
    # No secret value — top-level OR nested — leaks anywhere in the bundle bytes.
    assert "fake-should-never-appear" not in blob
    assert "fake-reader-nested-secret" not in blob
    assert "judge-token-secret" not in blob
    # No secret-bearing KEY leaks either.
    for key in ("api_key", "authorization", "headers", "bearer"):
        assert key not in blob.lower()
    # The config component carries only the declared reproduction pins.
    config_pins = json.loads(bundle[artifact.CONFIG_FILE])
    assert "api_key" not in config_pins


def test_nested_secrets_scrubbed_pins_preserved() -> None:
    bundle = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r1")
    config_pins = json.loads(bundle[artifact.CONFIG_FILE])
    # Nested secrets dropped at every depth ...
    assert "api_key" not in config_pins["reader"]
    assert "headers" not in config_pins["judge"]
    # ... while the legitimate reproduction pins survive.
    assert config_pins["split"] == "s_full_500"
    assert config_pins["reader"]["snapshot"] == "gpt-4o-2024-08-06"
    assert config_pins["reader"]["endpoint"] == "api.openai.com"
    assert config_pins["judge"]["model"] == "gpt-4o"
    assert config_pins["top_k"] == 20
    assert config_pins["scorer_version"] == "longmemeval@abc"


def test_deep_scrub_handles_lists_of_dicts() -> None:
    nested = {
        "memory_pipeline_llms": [
            {"role": "digest", "model": "gpt-4o-mini", "api_key": "fake-list-secret"},
        ],
        "model": "keep-me",
    }
    scrubbed = artifact._deep_scrub(nested)
    assert scrubbed["memory_pipeline_llms"][0] == {"role": "digest", "model": "gpt-4o-mini"}
    assert scrubbed["model"] == "keep-me"
    assert "fake-list-secret" not in json.dumps(scrubbed)


def test_deep_scrub_does_not_over_scrub_bare_key() -> None:
    # A legitimate reproduction pin literally named `key` must SURVIVE (no false
    # positive), while real secret-bearing keys are still dropped at any depth.
    nested = {
        "reader": {
            "model": "gpt-4o",
            "key": "some-non-secret-value",  # benign field named `key`
            "api_key": "fake-real-secret",
            "authorization": "Bearer t",
            "headers": {"x": "y"},
            "token": "tok",
            "secret": "shh",
        }
    }
    scrubbed = artifact._deep_scrub(nested)
    reader = scrubbed["reader"]
    # Benign `key` preserved with its value.
    assert reader["key"] == "some-non-secret-value"
    assert reader["model"] == "gpt-4o"
    # Real secrets dropped.
    for dropped in ("api_key", "authorization", "headers", "token", "secret"):
        assert dropped not in reader
    assert "fake-real-secret" not in json.dumps(scrubbed)


def test_bundle_components_present() -> None:
    bundle = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r1")
    assert set(bundle) == {
        artifact.HYPOTHESES_FILE,
        artifact.JUDGE_LABELS_FILE,
        artifact.RETRIEVAL_LOG_FILE,
        artifact.CONFIG_FILE,
        artifact.MANIFEST_FILE,
    }
    # hypotheses.jsonl has one line per record.
    assert bundle[artifact.HYPOTHESES_FILE].count("\n") == len(_RECORDS)


# --- checksum ---------------------------------------------------------------- #
def test_checksum_is_stable_and_well_formed() -> None:
    b1 = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r1")
    b2 = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r1")
    c1 = artifact.artifact_checksum(b1)
    assert c1 == artifact.artifact_checksum(b2)  # stable
    assert vr._CHECKSUM_RE.match(c1)  # sha256:<64 hex>


def test_checksum_changes_with_contents() -> None:
    b1 = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r1")
    changed = [
        *_RECORDS[:1],
        RunRecord(
            question_id="q2_abs",
            question_type="multi-session",
            hypothesis="different",
            correct=False,
            ranked_official_ids=[],
        ),
    ]
    b2 = artifact.build_artifact(config=_CONFIG, records=changed, result_id="r1")
    assert artifact.artifact_checksum(b1) != artifact.artifact_checksum(b2)


def test_write_artifact_roundtrip(tmp_path: Path) -> None:
    bundle = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r1")
    checksum = artifact.write_artifact(bundle, tmp_path / "r1")
    assert (tmp_path / "r1" / artifact.HYPOTHESES_FILE).exists()
    # The on-disk bundle recomputes to the same checksum the validator uses.
    assert vr._bundle_checksum(tmp_path / "r1") == checksum


# --- validation rules -------------------------------------------------------- #
def _valid_row(valid_sovantica_row: dict[str, Any]) -> dict[str, Any]:
    return valid_sovantica_row


def test_artifact_rule_requires_checksum(valid_sovantica_row: dict[str, Any]) -> None:
    row = valid_sovantica_row
    row["artifact_checksum"] = "sha256:tooshort"
    assert any("artifact_checksum" in e for e in vr._artifact_errors(row))


def test_artifact_rule_requires_url(valid_sovantica_row: dict[str, Any]) -> None:
    row = valid_sovantica_row
    row["reproduction_artifact_url"] = ""
    assert any("reproduction_artifact_url" in e for e in vr._artifact_errors(row))


def test_artifact_rule_skips_non_sovantica(valid_sovantica_row: dict[str, Any]) -> None:
    row = valid_sovantica_row
    row["provenance"] = "vendor-reported"
    row["artifact_checksum"] = ""
    assert vr._artifact_errors(row) == []


def test_validate_file_checks_local_artifact(
    tmp_path: Path, valid_sovantica_row: dict[str, Any]
) -> None:
    # Build a real bundle, set its checksum on the row, write the row + sibling
    # artifact bundle, and confirm validate_file passes; then corrupt and fail.
    bundle = artifact.build_artifact(config=_CONFIG, records=_RECORDS, result_id="r_local")

    row = valid_sovantica_row
    row["result_id"] = "r_local"
    row["reproduction_artifact_url"] = "results/longmemeval-s/engrava/r_local/"
    out = tmp_path / "longmemeval-s" / "engrava" / "r_local.json"
    out.parent.mkdir(parents=True)
    row["artifact_checksum"] = artifact.write_artifact(bundle, out.with_suffix(""))
    out.write_text(json.dumps(row))

    assert vr.validate_file(out, VALIDATOR, results_dir=tmp_path) == []

    # Corrupt the bundle on disk -> checksum mismatch is reported.
    (out.with_suffix("") / artifact.HYPOTHESES_FILE).write_text("tampered\n")
    errors = vr.validate_file(out, VALIDATOR, results_dir=tmp_path)
    assert any("checksum mismatch" in e for e in errors)
