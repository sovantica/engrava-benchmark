"""Shared test fixtures."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

# Make the repo root importable so tests can import the top-level ``integrations``
# package, which is intentionally NOT a distributed package (it is shim code copied
# into external harness checkouts, and is excluded from lint/typecheck/coverage).
# Under ``pytest -q`` (CI) the cwd is not on sys.path, unlike ``python -m pytest``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from runners.longmemeval import artifact, emit  # noqa: E402


@pytest.fixture
def valid_sovantica_row() -> dict[str, Any]:
    """A schema-valid, headline-eligible sovantica-run result row."""
    return copy.deepcopy(
        {
            "schema_version": "1.0",
            "result_id": "lme-s_engrava_0.4.0_2026-06-20_a1b2c3",
            "system": "Engrava",
            "system_version": "0.4.0",
            "tier": "engrava",
            "provenance": "sovantica-run",
            "verification_status": "verified",
            "benchmark": "longmemeval",
            "benchmark_version": "v1",
            "dataset_revision": "longmemeval_s@rev1",
            "split": "s_full_500",
            "partial": False,
            "date": "2026-06-20",
            "engrava_version": "0.4.0",
            "engrava_dist_hash": "sha256:deadbeef",
            "runner_commit": "engrava-benchmark@abc1234",
            "harness": {
                "name": "longmemeval-official",
                "source": "in-repo",
                "version": "engrava-benchmark@abc1234",
            },
            "system_config": {
                "adapter": "engrava_adapter",
                "embedder": "text-embedding-3-small",
                "embedder_endpoint": "api.openai.com",
                "memory_pipeline_llms": [],
                "params": {},
            },
            "group": "A",
            "reader_model": "gpt-4o",
            "reader_snapshot": "gpt-4o-2024-08-06",
            "reader_endpoint": "api.openai.com",
            "reader_sampling": {"temperature": 0.0},
            "judge_model": "gpt-4o",
            "judge_snapshot": "gpt-4o-2024-08-06",
            "judge_endpoint": "api.openai.com",
            "scorer_version": "longmemeval@def5678",
            "retriever": "search_hybrid",
            "granularity": "turn",
            "top_k": 20,
            "ingestion_regime": "one-thought-per-user-turn",
            "metrics": {
                "overall_micro": 0.76,
                "macro": 0.7775,
                "abstention": {"accuracy": 0.5, "n": 40},
                "per_category": {
                    "single-session-user": {"accuracy": 0.8, "n": 80},
                    "single-session-assistant": {"accuracy": 0.7, "n": 70},
                    "single-session-preference": {"accuracy": 0.75, "n": 30},
                    "knowledge-update": {"accuracy": 0.78, "n": 78},
                    "temporal-reasoning": {"accuracy": 0.72, "n": 130},
                    "multi-session": {"accuracy": 0.85, "n": 112},
                },
            },
            "n": 500,
            "reproduction_artifact_url": (
                "results/longmemeval-s/longmemeval-official/engrava/"
                "lme-s_engrava_0.4.0_2026-06-20_a1b2c3/"
            ),
            "artifact_checksum": "sha256:" + "c" * 64,
            "artifact_license": "MIT",
            "citation": None,
            "notes": "",
        }
    )


@pytest.fixture
def valid_artifact_bundle() -> dict[str, str]:
    """A tiny, checksumable reproduction artifact bundle for validator tests."""
    return {
        artifact.HYPOTHESES_FILE: '{"question_id":"q1","hypothesis":"a"}\n',
        artifact.JUDGE_LABELS_FILE: (
            '{"question_id":"q1","question_type":"multi-session","correct":true}\n'
        ),
        artifact.RETRIEVAL_LOG_FILE: (
            '{"question_id":"q1","retrieval_results":{"ranked_items":[]}}\n'
        ),
        artifact.CONFIG_FILE: '{"split":"s_full_500"}\n',
        artifact.MANIFEST_FILE: '{"components":[],"n":1,"result_id":"fixture"}\n',
    }


@pytest.fixture
def write_valid_artifact(valid_artifact_bundle: dict[str, str]):
    """Write a matching in-repo artifact bundle for a result row."""

    def _write(results_dir, row: dict[str, Any]) -> dict[str, Any]:
        row["reproduction_artifact_url"] = emit.artifact_reference(row)
        row["artifact_checksum"] = artifact.write_artifact(
            valid_artifact_bundle,
            emit.artifact_path(row, results_dir=results_dir),
        )
        return row

    return _write
