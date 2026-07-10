"""Reproduction-artifact bundling for a LongMemEval run.

A published result row links a **reproduction artifact**: the bundle a third party
needs to audit and replay the number. This module assembles that bundle from a run's
:class:`~runners.longmemeval.run.RunRecord` list + the run config, and computes a
``sha256`` over it so the row's ``artifact_checksum`` pins the exact bytes.

Leakage guard
-------------
The artifact carries NO oracle / gold:

* **Hypotheses** are the official 2-field form ``{"question_id", "hypothesis"}`` —
  the reader's answer only, never the gold answer.
* The **judge labels** carry the per-question verdict (``correct``), not the gold.
* The **retrieval log** carries the ranked official corpus ids per question (the
  retrieval *result*), not the dataset's gold/evidence fields — those live in the
  public dataset a reproducer already has.
* The **config** carries reproduction pins (split, models, retriever, versions),
  no secrets.

The result row records a repo-relative artifact directory path and the bundle
checksum. The artifact directory is committed beside the result row so a row and
its reproduction bundle land atomically in one reviewable change.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from runners.longmemeval.run import RunRecord

# The reproduction-pin keys copied from config into the artifact (number-affecting).
_CONFIG_PIN_KEYS: tuple[str, ...] = (
    "benchmark",
    "benchmark_version",
    "dataset_revision",
    "split",
    "retriever",
    "granularity",
    "top_k",
    "ingestion_regime",
    "embedder",
    "embedder_spec",
    "embedder_endpoint",
    "reader",
    "judge",
    "reader_version",
    "scorer_version",
)

# Generic secret-bearing key names (case-insensitive). Any matching key is dropped at
# any nesting depth before bundling, so a nested credential (e.g. ``reader.api_key`` or
# ``judge.headers.authorization``) can never reach the published config component.
# The names are specific/unambiguous: the bare ``key`` is intentionally excluded so a
# legitimate reproduction pin literally named ``key`` is not over-scrubbed (real API
# keys are caught by ``api_key`` / ``apikey``).
_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "passwd",
        "authorization",
        "auth",
        "credential",
        "credentials",
        "headers",
        "bearer",
    }
)


def _deep_scrub(value: Any) -> Any:  # noqa: ANN401 - recursive over arbitrary JSON
    """Recursively drop secret-bearing keys from a JSON-like value.

    Drops any mapping key whose lowercase name is in :data:`_SECRET_KEYS`, at every
    nesting depth, in dicts and in lists of dicts. Non-secret leaves are returned
    unchanged.

    Args:
        value: A JSON-serialisable value (dict, list, or scalar).

    Returns:
        The same shape with secret-bearing keys removed.

    """
    if isinstance(value, dict):
        return {k: _deep_scrub(v) for k, v in value.items() if str(k).lower() not in _SECRET_KEYS}
    if isinstance(value, list):
        return [_deep_scrub(item) for item in value]
    return value


# Artifact component filenames.
HYPOTHESES_FILE = "hypotheses.jsonl"
JUDGE_LABELS_FILE = "judge_labels.jsonl"
RETRIEVAL_LOG_FILE = "retrieval_log.jsonl"
CONFIG_FILE = "config.json"
MANIFEST_FILE = "manifest.json"


def build_hypotheses(records: Sequence[RunRecord]) -> list[dict[str, str]]:
    """Build the official 2-field hypotheses (no gold).

    Args:
        records: The per-question run records.

    Returns:
        ``[{"question_id", "hypothesis"}, ...]`` — the reader's answer only.

    """
    return [{"question_id": r.question_id, "hypothesis": r.hypothesis} for r in records]


def build_judge_labels(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """Build the per-question judge labels (verdict only, no gold).

    Args:
        records: The per-question run records.

    Returns:
        ``[{"question_id", "question_type", "correct"}, ...]``.

    """
    return [
        {
            "question_id": r.question_id,
            "question_type": r.question_type,
            "correct": r.correct,
        }
        for r in records
    ]


def build_retrieval_log(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """Build the official-format retrieval log (the ``id_map`` consumer).

    Each line is ``{"question_id", "retrieval_results": {"ranked_items": [...]}}``,
    where ``ranked_items`` are the ranked OFFICIAL corpus ids (best first) — the
    faithful retrieval-result portion of the official log. No gold/evidence fields.

    Args:
        records: The per-question run records (carry ranked official corpus ids).

    Returns:
        The retrieval-log lines.

    """
    return [
        {
            "question_id": r.question_id,
            "retrieval_results": {
                "ranked_items": [{"corpus_id": cid} for cid in r.ranked_official_ids],
            },
        }
        for r in records
    ]


def build_artifact(
    *,
    config: Mapping[str, Any],
    records: Sequence[RunRecord],
    result_id: str,
) -> dict[str, str]:
    """Assemble the artifact bundle as ``{filename: file_contents}``.

    Args:
        config: The run config (reproduction pins are copied out).
        records: The per-question run records.
        result_id: The result id (recorded in the manifest).

    Returns:
        A mapping ``{filename: serialized_contents}`` for every bundle component.

    """
    hypotheses = build_hypotheses(records)
    judge_labels = build_judge_labels(records)
    retrieval_log = build_retrieval_log(records)
    # Whitelist the reproduction pins, then deep-scrub any nested secret-bearing key
    # (a credential under reader/judge/etc. must never reach the published config).
    config_pins = _deep_scrub({k: config[k] for k in _CONFIG_PIN_KEYS if k in config})

    def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
        return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)

    return {
        HYPOTHESES_FILE: _jsonl(hypotheses),
        JUDGE_LABELS_FILE: _jsonl(judge_labels),
        RETRIEVAL_LOG_FILE: _jsonl(retrieval_log),
        CONFIG_FILE: json.dumps(config_pins, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        MANIFEST_FILE: json.dumps(
            {
                "result_id": result_id,
                "n": len(records),
                "components": sorted(
                    [HYPOTHESES_FILE, JUDGE_LABELS_FILE, RETRIEVAL_LOG_FILE, CONFIG_FILE]
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    }


def artifact_checksum(bundle: Mapping[str, str]) -> str:
    r"""Compute the ``sha256:`` checksum over the whole bundle, deterministically.

    The digest is over each component's ``name\\0contents`` in sorted-name order, so
    it is stable regardless of dict ordering and binds both names and contents.

    Args:
        bundle: The ``{filename: contents}`` bundle.

    Returns:
        ``"sha256:<hex>"``.

    """
    hasher = hashlib.sha256()
    for name in sorted(bundle):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(bundle[name].encode("utf-8"))
        hasher.update(b"\x00")
    return f"sha256:{hasher.hexdigest()}"


def write_artifact(bundle: Mapping[str, str], artifact_dir: Path) -> str:
    """Write the bundle to ``artifact_dir`` and return its ``sha256:`` checksum.

    Args:
        bundle: The ``{filename: contents}`` bundle.
        artifact_dir: Target directory (created if absent).

    Returns:
        The bundle's ``"sha256:<hex>"`` checksum (set on the result row).

    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in bundle.items():
        (artifact_dir / name).write_text(contents, encoding="utf-8")
    return artifact_checksum(bundle)
