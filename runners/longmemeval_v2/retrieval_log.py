"""Extract a comparable retrieval log from an upstream LongMemEval-V2 harness run.

The upstream harness, run with ``--skip-reader``, writes ``prompt_rows.jsonl``: one
JSON row per question carrying the ranked ``memory_context`` items engrava retrieved
(and then exits before any paid reader/judge). This module reduces that file to the
benchmark-agnostic retrieval-log shape that
:func:`runners.retrieval_diff.diff_retrieval_logs` consumes —
``{question_id: [item_key, ...]}`` — so an engrava code change can be screened for a
retrieval regression with zero reader spend.

Each item's identity is its parsed ``Trajectory ID`` + ``State index`` headers (the
deterministic V2 retrieval identity); when both headers are absent the item falls
back to a content hash. The parsing mirrors the storage-side screen so the two
produce byte-identical keys for the same run.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

_TRAJECTORY_RE = re.compile(r"Trajectory ID:\s*(\S+)")
_STATE_RE = re.compile(r"State index:\s*(\d+)")
_HASH_PREFIX = "h:"
_HASH_WIDTH = 16


class RetrievalLogError(ValueError):
    """Raised when a ``prompt_rows.jsonl`` file is missing or structurally malformed."""


def _hash_value(payload: str) -> str:
    """Return the content-hash fallback key for an item with no structured headers.

    Args:
        payload: The item text (or a stable JSON serialisation of a non-string value).

    Returns:
        A ``"h:"``-prefixed, truncated SHA-1 digest of ``payload``. SHA-1 is used as a
        fast content identity (never for security), matching the storage-side screen.

    """
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()  # noqa: S324 - identity, not security
    return _HASH_PREFIX + digest[:_HASH_WIDTH]


def item_key(item: object) -> str:
    """Derive the stable retrieval identity of one ``memory_context`` item.

    Args:
        item: A single ``memory_context`` entry from a ``prompt_rows.jsonl`` row.
            Normally a mapping with a ``value`` text; any other shape is hashed.

    Returns:
        ``"<trajectory_id>#<state_index>"`` when BOTH headers are present in the
        item's ``value`` text, else a content hash. Requiring both headers keeps two
        distinct states of one trajectory from collapsing to the same key.

    """
    value: object = item.get("value", "") if isinstance(item, dict) else item
    if isinstance(value, str):
        trajectory = _TRAJECTORY_RE.search(value)
        state = _STATE_RE.search(value)
        if trajectory is not None and state is not None:
            return f"{trajectory.group(1)}#{state.group(1)}"
        return _hash_value(value)
    return _hash_value(json.dumps(value, sort_keys=True))


def extract_retrieval_log(prompt_rows_path: Path) -> dict[str, list[str]]:
    """Extract the ranked retrieval log from a harness ``prompt_rows.jsonl`` file.

    Args:
        prompt_rows_path: Path to the ``prompt_rows.jsonl`` the upstream harness wrote
            under its ``--output-dir`` (produced by a ``--skip-reader`` run).

    Returns:
        The mapping ``{question_id: [item_key, ...]}`` (best first), ready for
        :func:`runners.retrieval_diff.diff_retrieval_logs`.

    Raises:
        RetrievalLogError: If the file is missing, a line is not valid JSON, a row is
            not an object, its ``question_id`` is absent or not a string, its
            ``memory_context`` is absent or not a list, or a ``question_id`` appears twice.

    """
    if not prompt_rows_path.is_file():
        msg = f"prompt rows file not found: {prompt_rows_path}"
        raise RetrievalLogError(msg)
    log: dict[str, list[str]] = {}
    lines = prompt_rows_path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"prompt rows line {lineno} is not valid JSON: {prompt_rows_path}"
            raise RetrievalLogError(msg) from exc
        if not isinstance(row, dict) or "question_id" not in row:
            msg = f"prompt rows line {lineno} has no 'question_id' object: {prompt_rows_path}"
            raise RetrievalLogError(msg)
        qid = row["question_id"]
        # Not coerced with str(): that would turn a null into "None" and a bool into "True",
        # inventing ids indistinguishable from real ones, and could collide two upstream rows
        # of different types into one key.
        if not isinstance(qid, str):
            msg = (
                f"prompt rows line {lineno} 'question_id' must be a string, got "
                f"{type(qid).__name__}: {prompt_rows_path}"
            )
            raise RetrievalLogError(msg)
        if qid in log:
            msg = f"duplicate question_id {qid!r} in {prompt_rows_path}"
            raise RetrievalLogError(msg)
        # Absent and empty are different facts: absent means the harness did not report what it
        # retrieved, empty means it retrieved nothing. Defaulting the first to the second turns a
        # corrupt row into a plausible retrieval result the diff would score as a full change.
        if "memory_context" not in row:
            msg = f"prompt rows line {lineno} has no 'memory_context': {prompt_rows_path}"
            raise RetrievalLogError(msg)
        context = row["memory_context"]
        if not isinstance(context, list):
            msg = f"prompt rows line {lineno} 'memory_context' must be a list: {prompt_rows_path}"
            raise RetrievalLogError(msg)
        log[qid] = [item_key(item) for item in context]
    return log
