"""Canonical path slugs for the partitioned results store.

Results are partitioned on disk as
``results/<benchmark>/<harness>/<system>/<result_id>.json``. The three directory
segments are **canonical slugs** derived from a result row's own ``benchmark`` (+
``split``), ``harness.name``, and ``system`` fields, so the directory is a faithful
projection of the content — never an independent label that can drift from it.

A slug must:

* match ``^[a-z0-9][a-z0-9-]*$`` (lowercase, leading alphanumeric, hyphen-separated);
* be a member of the small registered canonical list below.

Aliases, case-drift, and underscores are rejected at validation time. The registry
is intentionally tiny and explicit (no fuzzy slugification of arbitrary input) so a
new benchmark or system is a deliberate, reviewable addition here.
"""

from __future__ import annotations

import re
from typing import Any

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# Registered canonical benchmark slugs. Keyed by the in-file ``benchmark`` value and
# the leading token of ``split`` (the LongMemEval family distinguishes S vs M by
# split), mapping to the public directory slug.
#
#   (benchmark, split_family) -> slug
_BENCHMARK_SLUGS: dict[tuple[str, str], str] = {
    ("longmemeval", "s"): "longmemeval-s",
}

# Registered canonical harness slugs. Keyed by the in-file ``harness.name`` value.
# The harness (runner) fixes the reader, judge, prompt, context format, and scorer, so
# it is an explicit identity/comparability axis AND a path segment. A new harness is a
# deliberate, reviewable addition here.
#
#   harness_name -> slug
_HARNESS_SLUGS: dict[str, str] = {
    "longmemeval-official": "longmemeval-official",
}

# Registered canonical system slugs. Keyed by the in-file ``system`` value.
_SYSTEM_SLUGS: dict[str, str] = {
    "Engrava": "engrava",
}

# The flat set of every registered slug (for membership checks / messages).
REGISTERED_BENCHMARK_SLUGS: frozenset[str] = frozenset(_BENCHMARK_SLUGS.values())
REGISTERED_HARNESS_SLUGS: frozenset[str] = frozenset(_HARNESS_SLUGS.values())
REGISTERED_SYSTEM_SLUGS: frozenset[str] = frozenset(_SYSTEM_SLUGS.values())


def _split_family(split: str) -> str:
    """Return the leading family token of a split string (e.g. ``s_full_500`` -> ``s``)."""
    return split.split("_", 1)[0].lower()


def benchmark_slug(row: dict[str, Any]) -> str:
    """Return the registered canonical benchmark slug for a result row.

    Args:
        row: The result row (uses its ``benchmark`` + ``split`` fields).

    Returns:
        The canonical benchmark directory slug.

    Raises:
        KeyError: If the ``(benchmark, split-family)`` pair is not registered.

    """
    benchmark = str(row.get("benchmark", ""))
    family = _split_family(str(row.get("split", "")))
    key = (benchmark, family)
    if key not in _BENCHMARK_SLUGS:
        msg = (
            f"unregistered benchmark/split: benchmark={benchmark!r} "
            f"split-family={family!r}; register it in canonical_slugs.py"
        )
        raise KeyError(msg)
    return _BENCHMARK_SLUGS[key]


def harness_slug(row: dict[str, Any]) -> str:
    """Return the registered canonical harness slug for a result row.

    The harness (runner) fixes the reader, judge, prompt, context format, and
    scorer, so it is both a path segment and a comparability axis.

    Args:
        row: The result row (uses its ``harness.name`` field).

    Returns:
        The canonical harness directory slug.

    Raises:
        KeyError: If the ``harness.name`` value is not registered.

    """
    harness = row.get("harness")
    name = harness.get("name", "") if isinstance(harness, dict) else ""
    name = str(name)
    if name not in _HARNESS_SLUGS:
        msg = f"unregistered harness: {name!r}; register it in canonical_slugs.py"
        raise KeyError(msg)
    return _HARNESS_SLUGS[name]


def system_slug(row: dict[str, Any]) -> str:
    """Return the registered canonical system slug for a result row.

    Args:
        row: The result row (uses its ``system`` field).

    Returns:
        The canonical system directory slug.

    Raises:
        KeyError: If the ``system`` value is not registered.

    """
    system = str(row.get("system", ""))
    if system not in _SYSTEM_SLUGS:
        msg = f"unregistered system: {system!r}; register it in canonical_slugs.py"
        raise KeyError(msg)
    return _SYSTEM_SLUGS[system]
