"""LongMemEval-V2 mode wrapper over the upstream evaluation harness.

Unlike the LongMemEval-S runner, V2's reader, judge, and scorer are owned by the
external upstream harness (``xiaowu0162/LongMemEval-V2``, checked out separately —
NOT vendored here). This package therefore does not reimplement scoring: it shells
the upstream harness with mode-appropriate flags (via :mod:`runners._modes`) and
reuses :mod:`runners.retrieval_diff` for the free retrieval-regression signal.
"""

from __future__ import annotations
