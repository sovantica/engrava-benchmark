# Comparability rules

The leaderboard is **reader-segmented**, not one global list. This page explains why,
and the exact rule the render and the aggregator enforce.

## The #1 hazard: reader / judge

A memory/retrieval number is not produced by the memory system alone. After retrieval,
a **reader** model answers from the assembled context, and a **judge** model scores that
answer. Both move the number, often by more than the retrieval quality does:

- A strong reader can recover an answer from mediocre retrieval (masking a weak
  retriever); a weak reader can fail on good retrieval.
- Two rows that both say "gpt-4o" are *not* the same measurement if they used different
  model **snapshots**, different **endpoints** (a direct API vs a broker/proxy can
  differ), or a different **scorer** commit.

So presenting numbers from different readers/judges side-by-side as a single ranking is
a **mirage**: it compares reader+judge stacks, not memory systems. The board must never
do that.

## The rule: co-rank only within an identical comparability tuple

Rows are co-ranked **only** when they share this full tuple:

```
(benchmark, benchmark_version, dataset_revision, split,
 harness.name, harness.version,
 reader_snapshot, reader_endpoint, judge_snapshot, judge_endpoint, scorer_version)
```

- The **harness** (runner) fixes the reader, judge, prompt, context format, and scorer,
  so it changes what a number *means*. It is therefore an explicit comparability axis:
  cross-harness rows are never co-ranked, and the harness is also a path segment
  (`results/<benchmark>/<harness>/<system>/…`).
- It is the **full identity**, not just the model family — a `gpt-4o-2024-08-06` row and
  a `gpt-4o-mini` row never share a column, and neither do two "gpt-4o" rows judged on
  different endpoints or scorer commits.
- Rows with different tuples render in **separate, labelled segments**. The board is a
  set of per-reader/judge segments, never one global sorted list.
- `scripts/build_leaderboard.py` groups rows into exactly these segments and orders each
  segment by `overall_micro` descending. The render reads that grouping; it does not
  re-rank across segments.

Every row visibly carries its **reader**, **judge**, and **provenance**. A
row's metric *definition/aggregation* (micro vs macro) is a display axis, never a
cross-row rank.

## Verification (also gating)

- **Verification.** Only `verification_status == "verified"` rows enter a ranked
  segment. `pending`/`unverified` rows render, if at all, in a clearly separate area and
  are never ranked against verified numbers. `build_leaderboard.py` includes verified
  rows only.

## Consequence: the canonical segment is the honest headline

The canonical `gpt-4o-2024-08-06` reader/judge segment is the legitimate headline. Other
readers render in their own segments. Few external numbers were produced on the *exact*
canonical reader, so that segment may be sparse — that is the **truthful** state of the
field, not a defect, and it is exactly what prevents the reader mirage. A bigger,
mixed-reader "ranking" would be more impressive and less true; this board chooses true.
