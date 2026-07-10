# Methodology

How a number on this board is produced, pinned, and what it does — and does not —
claim.

## The run protocol

Every result comes from the same uniform pipeline (see `runners/<benchmark>/`):

```
dataset → adapter.ingest → adapter.search → context assembly (runner)
        → reader LLM (runner) → judge LLM (runner) → official scorer → result.json
```

- The **adapter** owns only the memory layer (`ingest` + `search`). It never sees the
  reader, the prompt, the judge, the scorer, or any gold/evidence label.
- The **runner** owns context assembly, the reader and reader prompt, the judge, and the
  official scorer — identical for every system. The memory layer is therefore the only
  independent variable (equal footing). A counted improvement must live behind the
  adapter's `ingest`/`search`, never in the harness.
- The **reader** and the **scorer** are the *unmodified official* implementations of the
  benchmark, replicated byte-faithfully from a pinned upstream commit (recorded as
  `reader_version` / `scorer_version`).

## Reproduction-first

A published number is only as good as its reproducibility. Every result pins every axis
that moves the number:

- the exact `engrava_version` + `engrava_dist_hash` (installed wheel identity),
- the `runner_commit` of this repo,
- the reader and judge model **snapshots** + **endpoints** + reader sampling,
- the `reader_version` and `scorer_version` (pinned official upstream commits),
- the retriever / granularity / `top_k` / ingestion regime,
- every generative-LLM use anywhere in the memory pipeline (`system_config`),
- a **reproduction artifact** (2-field hypotheses, judge labels, retrieval log, full
  config) with a `sha256` `artifact_checksum`.

Anyone can `pip install engrava==<version>`, run the named runner, and reproduce the
number; the artifact lets them audit it without re-running.

## Provenance + verification

Every row is labelled by `provenance` (`sovantica-run` for maintainer-produced Engrava
numbers) and carries a `verification_status`. Only `verified` rows enter the ranked
board. A `sovantica-run` row must carry a reproduction artifact + checksum.

Before a release, maintainers also run a private pre-publish audit over the docs and
result metadata. This audit lives in maintainer tooling, not in this repository or its
public CI; contributors do not need it.

## Comparability

Numbers from different readers/judges are **never** co-ranked. See
[comparability.md](comparability.md) for the segmentation rule and why.

## Groups (A vs B): is there an LLM in the memory pipeline?

Every row carries a `group` label — a two-way classification of the **memory system
under test**, based on a single question: does it invoke a generative LLM anywhere
inside its own pipeline?

- **Group A — no generative LLM in the memory pipeline.** `ingest` and `search` use no
  generative model (embeddings and classical retrieval do not count as generative-LLM
  use). `system_config.memory_pipeline_llms` is empty.
- **Group B — an LLM in the memory pipeline.** The memory layer invokes a generative LLM
  somewhere in `ingest` or `search` — for example to transform, summarise, or
  restructure what it stores at write time, or to select or rerank at retrieval time.
  Every such use is listed in `system_config.memory_pipeline_llms`.

This labels the **memory layer only**. The reader and judge LLMs live in the runner, not
the memory layer (see [The run protocol](#the-run-protocol)), so they are **not** what
`group` refers to — every row uses a reader and a judge regardless of group (the specific
reader/judge snapshot is itself a comparability axis, below).

Why disclose it: a memory layer that uses a generative LLM and one that does not are
different classes of system on cost, latency, and privacy, so a fair read needs to know
which is which. `group` is a descriptive label on each row, **not** a separate ranking
bucket: the board segments and ranks rows by their comparability key (reader, judge,
scorer, dataset, and split — see [Comparability](#comparability) for the exact tuple), and
a single segment may contain rows from either group — the label tells you *how* a system
reaches its number, compared on equal footing within the same comparability segment.

## Metric semantics (benchmark-owned — pointer)

> The precise metric definitions (what `overall_micro`, `macro`, the abstention subset,
> and each per-category cell mean, and how they are counted) are owned by the benchmark
> and pinned to the official scorer. They are **not** restated here to avoid drift — the
> single source of truth is the official scorer (`reader/scorer` upstream pin) plus the
> results [schema](../results/schema/results.schema.json) and the per-benchmark
> `runners/<benchmark>/README.md`. No numbers are quoted on this page.

## Known issues & validity caveats

A benchmark number is a point estimate under a specific protocol, not a universal
ranking. Read each result with these caveats:

- **Reader/judge sensitivity is the dominant confound.** The same retrieval can score
  differently under a different reader or judge snapshot/endpoint. This is why the board
  segments by reader/judge and never co-ranks across them; a cross-reader "ranking" is
  not meaningful.
- **LLM-judge noise.** The judge is itself a model; its verdicts have non-zero error and
  can drift across model snapshots. Two runs of the "same" config can differ within
  judge jitter. Treat small gaps inside a segment as noise, not a result.
- **Single-split point estimate.** A result is one split at one time; there are no
  confidence intervals on the headline number. The per-category `{accuracy, n}` cells
  expose the `n` so variance can be reasoned about, but the board does not compute CIs.
- **Dataset artifacts.** Public memory benchmarks contain idiosyncrasies (templated
  phrasing, dating conventions, distractor construction) that a system can fit without
  generalising. A high number on one benchmark does not transfer automatically.
- **Retrieval vs reading conflation.** Because a reader sits after retrieval, the
  headline measures the *stack*, not retrieval in isolation. The retrieval log in the
  artifact lets a third party inspect retrieval separately.
- **Context-window truncation.** When assembled context exceeds the reader's budget it
  is truncated per the official rule; very long haystacks can drop content the retriever
  ranked lower. This is part of the official protocol and is held constant across
  systems, but it bounds what any system can show.
- **Provenance asymmetry.** Maintainer (`sovantica-run`) rows ship a reproduction
  artifact; vendor/community rows (a later phase) are cited self-reports and carry that
  label — they are not the same evidentiary standard and are never silently mixed.

The honest reading: a number here is *reproducible evidence under a stated protocol*,
useful for comparing systems **within a segment**, and explicitly not a single global
"best memory system" verdict.
