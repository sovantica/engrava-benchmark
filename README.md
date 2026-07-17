# Engrava Benchmark

**The public reproduction surface for [Engrava](https://github.com/sovantica/engrava)'s
memory-benchmark numbers.** Per-benchmark **runners** + machine-readable **results**,
produced against the public `engrava` package so anyone can reproduce a number.

This repo is **not** a pip package. The unit of distribution is the repo itself:
clone it, `pip install engrava==<version>`, run a runner, reproduce a number.

## What's here

- **`adapters/`** — the pluggable [`MemoryAdapter`](adapters/base.py) seam (`ingest` +
  `search`) and the maintainer-supported [`engrava_adapter.py`](adapters/engrava_adapter.py).
  Any memory database can plug in by adding one adapter file
  ([guide](adapters/README.md)).
- **`runners/`** — uniform benchmark runners. The runner owns context assembly, the
  reader, the reader prompt, the judge, and the official scorer; the adapter owns only
  the memory. (LongMemEval first.)
- **`integrations/`** — thin Engrava adapters for running inside **external** benchmark
  harnesses (as opposed to this repo's own `runners/`): the
  [LongMemEval-V2](integrations/longmemeval_v2/README.md) memory backend and the
  [Agent Memory Benchmark (AMB)](integrations/amb_vectorize/README.md) provider. Each is a
  drop-in shim copied into the upstream checkout; public `engrava` only, no LLM in the memory
  path.
- **`results/`** — machine-readable result JSON (one per result), validated against
  [`results/schema/results.schema.json`](results/schema/results.schema.json).
- **`leaderboard.json`** — generated aggregate of the **verified** rows; consumed by
  [engrava.ai](https://engrava.ai) at build time.
- **`docs/`** — [methodology](docs/methodology.md) + [comparability rules](docs/comparability.md).

## Quick reproduce (copy-paste)

```bash
# 1. clone
git clone https://github.com/sovantica/engrava-benchmark.git
cd engrava-benchmark

# 2. set up + pin the engrava version named by the result you want to reproduce
python -m venv .venv && source .venv/bin/activate
make install
pip install "engrava==0.5.0"   # <- the engrava_version from the result row

# 3. point the runner at the dataset this result pins: the CLEANED LongMemEval-S split
#    from Hugging Face `xiaowu0162/longmemeval-cleaned` (file: longmemeval_s_cleaned.json),
#    NOT the raw longmemeval_s.json. Verify you have the right file — its sha256 must equal
#    the `dataset_revision` in the result row
#    (longmemeval_s_cleaned@sha256:d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442).
export OPENAI_API_KEY=...          # OpenAI-direct reader + judge; never committed
export ENGRAVA_BENCH_LONGMEMEVAL_S=<path-to>/longmemeval_s_cleaned.json

# 4. run the OFFICIAL config — no flags. The bare command IS the canonical run.
python runners/longmemeval/run.py

# 5. validate the emitted result + rebuild the leaderboard
make validate
make leaderboard
```

> **Run it with no flags — strongly recommended.** The bare
> `python runners/longmemeval/run.py` is the canonical configuration used for every
> published number (`gpt-4o-2024-08-06` reader + judge OpenAI-direct, `top_k=20`, the
> pinned official reader/scorer). Any flag that overrides a reader/judge/model/endpoint
> default moves the row out of the canonical comparability segment and it is no longer
> a headline — see [`runners/longmemeval/README.md`](runners/longmemeval/README.md)
> and [comparability rules](docs/comparability.md). The overrides are for exploratory
> or free-local runs only.

Want to verify the wiring **without spending** first? Run the free offline smoke
(real Engrava retrieval + a local embedder + mock reader/judge):

```bash
python runners/longmemeval/run.py --smoke
```

The reader/judge/scorer for a published `sovantica-run` headline are the official
LongMemEval reader prompt, the `gpt-4o-2024-08-06` judge (OpenAI-direct), and the
unmodified official scorer (pinned at a known upstream commit); each is recorded in
the result row (`reader_snapshot`, `judge_snapshot`, `scorer_version`). The canonical
headline requires `reader_endpoint = api.openai.com`. See
[`runners/longmemeval/README.md`](runners/longmemeval/README.md) for exact steps.

## How the leaderboard works

- Each result is a `results/<benchmark>/<harness>/<system>/<result_id>.json` row pinning every axis that moves the number (including the harness that fixes reader+judge+prompt+scorer)
  (engrava version + dist hash, runner commit, reader/judge snapshots + endpoints,
  scorer version, retriever/granularity/`top_k`, plus a reproduction artifact + checksum).
- CI validates each row against the schema + cross-field rules, then rebuilds
  `leaderboard.json` from the **verified** rows only.
- Rows are co-ranked **only within an identical reader/judge/scorer segment** (the
  [comparability guard](docs/comparability.md)) — the board is a set of per-reader
  segments, not one global list. The canonical `gpt-4o-2024-08-06` segment is the
  honest headline.
- [engrava.ai](https://engrava.ai) pinned-reads `leaderboard.json` at build time
  (static, zero-runtime-JS) — a new number goes live via an explicit, reviewable pin
  bump.

## Adding your own system

Implement [`MemoryAdapter`](adapters/base.py) in `adapters/<your_db>.py`, run a runner,
and produce a result. See the [adapter guide](adapters/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every result ships a reproduction artifact and a
provenance label; numbers from different readers/judges are never shown as directly
rank-comparable.

## License

[MIT](LICENSE). Reproduction artifacts declare their own `artifact_license`.
