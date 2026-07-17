# runners/longmemeval/

The uniform LongMemEval runner. One pipeline, system-independent except for the
memory layer:

```
dataset → adapter.ingest → adapter.search → context assembly (runner)
        → reader LLM (runner) → judge LLM (runner) → official scorer → result.json
```

The **runner** owns context assembly, the reader, the reader prompt, the judge, and
the official scorer. The **adapter** owns only ingest + retrieve. This is the
equal-footing contract.

## Files

- `run.py` — the uniform pipeline + dataset loading + context assembly + the model
  factory + result emission. Exposes pluggable `Reader` / `Judge` / `Scorer` seams.
- `config/default.json` — every parameter that affects the number (split, `top_k`,
  granularity, embedder, reader/judge model + snapshot + endpoint, scorer version).
  Each value is written into the result JSON; nothing that moves the number is
  hard-coded silently.
- `openai_models.py` — the OpenAI-direct reader + judge (real, paid). Key from the
  `OPENAI_API_KEY` env var; never hard-coded.
- `mock_models.py` — the free, offline mock reader + judge for the local smoke path.
- `official_reader.py` — the **upstream-verbatim** reader: the official CoT prompt +
  the context assembly (round-expansion, chronological re-sort, JSON history
  formatting, tiktoken truncation). Pin recorded in `READER_UPSTREAM.md`.
  `reader_version = longmemeval@9e0b455f…`.
- `scorer.py` — the official metric aggregation (mirrors `print_qa_metrics.py`).
- `official_scorer/` — the upstream-verbatim, **pinned** judge prompts + the pin
  record (`UPSTREAM.md`). `scorer_version = longmemeval@9e0b455f…`.
- `emit.py` — assembles a schema-valid `results/<id>.json` (dist hash, runner commit,
  etc.) and validates it.

## Canonical reader/judge (D9 — REQUIRED for the headline)

The canonical headline runs the **reader** at `reader_endpoint = api.openai.com`
with snapshot `gpt-4o-2024-08-06` (temperature 0.0) so the row lands in the
**canonical comparability segment** (`../../docs/comparability.md`). A row produced
with the reader on any other endpoint lands in a *non-canonical reader segment* and
is not the headline. The **judge** is OpenAI-direct `gpt-4o-2024-08-06` regardless
(no broker — endpoint-faithful). Both defaults are pinned in `config/default.json`.

## The paid full-500 run (benchmarks-owned)

The cost-bearing canonical run is executed by the benchmark maintainers (they own
cost discipline + the reproduction artifact). The default command is the official
run; set the public dataset path first:

```bash
export OPENAI_API_KEY=...      # OpenAI-direct; never committed
export ENGRAVA_BENCH_LONGMEMEVAL_S=<path-to>/longmemeval_s_cleaned.json  # the CLEANED split (see Dataset below)
pip install "engrava==<version>"   # the version the result will disclose
python runners/longmemeval/run.py
# then: make validate
```

By default the runner uses `config/default.json`, `--models openai`, and `--emit`.
It writes both the result row and the sibling reproduction-artifact directory under
`results/<benchmark>/<harness>/<system>/`. The emitted row starts at
`verification_status: unverified`; a maintainer promotes it to `verified` after
review. Do not run this without the cost owner's go-ahead.

> **Run the headline with no flags.** The bare `python runners/longmemeval/run.py`
> **is** the canonical configuration — `config/default.json` + `--models openai` +
> `--emit`, i.e. the `gpt-4o-2024-08-06` reader + judge OpenAI-direct, `top_k=20`,
> the pinned official reader/scorer. This is strongly recommended and is what every
> published number uses. Any flag that overrides a reader/judge/model/endpoint
> default (`--models`, `--endpoint`, `--reader-*`, `--judge-*`, `--embedder-spec`)
> moves the row **out of** the canonical comparability segment (see
> [Canonical reader/judge](#canonical-readerjudge-d9--required-for-the-headline) and
> [`../../docs/comparability.md`](../../docs/comparability.md)) — **do not pass them
> for a headline run.** The overrides exist only for exploratory / free-local runs
> (`--models ollama`, `--limit`, `--no-emit`), which are never the published number.
> The one thing you set outside the runner is the dataset (`ENGRAVA_BENCH_LONGMEMEVAL_S`,
> below) — a path, not a run parameter.

If `--dataset` is omitted, the runner resolves the dataset from
`ENGRAVA_BENCH_LONGMEMEVAL_S`, then from the ignored repo-local cache path
`runners/longmemeval/_cache/longmemeval_s_cleaned.json`.

## Free local smoke (NO spend, NO paid API)

Exercises the full pipeline end-to-end offline — real Engrava retrieval with the
free local embedder + mock reader/judge:

```bash
pip install -e ".[engrava]"
python runners/longmemeval/run.py --smoke
```

A pytest version (no Engrava dependency at all — a tiny in-test adapter) lives in
`tests/test_runner_smoke.py` and runs as part of `make check`. Smoke numbers are
artifacts, never published.

## Free end-to-end run with a local LLM (Ollama)

`--models ollama` points the reader + judge at a local OpenAI-compatible server
(e.g. [Ollama](https://ollama.com)) so a full end-to-end run — real reader + judge,
not mocks — costs nothing. Start the server, then:

```bash
pip install -e ".[engrava,openai]"
python runners/longmemeval/run.py \
  --dataset <longmemeval_s.json> \
  --models ollama \
  --endpoint http://<host>:11434 \
  --reader-model gemma3:4b \
  --judge-model gemma3:4b \
  --embedder-spec ollama:nomic-embed-text \
  --limit 10
```

- The `endpoint` carries a scheme (`http://…:11434`) and resolves to
  `http://…:11434/v1`; a bare host like `api.openai.com` keeps `https://…/v1`. No
  API key is needed (a dummy placeholder is used; a local server ignores auth).
- A local run is **exploratory, not the canonical headline.** It records the actual
  local reader/judge model + endpoint, so it lands in a non-canonical segment. The
  schema reserves `provenance: sovantica-run` for the canonical OpenAI-direct judge,
  so a fully-local run is not emitted as a publishable row — run it with `--no-emit`
  to compute metrics only, or use a local reader with the canonical judge
  for a non-canonical-reader `sovantica-run` row.

## Dataset

The LongMemEval `_s` full-500 split is **not vendored** here (load it from a
configured local path via `--dataset` or `ENGRAVA_BENCH_LONGMEMEVAL_S`). The published
result uses the authors' **cleaned** release — Hugging Face
[`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
file `longmemeval_s_cleaned.json` — **not** the raw `longmemeval_s.json` from the code repo.
To reproduce the exact number, use that cleaned file and confirm its **sha256 matches the
`dataset_revision`** pinned in the config / result row
(`longmemeval_s_cleaned@sha256:d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`);
a different revision or the raw split will produce a different score.

## Official scorer (UNMODIFIED, pinned)

Scoring is the official LongMemEval contract, pinned at commit
`9e0b455f4ef0e2ab8f2e582289761153549043fc` — see `official_scorer/UPSTREAM.md`. The
judge prompts are reproduced byte-faithfully; the metric aggregation mirrors
`print_qa_metrics.py`. `scorer_version` in every result records the pin.

## Metrics

- `overall_micro` — micro accuracy over **all** questions (includes abstention items).
- `macro` — unweighted mean of the 6 categories.
- `abstention` — a cross-cutting subset (`_abs` question-id suffix; overlaps the
  categories, not a 7th category, not additive).
- `per_category` — the 6 official `question_type` strings (hyphenated, 1:1 with the
  official scorer), each a `{accuracy, n}` cell.
