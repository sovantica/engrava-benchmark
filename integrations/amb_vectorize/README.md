# Agent Memory Benchmark (AMB) Integration

This directory contains the public integration artifact for evaluating
`engrava==0.5.0` as an Agent Memory Benchmark memory provider.

The upstream benchmark repository is:

<https://github.com/vectorize-io/agent-memory-benchmark>

The integration is intentionally small:

- [`engrava_provider.py`](engrava_provider.py) is a single AMB `MemoryProvider`
  implementation (`EngravaMemoryProvider`).

The provider stays in **Group A**: there is no LLM anywhere in the memory path.
Ingest indexes one Engrava thought per document with an explicit embedding, and
retrieval is a single default-weighted `search_hybrid` (dense + sparse/BM25
fusion, recency inactive, reflections excluded). The reader (answer generation),
the judge, the prompts, the scorer, and the datasets remain owned by the upstream
AMB harness.

## Upstream Pin

The current implementation target is:

```text
vectorize-io/agent-memory-benchmark@aa9273ab9e34bbeaff3c6ef2f694142a552d5b22
```

The provider implements the `MemoryProvider` ABC as pinned above
(`ingest(documents)` and `retrieve(query, k, user_id, query_timestamp)`, plus the
`cleanup()` teardown hook and the `name` / `description` / `kind` / `provider` /
`variant` / `link` / `concurrency` class attributes). Re-check upstream before a
run: AMB's answer generation and judging use Gemini, and the exact reader/judge
models are part of the comparability contract.

## Install

Inside an AMB checkout:

```bash
uv pip install "engrava==0.5.0" aiosqlite tiktoken
```

The default embedding backend is a network-free local SentenceTransformer, so the
memory path needs no API key. Add `sentence-transformers` if the AMB environment
does not already provide it.

## Wire the Provider

Copy the provider into the upstream memory modules:

```bash
cp path/to/engrava-benchmark/integrations/amb_vectorize/engrava_provider.py \
  src/memory_bench/memory/engrava.py
```

Register it in `src/memory_bench/memory/__init__.py`:

```python
from .engrava import EngravaMemoryProvider  # noqa: F401

REGISTRY: dict[str, type[MemoryProvider]] = {
    # ...existing entries...
    "engrava": EngravaMemoryProvider,
}
```

## Run

Use AMB's own CLI, selecting the `engrava` provider:

```bash
# Quick, small smoke: cap the number of queries.
uv run amb run --dataset personamem --domain 32k --memory engrava --query-limit 20

# Full domain run.
uv run amb run --dataset personamem --domain 32k --memory engrava
```

### Embedding configuration

The memory path reads its embedding backend from the environment (all optional):

```bash
# Local SentenceTransformer (default, no API):
export ENGRAVA_AMB_EMBED_BACKEND=local
export ENGRAVA_AMB_EMBED_MODEL=all-MiniLM-L12-v2   # optional override

# OpenAI-compatible embeddings:
export ENGRAVA_AMB_EMBED_BACKEND=openai
export ENGRAVA_AMB_EMBED_MODEL=text-embedding-3-small
export OPENAI_API_KEY=...
# export ENGRAVA_AMB_EMBED_BASE_URL=http://localhost:8114/v1   # optional

# Offline, network-free adapter smoke ONLY (never for publishable results):
export ENGRAVA_AMB_EMBED_BACKEND=deterministic
```

The `deterministic` backend encodes byte statistics, not semantics; use it only to
exercise the ingest/retrieve wiring without a network, never for a published score.

## Public Artifact Hygiene

Do not commit raw AMB run folders (`outputs/`) to this repository by default.
Upstream run outputs can contain per-query retrieved context, prompt messages,
gold answers, raw provider responses, and absolute local paths.

Before adding any AMB result artifact to `engrava-benchmark`, remove local
filesystem paths and keep only reviewed public data. Public files should use
repo-relative paths, upstream-package-relative paths, or checksums — never a
local absolute path, and never gold-answer leakage.

The harness slug for AMB runs produced through this integration is
`amb-vectorize` (registered in `scripts/canonical_slugs.py`). The actual run and
any leaderboard submission are gated and owner-approved.
