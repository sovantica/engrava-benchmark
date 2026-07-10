# adapters/

A memory system enters the benchmark by implementing **one** small interface:
[`MemoryAdapter`](base.py) — `ingest` + `search`. Everything else (dataset loading,
context assembly, the reader LLM, the reader prompt, the judge LLM, the official
scorer) is owned by the runner and is identical for every system. The memory layer
is the only independent variable — so no system can win in the reader or the prompt.

## The interface

```python
from adapters.base import CorpusTurn, MemoryAdapter, RankedItem, RunContext


class MyDatabaseAdapter:
    adapter_name = "my_database_adapter"

    def ingest(self, corpus: list[CorpusTurn], *, run_ctx: RunContext) -> None:
        # Store/index every CorpusTurn however your system likes.
        ...

    def search(self, query: str, *, top_k: int) -> list[RankedItem]:
        # Return up to top_k RankedItem(unit_id, score), best first.
        # Each unit_id MUST be a CorpusTurn.unit_id you were given in ingest().
        ...
```

`MemoryAdapter` is a `@runtime_checkable` `Protocol`, so you do **not** subclass it —
just match the shape.

## The rules (equal footing)

An adapter owns **only** the memory layer. It MUST NOT:

- override or influence the reader, the reader prompt, context assembly, the judge,
  or the official scorer;
- access benchmark answers, evidence/gold labels, or any leaderboard hint — the
  `RunContext` deliberately does not expose them, and `CorpusTurn` carries no
  evidence flag;
- do retrieval-time work that is really reader-side work in disguise.

A counted improvement (a "lever") must live behind your `ingest`/`search` — that is
what makes the number reproducible and the comparison fair.

## Disclosing pipeline LLMs

If your system uses a generative LLM **anywhere** in the memory pipeline — write-time
summarization/extraction, query rewriting, LLM rerank, compression, etc. — every such
use must be disclosed in the result's `system_config.memory_pipeline_llms`. That list
decides the Group A / Group B axis (empty = Group A; any generative LLM = Group B), and
the value is recomputed and checked in CI. Embeddings alone do not make a system
Group B.

## Submitting a result

1. Drop `adapters/<your_db>.py` implementing `MemoryAdapter`.
2. Run the same runner (`runners/<benchmark>/`) to produce a `results/<id>.json`.
3. Validate: `make validate`.
4. Open a PR. (Community contribution + curation flow is a later phase — see the repo
   README for the current contribution state.)

## Reference adapter

[`engrava_adapter.py`](engrava_adapter.py) is the maintainer-supported adapter for the
open-source `engrava` package. It is the worked example for your own.
