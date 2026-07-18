# LongMemEval-V2 Integration

This directory contains the public integration artifact for evaluating
`engrava==0.5.0` as a LongMemEval-V2 memory backend.

The upstream benchmark repository is:

<https://github.com/xiaowu0162/LongMemEval-V2>

The harness slug for runs produced through this integration is `lme-v2-official`
(registered in `scripts/canonical_slugs.py`). The actual run and any leaderboard
submission are gated and owner-approved.

The integration is intentionally small:

- [`engrava_memory.py`](engrava_memory.py) is a single LongMemEval-V2 memory
  backend file.
- [`memory_config.json`](memory_config.json) is the canonical Engrava memory
  config for the LME-V2 `small` submission path. `embedding_params` also accepts two
  **operational** retry knobs passed straight to the public engrava embedding provider —
  `max_attempts` and `base_retry_delay_s` — bounded backoff to ride out a hosted
  embeddings rate limit (HTTP 429) on a low usage tier. They change resilience only, never
  the embeddings (a retried batch is byte-identical), so the result is unaffected.
- [`SYSTEM_DESCRIPTION.md`](SYSTEM_DESCRIPTION.md) is the draft description used
  by the upstream leaderboard package builder.

## Upstream Pin

The current implementation target is:

```text
xiaowu0162/LongMemEval-V2@be15ea6e995462f3391c1a610892df3f67dfa7bd
```

Re-check upstream before a paid run. The leaderboard package builder validates
the reader and judge model families.

## Install

Inside a LongMemEval-V2 checkout:

```bash
pip install -e .
pip install "engrava==0.5.0" aiosqlite tiktoken
```

For the official LME-V2 stack, provide OpenAI-compatible reader and embedding
endpoints as described by upstream:

```bash
export READER_BASE_URL=http://localhost:8023/v1
export READER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_BASE_URL=http://localhost:8114/v1
export LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export OPENAI_API_KEY=...
```

## Wire the Backend

Copy the backend into the upstream memory modules:

```bash
cp path/to/engrava-benchmark/integrations/longmemeval_v2/engrava_memory.py \
  memory_modules/engrava_memory.py
```

Register it by adding this import to the bottom of upstream
`memory_modules/memory.py`:

```python
from .engrava_memory import EngravaMemory  # noqa: E402,F401
```

Then run `evaluation/harness.py` directly with
`--memory-config-path path/to/memory_config.json`, or add a local `engrava`
method entry to upstream `evaluation/run_eval.py`.

The leaderboard package can still infer the method from folder names or from the
`--method engrava` override in `leaderboard/build_submission_step_1_single_operating_point.py`.

## Public Artifact Hygiene

Do not commit raw LongMemEval-V2 run folders to this repository by default.
Upstream run outputs can contain per-question memory context, prompt messages,
gold answers, raw responses, runtime inputs, and absolute local paths.

Before adding any LongMemEval-V2 result artifact to `engrava-benchmark`, remove
local filesystem paths and keep only reviewed public data. Public files should
use repo-relative paths, upstream-package-relative paths, or checksums.

## Smoke Path

Use a tiny question limit before any paid run:

```bash
python evaluation/run_eval.py \
  --data-root "$DATA_ROOT" \
  --domain web \
  --tier small \
  --method no_retrieval \
  --limit 1 \
  --output-dir runs/no_retrieval_web_small_smoke
```

Then run the Engrava backend through `evaluation/harness.py` on the same
materialized runtime inputs. For a synthetic no-network adapter smoke, set
`embedding_params.backend` to `deterministic` in a temporary copy of
`memory_config.json`. Do not use the deterministic backend for publishable
results.

## Official Small-Tier Shape

The publishable operating point requires both domains:

```text
runs/engrava_web_small/
runs/engrava_enterprise_small/
```

Then package:

```bash
python leaderboard/build_submission_step_1_single_operating_point.py \
  runs/engrava_web_small \
  runs/engrava_enterprise_small \
  engrava_0_5_0_small \
  canonical \
  small \
  --method engrava

python leaderboard/build_submission_step_2_build_package.py \
  engrava_0_5_0_small \
  path/to/SYSTEM_DESCRIPTION.md \
  memory_modules/engrava_memory.py \
  leaderboard/submissions/engrava_0_5_0_small/operating_points/canonical
```

Do not submit through GitHub issues. Use the official LongMemEval-V2 submission
form linked from upstream `leaderboard/README.md`.
