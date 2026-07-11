# results/

Machine-readable benchmark results, one JSON object per result.

## Layout (partitioned, benchmark-first)

```
results/
├── schema/
│   └── results.schema.json
└── <benchmark>/                    # canonical benchmark slug, e.g. longmemeval-s
    └── <harness>/                  # canonical harness slug, e.g. longmemeval-official
        └── <system>/               # canonical system slug, e.g. engrava
            ├── <result_id>.json    # result row
            └── <result_id>/        # sibling reproduction artifact bundle
                ├── manifest.json
                ├── config.json
                ├── hypotheses.jsonl
                ├── judge_labels.jsonl
                └── retrieval_log.jsonl
```

- **No flat dump:** a result JSON never lives directly under `results/` — only
  `schema/` and `<benchmark>/<harness>/<system>/…` are allowed.
- The `<benchmark>`, `<harness>`, and `<system>` directory segments are the row's own
  **canonical slugs**, so the path is a faithful projection of the content (no
  independent label that can drift). The registered slugs live in
  [`../scripts/canonical_slugs.py`](../scripts/canonical_slugs.py); each segment must
  match `^[a-z0-9][a-z0-9-]*$` and be on that registered list (aliases/case-drift are
  rejected at validation).
- The **harness** (runner) is a first-class axis: it fixes the reader, judge, prompt,
  context format, and scorer, so it changes what a number *means*. It is therefore both
  a path segment AND part of the comparability key — cross-harness rows are never
  co-ranked.
- **`result_id` is globally unique** across the whole `results/` tree — it is the stable
  render key and is collision-safe across configs, readers, and dates, so two rows can
  never share one even in different directories.

## Tooling

- **Schema:** [`schema/results.schema.json`](schema/results.schema.json) — every result
  is validated against it (structure, enums, required fields, the
  provenance/judge/citation conditionals).
- **Validate locally:** `make validate` (or `python scripts/validate_results.py`) —
  schema + cross-field rules + the layout rules (slug shape, path/content agreement,
  global `result_id` uniqueness).
- **Aggregate:** `leaderboard.json` (repo root) is generated from the **verified** rows
  in the tree — `make leaderboard`. It reads only result rows
  (`results/<benchmark>/<harness>/<system>/<result_id>.json`); artifact `config.json`
  and `manifest.json` files are never treated as result rows. Rows with
  `verification_status` other than `verified` are excluded. Comparability segmentation
  is keyed off in-file fields, not directory position — and the harness is now both a
  path segment AND part of that comparability key.

## Per-row provenance

Every row carries a `provenance` label:

- `sovantica-run` — produced by the maintainers against the public `engrava` package,
  judged by the official `gpt-4o-2024-08-06` via `api.openai.com`, with a reproduction
  artifact. (Currently the only kind of row.)
- `vendor-reported` / `community-submitted` — reserved; these require a `citation` block.

Every row also carries a `harness` provenance block — `{name, source, version}`:

- `name` — the canonical harness slug (must be registered in
  [`../scripts/canonical_slugs.py`](../scripts/canonical_slugs.py)) and agrees with the
  `<harness>` path segment.
- `source` — the harness origin: `in-repo` for this repo's native runner, or a URL for
  an external runner we pull.
- `version` — the exact ref the run used. For the native runner this is the
  `engrava-benchmark@<sha>` runner commit; for an external runner it is that runner's
  pinned version.

The harness fixes the reader, judge, prompt, context format, and scorer, so it is a
coarse identity axis above the finer `reader_version` / `scorer_version` pins. When the
harness is `longmemeval-official`, the judge is pinned to the official
`gpt-4o` / `gpt-4o-2024-08-06` via `api.openai.com`; a different harness may use a
different judge.

## Headline eligibility

A row is a **headline** number only if `partial == false` AND
`verification_status == "verified"` AND it reports all six per-category cells plus
abstention over the full split.

## Reproduction artifacts

Reproduction artifacts (hypothesis files + judge labels + retrieval log + full
config) are committed in this tree beside their result row. The row records a
repo-relative `reproduction_artifact_url` such as
`results/longmemeval-s/longmemeval-official/engrava/<result_id>/` plus
`artifact_checksum`. Validation recomputes that checksum from the sibling directory.

## Path hygiene

Public artifacts must not record local filesystem paths. Local checkout paths,
home-directory paths, storage/cache paths, scratch paths, runtime output paths,
and endpoint-local file paths are data leaks in this repository.

Use repo-relative paths for committed artifacts and URLs. If an upstream
benchmark output requires absolute local paths for execution, keep that raw
output outside this repository or sanitize it before publication.
