# results/

Machine-readable benchmark results, one JSON object per result.

## Layout (partitioned, benchmark-first)

```
results/
├── schema/
│   └── results.schema.json
└── <benchmark>/                # canonical benchmark slug, e.g. longmemeval-s
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
  `schema/` and `<benchmark>/<system>/…` are allowed.
- The `<benchmark>` and `<system>` directory segments are the row's own **canonical
  slugs**, so the path is a faithful projection of the content (no independent label
  that can drift). The registered slugs live in
  [`../scripts/canonical_slugs.py`](../scripts/canonical_slugs.py); each segment must
  match `^[a-z0-9][a-z0-9-]*$` and be on that registered list (aliases/case-drift are
  rejected at validation).
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
  in the tree — `make leaderboard`. It reads only depth-3 result rows
  (`results/<benchmark>/<system>/<result_id>.json`); artifact `config.json` and
  `manifest.json` files are never treated as result rows. Rows with
  `verification_status` other than `verified` are excluded. Comparability
  segmentation is keyed off in-file fields, not directory position.

## Per-row provenance

Every row carries a `provenance` label:

- `sovantica-run` — produced by the maintainers against the public `engrava` package,
  judged by the official `gpt-4o-2024-08-06` via `api.openai.com`, with a reproduction
  artifact. (Currently the only kind of row.)
- `vendor-reported` / `community-submitted` — reserved; these require a `citation` block.

## Headline eligibility

A row is a **headline** number only if `partial == false` AND
`verification_status == "verified"` AND it reports all six per-category cells plus
abstention over the full split.

## Reproduction artifacts

Reproduction artifacts (hypothesis files + judge labels + retrieval log + full
config) are committed in this tree beside their result row. The row records a
repo-relative `reproduction_artifact_url` such as
`results/longmemeval-s/engrava/<result_id>/` plus `artifact_checksum`. Validation
recomputes that checksum from the sibling directory.
