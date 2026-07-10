# Contributing to engrava-benchmark

Thanks for your interest! This repo hosts reproduction runners and machine-readable
benchmark results. There are two kinds of contribution: **code** (adapters, runners,
tooling) and **results** (benchmark numbers).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
make install        # installs dev deps + the engrava extra
make check          # lint + format check + typecheck + tests + result validation
```

`make check` must be green before opening a PR.

## Commit message format

[Conventional Commits 1.0.0](https://www.conventionalcommits.org/):
`<type>(<scope>): <description>`. Types: `feat`, `fix`, `perf`, `docs`, `style`,
`refactor`, `test`, `build`, `ci`, `chore`, `revert`. Header ≤ 100 chars. Commitlint
validates PR titles and commits in CI.

Assume every commit message, PR title, branch name, comment, and docstring is
world-visible — keep them generic and public-safe. Branch names: `<type>/<kebab>`,
lowercase, ≤ 50 chars, no exotic identifiers. Use `results/<kebab>` for changes
that primarily add or update benchmark result rows and their reproduction artifacts.
Result-row PRs use `chore(results): ...` because they are data updates, not feature
releases.

## Contributing an adapter

To benchmark your own memory database, implement
[`MemoryAdapter`](adapters/base.py) in `adapters/<your_db>.py` (`ingest` + `search`)
and follow the [adapter guide](adapters/README.md). An adapter owns **only** the memory
layer — it must not touch the reader, the prompt, the judge, or the scorer, and must not
read benchmark answers/labels. A counted improvement lives behind `ingest`/`search`.

## Contributing a result

Create result-row PRs from a `results/<kebab>` branch. Every result row is a
`results/<benchmark>/<system>/<result_id>.json` file validated against
[`results/schema/results.schema.json`](results/schema/results.schema.json):

1. Produce the row by running a runner (do not hand-write metrics).
2. Validate it locally: `make validate`.
3. Provide a reproduction artifact (hypothesis files + judge labels + full config) and
   reference it with `reproduction_artifact_url` + `artifact_checksum`.
4. Label `provenance` honestly and supply a `citation` when the row is not a maintainer
   run.

A row enters the ranked leaderboard only at `verification_status: verified`.

## What CI enforces

- JSON Schema + `schema_version` validity, and the cross-field rules (`group`
  recomputed, headline eligibility, provenance-specific required fields).
- Reproduction-artifact presence + a well-formed checksum on maintainer-run rows.
- Conventional Commits + branch-name shape.

Maintainers additionally run a private pre-publish audit before changing repository
visibility. External contributors need no extra setup; reviewers backstop it.

## Code style

- `ruff` clean (lint + format); `mypy --strict` clean.
- Type hints + Google-style docstrings on public symbols.
- No `# type: ignore` without a justification comment.

## License

By contributing you agree your contributions are licensed under the repo's
[MIT](LICENSE) license. Reproduction artifacts declare their own `artifact_license`.
