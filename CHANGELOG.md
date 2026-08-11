# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-08-11

### Added

- `--mode` on the runners: one dial for which of the embedder, reader and judge are real, whether the run may write a result row, and whether it emits a ranked retrieval log.
- A LongMemEval-V2 runner that drives the upstream harness and can emit a retrieval log for a deterministic diff.
- `runners/retrieval_diff.py` to compare two retrieval logs, so a memory change can be screened without paying for a reader.
- Documentation for the run modes and for the LongMemEval-V2 runner.

### Changed

- Track the engrava 0.6.0 line; the packaged extra and the documented install now name the same version.

### Fixed

- Bug fixes and stability improvements.

## [0.2.1] - 2026-07-17

### Added

- `--results-dir` runner flag to emit a run's result + artifact bundle into a chosen
  directory instead of the repo `results/` tree.

## [0.2.0] - 2026-07-14

### Added

- Record the harness (runner) that produced each result — its name, source
  (in-repo or an external runner's origin), and the exact version the run used —
  so external-runner results are reproducible and auditable.
- LongMemEval-V2 Engrava adapter.
- Agent Memory Benchmark (AMB) Engrava provider under `integrations/` for running Engrava
  inside the AMB harness.

### Changed

- Partition results by harness: results now live at
  `results/<benchmark>/<harness>/<system>/<result_id>.json` (previously
  `results/<benchmark>/<system>/…`), and leaderboard comparability segments key on
  the harness — results from different harnesses are never co-ranked.

### Fixed

- Let the AMB provider's offline smoke embedding backend answer queries, not just
  index documents (it was missing the single-string embed used at retrieval time).

## [0.1.0]

### Added

- Initial repository scaffold: the pluggable `MemoryAdapter` seam (`adapters/base.py`),
  the public-`engrava` reference adapter (`adapters/engrava_adapter.py`), the uniform
  LongMemEval runner skeleton (`runners/longmemeval/`), the machine-readable results
  JSON Schema (`results/schema/results.schema.json`), the results validator and
  leaderboard builder (`scripts/`), and methodology + comparability docs.
- Quality gates: `pyproject.toml` (ruff/mypy/pytest), `Makefile`, CI workflows, and
  Conventional Commits enforcement.
- Executable LongMemEval runner: OpenAI-direct reader + judge (`openai_models.py`),
  free offline mock reader + judge (`mock_models.py`), the official judge prompts
  pinned upstream + the official metric aggregation (`official_scorer/`, `scorer.py`),
  config-driven dataset loading, and schema-validating result emission (`emit.py`).
- Free end-to-end smoke path (`tests/test_runner_smoke.py` + a 2-question fixture)
  exercising ingest → search → assemble → reader → judge → score → emit → validate with
  no network and no paid call.
