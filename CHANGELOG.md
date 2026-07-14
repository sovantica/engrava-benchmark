# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
