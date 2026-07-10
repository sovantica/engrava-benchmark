# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
