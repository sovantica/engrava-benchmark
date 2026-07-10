SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))
PIP    ?= $(PYTHON) -m pip

.PHONY: help install clean lint fmt fmt-check typecheck test check validate leaderboard

help:
	@echo ""
	@echo "engrava-benchmark — Development Commands"
	@echo ""
	@echo "  install      Install package + dev dependencies (with [engrava] extra for the adapter)"
	@echo "  lint         Ruff lint check"
	@echo "  fmt          Ruff format (write changes)"
	@echo "  fmt-check    Ruff format check (read-only)"
	@echo "  typecheck    Mypy strict"
	@echo "  test         pytest"
	@echo "  validate     Validate results/*.json against schema + rules"
	@echo "  leaderboard  Rebuild leaderboard.json from verified rows"
	@echo "  check        Full quality gate: lint + fmt-check + typecheck + test + validate"
	@echo ""

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,engrava]"
	@echo ">> Done. Run 'make check' to verify the quality gate."

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	find . \( -type d -name __pycache__ -o -type f -name '*.pyc' \) -prune -exec rm -rf {} +

lint:
	$(PYTHON) -m ruff check adapters runners scripts

fmt:
	$(PYTHON) -m ruff format adapters runners scripts

fmt-check:
	$(PYTHON) -m ruff format --check adapters runners scripts

typecheck:
	$(PYTHON) -m mypy --strict adapters scripts runners

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_results.py

leaderboard:
	$(PYTHON) scripts/build_leaderboard.py

check: lint fmt-check typecheck test validate
	@echo ">> All quality gates passed."
