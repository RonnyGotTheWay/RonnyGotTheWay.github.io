PYTHON ?= .venv/bin/python
UV ?= python -m uv
export UV_CACHE_DIR ?= .uv-cache

.PHONY: help venv install lint typecheck test synthetic eastmoney-daily weekly-report api api-dev web services-up services-down init-db daily
help:
	@echo "ART-Rank commands: venv install lint typecheck test synthetic eastmoney-daily weekly-report api web services-up init-db daily"

venv:
	python3.12 -m venv .venv

install:
	$(UV) sync --dev

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

synthetic:
	$(UV) run art-rank synthetic --start 2020-01-01 --end 2025-12-31

eastmoney-daily:
	$(UV) run art-rank crawl-eastmoney

weekly-report:
	$(UV) run art-rank weekly-report

api:
	$(PYTHON) -m uvicorn art_rank_quant.api.main:app --host 127.0.0.1 --port 8000

api-dev:
	$(PYTHON) -m uvicorn art_rank_quant.api.main:app --host 127.0.0.1 --port 8000 --reload

web:
	npm run dev:web

services-up:
	docker compose up -d postgres

services-down:
	docker compose down

init-db:
	$(UV) run alembic upgrade head

daily:
	$(UV) run art-rank daily
