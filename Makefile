.PHONY: help install dev test lint type fmt up down logs build push clean migrate

PYTHON  ?= python
UV      ?= uv
APP     ?= mldm-agent
IMAGE   ?= $(APP):dev
COMPOSE ?= docker compose

help:  ## list targets
	@awk 'BEGIN{FS=":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## install runtime + dev deps via uv (fallback: pip)
	@command -v $(UV) >/dev/null 2>&1 \
		&& $(UV) pip install -e '.[dev]' \
		|| $(PYTHON) -m pip install -e '.[dev]'

dev:  ## run uvicorn with --reload
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:  ## run pytest
	$(PYTHON) -m pytest -v --cov=app tests/

lint:  ## ruff check
	$(PYTHON) -m ruff check app tests

fmt:  ## ruff format
	$(PYTHON) -m ruff format app tests

type:  ## mypy
	$(PYTHON) -m mypy app

up:  ## docker compose up
	$(COMPOSE) up -d

down:  ## docker compose down
	$(COMPOSE) down

logs:  ## tail compose logs
	$(COMPOSE) logs -f api

build:  ## docker build the runtime image
	docker build -t $(IMAGE) .

push:  ## docker push (set IMAGE=registry/path:tag)
	docker push $(IMAGE)

migrate:  ## run alembic migrations
	alembic upgrade head

clean:  ## drop caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
