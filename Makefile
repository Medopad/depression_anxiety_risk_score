ifndef VERBOSE
MAKEFLAGS += --no-print-directory
endif

SHELL:=/bin/bash

GIT_SHA:=$(shell git rev-parse --short HEAD)
GIT_TAG:=$(shell git describe --tags --abbrev=0)
GIT_BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)

setup/startup: ## startup installation, pyenv update, python git and venv
	@echo "+ Inititalized git, updates pyenv and install python version"
	@brew upgrade pyenv \
		&& if [ -d ".git" ]; then echo "+git already present"; else git init --initial-branch=main .; fi \
		&& cat .python-version | pyenv install || true \
		&& make setup/venv


setup/venv: ## Setup venv and install requirements
	@echo "+ Setting up virtual environment"
	@python -m venv .venv \
		&& source .venv/bin/activate \
		&& python -m pip install --upgrade pip \
		&& poetry update \
		&& poetry install

setup/re-install: ## Remove venv and install it again
	@echo "+ Setting up virtual environment"
	@rm -rf .venv \
		&& make setup/venv

run/tests: ## Run tests
	@echo "+ Running tests"
	@source .venv/bin/activate \
		&& pytest tests -s --disable-pytest-warnings

open/docs: ## Builds and opens documentation
	@echo "+ Creating documentation with sphinx"
	@source .venv/bin/activate \
		&& sphinx-build docs docs/_build \
		&& open docs/_build/index.html

common/git-info: ## Get git info
	@echo "+ GIT Branch: "$(GIT_BRANCH)
	@echo "+ GIT SHA   : "$(GIT_SHA)

.PHONY: help all
help:
	@grep -hE '^[a-zA-Z0-9\._/-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help