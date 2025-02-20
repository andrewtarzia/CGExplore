# List all recipes.
default:
  @just --list

# Install development environment.
dev:
  pip install -e '.[dev]'
  mamba install -y openmm openmmtools

# Run code checks.
check:
  #!/usr/bin/env bash

  error=0
  trap error=1 ERR

  echo
  (set -x; ruff check . )

  echo
  ( set -x; ruff format --check . )

  echo
  ( set -x; mypy src )

  echo
  ( set -x; pytest --cov=src --cov-report term-missing )

  test $error = 0


# Auto-fix code issues.
fix:
  ruff format .
  ruff check --fix .


# Build docs.
docs:
  rm -rf ./docs/build docs/source/_autosummary
  make -C docs html
  echo Docs are in $PWD/docs/build/html/index.html
