---
name: workflow-training-dev
description: >-
  Use this skill to understand the software engineering stack, project setup, and standard workflows for the workflow-training repository. Activate when you need to run tests, add dependencies, lint code, or manage the development environment.
---

# Workflow Training Development Guide

This skill outlines the software engineering practices, tooling stack, and local development setup for the `workflow-training` project.

## 1. Environment & Dependency Management
- **Environment**: Micromamba environment `janus` (`/opt/micromamba/envs/janus`).
  - Prefix/activation command: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python <command>` or `micromamba run -n janus <command>`
- **Dependencies**: Package management via `uv`.
  - Install dependencies: `uv pip install -e ".[test]"`
  - Add new package: `uv add <package>`
- **Build System**: Hatchling build backend.
  ```toml
  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
  ```

## 2. Linting and Formatting
- **Linters & Formatters**: Ruff.
  - Run linter: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check src tests`
  - Run formatter: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff format src tests`
  - Auto-fix issues: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check --fix src tests`
- **Pre-commit**: Pre-commit hook configuration in `.pre-commit-config.yaml`.

## 3. Testing & Verification
- **Framework**: Pytest.
- **Running Tests**:
  - Full suite: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest`
  - Single test: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest tests/test_feature.py -k test_name`
- **Coverage**:
  - `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest --cov=src --cov-report=term-missing`
- **Special Execution Modes**:
  - Headless atomistic simulations: Always pass `--no-tracker` in scripts and automated pipelines to prevent CodeCarbon emission overhead.
  - Cell relaxation: Always prefer `--filter FrechetCellFilter` for periodic geometry optimization.
  - Structure verification: Check coordinates using `python tools/structure_inspector.py <structure>` before executing heavy simulations.

## 4. CI/CD Workflows
- **Test Matrix Workflow**: `.github/workflows/test.yml` running linting, formatting, and pytest suites.
- **Release / Publishing Workflow**: `.github/workflows/publish.yml` for distribution.

## 5. Local Execution & Debugging
- **Run Application / Script**: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python -m src`
- **Common Gotchas**:
  - Always execute Python within the `janus` micromamba environment (`/opt/micromamba/envs/janus/bin/python`) or via `uv run --no-project --python /opt/micromamba/envs/janus/bin/python`.
  - Ensure AiiDA daemon (`verdi daemon status`) is running when executing AiiDA workflows.

