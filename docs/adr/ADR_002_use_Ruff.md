# ADR-002 | Use Ruff

## Status 

Accepted 

## Context 

The prpject requiers a tool to ensure consistent code quality and formatting. Several options were considered such as flake8, isort, black, pylint and Ruff.

## Decision 

Use Ruff for both linting and formatting

## Consequences 

### Positive 

- Single tool for linting and formatting.
- Very fast execution.
- Easy integration with pre-commit.
- Actively maintained.

### Negative

- Developers coming from Black/Flake8 may need to adapt

