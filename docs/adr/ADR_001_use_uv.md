# ADR-001 | Use uv 

## Status 

Accepted

## Context 

The project requiers a modorn dependency and environment manager for Python. Several solutions were considered, including pip with venv, peotry and uv.

## Decision 

Use uv as the dependency manager and virtual environment tool.

## Consequenses 

### Positive 

- Extremely fast dependency resolution. 
- Native support for pyproject.toml.
- Automatic lockfile generation.
- Modern workflow with single tool replacing pip and venv.
- Growing adoption in the Python exosystem. 

### Negative

- Less widespread than traditional pip.
- New contributors may need to install uv.