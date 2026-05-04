# GemsPy — Compatibility

## Python Version Support

| GemsPy | Python 3.10 | Python 3.11 | Python 3.12 |
|--------|-------------|-------------|-------------|
| 0.1.0  | ✓           | ✓           | ✓           |

## GEMS Language Compatibility

GemsPy tracks the [GEMS language specification](https://gems-energy.readthedocs.io/) directly. There is no formal GEMS Language version yet. This table will be updated when the GEMS Language introduces official versioning.

## Key Dependency Versions

| GemsPy | linopy  | HiGHS (highspy) | antlr4-python3-runtime | Notes |
|--------|---------|-----------------|------------------------|-------|
| 0.1.0  | ≥ 0.6   | ≥ 1.14          | ≥ 4.13.2               | Initial release |

## Versioning Policy

- **GemsPy** — version in `pyproject.toml` (`[project] version`). Follows semantic versioning:
  - **Major** — backward-incompatible change, or major new GEMS Language feature (e.g. rolling optimization horizon)
  - **Minor** — bug fix, new backward-compatible feature, or external dependency version update
  - **Patch** — internal code optimisation, refactor with no syntax impact, or dependency update with no behavior change

## Version Files

| Component | Current Version | Version File |
|-----------|----------------|--------------|
| GemsPy    | 0.1.0          | `pyproject.toml` |
| linopy    | ≥ 0.6          | `pyproject.toml` |
| highspy   | ≥ 1.14         | `pyproject.toml` |
| antlr4-python3-runtime | ≥ 4.13.2 | `pyproject.toml` |
