# Developer guide

This page provides useful information for contributors on GemsPy-specific local development tasks. The authoritative branching, versioning, CI/CD, and release process shared across all GEMS repositories (including GemsPy) lives in the **[GEMS Developer Guidelines](https://gems-energy.readthedocs.io/en/latest/support/dev-guidelines/)** — read it before any branching, versioning, or release work.

## Install dev requirements

Install dev requirements with `uv sync` (or `pip install -e ".[dev]"` for pip users)

## Linting and formatting

To reformat your code, use this command line: `python -m black src tests && python -m isort --profile black  src, tests`

## Typechecking

To typecheck your code, use this command line: `mypy`

## Input-file schemas

The model-library, system, and optimization-config parsers validate their normalized YAML
inputs against the bundled Draft 2020-12 schemas in `src/gems_craft/schemas/`. Each shared
GEMS schema must validate; Pydantic then applies GemsPy's typed parsing and business rules. If a
GemsPy feature is missing from a shared schema, add a targeted schema extension and test rather
than bypassing the shared validation. These files are a snapshot of GEMS PR #284; update the
snapshot and parser conformance tests together when the shared schemas change. The
optimization-config schema is provisional.

## Documentation

1. To preview the docs on your local machine run `mkdocs serve`.
2. To build the static site for publishing for example on [Read the Docs](https://readthedocs.io) use `mkdocs build`.
3. To flesh out the documentation see [mkdocs guides](https://www.mkdocs.org/user-guide/).
