# Developer guide

This page aims at providing useful information for contributors.

## Install dev requirements

Install dev requirements with `uv sync` (or `pip install -e ".[dev]"` for pip users)

## Linting and formatting

To reformat your code, use this command line: `python -m black src tests && python -m isort --profile black  src, tests`

## Typechecking

To typecheck your code, use this command line: `mypy`

## Documentation

1. To preview the docs on your local machine run `mkdocs serve`.
2. To build the static site for publishing for example on [Read the Docs](https://readthedocs.io) use `mkdocs build`.
3. To flesh out the documentation see [mkdocs guides](https://www.mkdocs.org/user-guide/).