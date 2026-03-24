# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GemsPy** is a Python interpreter for the GEMS (Generic Energy Modeling System) framework — a high-level modeling language for simulating energy systems under uncertainty. It allows users to define energy system models via YAML without writing solver code directly.

## Commands

**Install:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

**Test:**
```bash
pytest                                          # run all tests
pytest tests/path/to/test_file.py::test_name   # run a single test
pytest --cov gems --cov-report xml         # with coverage
```

**Lint & Format:**
```bash
black --config pyproject.toml src/ tests/
isort --profile black --filter-files src/ tests/
mypy
```

**Running:**
```bash
# CLI entry point
gemspy \
  --model-libs  path/to/lib1.yml path/to/lib2.yml \
  --components   path/to/components.yml \
  --timeseries   path/to/timeseries/ \
  --duration     8760 \
  --scenarios    1

# Python API (minimal example)
from gems.main.main import build_problem
```

## Architecture

The pipeline flows: **YAML input → parsing → model resolution → network building → optimization problem → OR-Tools solver → results**

### Core Modules (`src/gems/`)

**`model/`** — Immutable model templates.
- `Model`: defines component behavior (parameters, variables, constraints, ports)
- `Library`: a collection of models, loaded from YAML
- Models are never instantiated directly — they are referenced by components

**`expression/`** — Mathematical expression language and AST.
- `ExpressionNode`: base frozen dataclass for all expression tree nodes
- Node types cover: arithmetic (`+`, `-`, `*`, `/`), comparisons (`<=`, `>=`, `==`), time/scenario operators (`time_sum()`), and functions (`max()`, `min()`, `ceil()`, `floor()`)
- Grammar is defined in `/grammar/` and parsed via ANTLR4 (generated files live in `expression/parsing/antlr/`)
- `ExpressionVisitor` is the dominant pattern for traversing and transforming expression trees (evaluation, linearization, printing, degree analysis)
- Expressions support operator overloading: `var('x') + 5 * param('p')`

**`study/`** — Study definition and network instantiation.
- `System`: top-level structure parsed from YAML (before instantiation)
- `Network`: instantiated graph of `Node`s, `Component`s, and connections
- `Component`: an instance of a `Model` with concrete parameter values
- `DataBase`: manages time series and scenario data

**`simulation/`** — Optimization problem construction and solving.
- `OptimizationProblem`: main interface; translates network + database into OR-Tools constraints
- `LinearExpression`: the linearized form of model constraints used by the solver
- `BendersDecomposedProblem`: temporal decomposition strategy for large problems
- `TimeBlock`: structure for defining temporal decomposition
- `OutputValues`: result extraction and formatting

### Key Design Patterns

- **Frozen dataclasses** throughout for immutability (models, expressions, constraints)
- **Visitor pattern** for all expression tree operations (`ExpressionVisitor` subclasses)
- **Indexing dimensions**: parameters and variables carry time and scenario indices explicitly; expressions track these automatically
- **`ValueType`** enum (`INTEGER`, `CONTINUOUS`, `BOOLEAN`) for variable typing

### Type Checking

Strict mypy is enforced (`disallow_untyped_defs`, `disallow_untyped_calls`). All new code must be fully typed. Configuration is in `mypy.ini`.

## Further Reading

- [Python Convention](docs/agents/python-convention.md) — Code style, conventions, and agent guardrails
- [Testing](docs/agents/testing.md) — Testing strategy and layer overview
- [docs/getting-started.md](docs/getting-started.md) — Installation and first study walkthrough
- [docs/user-guide.md](docs/user-guide.md) — Full user documentation
- [docs/developer-guide.md](docs/developer-guide.md) — Contributor guide
- [grammar/](grammar/) — ANTLR4 grammar source (`Expr.g4`)
