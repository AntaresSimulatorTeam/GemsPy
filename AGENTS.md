# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GemsPy** is a Python interpreter for the GEMS (Generic Energy Systems Modelling Schema) framework — a high-level modeling language for simulating energy systems under uncertainty. It allows users to define energy system models via YAML without writing solver code directly.

## Commands

**Install:**
```bash
uv sync          # installs all dependencies including dev group
# or: pip install -e ".[dev]"
```

**Test:**
```bash
pytest                                          # run all tests
pytest tests/path/to/test_file.py::test_name   # run a single test
pytest --cov gems --cov-report xml             # with coverage
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

# Python API — directory-based study
from gems.study.folder import load_study
from gems.study.runner import run_study

study = load_study(Path("path/to/study_dir"))   # reads input/, model-libraries/, data-series/
run_study(Path("path/to/study_dir"))            # loads study, solves, writes CSV to output/

# Python API — programmatic study
from gems.study import Study
from gems.simulation import build_problem, TimeBlock

study = Study(system=system, database=database)
problem = build_problem(study, TimeBlock(1, list(range(8760))), scenarios=1)
problem.solve(solver_name="highs")
```

## Architecture

The pipeline flows: **YAML input → parsing → model resolution → system instantiation → optimization problem → HiGHS solver (via linopy) → results**

An optional `optim-config.yml` activates decomposition: variables and constraints are split across a master problem and subproblems, with either sequential resolution or full Benders decomposition.

### Core Modules (`src/gems/`)

**`model/`** — Immutable model templates.
- `Model`: defines component behavior (parameters, variables, constraints, ports)
- `Library`: a collection of models, loaded from YAML

**`expression/`** — Mathematical expression language and AST.
- `ExpressionNode`: base frozen dataclass for all expression tree nodes
- Grammar is defined in `grammar/Expr.g4` and parsed via ANTLR4 (generated files live in `expression/parsing/antlr/` — do not edit directly)
- `ExpressionVisitor` is the dominant pattern for traversing and transforming expression trees (evaluation, linearization, printing, degree analysis)

**`study/`** — Study definition and instantiation.
- `System` (`system.py`): resolved topology — graph of `Component`s, `PortRef`s, and `PortsConnection`s after library references are substituted
- `Study` (`study.py`): dataclass pairing a `System` with a `DataBase`; validates that the database supplies every parameter required by the system
- `DataBase` (`data.py`): manages time-series and scenario data;
- `load_study` / `run_study` (`folder.py`): convenience functions for directory-based studies (`input/system.yml`, `input/model-libraries/`, `input/data-series/`)

**`simulation/`** — Optimization problem construction and solving.
- `OptimizationProblem` (`optimization.py`): main interface; translates a `Study` into a linopy model solved by HiGHS
- `DecomposedProblems` (`optimization.py`): holds the master problem and subproblem produced by temporal decomposition
- `VectorizedLinearExprBuilder` (`linearize.py`): `ExpressionVisitor` subclass that converts an expression AST into a `VectorizedExpr`
- `VectorizedBuilderBase` (`vectorized_builder.py`): shared base for all vectorized visitors (used by both `linearize.py` and `extra_output.py`)
- `TimeBlock` (`time_block.py`): defines the temporal window for one solve
- `SimulationTableBuilder` / `SimulationTableWriter` (`simulation_table.py`): result extraction as a flat pandas `DataFrame`

**`optim_config/`** — Optional decomposition configuration.
- `OptimConfig` (`parsing.py`): top-level config loaded from `optim-config.yml`
- `ResolutionMode` (`parsing.py`): `FRONTAL` (default), `SEQUENTIAL_SUBPROBLEMS`, `PARALLEL_SUBPROBLEMS`, or `BENDERS_DECOMPOSITION`
- `ModelDecompositionConfig` (`parsing.py`): per-model assignment of variables/constraints/objective contributions to master or subproblems

**`libs/`** — Resolves the path to bundled YAML model libraries shipped with the package.

### Key Design Patterns

- **Visitor pattern** for all expression tree operations (`ExpressionVisitor` subclasses). Use `ExpressionVisitorOperations` as a base when the return type supports `+, -, *, /` — it provides those four method implementations for free.
- **Template-method via single abstract method**: `VectorizedBuilderBase` implements all 18+ visitor methods once with `xr.DataArray`-compatible semantics; concrete subclasses only override `variable()` (and optionally a few linopy-specific methods).
- **Indexing dimensions**: parameters and variables carry time and scenario indices explicitly via `IndexingStructure`; expressions track these automatically.

## Further Reading

- [Python Convention](docs/agents/python-convention.md) — Code style, conventions, and agent guardrails
- [Testing](docs/agents/testing.md) — Testing strategy and layer overview
- [docs/getting-started.md](docs/getting-started.md) — Installation and first study walkthrough
- [docs/user-guide.md](docs/user-guide.md) — Full user documentation
- [docs/developer-guide.md](docs/developer-guide.md) — Contributor guide
- [grammar/](grammar/) — ANTLR4 grammar source (`Expr.g4`)
