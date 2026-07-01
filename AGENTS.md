# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GemsPy** is a Python interpreter for the GEMS (Generic Energy Systems Modelling Schema) framework — a high-level modeling language for simulating energy systems under uncertainty. It allows users to define energy system models via YAML without writing solver code directly.

## Commands

**Install:**
```bash
uv sync --group dev
```

**Test:**
```bash
uv run pytest                                          # run all tests
uv run pytest tests/path/to/test_file.py::test_name   # run a single test
uv run pytest --cov gems_craft --cov gems_runner --cov gems_craft_hybrid --cov-report xml  # with coverage
```

**Lint & Format:**
```bash
uv run black src tests
uv run isort --profile black src tests
uv run mypy
```

**Running:**
```bash
# CLI entry point
gemspy --study path/to/study_dir

# Python API — directory-based study
from gems_runner.study.folder import load_study
from gems_runner.study.runner import run_study

study = load_study(Path("path/to/study_dir"))   # reads input/, model-libraries/, data-series/
run_study(Path("path/to/study_dir"))            # loads study, solves, writes CSV to output/

# Python API — programmatic study
from gems_runner.study import Study
from gems_runner.simulation import build_problem, TimeBlock

study = Study(system=system, database=database)
problem = build_problem(study, TimeBlock(1, list(range(8760))), scenario_ids=[0])
problem.solve(solver_name="highs")
```

## Architecture

The pipeline flows: **YAML input → parsing → model resolution → system instantiation → optimization problem → HiGHS solver (via linopy) → results**

An optional `optim-config.yml` activates decomposition: variables and constraints are split across a master problem and subproblems, with either sequential resolution or full Benders decomposition.

### Core Modules

The source tree is split into three packages under `src/`:

#### `gems_craft/` — I/O and data structures (no solver dependency)

**`model/`** — Immutable model templates.
- `Model`: defines component behavior (parameters, variables, constraints, ports)
- `Library`: a collection of models, loaded from YAML
- `Taxonomy` (`taxonomy.py`): categories naming the items a model must expose. Models opt in via `taxonomy-category`; `check_library_against_taxonomy` enforces conformance.
- Key functions: `load_yaml_library(path, schema=LibrarySchema)`, `write_yaml_library(library, path)`

**`study/`** — Study schema and parsing.
- `SystemSchema` (`parsing.py`): Pydantic model for `system.yml`
- Key functions: `load_yaml_system(path, schema=SystemSchema)`, `write_yaml_system(system, path)`
- `ScenarioBuilder` (`scenario_builder.py`): maps MC scenarios to data-series column indices; `load_dat` / `write_dat`

**`optim_config/`** — Optional decomposition configuration.
- `OptimConfig` (`parsing.py`): top-level config loaded from `optim-config.yml`
- `ResolutionMode` (`parsing.py`): `FRONTAL` (default), `SEQUENTIAL_SUBPROBLEMS`, `PARALLEL_SUBPROBLEMS`, or `BENDERS_DECOMPOSITION`
- `ModelDecompositionConfig` (`parsing.py`): per-model assignment of variables/constraints/objective contributions to master or subproblems
- Key functions: `load_yaml_optim_config(path)`, `write_yaml_optim_config(config, path)`

#### `gems_craft_hybrid/` — Hybrid study I/O (read/write only, no simulation)

Extends `gems_craft` schemas for hybrid GEMS studies. **Hybrid studies cannot be simulated with GemsPy** — simulation support is not yet implemented.

**`study/`** — `HybridSystemSchema(SystemSchema)`: adds `area-connections` and `thermal-capacity-connections`.
- Use `load_yaml_system(path, HybridSystemSchema)` and `write_yaml_system(system, path)` from `gems_craft`

**`model/`** — `HybridPortTypeSchema(PortTypeSchema)`: adds `area-connection` and `thermal-capacity-connection`. `HybridLibrarySchema(LibrarySchema)`: overrides `port-types` with `HybridPortTypeSchema`.
- Use `load_yaml_library(path, HybridLibrarySchema)` and `write_yaml_library(library, path)` from `gems_craft`

#### `gems_runner/` — Solver and execution (depends on `gems_craft`)

**`model/`** — Resolves YAML schemas into runtime objects.
- `resolve_library`: validates and cross-links models across libraries

**`expression/`** — Mathematical expression language and AST.
- `ExpressionNode`: base frozen dataclass for all expression tree nodes
- Grammar is defined in `grammar/Expr.g4` and parsed via ANTLR4 (generated files live in `expression/parsing/antlr/` — do not edit directly)
- `ExpressionVisitor` is the dominant pattern for traversing and transforming expression trees (evaluation, linearization, printing, degree analysis)

**`study/`** — Study definition and instantiation.
- `System` (`system.py`): resolved topology — graph of `Component`s, `PortRef`s, and `PortsConnection`s after library references are substituted
- `Study` (`study.py`): dataclass pairing a `System` with a `DataBase`; validates that the database supplies every parameter required by the system
- `DataBase` (`data.py`): manages time-series and scenario data
- `load_study` / `run_study` (`folder.py`): convenience functions for directory-based studies (`input/system.yml`, `input/model-libraries/`, `input/data-series/`)

**`simulation/`** — Optimization problem construction and solving.
- `OptimizationProblem` (`optimization.py`): main interface; translates a `Study` into a linopy model solved by HiGHS
- `DecomposedProblems` (`optimization.py`): holds the master problem and subproblem produced by temporal decomposition
- `VectorizedLinearExprBuilder` (`linearize.py`): `ExpressionVisitor` subclass that converts an expression AST into a `VectorizedExpr`
- `VectorizedBuilderBase` (`vectorized_builder.py`): shared base for all vectorized visitors (used by both `linearize.py` and `extra_output.py`)
- `TimeBlock` (`time_block.py`): defines the temporal window for one solve
- `SimulationTableBuilder` / `SimulationTableWriter` (`simulation_table.py`): result extraction as a flat pandas `DataFrame`

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

> **Full ecosystem developer guide:** the authoritative branching, versioning, CI/CD, and release process for all repositories (including GemsPy) lives in the GEMS Developer Guidelines, published at <https://gems-energy.readthedocs.io/en/latest/support/dev-guidelines/>. Fetch this page (e.g. via WebFetch) before any branching, versioning, or release work.
