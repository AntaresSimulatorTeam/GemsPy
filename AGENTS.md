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
uv run pytest --cov gems_craft --cov gems_craft_hybrid --cov gems_runner --cov-report xml  # with coverage
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
gemspy \
  --model-libs  path/to/lib1.yml path/to/lib2.yml \
  --components   path/to/components.yml \
  --timeseries   path/to/timeseries/ \
  --duration     8760 \
  --scenarios    1

# Python API — directory-based study
from gems_craft.study.folder import load_study
from gems_runner.study.runner import run_study

study = load_study(Path("path/to/study_dir"))   # reads input/, model-libraries/, data-series/
run_study(Path("path/to/study_dir"))            # loads study, solves, writes CSV to output/

# Python API — programmatic study
from gems_craft.study import Study
from gems_runner.simulation import build_problem, TimeBlock

study = Study(system=system, database=database)
problem = build_problem(study, TimeBlock(1, list(range(8760))), scenarios=1)
problem.solve(solver_name="highs")
```

## Architecture

The pipeline flows: **YAML input → parsing → model resolution → system instantiation → optimization problem → HiGHS solver (via linopy) → results**

An optional `optim-config.yml` activates decomposition: variables and constraints are split across a master problem and subproblems, with either sequential resolution or full Benders decomposition.

### Three packages: `gems_craft`, `gems_craft_hybrid`, and `gems_runner`

The codebase is split into three packages along a solver-dependency boundary:

- **`gems_craft`** (`src/gems_craft/`) — the domain model and all YAML I/O. No solver dependency at the module level; importable on its own for building, editing, validating, and querying systems (e.g. an API layer) without touching solve-time code. Deps: `numpy`, `pandas`, `PyYAML`, `pydantic`, `anytree`, `antlr4-python3-runtime`.
- **`gems_craft_hybrid`** (`src/gems_craft_hybrid/`) — read/write support for *hybrid* GEMS studies, an extended format used to interoperate with Antares Simulator. No solver dependency; depends only on `gems_craft`. Hybrid studies cannot be simulated by GemsPy — this package only extends the `gems_craft` parsing schemas (`HybridLibrarySchema`, `HybridSystemSchema`) with the extra fields (`area-connection`, `thermal-capacity-connection` on port-types; `area-connections`, `thermal-capacity-connections` on systems), reusing `gems_craft`'s schema-parameterized `parse_yaml_library`/`load_input_system`/`parse_yaml_system`/`write_yaml_library`/`write_yaml_system`.
- **`gems_runner`** (`src/gems_runner/`) — solve-time execution. Depends on `gems_craft` plus `linopy`, `xarray`, `highspy`, all installed as part of the base `gemspy` package (`pip install gemspy` installs the full solver stack, not just `gems_craft`'s dependencies).

### Core Modules

**`gems_craft/model/`** — Immutable model templates.
- `Model`: defines component behavior (parameters, variables, constraints, ports)
- `Library`: a collection of models, loaded from YAML
- `Taxonomy` (`taxonomy.py`): categories naming the items a model must expose. Models opt in via `taxonomy-category`; `check_library_against_taxonomy` enforces conformance.

**`gems_craft/expression/`** — Mathematical expression language and AST (structural/static analysis only — no numeric evaluation).
- `ExpressionNode`: base frozen dataclass for all expression tree nodes
- Grammar is defined in `grammar/Expr.g4` and parsed via ANTLR4 (generated files live in `expression/parsing/antlr/` — do not edit directly)
- `ExpressionVisitor` is the dominant pattern for traversing and transforming expression trees (linearization support, printing, degree analysis, indexing)
- Numeric evaluation (`EvaluationVisitor`) lives in `gems_runner.expression.evaluate`, not here — several of its node handlers (`dual()`, `reduced_cost()`, `variable()`) are solver-output-shaped and only make sense at solve time.

**`gems_craft/study/`** — Study definition and instantiation.
- `System` (`system.py`): resolved topology — graph of `Component`s, `PortRef`s, and `PortsConnection`s after library references are substituted
- `Study` (`study.py`): dataclass pairing a `System` with a `DataBase`; validates that the database supplies every parameter required by the system
- `DataBase` (`data.py`): manages time-series and scenario data
- `load_study` (`folder.py`): convenience function for directory-based studies (`input/system.yml`, `input/model-libraries/`, `input/data-series/`)
- `run_study` lives in `gems_runner.study.runner` (it drives an actual solve via `SimulationSession`)

**`gems_craft/optim_config/`** — Decomposition configuration schema (no solver dependency; consumed by `gems_runner` at solve time).
- `OptimConfig` (`parsing.py`): top-level config loaded from `optim-config.yml`
- `ResolutionMode` (`parsing.py`): `FRONTAL` (default), `SEQUENTIAL_SUBPROBLEMS`, `PARALLEL_SUBPROBLEMS`, or `BENDERS_DECOMPOSITION`
- `ModelDecompositionConfig` (`parsing.py`): per-model assignment of variables/constraints/objective contributions to master or subproblems
- `HeuristicConfig` (`parsing.py`): per-model binding of a built-in heuristic (`fast`/`accurate`) to the model's own parameters/variables

**`gems_craft/libs/`** — Resolves the path to bundled YAML model libraries shipped with the package.

**`gems_craft_hybrid/model/`** — `HybridLibrarySchema(LibrarySchema)`, `HybridPortTypeSchema(PortTypeSchema)`: adds `area-connection` (`AreaConnectionSchema`) and `thermal-capacity-connection` (`PortThermalCapacitySchema`) per port-type.

**`gems_craft_hybrid/study/`** — `HybridSystemSchema(SystemSchema)`: adds `area-connections` (`AreaConnectionsSchema`) and `thermal-capacity-connections` (`ThermalCapacityConnectionSchema`, referencing a `ThermalComponentSchema`).

**`gems_runner/simulation/`** — Optimization problem construction and solving.
- `OptimizationProblem` (`optimization.py`): main interface; translates a `Study` into a linopy model solved by HiGHS
- `DecomposedProblems` (`optimization.py`): holds the master problem and subproblem produced by temporal decomposition
- `VectorizedLinearExprBuilder` (`linearize.py`): `ExpressionVisitor` subclass that converts an expression AST into a `VectorizedExpr`
- `VectorizedBuilderBase` (`vectorized_builder.py`): shared base for all vectorized visitors (used by both `linearize.py` and `extra_output.py`)
- `TimeBlock` (`time_block.py`): defines the temporal window for one solve
- `SimulationTableBuilder` / `SimulationTableWriter` (`simulation_table.py`): result extraction as a flat pandas `DataFrame`
- `apply_thermal_heuristics` (`heuristic_runner.py`): injects heuristic-derived bounds into a solved problem
- `find_min_generation_fast` / `find_num_units_accurate` (`thermal_heuristic.py`): the `fast`/`accurate` heuristic functions

**`gems_runner/session/`** — `SimulationSession`, the stateful entry point wrapping a loaded study for interactive solving.

**`gems_runner/main/`** — the `gemspy` CLI entry point (`main_cli`).

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
