# Changelog

All notable changes to GemsPy are documented here.

## [Unreleased]

### Added

- **System components: `properties`** - introduces optional `properties` on components in `system.yml` (a list of `id`/`value` pairs). These are normalized into a `dict[str, str]` on the resolved `Component` (duplicate ids raise a `ValueError`).
- **Model schema: `taxonomy-category`** - introduces optional `taxonomy-category` on models in library YAML files, exposed as `ModelSchema.taxonomy_category`.
- **Taxonomy check** - new `gems.model.taxonomy` module with `load_taxonomy(path)` and `check_library_against_taxonomy(library, taxonomy)`. Validates that every model declaring a `taxonomy-category` references a category that exists in the taxonomy file, and exposes all variables, parameters, constraints, ports, extra-outputs and properties required by that category. Taxonomy classes (`TaxonomyItem`, `TaxonomyCategory`, `Taxonomy`) mirror the structure defined in GEMS-ViewsBuilder.

## [0.1.0] - 2026-04-30

### Study folder structure

Studies are now loaded from a conventional directory layout via `load_study(study_dir)`:

```
<study>/
  input/
    system.yml                          # component topology and connections
    model-libraries/*.yml               # model library files
    data-series/                        # time-series parameter files (TSV)
    data-series/modeler-scenariobuilder.dat   # optional scenario mapping
```

The `Study` object (pairing a `System` with a `DataBase`) is the single input to `SimulationSession`.

### Optimization configuration (`optim-config.yml`)

A new `optim-config.yml` file controls all aspects of a simulation run:

- **`resolution.mode`** — four strategies: `frontal`, `sequential-subproblems`, `parallel-subproblems`, `benders-decomposition`
- **`resolution.block_length` / `block_overlap`** — time-window size and overlap for sequential/parallel modes
- **`time_scope`** — `first_time_step` / `last_time_step`
- **`scenario_scope.nb_scenarios`** — number of Monte-Carlo scenarios to run
- **`solver_options`** — solver name (default: HiGHS), log verbosity, and free-form solver parameters
- **`models[].model_decomposition`** — per-model assignment of variables, constraints, and objective contributions to `master`, `subproblems`, or `master-and-subproblems` (used for Benders decomposition)
- **`models[].out_of_bounds_processing`** — per-constraint handling of time indices that fall outside the horizon (`cyclic` or `drop`)

### Scenario builder

`modeler-scenariobuilder.dat` maps Monte-Carlo scenario indices to data-series column indices on a per-scenario-group basis, allowing different components to draw from different stochastic draws.

### Vectorized solver

The internal optimizer was migrated from a scalar OR-Tools pipeline to a fully vectorized [linopy](https://github.com/PyPSA/linopy) / xarray pipeline. All constraints for all components, time steps, and scenarios are now built in a single pass and solved in one call — significantly reducing build time for large horizons.

### Removed
- OR-Tools dependency.

---

## [0.0.6] - 2025

### Added
- Math operators `floor`, `ceil`, `min`, and `max` in the GemsPy expression language, with `max`/`min` accepting a variadic argument list (`argList` grammar rule).
- Support for complex variable bounds expressions: operator expansion now runs before bounds evaluation, enabling expressions such as `min(p_max_cluster, min_gen_mod * unit_count * p_max_unit)`.
- `visitTimeShiftExpr` and `visitTimeIndexExpr` in the expression visitor interface.
- `AGENTS.md` and agent guidance documentation.
- Status and quality badges in README.

### Fixed
- O(T²) list-copy performance bug in `CopyVisitor.addition` (now O(T)).
- Pandas and Pydantic deprecation warnings resolved.
- `floor`/`ceil`/`max`/`min` visitor implementations in `LinearExpressionBuilder`.
- Degree of `floor`/`ceil`/`max`/`min` returns `inf` for non-constant operands.

---

## [0.0.5] - 2025

### Added
- `load_input_system`: load and validate an input system into memory from a file path.

---

## [0.0.4] - 2025

### Removed
- PyPSA → GEMS converter and related files (out of scope).

---

## [0.0.3] - 2025

### Changed
- Version naming aligned; package prepared for PyPI upload.
- README revised to consistently use GEMS / GemsPy naming.

---

## [0.0.2] - 2025

### Changed
- Initial PyPI preparation: project metadata, classifiers, and packaging configuration.
