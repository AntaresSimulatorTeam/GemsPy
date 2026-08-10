# Changelog

All notable changes to GemsPy are documented here.

## [Unreleased]

### Added
- **Integer strategy and thermal heuristics** - components can now set
  `integer-strategy` (`exact` (default), `relaxed`, or `heuristic` +
  `heuristic-id`) to control how their model's integer/binary variables are
  built. `heuristic` relaxes to continuous and triggers a second solve in
  `SimulationSession`, after which built-in heuristics (`fast`, `accurate`)
  compute tighter variable bounds from the first solve. Each model declares
  what a heuristic reads/writes via `models[].heuristics` in
  `optim-config.yml`.
- **`lower_bound(variable_name)`** and **`upper_bound(variable_name)`** operators in the
  expression language, usable in `extra-outputs` and port-field-definitions. Both take a bare
  variable identifier and return its *current* lower/upper bound post-solve — in particular
  reflecting mutations made by thermal heuristics (see "Integer strategy and thermal
  heuristics" above), which previously had no way to be surfaced in results. Validated at
  model-build time; using them inside constraints, binding-constraints, objective
  contributions, or variable bounds raises a `ValueError`.

## [0.1.3] - 2026-07-24

### Added
- **`taxonomy` field on `LibrarySchema`** - optional free-form string field on
  the library YAML schema; no validation is attached. `taxonomy` is also
  carried through to the resolved `Library` class.

- **`version` field on library YAML** - `LibrarySchema` (standard, not
  hybrid-specific) gains an optional top-level `version` string, for
  tracking the version of a library.
- **Hybrid studies (`gems_craft_hybrid`)** - new package for reading and
  writing GEMS studies extended with fields used to interoperate with
  Antares Simulator. Hybrid studies cannot be simulated by GemsPy.
  - `HybridLibrarySchema` / `HybridPortTypeSchema` add `area-connection` and
    `thermal-capacity-connection` per port-type.
  - `HybridSystemSchema` adds `area-connections` and
    `thermal-capacity-connections`. `AreaConnectionsSchema` moved from the
    standard `SystemSchema` into `HybridSystemSchema` — it was parsed but
    never consumed by the standard resolve/solve pipeline.
  - `gems_craft.model.parsing.parse_yaml_library` and
    `gems_craft.study.parsing.load_input_system` /
    `parse_yaml_system` now accept an optional `schema` parameter to
    validate against a subclass (e.g. the hybrid schemas above); new
    `write_yaml_library` / `write_yaml_system` functions support the
    corresponding round-trip.

### Changed

- **Package split** - the monolithic `gems` package is split into `gems_craft`
  and `gems_runner`.
  - `gems_craft` holds the solver-independent domain model and I/O: `expression`
    (AST, parsing, linearity/indexing analysis), `model`, `optim_config`,
    `study` (`System`, `Component`, `Study`, data, YAML parsing/resolution),
    `utils`, `libs`. Its dependencies are `numpy`, `pandas`, `PyYAML`,
    `pydantic`, `anytree`, `antlr4-python3-runtime` - no solver required.
  - `gems_runner` holds solve-time execution: `expression.evaluate`,
    `session`, `simulation`, `study.runner`, `main` (the `gemspy` CLI). It
    depends on `gems_craft` plus `linopy`, `xarray`, `highspy`.
  - All `gems.*` imports must be updated to `gems_craft.*` or `gems_runner.*`
    accordingly; see the module lists above. The `gemspy` console script now
    points at `gems_runner.main.main:main_cli`.

### Fixed
- Comparison operators (`>=`, `<=`, `=`) in extra-output expressions no longer raise
  `NotImplementedError` at post-solve evaluation; they now evaluate to a 0/1 indicator
  (e.g. `dual(balance) >= unsupplied_cost - 5`).
- Extra-output `minimum()`/`maximum()` no longer break when several models coexist
  in the same problem: solution and constraint-dual arrays are now filtered back
  down to each model's own components before evaluation, instead of keeping the
  NaN-padded entries introduced by linopy's outer join across models.

---

## [0.1.2] - 2026-06-11

### Added
- **Properties in Models and Components** - models in library YAML files may declare an optional `properties` list (entries with an `id`). Components in system YAML files carry optional `properties`.
- **Taxonomy** - new `gems.model.taxonomy` module to represent Taxonomy, i.e. a classification of GEMS models.  Optional `taxonomy-category` on models in library YAML files, exposed as `ModelSchema.taxonomy_category`.
- Math operators `abs` and `round` in the GEMS expression language.
  - Can be applied to parameters and literals in constraints, bounds, and objective contributions (degree-0 context).
  - Can be applied to any expression in extra-outputs (post-solve evaluation), including decision variables.
  - Use `.abs()` and `.round()` methods on expression objects, or `abs(expr)` and `round(expr)` in parsed expression strings.
- **`dual(constraint_name)`** and **`reduced_cost(variable_name)`** operators in
  the expression language, usable in `extra-outputs`.  Both are validated at
  model-build time; using them inside constraints or objective contributions
  raises a `ValueError`.
- **Xpress** (≥ 9.8) and **Gurobi** (≥ 10.0) solver support alongside HiGHS.
  Reduced costs use each solver's native API (`getLpSol` / `getAttr("RC")` /
  `col_dual`) since linopy has no unified reduced-cost interface.
- New `solvers` optional dependency group (`uv sync --group solvers`) for
  running solver-specific tests with a licence; CI now installs it.

### Fixed

- Pre-commit `black` hook no longer passes redundant `--config pyproject.toml`.
- Pre-commit `isort` hook simplified to `--profile black` (removed
  `--filter-files` which conflicted with pre-commit's own file filtering).


### Changed
- Modernized README design: new layout, GEMS favicon, quick-link navigation, and `uv` install instructions.

---

## [0.1.1] - 2026-05-29

### Scenario-scope playlist (replaces `nb-scenarios`)

The `scenario-scope` section of `optim-config.yml` now supports a full
playlist mechanism.  The old `nb-scenarios` integer key is removed and raises
a validation error if still present.

Scenario indices are **0-based** throughout, consistent with the
`modeler-scenariobuilder.dat` convention.

**Inline form** — specify scenarios with integers, string-integers, and
inclusive `"a-b"` range strings:

~~~ yaml
scenario-scope:
  include:
    - "0-49"
    - 74
    - "89-99"
  exclude:
    - 9
    - 14
~~~

**Playlist-file form** — point to a flat JSON array of 0-based integers,
useful for machine-generated playlists:

~~~ yaml
scenario-scope:
  playlist-file: mc_playlist.json
~~~

Other changes:

- `exclude` is now compatible with both `include` and `playlist-file`.
  Use it to subtract a few scenarios at run time without modifying the
  playlist file.
- `validate_optim_config()` now accepts an optional `scenario_builder`
  argument and cross-checks all playlist indices against every scenario group,
  raising a `ValueError` for out-of-bounds indices.
- The playlist is resolved and cached exactly once at `load_optim_config()`
  time; I/O and format errors surface immediately as `ValueError`.
- Boolean values (`true`/`false`) are explicitly rejected in both inline
  lists and JSON playlist files.

---

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
- **`scenario_scope.nb_scenarios`** — number of Monte-Carlo scenarios to run (replaced by the playlist mechanism in a later release)
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
