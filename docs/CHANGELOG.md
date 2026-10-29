# Changelog

All notable changes to GemsPy are documented here.

## [Unreleased]

### Added
- **`write_yaml_hybrid_library`** - writes a hybrid library and keeps
  `injection-to-balance`, `spillage-bound` and `unsupplied-energy-bound`
  (as `null` when empty), which Antares Simulator requires. `write_yaml_library`
  stays the full GEMS writer and raises when a port type has
  `area-connection` or `thermal-capacity-connection`, naming
  `write_yaml_hybrid_library`. That writer also keeps a
  `thermal-capacity-connection` when the port type has one. The optional
  `price` key is still omitted when empty.
- **`price` in the `area-connection` of port types** - `AreaConnectionSchema`
  accepts the optional `price` key of hybrid libraries (Antares Simulator
  10.2), which names the port field through which a component reads the
  marginal price of the legacy area it is connected to. `parse_yaml_library`
  rejected libraries using it (`Extra inputs are not permitted`).
- **Plain numbers in expression fields** - in a model library, every
  expression field (constraint expressions and bounds, port-field definitions,
  objective contributions, extra outputs) now accepts a YAML number, such as
  `definition: 0`, read as if it were quoted. Only variable bounds accepted one
  before; elsewhere parsing failed with `Input should be a valid string`.
- **`SimulationTable.objective_values()`** - returns the objective value of
  every solved problem, with columns `block`, `scenario_index` and `value`.
- **`block=` on the simulation table views** - `component(...).output(name,
  block=...)` and `OutputView.value(..., block=...)` select one block.
- **Simulation table read API** - `SimulationTable.to_pandas(components,
  outputs, scenarios)` returns part of the table, `iter_scenario_frames()`
  yields it one scenario at a time, `scan()` returns a
  `pyarrow.dataset.Dataset`, and `scenarios` lists the scenarios of the run.
  Scenario filters keep the rows shared by all scenarios, which
  `iter_scenario_frames()` yields once, first.
  `SimulationTable(df, table_id, scenarios=...)` takes the scenarios of the
  run, checked against the rows; the builder and the session pass them.
  `SimulationTable` is exported from `gems_runner.simulation`.
- **Simulation tables stored on disk** - `SimulationTable.save(path, format)`
  writes the table as a folder: one file per scenario of the run
  (`scenario-<N>`, empty for a scenario without rows of its own), a
  `scenario-common` file for the rows shared by all scenarios, and a
  `manifest.json` written last, in Parquet (default) or CSV with one fixed
  schema. `SimulationTable.open(path)` reads it back lazily, opening only the
  files a call needs, and refuses a folder without a manifest. On such a table
  `data` raises and `to_pandas()` without filter needs `load_all=True`.
- **Taxonomy conformance checked when a study is loaded** - `load_study` reads
  the optional `input/taxonomy.yml` and calls
  `validate_libraries_against_taxonomy` on every library declaring a `taxonomy`
  field. `parse_yaml_library` is unchanged and performs no validation.
- **Absolute bounds in time sums** - in `sum(S .. E, X)`, a bound can now be an
  absolute time index of the block (an expression that does not depend on `t`, e.g.
  `sum(0 .. 2, x)`) as well as a bound relative to `t` (with a shift that does not depend on `t`), and both can be mixed
  (e.g. `sum(0 .. t, x)`).
- **Constant terms in objective contributions** - an objective contribution can
  now mix variables and constants (e.g. `sum(cost * generation + 1)`); this
  previously raised `ValueError: Constant values in objective function not
  supported`. `OptimizationProblem.objective_value` and the `objective-value`
  row of the simulation table include the constant; `linopy_model.objective.value`
  and the LP file written by `export_lp` do not.

### Changed
- **Breaking** - `run_study` and `gemspy` write the simulation table as a folder,
  `output/<run_id>/simulation_table/` (one file per scenario, a
  `scenario-common` file and a `manifest.json`), instead of a single
  `simulation_table_<run_id>.csv` / `.parquet` file. In sequential and
  parallel modes, the rows of each block are passed to their scenario's file
  as soon as the block is solved (Parquet writes them in row groups of up to
  64,000 rows); frontal mode solves once and writes one scenario at a time.
  The whole table is never held in memory; reopen it with
  `SimulationTable.open()`. Benders mode writes no simulation table (it wrote
  an empty `simulation_table_.csv`), and `SimulationSession.run()` returns an
  empty table with the schema's columns.
- **Breaking** - the default output format of `run_study` and of the CLI
  `--output-format` option is now Parquet (it was CSV). CSV is still available
  with `--output-format csv`.
- **Breaking** - `SimulationSession(..., output_dir=...).run()` now stores the
  results on disk and returns the stored table, on which `data` raises: use
  `to_pandas(load_all=True)`, the filters of `to_pandas()`, or
  `iter_scenario_frames()`. A new `output_format` argument chooses the format
  (Parquet by default). `output_dir/simulation_table/` must be missing or
  empty, so an `output_dir` holds a single run. If the run fails or is
  interrupted, the completed scenarios stay without a manifest (the
  unfinished one is removed), and `SimulationTable.open()` refuses the folder.
  Without `output_dir`, results stay in memory as before.
  `merge_simulation_tables` is removed.
- **Clearer error for a single-time-step sequential run** - in
  `sequential-subproblems` mode, a time scope whose `first-time-step` equals
  its `last-time-step` still solves no block, but `run()` now fails
  before solving anything with "Sequential mode solves no time block when the
  time scope has a single time step (…): use frontal or parallel-subproblems
  mode." instead of `ValueError: No objects to concatenate`.
- **Scenario playlist removed** - scenario selection is now configured inline
  through `scenario-scope`; the `playlist-file` option is no longer supported.
- **Breaking** - a time-sum bound (an absolute bound, or the offset `k` of a
  relative bound `t + k`) must now be declared time-independent and contain no
  variable. This is checked when the model is created, whatever the data.
  Since a parameter that omits `time-dependent` is time-dependent, parameters
  used in bounds must declare `time-dependent: false`, e.g. `d_min_up` in
  `sum(t - d_min_up + 1 .. t, x)`. Such models were previously accepted in
  `cyclic` mode.
- **Breaking** - `SimulationTable.data` has one fixed schema, whatever the
  content: nullable `Int64` indices (empty for a missing dimension), `string`
  labels and `float64` values (`SIMULATION_TABLE_SCHEMA` /
  `SIMULATION_TABLE_DTYPES`). `scenario_index` was an `object` column mixing
  ints and `None`. Comparisons on empty cells give `<NA>`, which filters treat
  as no match.
- **Breaking** - when the values requested from an `OutputView` include a
  (time, scenario) solved in several blocks (overlapping sequential blocks, or
  the per-block rows of an output without a time dimension), `value()` and
  `data` raise unless `block=` is given. They silently returned the first row.
- **Each run gets its own output folder** - the run id is now the start time to
  the second plus a short random suffix (e.g. `20261009T143502-3f9a1c`, it was
  `20261009T1435`), and `run_study` creates `output/<run_id>/` before solving,
  failing over to a new id rather than reusing an existing folder. Runs started
  in the same minute shared a folder, and the second run's files replaced or
  mixed with the first run's.
- **Breaking** - a model parameter that omits `time-dependent` or
  `scenario-dependent` is now time- or scenario-dependent, as in Antares
  Simulator (it was constant). A component may still give it a constant value.
  Declare `time-dependent: false` / `scenario-dependent: false` for parameters
  that must stay constant.
- **Breaking** - loading a study whose library declares a `taxonomy` raises
  `ValueError` if no taxonomy is supplied, or if its id differs from the declared
  one.
- **Breaking** - `TaxonomyData` renamed to `TaxonomySchema`; no alias kept.
- **Breaking** - `check_library_against_taxonomy` moved from
  `gems_craft.model.taxonomy` to the new `gems_craft.model.validation`, and
  `consistency_check` from `gems_craft.study.resolve_components` to the new
  `gems_craft.study.validation`, where it is renamed
  `check_component_models`. The `Study.check_consistency()` method moved to
  that same module as the function `check_data_requirements(study)` — the two
  checks are disjoint (component model ids vs. database coverage) and their
  near-identical old names invited confusion. Reading and validating are now
  separate modules, as in `optim_config/`. No behavior change; names and
  import paths only.

### Deprecated
- **`SimulationTable.to_csv()`, `to_parquet()`, `to_dataset()` and
  `to_netcdf()`** - they emit a `DeprecationWarning` and will be removed in a
  following release. Use `save()`, or build an xarray object from
  `to_pandas()`. They raise on a table stored on disk.

### Fixed
- **Empty scenario scope** - a `scenario-scope` that selects no scenario (e.g.
  every included scenario excluded) is rejected by `validate_optim_config` at
  the start of `SimulationSession.run()`, before anything is solved, instead of
  failing during the run with `No objects to concatenate` or `cannot reshape
  array of size 0`.
- **Outputs without a time or scenario dimension** - `OutputView.value()`
  accepts any index for the missing dimension and returns the single stored
  value; it raised `KeyError` for any index but 0. `OutputView.data` keeps one
  row (or column) for that dimension, labelled `<NA>` (it was `0`).
- **Scenario index of single-scenario solves** - a problem solved for a single
  scenario (every sequential/parallel block, a frontal run with one scenario)
  now fills `scenario_index` on every row, including the objective value and
  outputs without a scenario dimension. They were left empty, so in sequential
  and parallel modes the rows of different scenarios could not be told apart.
- **`solver-options.logs` now controls solver output** - it was passed to the
  solver as an unknown `solver_logs` option (HiGHS printed
  `getOptionIndex: Option "solver_logs" is unknown` on every solve) and had no
  effect. It is now translated into the solver's own output option
  (`output_flag` for HiGHS, `OutputFlag` for Gurobi, `outputlog` for Xpress);
  with `logs: false` HiGHS only prints its startup banner.
- **Time sums with bounds relative to `t`** - a start after the end now gives an
  empty sum, a bound can contain a time operator (e.g. `t - (d[0])`), and a
  scenario-dependent bound makes the sum scenario-dependent. In `drop` mode,
  the operand is checked at every summed time step.
- **`first-time-step` greater than `last-time-step`** is now rejected when the
  optim-config is loaded. The run used to fail later, with an error that did
  not mention the time scope.

---

## [0.2.0] - 2026-08-24

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
  heuristics" above).

### Changed
- **Sequential mode carry-over length is now user-controlled** through the new
  `resolution.carry-over-length` parameter (default: `block-overlap`), which
  sets how many of the shared leading timesteps of block *N+1* are pinned to
  block *N*'s already-solved values. This also fixes an incorrect stitching
  for `block-overlap >= 2`, where the previous hardcoded behaviour pinned block
  *N+1*'s first timestep to block *N*'s **last** timestep — a different
  absolute timestep.
- **linopy upgraded to `>=0.9.0`** - the minimum supported Python version rises
  to **3.11** accordingly (linopy 0.9 requires Python >= 3.11).

### Fixed
- **Standard library parsing now accepts hybrid port-type fields** -
  `parse_yaml_library` (the standard, non-hybrid parser) no longer rejects
  `area-connection` / `thermal-capacity-connection` on port-types, so a
  library YAML file can be used as-is in both full GEMS and hybrid studies.
  `AreaConnectionSchema` and `PortThermalCapacitySchema` moved from the
  deleted `gems_craft_hybrid/model/` (`HybridLibrarySchema` /
  `HybridPortTypeSchema`) directly onto `gems_craft`'s own `PortTypeSchema`;
  the fields are parsed and validated but otherwise ignored outside hybrid
  consumers.

---

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
