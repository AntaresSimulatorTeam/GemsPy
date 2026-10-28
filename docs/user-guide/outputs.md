# Outputs: retrieving outputs with GemsPy

The results of a run are a `SimulationTable`. It is held in memory, or stored on
disk as a folder (one file per scenario), with the same API in both cases.

## Getting the results

### In memory: `SimulationSession` without `output_dir`

~~~ python
from pathlib import Path
from gems_craft.study.folder import load_study
from gems_runner.session import SimulationSession
from gems_craft.optim_config import load_optim_config

study = load_study(Path("my_study"))
optim_config = load_optim_config(Path("my_study/input/optim-config.yml"))

results = SimulationSession(study=study, optim_config=optim_config).run()
~~~

This suits tests, notebooks and small studies: the whole table stays in memory.

### On disk: `SimulationSession` with `output_dir`

~~~ python
from gems_craft.study.parsing import OutputFormat

results = SimulationSession(
    study=study,
    optim_config=optim_config,
    output_dir=Path("my_run"),
    output_format=OutputFormat.PARQUET,  # default; or OutputFormat.CSV
).run()
~~~

The results are written to `my_run/simulation_table/` as they are solved, so
the whole table is never held in memory, and `run()` returns the stored table.
In sequential and parallel modes, the rows of each block are passed to the
file of their scenario as soon as the block is solved: CSV appends them at
once, Parquet buffers them and writes row groups of up to 64,000 rows, so
about one row group of rows is held, and the memory of the solved problems is
freed after each scenario. In frontal mode, the problem is solved once, then
the rows of each scenario are built from the solution and written one scenario
at a time.

- `my_run/simulation_table/` must be missing or empty: use a new `output_dir`
  for each run. Otherwise `run()` raises `FileExistsError`, before anything is
  solved.
- If the run fails or is interrupted, the files of the scenarios completed so
  far stay in the folder (the unfinished scenario's file is removed), but no
  manifest is written, so `SimulationTable.open()`
  refuses it. Delete the folder, or use another `output_dir`, before running
  again.
- Benders mode produces no simulation table: no `simulation_table/` folder is
  written and `run()` returns an empty table.

### From the command line: `gemspy` and `run_study`

`gemspy` and `run_study` store the results in a new run folder,
`<study>/output/<run_id>/simulation_table/`, in Parquet by default (Benders
mode writes no table). The run folder is created before the run starts, so a
run that fails leaves it in `output/`, empty or with the scenarios written so
far:

~~~ bash
gemspy --study path/to/study_dir                       # Parquet
gemspy --study path/to/study_dir --output-format csv   # CSV
~~~

The equivalent Python call is
`run_study(Path("path/to/study_dir"), output_format=OutputFormat.CSV)`.

### Reopening a stored run

~~~ python
from gems_runner.simulation import SimulationTable

results = SimulationTable.open(Path("my_study/output/<run_id>/simulation_table"))
~~~

`open()` reads only the manifest; rows are read when asked for. A folder
without a manifest (an interrupted run) is refused.

### Low-level: `SimulationTableBuilder`

When using `build_problem()` directly, build the table from the solved problem:

~~~ python
from gems_runner.simulation.simulation_table import SimulationTableBuilder

results = SimulationTableBuilder().build(problem)
~~~

---

## The table folder

~~~
simulation_table/
├── manifest.json            # written last: a folder without it is incomplete
├── scenario-common.parquet  # rows shared by all scenarios, if any
├── scenario-0.parquet
└── scenario-1.parquet
~~~

- One file per scenario of the run, each keeping the `scenario_index` column
  (a scenario without rows of its own gets an empty file).
- `scenario-common` holds the rows shared by all scenarios: outputs without a
  scenario dimension and the objective value of a frontal run with several
  scenarios. Their `scenario_index` is empty.
- In sequential and parallel modes (and in frontal mode with one scenario),
  each scenario is solved on its own: these rows carry its `scenario_index` and
  go to its file, once per block, and there is no `scenario-common` file.
- `manifest.json` holds the schema version, the `run_id`, the format, the
  scenarios of the run, whether there is a `scenario-common` file
  (`has_common`) and the column schema.
- Every file has the same columns and types: `block`, `absolute_time_index`,
  `block_time_index` and `scenario_index` are integers (empty for a missing
  dimension), `component`, `output` and `basis_status` are strings, `value` is
  a float.
- In CSV, only an empty cell is read as missing, so a label that is an empty
  string and a missing value both read back as `<NA>`. Parquet keeps them
  apart.

Other tools can read the files directly, e.g. `scenario-*.parquet` with polars
or duckdb: include the `scenario-common` file, which the glob matches. Check
that `manifest.json` exists first: without it, the folder is an incomplete run.

---

## Reading results

The same calls work on a table in memory and on a stored one. On a stored
table, only the files and rows a call needs are read.

### Fluent API

~~~ python
output_view = results.component("gen_de").output("generation")

val = output_view.value(time_index=0, scenario_index=0)  # a float
series = output_view.value(scenario_index=0)             # Series over time
frame = output_view.data                                 # Time × Scenario DataFrame
~~~

- An output without a time (or scenario) dimension has a single row (or
  column), labelled `<NA>`, and `value()` accepts any index for that dimension.
- When the same (time, scenario) was solved in several blocks (overlapping
  sequential blocks, or outputs without a time dimension in sequential or
  parallel mode), the call raises unless `block=` says which block to read:
  `output("generation", block=1)` or `value(..., block=1)`.

### Objective values

~~~ python
results.objective_values()  # columns: block, scenario_index, value
~~~

### Parts of the table

~~~ python
results.scenarios                                   # [0, 1, ...]
results.to_pandas(components=["gen_de"], scenarios=[0, 1])
results.to_pandas(outputs=["generation"])
for scenario, df in results.iter_scenario_frames():  # one scenario at a time
    ...
dataset = results.scan()                            # pyarrow.dataset.Dataset
~~~

- A scenario filter keeps the rows shared by all scenarios.
  `iter_scenario_frames()` yields them once, first, as `(None, rows)`.
- `scan()` hands the table to polars, duckdb or pyarrow, e.g.
  `polars.scan_pyarrow_dataset(results.scan())`.

### The whole table

~~~ python
df = results.data                        # in memory
df = results.to_pandas(load_all=True)    # in memory or on disk
~~~

On a stored table, `data` raises and `to_pandas()` without a filter needs
`load_all=True`, so that a large run is not loaded by accident.

---

## Saving a table

~~~ python
results.save(Path("copy"))                    # a table folder, in Parquet
results.save(Path("copy_csv"), OutputFormat.CSV)
~~~

`save()` writes the folder described above, from a table in memory or on disk.
The folder is created if missing and must be empty otherwise. Parquet files are
written with zstd compression (level 3) and row groups of 64,000 rows. If the
write fails or is interrupted, the files written so far are removed and the
folder is left as it was, so `save()` can be retried with the same path; a run
given an `output_dir`, by contrast, keeps its partial files (see above).

`to_csv()`, `to_parquet()`, `to_dataset()` and `to_netcdf()` are deprecated and
will be removed in a following release: use `save()`, or build an xarray object
from `to_pandas()`.
