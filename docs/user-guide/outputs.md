# Outputs: retrieving outputs with GemsPy

## Via `SimulationSession` (recommended)

`SimulationSession.run()` returns a `SimulationTable` directly after solving:

~~~ python
from pathlib import Path
from gems_craft.study.folder import load_study
from gems_runner.session import SimulationSession
from gems_craft.optim_config import load_optim_config

study = load_study(Path("my_study"))
optim_config = load_optim_config(Path("my_study/input/optim-config.yml"))

session = SimulationSession(study=study, optim_config=optim_config)
results = session.run()  # SimulationTable
~~~

## Via `SimulationTableBuilder` (low-level)

When using `build_problem()` directly, build the table from the solved problem:

~~~ python
from gems_runner.simulation.simulation_table import SimulationTableBuilder

results = SimulationTableBuilder().build(problem)
~~~

---

## Accessing results

`SimulationTable` exposes a fluent accessor API and a raw DataFrame.

### Fluent API

~~~ python
# All values for a component
component_view = results.component("gen_de")

# Pivot for a specific output variable
output_view = component_view.output("generation")

# Single value (single timestep, single scenario)
val = output_view.value(time_index=0, scenario_index=0)

# All timesteps for scenario 0
series = output_view.value(scenario_index=0)  # returns a pandas Series
~~~

### Raw DataFrame

~~~ python
df = results.data
~~~

The DataFrame has columns: `block`, `component`, `output`,
`absolute_time_index`, `block_time_index`, `scenario_index`, `value`, `basis_status`.

Reading the value of the optimisation variable `var_id` of component `component_id`
for a single time step and scenario:

~~~ python
value = df[(df["component"] == component_id) & (df["output"] == var_id)]["value"].iloc[0]
~~~

For multi-time or multi-scenario results, filter additionally by `block_time_index`
and `scenario_index`:

~~~ python
sub = df[(df["component"] == component_id) & (df["output"] == var_id)]
value_s0_t1 = sub[(sub["scenario_index"] == 0) & (sub["block_time_index"] == 1)]["value"].iloc[0]
~~~

---

## Exporting results

~~~ python
from gems_craft.study.parsing import OutputFormat
from gems_runner.simulation import SimulationTableWriter

SimulationTableWriter(OutputFormat.PARQUET).write(results, Path("output/"))  # one file per scenario
results.to_netcdf(Path("output/"))    # writes a NetCDF file
ds = results.to_dataset()             # returns an xarray Dataset
~~~

Parquet files are written with zstd compression (level 3) and row groups of
64,000 rows.

To write the results without holding all scenarios in memory, iterate over
them one scenario at a time, as `gemspy` does. Each table holds the rows of one
scenario, or the rows shared by all scenarios:

~~~ python
writer = SimulationTableWriter(OutputFormat.PARQUET)
for table in session.iter_scenario_tables():
    writer.write(table, Path("output/"))
~~~

### Output files of `gemspy`

`gemspy` (and `run_study`) writes the simulation table as **one file per MC
scenario**, as CSV by default or as Parquet with `--output-format parquet`:

~~~ bash
gemspy --study path/to/study_dir --output-format parquet
~~~

~~~
output/<run_id>/
├── simulation_table_<run_id>_scenario-0.parquet
├── simulation_table_<run_id>_scenario-1.parquet
├── ...
└── simulation_table_<run_id>_scenario-common.parquet
~~~

Rows shared by all scenarios, i.e. with an empty `scenario_index`, go to the
`scenario-common` file. This only happens in `frontal` mode with several
scenarios, for scenario-independent variables (e.g. an investment) and the
objective value. In `sequential-subproblems` and `parallel-subproblems` modes
each scenario is solved separately, so every row belongs to a scenario and no
common file is written.

In `benders-decomposition` mode no simulation table is written: the solve is
done by Antares Xpansion.

`output/<run_id>/` only appears once the run has completed. While the run is
going, files are written to `output/incomplete/<run_id>/`, which is moved to
`output/<run_id>/` at the end:

- if the run fails, the incomplete folder is removed, and the error names the
  scenario and the block being solved (sequential and parallel subproblem
  modes);
- if the process is killed, its folder stays in `output/incomplete/` and can be
  deleted.

`<run_id>` is the start time to the minute (`YYYYMMDDTHHMM`). A run started in
the same minute as an earlier one gets a `-2` suffix (then `-3`, ...), in its
folder and file names.

The equivalent Python call is
`run_study(Path("path/to/study_dir"), output_format=OutputFormat.PARQUET)`,
with `OutputFormat` imported from `gems_craft.study.parsing`.
