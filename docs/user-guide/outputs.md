# Outputs: retrieving outputs with GemsPy

## Via `SimulationSession` (recommended)

`SimulationSession.run()` returns a `SimulationTable` directly after solving:

~~~ python
from pathlib import Path
from gems.study.folder import load_study
from gems.session import SimulationSession
from gems.optim_config import load_optim_config

study = load_study(Path("my_study"))
optim_config = load_optim_config(Path("my_study/input/optim-config.yml"))

session = SimulationSession(study=study, optim_config=optim_config)
results = session.run()  # SimulationTable
~~~

## Via `SimulationTableBuilder` (low-level)

When using `build_problem()` directly, build the table from the solved problem:

~~~ python
from gems.simulation.simulation_table import SimulationTableBuilder

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
`absolute-time-index`, `block-time-index`, `scenario-index`, `value`, `basis-status`.

Reading the value of the optimisation variable `var_id` of component `component_id`
for a single time step and scenario:

~~~ python
value = df[(df["component"] == component_id) & (df["output"] == var_id)]["value"].iloc[0]
~~~

For multi-time or multi-scenario results, filter additionally by `block-time-index`
and `scenario-index`:

~~~ python
sub = df[(df["component"] == component_id) & (df["output"] == var_id)]
value_s0_t1 = sub[(sub["scenario-index"] == 0) & (sub["block-time-index"] == 1)]["value"].iloc[0]
~~~

---

## Exporting results

~~~ python
results.to_csv(Path("output/"))       # writes one CSV per component
results.to_parquet(Path("output/"))   # writes Parquet files
results.to_netcdf(Path("output/"))    # writes a NetCDF file
ds = results.to_dataset()             # returns an xarray Dataset
~~~
