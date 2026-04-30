# Scenario builder

The scenario builder lets you control which data-series column is used for each
Monte-Carlo (MC) scenario, independently per component group.  Without it,
scenario index `i` always maps to column `i` of every timeseries file.

---

## Motivation

A study with `nb-scenarios = 100` may have some components (e.g. wind farms)
whose data only has 10 distinct columns, while others (e.g. demand) have 100.
The scenario builder lets you express this mapping explicitly, so GemsPy reads
the right column for each component and scenario.

---

## The `modeler-scenariobuilder.dat` file

Place the file at:

```
my_study/
└── input/
    └── data-series/
        └── modeler-scenariobuilder.dat
```

Each non-empty, non-comment line has the form:

```
<scenario_group>, <mc_scenario> = <time_serie_number>
```

- `<scenario_group>` — a group name you assign to one or more components
- `<mc_scenario>` — the MC scenario index (**0-based** in the file)
- `<time_serie_number>` — the column of the timeseries file to use (1-based)

### Example

Suppose you have 4 MC scenarios, 2 wind-power columns, and 4 load columns:

```
# Wind turbines: map 4 MC scenarios onto 2 data columns (round-robin)
wind, 0 = 1
wind, 1 = 2
wind, 2 = 1
wind, 3 = 2

# Demand: one distinct column per scenario
demand, 0 = 1
demand, 1 = 2
demand, 2 = 3
demand, 3 = 4
```

!!! note
    All MC scenario indices (`0` to `nb-scenarios - 1`) must be listed for every
    group that appears in the file.  Missing entries raise a `ValueError` at
    load time.

---

## Linking components to a scenario group

In the system YAML, set `scenario-group` on any component whose parameters
should use this mapping:

~~~ yaml
system:
  model-libraries: my_library
  components:
    - id: wind_north
      model: my_library.wind
      scenario-group: wind   # ← links this component to the "wind" group
      parameters:
        - id: p_max
          time-dependent: true
          scenario-dependent: true
          value: wind_north_data

    - id: load_city
      model: my_library.load
      scenario-group: demand
      parameters:
        - id: load
          time-dependent: true
          scenario-dependent: true
          value: city_load_data
~~~

Components without a `scenario-group` use the identity mapping (MC scenario `i`
→ column `i`).

---

## How `load_study()` handles the file

`load_study()` automatically reads `input/data-series/modeler-scenariobuilder.dat` if present and
attaches the resulting `ScenarioBuilder` to the returned `Study` object:

~~~ python
from pathlib import Path
from gems.study import load_study

study = load_study(Path("my_study"))
# study.scenario_builder is populated from modeler-scenariobuilder.dat
~~~

When the file is absent, `study.scenario_builder` is an empty `ScenarioBuilder`
that passes MC scenario indices through unchanged.

---

## Python API

~~~ python
import numpy as np
from pathlib import Path
from gems.study.scenario_builder import ScenarioBuilder

# Load from file
sb = ScenarioBuilder.load(Path("my_study/input/data-series/modeler-scenariobuilder.dat"))

# Resolve a batch of MC scenarios for the "wind" group
mc_scenarios = np.array([0, 1, 2, 3])          # 0-based internally
col_indices = sb.resolve_vectorized("wind", mc_scenarios)
# → array([0, 1, 0, 1])  (0-based column indices)

# Components without a group use None → identity mapping
col_indices = sb.resolve_vectorized(None, mc_scenarios)
# → array([0, 1, 2, 3])
~~~

!!! note
    `resolve_vectorized` works with **0-based** indices internally, even though
    the `.dat` file uses 1-based numbering.  The conversion is handled
    automatically by `ScenarioBuilder.load()`.
