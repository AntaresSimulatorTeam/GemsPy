# Scenario builder

The scenario builder lets you control which data-series column is used for each
Monte-Carlo (MC) scenario, independently per component group.  Without it,
scenario index `i` always maps to column `i` of every timeseries file.

---

## Motivation

A study that simulates 100 MC scenarios may have some components (e.g. wind
farms) whose data only has 10 distinct columns, while others (e.g. demand)
have 100.  The scenario builder lets you express this mapping explicitly, so
GemsPy reads the right column for each component and scenario.

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
- `<time_serie_number>` — the column of the timeseries file to use (1-based:
  a number below 1 raises a `ValueError`)

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

### Scenarios without a line

A scenario that has no line in its group uses the default: MC scenario `n`
uses time series `n + 1`, as for components without a group. So only the
lines that differ from the default are needed:

```
# Scenario 2 uses time series 1; scenarios 0, 1, 3, ... use 1, 2, 4, ...
wind, 2 = 1
```

This applies to a gap (no line for scenario 1 between lines for 0 and 2) and
to the scenarios after the group's last line. Before the run, a `UserWarning`
per group lists the selected scenarios that use the default, so that a
forgotten line does not go unnoticed:

```
Scenario group 'wind': scenarios 0, 1, 3 have no line in modeler-scenariobuilder.dat and use the default (scenario n uses time series n+1).
```

!!! note
    Before the run starts, `validate_optim_config()` also raises a
    `ValueError`:

    - for a scenario whose time series is not in the data-series file, naming
      the component, the parameter and the file, and whether the time series
      number comes from a line or from the default;
    - for a `scenario-group` of a component that is not in the file (most
      likely a typo).

    MC scenarios start at 0 and time series numbers at 1: a negative scenario
    or a time series number below 1 raises a `ValueError` when the file is
    loaded.

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
from gems_craft.study.folder import load_study

study = load_study(Path("my_study"))
# study.scenario_builder is populated from modeler-scenariobuilder.dat
~~~

When the file is absent, `study.scenario_builder` is an empty `ScenarioBuilder`:
components without a `scenario-group` use the default. A component with a
`scenario-group` and scenario-dependent parameters raises a `ValueError` before
the run, since its group is not in the file.

---

## Python API

~~~ python
import numpy as np
from pathlib import Path
from gems_craft.study.scenario_builder import ScenarioBuilder

# Load from file
sb = ScenarioBuilder.load(Path("my_study/input/data-series/modeler-scenariobuilder.dat"))

# Resolve a batch of MC scenarios for the "wind" group
mc_scenarios = np.array([0, 1, 2, 3])          # 0-based internally
col_indices = sb.resolve_vectorized("wind", mc_scenarios)
# → array([0, 1, 0, 1])  (0-based column indices)

# Components without a group use None → identity mapping
col_indices = sb.resolve_vectorized(None, mc_scenarios)
# → array([0, 1, 2, 3])

# Which scenarios have no line in each group, and use the default
sb.scenarios_without_line([0, 1, 2, 3, 4, 5])
# → {"wind": [4, 5], "demand": [4, 5]}
~~~

!!! note
    `resolve_vectorized` works with **0-based** indices internally, even though
    the `.dat` file uses 1-based numbering.  The conversion is handled
    automatically by `ScenarioBuilder.load()`.
