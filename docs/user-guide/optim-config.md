# Optimisation configuration

The optimisation configuration controls the time scope, scenario scope, solver
options, and resolution strategy used when running a study.

## File location

By convention the file lives at:

```
my_study/
└── input/
    └── optim-config.yml   ← read automatically by run_study() / load_optim_config()
```

When the file is absent, `run_study()` and `SimulationSession` use the defaults
described below.

---

## Full annotated example

~~~ yaml
# Time range (0-based indices, inclusive on both ends)
time-scope:
  first-time-step: 0
  last-time-step: 8759   # 8760 hourly timesteps → one year

# Number of Monte-Carlo scenarios to run
scenario-scope:
  nb-scenarios: 10

# Solver settings
solver-options:
  name: highs            # only HiGHS is currently supported
  logs: false            # set to true to print solver output
  parameters: "threads=4 time_limit=300"  # space-separated key=value pairs

# Resolution strategy
resolution:
  mode: sequential-subproblems   # see section below
  block-length: 168               # one week (in timesteps)
  block-overlap: 0

# Per-model configuration (optional)
models:
  - id: storage
    out-of-bounds-processing:
      constraints:
        - id: soc_balance
          mode: cyclic   # wrap time index at horizon boundaries
~~~

---

## `time-scope`

| Key | Type | Default | Description |
|---|---|---|---|
| `first-time-step` | int | `0` | First timestep index (0-based, inclusive) |
| `last-time-step` | int | `0` | Last timestep index (0-based, inclusive) |

The total number of timesteps solved is `last-time-step − first-time-step + 1`.

---

## `scenario-scope`

| Key | Type | Default | Description |
|---|---|---|---|
| `nb-scenarios` | int | `1` | Number of Monte-Carlo scenarios |

Scenario indices run from `0` to `nb-scenarios − 1`.  If a
[scenario builder](scenario-builder.md) file is present, these indices are
mapped to data-series columns by that file.

---

## `solver-options`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"highs"` | Solver name (currently only `"highs"` is supported) |
| `logs` | bool | `false` | Print solver output to stdout |
| `parameters` | str | `""` | Space-separated `key=value` HiGHS parameters |

---

## `resolution`

The `resolution` block selects how the time horizon is decomposed into
optimisation subproblems.

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | str | `"frontal"` | Resolution strategy (see below) |
| `block-length` | int | — | Timesteps per window; required for windowed modes |
| `block-overlap` | int | `0` | Extra overlap timesteps between consecutive blocks |

### `frontal` (default)

The entire time horizon is solved as a single LP.

~~~ yaml
resolution:
  mode: frontal
~~~

**When to use**: Small to medium horizons where the full problem fits in memory.
Produces globally optimal results.

### `sequential-subproblems`

The horizon is split into non-overlapping (or slightly overlapping) windows of
`block-length` timesteps.  Blocks are solved **one after the other**; the state
of inter-block dynamics (e.g. storage level) is carried over from one block to
the next.

~~~ yaml
resolution:
  mode: sequential-subproblems
  block-length: 168   # one week
  block-overlap: 0
~~~


### `parallel-subproblems`

The horizon is split into independent windows of `block-length` timesteps.
Blocks are solved **independently** (no carry-over state between them).

~~~ yaml
resolution:
  mode: parallel-subproblems
  block-length: 168
~~~



### `benders-decomposition`

A Benders decomposition is applied via AntaresXpansion.  Investment decisions
are placed in a master problem; operational subproblems are solved per scenario.

~~~ yaml
resolution:
  mode: benders-decomposition
~~~

---

## `models` — per-model configuration

The optional `models` list lets you override behaviour for specific models.

### `out-of-bounds-processing`

When a constraint references a time-shifted variable (e.g. `x[t-1]`), timestep
`t = 0` refers to a time index *before* the start of the horizon.  Two
strategies are available:

| Mode | Behaviour |
|---|---|
| `cyclic` (default) | Wrap around: time shift are defined modulo `block-length` |
| `drop` | Skip the constraint entirely for out-of-bounds timesteps |

~~~ yaml
models:
  - id: storage
    out-of-bounds-processing:
      constraints:
        - id: soc_balance
          mode: drop   # do not enforce at t=0 where previous state is unknown
~~~

### `model-decomposition`

The `model-decomposition` block assigns individual model elements to the master
problem or subproblems when using Benders decomposition
(`resolution.mode: benders-decomposition`).  It is ignored for other resolution
modes.

Each element (variable, constraint, or objective contribution) can be placed in
one of three locations:

| Location | Description |
|---|---|
| `subproblems` (default) | Element lives in each operational subproblem |
| `master` | Element lives only in the investment master problem |
| `master-and-subproblems` | Variable is decided in the master problem and used in the subproblems (coupling variable) |

Elements not listed keep their default location (`subproblems`).

~~~ yaml
models:
  - id: my_lib.generator_with_invest
    model-decomposition:
      variables:
        - id: nb_units
          location: master
        - id: p_max
          location: master-and-subproblems
        # unlisted variables default to subproblems
      constraints:
        - id: p_max_nb_units_relation
          location: master
      objective-contributions:
        - id: invest_objective
          location: master
        - id: operational_objective
          location: subproblems
~~~

!!! note
    Master variables must be time-independent.  Master constraints and objective
    contributions may only reference variables whose location is `master` or
    `master-and-subproblems`.  GemsPy validates these rules at config-load time
    and raises an error for any violation.

---

## Python API

You can load, inspect, and build the config programmatically:

~~~ python
from pathlib import Path
from gems.optim_config import (
    load_optim_config,
    OptimConfig,
    ResolutionConfig,
    ResolutionMode,
    TimeScopeConfig,
    ScenarioScopeConfig,
    SolverOptionsConfig,
)

# Load from file (returns None if the file does not exist)
config = load_optim_config(Path("my_study/input/optim-config.yml"))

# Build programmatically
config = OptimConfig(
    time_scope=TimeScopeConfig(first_time_step=0, last_time_step=8759),
    scenario_scope=ScenarioScopeConfig(nb_scenarios=10),
    solver_options=SolverOptionsConfig(name="highs", logs=False),
    resolution=ResolutionConfig(
        mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
        block_length=168,
    ),
)

# Pass to SimulationSession
from gems.session import SimulationSession
from gems.study import load_study

study = load_study(Path("my_study"))
session = SimulationSession(study=study, optim_config=config)
results = session.run()
~~~
