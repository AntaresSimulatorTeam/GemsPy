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

# Monte-Carlo scenarios to simulate (0-based, inline form)
scenario-scope:
  include:
    - "0-9"   # scenarios 0 through 9 (10 scenarios)

# Solver settings
solver-options:
  name: highs            # highs (default), xpress, or gurobi
  logs: false            # set to true to print solver output
  parameters: "threads=4 time_limit=300"  # space-separated key=value pairs passed to the solver

# Resolution strategy
resolution:
  mode: sequential-subproblems   # see section below
  block-length: 168               # one week (in timesteps)
  block-overlap: 24               # consecutive blocks share one day
  carry-over-length: 24           # optional; omitted → defaults to block-overlap

# Per-model configuration (optional)
models:
  - id: storage
    out-of-bounds-processing:
      constraints:
        - id: soc_balance
          mode: cyclic   # wrap time index at horizon boundaries

  - id: thermal
    heuristics:                          # see section below
      - id: accurate
        inputs:
          - heuristic-element: num_units_on_opt
            id: num_units_on
            type: variable-solution
          - heuristic-element: num_units_max
            id: num_units_on
            type: variable-upper-bound
          - heuristic-element: min_up_duration
            id: min_up_duration
          - heuristic-element: min_down_duration
            id: min_down_duration
        outputs:
          - heuristic-element: minimum_num_units_on
            id: num_units_on
            type: variable-lower-bound
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

Selects which Monte-Carlo scenarios to simulate.  Indices are **0-based**,
consistent with the `modeler-scenariobuilder.dat` file convention.

The base scenario set is defined by exactly one of two mutually exclusive
keys: `include` (inline) or `playlist-file` (from a JSON file).  `exclude`
is optional and applies to **either** form.

### Inline form (`include` / `exclude`)

Specify scenarios directly in the YAML using individual integers, string
integers, and inclusive `"a-b"` range strings.

| Key | Type | Default | Description |
|---|---|---|---|
| `include` | list | — | Scenarios to run (required in inline form) |
| `exclude` | list | — | Scenarios to remove from the base set (optional) |

Each entry in `include` or `exclude` may be:

- An integer: `5` → scenario 5
- A string integer: `"5"` → scenario 5 (identical to `5`)
- A range: `"0-9"` → scenarios 0 through 9 inclusive (10 scenarios)

**Examples:**

~~~ yaml
# Run a single scenario
scenario-scope:
  include:
    - 0

# Run scenarios 0 to 99
scenario-scope:
  include:
    - "0-99"

# Run scenarios 0–19 and 49–59, but skip 9 and 14
scenario-scope:
  include:
    - "0-19"
    - "49-59"
  exclude:
    - 9
    - 14
~~~

**Rules:**

- All indices must be ≥ 0.
- Overlapping entries in `include` are deduplicated automatically.
- Excludes that do not appear in the base set produce a warning and have no effect.
- Output is always sorted in ascending order.
- `exclude` cannot be used without `include` or `playlist-file`.

**Default behaviour** (no `scenario-scope` key at all, or an empty block):
runs scenario 0 only.

---

### Playlist-file form (`playlist-file`)

Point to a JSON file containing a flat array of 0-based integer scenario
indices.  Useful when the list of scenarios is generated programmatically or
is too large to embed in YAML.

| Key | Type | Description |
|---|---|---|
| `playlist-file` | path | Path to a JSON playlist (relative to `optim-config.yml`) |

~~~ yaml
scenario-scope:
  playlist-file: mc_playlist.json   # resolved relative to optim-config.yml
~~~

The referenced file must contain a flat JSON array of non-negative integers:

~~~ json
[0, 2, 4, 6, 8, 10, 12]
~~~

`exclude` can be combined with `playlist-file` to subtract specific scenarios
at run time without modifying the file:

~~~ yaml
scenario-scope:
  playlist-file: mc_playlist.json
  exclude:
    - 4
    - "8-10"
~~~

GemsPy reads and validates the playlist eagerly when `load_optim_config()` is
called, so any I/O or format errors surface immediately at load time.

**Rules:**

- The file must be a flat JSON array of integers (no booleans, strings, or objects).
- All indices must be ≥ 0.
- Duplicates are silently removed; the result is sorted ascending.
- `include` and `playlist-file` are mutually exclusive.

---

### ScenarioBuilder cross-validation

If a [scenario builder](scenario-builder.md) file is present,
`validate_optim_config()` checks that every scenario index in the playlist is
defined for every scenario group.  Out-of-bounds indices raise a `ValueError`
listing the affected groups.

---

## `solver-options`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"highs"` | Solver name: `"highs"` (default), `"xpress"`, or `"gurobi"` |
| `logs` | bool | `false` | Print solver output to stdout |
| `parameters` | str | `""` | Space-separated `key=value` pairs forwarded as solver options |

> **Note** — Xpress (≥ 9.8) and Gurobi (≥ 10.0) require their respective Python
> packages and a valid licence.

---

## `resolution`

The `resolution` block selects how the time horizon is decomposed into
optimisation subproblems.

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | str | `"frontal"` | Resolution strategy (see below) |
| `block-length` | int | — | Timesteps per window; required for windowed modes |
| `block-overlap` | int | `0` | Shared timesteps between consecutive blocks; must satisfy `0 <= block-overlap < block-length` |
| `carry-over-length` | int | `block-overlap` | Sequential mode only: how many of the shared timesteps are pinned to the previous block's values; must satisfy `0 <= carry-over-length <= block-overlap` |

### `frontal` (default)

The entire time horizon is solved as a single LP.

~~~ yaml
resolution:
  mode: frontal
~~~

**When to use**: Small to medium horizons where the full problem fits in memory.
Produces globally optimal results.

### `sequential-subproblems`

The horizon is split into windows of `block-length` timesteps, each starting
`block-length - block-overlap` timesteps after the previous one.  Blocks are
solved **one after the other**; the state of inter-block dynamics (e.g. storage
level) is carried over from one block to the next by pinning the leading
timesteps of each block to the values the previous block already computed.

~~~ yaml
resolution:
  mode: sequential-subproblems
  block-length: 168       # one week
  block-overlap: 24       # one day shared between consecutive blocks
  carry-over-length: 24   # optional; omitted → defaults to block-overlap (full pin)
                          # 0 is legal and explicit: overlap solved twice, no stitching
~~~

Three parameters shape the stitching between consecutive blocks.  Illustrative
example with `block-length: 10`, `block-overlap: 4`, `carry-over-length: 3`
(a partial pin, so all three parameters are visible at once):

~~~ text
abs t     0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
Block N   0   1   2   3   4   5   6   7   8   9
          └──────────────────────────────────────┘
                     block-length = 10

Block N+1                         0   1   2   3   4   5   6   7   8   9
                                  └──────────────────────────────────────┘
                                             block-length = 10

                                  |------------|  overlap = 4  (t=6..9: solved by BOTH blocks)
                                  |========|  carry-over = 3  (t=6..8: PINNED to block N's value)
                                              ^  t=9: still shared, but free in N+1 (re-optimized)
~~~

Reading it:

- **`block-length`** — width of each block's own window (10 for both here).
- **`block-overlap`** — how far block *N+1*'s start reaches back into block
  *N*'s window (4 → t=6..9 exist in both solves).  The overlap gives block
  *N+1* real historical values for lag-dependent constraints (e.g. a storage
  balance using `soc[t-1]`, or min up/down durations spanning several hours).
- **`carry-over-length`** — how many of those *shared* leading timesteps of
  block *N+1* get hard-pinned (`var[t] == value from block N`) to block *N*'s
  already-solved values, counted from the earliest shared timestep (t=6), not
  from t=9.  Here `carry-over-length: 3 < overlap: 4`, so t=6,7,8 are frozen
  but t=9 is left free — an MPC-style partial pin where the optimizer may
  revise the tail of the overlap with more lookback context.

Defaults and special values:

- **Omitted** `carry-over-length` resolves to `block-overlap`: the whole
  overlap zone is pinned.  This is the right default when the overlap exists
  to provide history for lag-dependent constraints without re-litigating
  decisions the previous block already made.
- **Explicit `carry-over-length: 0`** is legal and distinct from omitting the
  field: blocks overlap for lag-constraint history, but no timestep is pinned
  — block *N+1* re-solves the whole overlap window independently.
- Validation requires `0 <= carry-over-length <= block-overlap` (and
  `0 <= block-overlap < block-length`), with no special case at
  `block-overlap: 0`.

Overlapping timesteps appear once per block in the simulation table, tagged
with the `block` column — nothing is lost or silently merged.  Downstream
tooling decides which block's version of a shared timestep is authoritative;
`carry-over-length` only controls how much two consecutive blocks may
*disagree* on that shared window.



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

### `heuristics` — integer strategy and thermal heuristics

A component's `integer-strategy` controls how its model's integer/binary
variables are built:

| `id` | Effect |
|---|---|
| `exact` (default) | Keep integer/binary types (MILP) |
| `relaxed` | Relax to continuous |
| `heuristic` | Relax to continuous, then refine with `heuristic-id` after a first solve |

~~~ yaml
components:
  - id: G1
    model: my_lib.thermal
    integer-strategy:
      id: heuristic
      heuristic-id: fast   # must match a heuristics entry declared below
                            # for model "my_lib.thermal"
~~~

Each heuristic's inputs/outputs are bound to the model's own parameters and
variables via `models[].heuristics`:

| Key | Type | Default | Description |
|---|---|---|---|
| `id` | str | — | Heuristic to run: `fast` or `accurate` |
| `inputs` | list | `[]` | Model elements fed into the heuristic |
| `outputs` | list | `[]` | Model elements the heuristic result is written to |

Each `inputs`/`outputs` entry binds a fixed `heuristic-element` name (see
table below) to one of the model's own parameter/variable ids:

| Key | Type | Default | Description |
|---|---|---|---|
| `heuristic-element` | str | — | Fixed name the heuristic function expects |
| `id` | str | — | The model's own parameter or variable id |
| `type` | str | `parameter` | How the element is read/written |

| `type` | Meaning |
|---|---|
| `parameter` (default) | Read the model's parameter value (inputs only) |
| `variable-solution` | Read the variable's solved value from the first solve (inputs only) |
| `variable-lower-bound` | Read/write the variable's lower bound |
| `variable-upper-bound` | Read/write the variable's upper bound |

`outputs` entries must use `variable-lower-bound` or `variable-upper-bound`.

~~~ yaml
models:
  - id: my_lib.thermal
    heuristics:
      - id: accurate
        inputs:
          - heuristic-element: num_units_on_opt
            id: num_units_on
            type: variable-solution
          - heuristic-element: num_units_max
            id: num_units_on
            type: variable-upper-bound
          - heuristic-element: min_up_duration
            id: min_up_duration
          - heuristic-element: min_down_duration
            id: min_down_duration
        outputs:
          - heuristic-element: minimum_num_units_on
            id: num_units_on
            type: variable-lower-bound
~~~

Two heuristics are built in, each expecting a fixed set of
`heuristic-element` names:

| Heuristic | `inputs` elements | `outputs` elements |
|---|---|---|
| `fast` | `generation_power`, `cluster_max_generation`, `min_power_per_unit`, `max_power_per_unit`, `min_up_duration`, `min_down_duration` | `minimum_generation_power` |
| `accurate` | `num_units_on_opt`, `num_units_max`, `min_up_duration`, `min_down_duration` | `minimum_num_units_on` |

!!! note
    The `heuristic` strategy triggers an automatic second solve in
    `SimulationSession` and is incompatible with `resolution.mode:
    benders-decomposition`. `validate_optim_config()` checks both the
    `heuristic-id` ↔ `heuristics` consistency and that every bound `id` exists
    on the model with the expected time-dependence.

---

## Python API

You can load, inspect, and build the config programmatically:

~~~ python
from pathlib import Path
from gems_craft.optim_config import (
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

# Build programmatically — inline form (scenarios 0–9)
config = OptimConfig(
    time_scope=TimeScopeConfig(first_time_step=0, last_time_step=8759),
    scenario_scope=ScenarioScopeConfig(include=["0-9"]),
    solver_options=SolverOptionsConfig(name="highs", logs=False),
    resolution=ResolutionConfig(
        mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
        block_length=168,
    ),
)

# Build programmatically — playlist-file form
config_pf = OptimConfig(
    time_scope=TimeScopeConfig(first_time_step=0, last_time_step=8759),
    scenario_scope=ScenarioScopeConfig(playlist_file=Path("mc_playlist.json")),
)

# Pass to SimulationSession
from gems_runner.session import SimulationSession
from gems_craft.study.folder import load_study

study = load_study(Path("my_study"))
session = SimulationSession(study=study, optim_config=config)
results = session.run()
~~~
