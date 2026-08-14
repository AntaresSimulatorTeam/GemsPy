# Reading input files with GemsPy

GemsPy supports two loading strategies: a directory-based approach that reads
an entire study at once, and a file-by-file approach for programmatic control.

---

## Directory-based loading (recommended)

When your study follows the standard directory layout, use `load_study()`:

```
my_study/
├── input/
│   ├── system.yml
│   ├── optim-config.yml
│   ├── model-libraries/
│   │   └── *.yml
│   └── data-series/
│       ├── modeler-scenariobuilder.dat   ← optional
│       └── *.txt / *.csv
```

~~~ python
from pathlib import Path
from gems_craft.study.folder import load_study

study = load_study(Path("my_study"))
~~~

`load_study()` returns a `Study` object that bundles the resolved `System`,
the `DataBase` with all parameter values, and the `ScenarioBuilder` loaded
from `input/data-series/modeler-scenariobuilder.dat` (if present).

---

## File-by-file loading (programmatic)

Use the lower-level functions when you want to load files individually or build
parts of the study from in-memory data.

### Loading the library and the system

~~~ python
from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import resolve_system, build_data_base
from pathlib import Path

with open("simple_library.yml") as lib_file:
    input_libraries = [parse_yaml_library(lib_file)]

with open("system_example.yml") as compo_file:
    input_system = parse_yaml_system(compo_file)

result_lib = resolve_library(input_libraries)
system = resolve_system(input_system, result_lib)
~~~

### Loading timeseries data

~~~ python
database = build_data_base(input_system, Path(series_dir))
~~~

`build_data_base()` reads all timeseries files referenced by the system
(`.txt` or `.csv`) from `series_dir`.

---

## Parameter time and scenario dependency

Every parameter is declared twice: once in the model library, where
`time-dependent` and `scenario-dependent` state the **maximum** dependency the
model's expressions are written for, and once per component in the system file,
where the same two flags state how that particular component's data actually
varies.

### The system file may narrow, never extend

A component may declare a parameter *less* dependent than its model does, but
never *more*:

| Model declares | Component may declare |
|---|---|
| `true` | `true` or `false` |
| `false` | `false` only |

The rule applies to each axis independently. Narrowing is how you share one
value across scenarios, or one value across the whole horizon, without changing
the model — the value is simply broadcast over the axes the component declared
independent, and the optimization problem keeps exactly the shape the model
declares.

Declaring an axis the model does not is rejected when the system is resolved,
with an error naming the component, the parameter, and both declarations.

### Data shape must match the component's declaration

The two flags determine the shape of the data expected for the parameter:

| `time-dependent` | `scenario-dependent` | Expected data |
|---|---|---|
| `false` | `false` | a number written inline in the system file |
| `true` | `false` | a data-series file of one column and `T` rows |
| `false` | `true` | a data-series file of one row and `S` columns |
| `true` | `true` | a data-series file of `T` rows and `S` columns |

A file whose shape does not match is rejected. In particular, a parameter
declared time-dependent but scenario-independent takes exactly one timeseries:
to supply several, declare it `scenario-dependent: true` as well — which
requires the model to declare it scenario-dependent too. Which column each
scenario then reads is controlled by the
[scenario builder](scenario-builder.md).

### Assembling a Study

Once you have `system` and `database`, wrap them in a `Study`:

~~~ python
from gems_craft.study import Study

study = Study(system=system, database=database)
~~~

This `study` object can then be passed directly to
[`build_problem()`](optimisation.md) or `SimulationSession`.

---

## Hybrid studies (`gems_craft_hybrid`)

Hybrid GEMS studies extend the standard format with additional fields used to
interoperate with [Antares Simulator](https://antares-simulator.org/).

> **Limitation:** hybrid studies cannot be simulated with GemsPy. The
> `gems_craft_hybrid` package only provides reading and writing of hybrid
> files.

### Hybrid libraries and hybrid systems

For the YAML fields in library and system files associated with the hybrid mode, see [this page](https://gems-energy.readthedocs.io/en/latest/interoperability/hybrid/hybrid-connections/) of the GEMS documentation website.

Load and write hybrid libraries with `parse_yaml_hybrid_library` / `write_yaml_library`:

~~~ python
from gems_craft.model.parsing import write_yaml_library
from gems_craft_hybrid.model.parsing import parse_yaml_hybrid_library

with open("hybrid_lib.yml") as f:
    library = parse_yaml_hybrid_library(f)
write_yaml_library(library, Path("output/hybrid_lib.yml"))
~~~


Load and write hybrid systems with `parse_yaml_hybrid_system` / `write_yaml_system`:

~~~ python
from gems_craft.study.parsing import write_yaml_system
from gems_craft_hybrid.study.parsing import parse_yaml_hybrid_system

with open("hybrid_system.yml") as f:
    system = parse_yaml_hybrid_system(f)
write_yaml_system(system, Path("output/hybrid_system.yml"))
~~~




