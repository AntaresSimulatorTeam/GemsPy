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
from gems_runner.study.folder import load_study

study = load_study(Path("my_study"))
~~~

`load_study()` returns a `Study` object that bundles the resolved `System`,
the `DataBase` with all parameter values, and the `ScenarioBuilder` loaded
from `input/data-series/modeler-scenariobuilder.dat` (if present).

---

## File-by-file loading (programmatic)

Use the lower-level functions when you want to load files individually or build
parts of the study from in-memory data.

---

## System YAML: optional field `properties` for components

In `system.yml`, each component may define an optional `properties` section as a
list of id/value pairs:

~~~ yaml
system:
  components:
    - id: nuclear_1
      model: basic.generator
      properties:
        - id: technology
          value: nuclear
        - id: company
          value: rhonepower
~~~

Duplicate ids for properties are rejected.

### Loading the library and the system

~~~ python
from gems_craft.model.parsing import load_yaml_library
from gems_runner.model.resolve_library import resolve_library
from gems_craft.study.parsing import load_yaml_system
from gems_runner.study.resolve_components import resolve_system, build_data_base
from pathlib import Path

input_libraries = [load_yaml_library(Path("simple_library.yml"))]
input_system = load_yaml_system(Path("system_example.yml"))

result_lib = resolve_library(input_libraries)
system = resolve_system(input_system, result_lib)
~~~

### Loading timeseries data

~~~ python
database = build_data_base(input_system, Path(series_dir))
~~~

`build_data_base()` reads all timeseries files referenced by the system
(`.txt` or `.csv`) from `series_dir`.

### Assembling a Study

Once you have `system` and `database`, wrap them in a `Study`:

~~~ python
from gems_runner.study import Study

study = Study(system=system, database=database)
~~~

This `study` object can then be passed directly to
[`build_problem()`](optimisation.md) or `SimulationSession`.
