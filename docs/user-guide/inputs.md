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




