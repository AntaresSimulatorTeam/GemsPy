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
│   ├── taxonomy.yml                      ← optional
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

Libraries are "cold" files: the same library YAML is often reused unchanged between a full GEMS study and a hybrid study of the same system. Hybrid-format libraries therefore load with the standard `parse_yaml_library` / `write_yaml_library` from `gems_craft` — the hybrid-only `area-connection` and `thermal-capacity-connection` port-type fields are recognized directly by `gems_craft`'s own schema (and simply ignored outside the hybrid workflow), so no separate hybrid library function is needed, and the same file works for either kind of study without a format change:

~~~ python
from gems_craft.model.parsing import parse_yaml_library, write_yaml_library

with open("hybrid_lib.yml") as f:
    library = parse_yaml_library(f)
write_yaml_library(library, Path("output/hybrid_lib.yml"))
~~~


System files, by contrast, are necessarily different between a full GEMS study and a hybrid study of the same setup — a hybrid system file describes area/thermal-capacity connections a full GEMS system wouldn't have. Since there's no cross-format file to keep compatible, parsing can afford to stay strict: hybrid systems still require the dedicated `gems_craft_hybrid` functions — load and write them with `parse_yaml_hybrid_system` / `write_yaml_system` — and `parse_yaml_system` (standard) rejects hybrid-only fields outright.

~~~ python
from gems_craft.study.parsing import write_yaml_system
from gems_craft_hybrid.study.parsing import parse_yaml_hybrid_system

with open("hybrid_system.yml") as f:
    system = parse_yaml_hybrid_system(f)
write_yaml_system(system, Path("output/hybrid_system.yml"))
~~~




