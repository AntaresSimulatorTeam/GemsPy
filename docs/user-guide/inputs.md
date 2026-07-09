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
from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study.parsing import parse_yaml_components
from gems_craft.study.resolve_components import resolve_system, build_data_base
from pathlib import Path

with open("simple_library.yml") as lib_file:
    input_libraries = [parse_yaml_library(lib_file)]

with open("system_example.yml") as compo_file:
    input_system = parse_yaml_components(compo_file)

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

### Hybrid system

A hybrid `system.yml` builds on two optional `system.yml` sections:

- `area-connections`: maps component ports to areas (`component`, `port`,
  `area`) — already available on the standard `SystemSchema`.
- `thermal-capacity-connections`: maps component ports to Antares thermal
  clusters, each identified by `area` and `cluster-id` — added by
  `HybridSystemSchema`.

~~~ yaml
system:
  components:
    - id: G
      model: basic.generator
      parameters:
        - id: cost
          value: 30
        - id: p_max
          value: 100
  connections:
    - component1: N
      port1: injection_port
      component2: G
      port2: injection_port
  area-connections:
    - component: G
      port: injection_port
      area: fr
  thermal-capacity-connections:
    - component: G
      port: injection_port
      thermal-component:
        area: fr
        cluster-id: nuclear1
~~~

Load and write hybrid systems with `parse_yaml_components` / `write_yaml_system`
passing `HybridSystemSchema` as the schema:

~~~ python
from gems_craft.study.parsing import parse_yaml_components, write_yaml_system
from gems_craft_hybrid.study.parsing import HybridSystemSchema

with open("system.yml") as f:
    system = parse_yaml_components(f, HybridSystemSchema)
write_yaml_system(system, Path("output/system.yml"))
~~~

### Hybrid library

A hybrid library YAML may include a top-level `version` field, and each
port-type may carry two additional sub-fields:

- `area-connection`: maps port roles to port fields for area coupling
  (`injection-to-balance`, `spillage-bound`, `unsupplied-energy-bound`).
- `thermal-capacity-connection`: identifies the port field that carries
  thermal capacity (`capacity-field`).

~~~ yaml
library:
  id: my_lib
  version: "1.0"
  port-types:
    - id: flow
      fields:
        - id: flow
      area-connection:
        injection-to-balance: flow
        spillage-bound: flow
        unsupplied-energy-bound:
      thermal-capacity-connection:
        capacity-field: flow
~~~

Load and write hybrid libraries with `parse_yaml_library` / `write_yaml_library`
passing `HybridLibrarySchema` as the schema:

~~~ python
from gems_craft.model.parsing import parse_yaml_library, write_yaml_library
from gems_craft_hybrid.model.parsing import HybridLibrarySchema

with open("lib.yml") as f:
    library = parse_yaml_library(f, HybridLibrarySchema)
write_yaml_library(library, Path("output/lib.yml"))
~~~
