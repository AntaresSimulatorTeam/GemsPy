# Getting started with GemsPy

The [GEMS](https://gems-energy.readthedocs.io/en/latest/) framework consists of an **algebraic modelling language**, close to mathematical syntax, and a **data structure** for describing energy systems.

More specifically, three main types of input files can be defined with the [GEMS](https://gems-energy.readthedocs.io/en/latest/) framework:

1. **Library files**: describe abstract component models.  
2. **System files**: describe the graph of components that make up a system of interest; refer to model libraries (instantiation of abstract models) and to timeseries files.  
3. **Timeseries files**: the data of timeseries.

To get started with the syntax of these files, the reader can find basic examples below. A detailed introduction to the language is available on the [GEMS documentation website](https://gems-energy.readthedocs.io/en/latest/).

## Simple example of a library file

The first category of input files mentioned above comprises libraries of models. A simple `library.yml` file might look like this:

~~~ yaml
library:
  id: basic
  description: Basic library

  port-types:
    - id: flow
      description: A port which transfers power flow
      fields:
        - id: flow

  models:

    - id: generator
      description: A basic generator model
      parameters:
        - id: marginal_cost
          time-dependent: false
          scenario-dependent: false
        - id: p_max
          time-dependent: false
          scenario-dependent: false
      variables:
        - id: generation
          lower-bound: 0
          upper-bound: p_max
      ports:
        - id: injection_port
          type: flow
      port-field-definitions:
        - port: injection_port
          field: flow
          definition: generation
      objective-contributions:
        - id: obj
          expression: expec(sum(marginal_cost * generation))

    - id: node
      description: A basic balancing node model
      ports:
        - id: injection_port
          type: flow
      binding-constraints:
        - id: balance
          expression:  sum_connections(injection_port.flow) = 0


    - id: load
      description: A basic fixed demand model
      parameters:
        - id: load
          time-dependent: true
          scenario-dependent: true
      ports:
        - id: injection_port
          type: flow
      port-field-definitions:
        - port: injection_port
          field: flow
          definition: -load

~~~

## Simple example of system file

The second category of input files mentioned above corresponds to system files. A system file describes a practical instance that the user wants to simulate. Such a `system.yml` file might look like this:


~~~yaml
system:
  model-libraries: basic
  nodes:
    - id: N
      model: basic.node

  components:
    - id: G1
      model: basic.generator
      parameters:
        - id: marginal_cost
          time-dependent: false
          scenario-dependent: false
          value: 30
        - id: p_max
          time-dependent: false
          scenario-dependent: false
          value: 100
    - id: G2
      model: basic.generator
      parameters:
        - id: marginal_cost
          time-dependent: false
          scenario-dependent: false
          value: 10
        - id: p_max
          time-dependent: false
          scenario-dependent: false
          value: 50
    - id: D
      model: basic.load
      parameters:
        - id: load
          time-dependent: true
          scenario-dependent: true
          value: load_data

  connections:
    - component1: N
      port1: injection_port
      component2: D
      port2: injection_port

    - component1: N
      port1: injection_port
      component2: G1
      port2: injection_port

    - component1: N
      port1: injection_port
      component2: G2
      port2: injection_port
~~~

## Example of a timeseries file
Here is an example for the data file ~load_data~ mentioned in the system file above, in the case with 4 timesteps and 2 scenarios.

~~~
50 55
60  80
120 110
150 150
~~~
A data file may have a `.txt` or `.csv` extension.

# Getting started with GemsPy

## Installation

You can directly clone the [GitHub repo](https://github.com/AntaresSimulatorTeam/GemsPy) of the project.

## Interpretation and simulation with GemsPy

Here is an example of how to load component and library files, resolve the system, and solve the optimisation problem using the ***GemsPy*** package.

### Option A — directory-based (recommended)

If your inputs are organised in a study directory (see [Reading input files](user-guide/inputs.md)), the simplest way is:

~~~ python
from pathlib import Path
from gems.study.folder import load_study
from gems.session import SimulationSession
from gems.optim_config import load_optim_config

study = load_study(Path("my_study"))
optim_config = load_optim_config(Path("my_study/input/optim-config.yml"))

session = SimulationSession(study=study, optim_config=optim_config)
results = session.run()
~~~

Or, in a single call:

~~~ python
from pathlib import Path
from gems.study.runner import run_study

run_study(Path("my_study"))
~~~

### Option B — file-by-file (programmatic)

Here is the GemsPy syntax to read a test case described by

-  A library of models: `library.yml`
-  A system file: `system.yml`
-  A set of timeseries located in the directory: `series_dir`.

~~~ python
with open("library.yml") as lib_file:
    input_libraries = [parse_yaml_library(lib_file)]

with open("system.yml") as compo_file:
    input_system = parse_yaml_components(compo_file)

result_lib = resolve_library(input_libraries)
system = resolve_system(input_system, result_lib)
database = build_data_base(input_system, Path(series_dir))
~~~

### Building the optimisation problem

~~~ python
from gems.study import Study

problem = build_problem(
    Study(system, database),
    TimeBlock(1, list(range(timespan))),
    scenario_ids=list(range(nb_scenarios)),
)
~~~

### Solving the optimisation problem
~~~ python
problem.solve()
print(problem.objective_value)
~~~
