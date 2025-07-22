# Getting started with Gems

The Gems framework consists of a **high-level modelling language**, close to mathematical syntax, and a **data structure** for describing energy systems.

More specifically, three main types of input files can be defined with the Gems framework:

1. **Model libraries**: describe abstract component models.  
2. **System files**: describe the graph of components that make up a system of interest; refer to model libraries (instantiation of abstract models) and to timeseries files.  
3. **Timeseries files**: the data of timeseries.

To get started with the syntax of these files, the reader can find basic examples below. More details are available in the dedicated sections of the documentation.

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
      objective: expec(sum(marginal_cost * generation))

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

### Loading the library, the system and the timeseries:

Here is the GemsPy syntax to read a test case described by

-  A library of models: `library.yml`
-  A system file: `system.yml`
-  A set of timeseries located in the directory: `series_dir`.

~~~ python
with open("library.yml") as compo_file:
    input_system = parse_yaml_components(compo_file)

with open("system.yml") as lib_file:
    input_libraries = [parse_yaml_library(lib_file)]

result_lib = resolve_library(input_libraries)
components_input = resolve_system(input_system, result_lib)
database = build_data_base(input_system, Path(series_dir))
~~~

### Building the optimisation problem

~~~ python

network = build_network(components_input)

problem = build_problem(
    network,
    database,
    TimeBlock(1, [i for i in range(0, timespan)]),
    scenarios,
)
~~~

### Solving the optimisation problem
~~~ python
status = problem.solver.Solve()
print(problem.solver.Objective().Value())
~~~
