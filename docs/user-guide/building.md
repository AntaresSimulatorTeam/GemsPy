# Building systems with the Python API

Instead of reading a `.yml` file, one can also build a system with GemsPy by using the API of the package.

The Pydantic schema classes used to describe systems programmatically follow the `*Schema` naming convention.


## Defining a model library

Models and libraries can also be built with the Python API. Here is the
`simple_library` used below, with `bus`, `load` and `generator` models:

~~~ python
from gems_craft.model.parsing import (
    LibrarySchema,
    ModelSchema,
    ParameterSchema,
    VariableSchema,
    PortTypeSchema,
    FieldSchema,
    ModelPortSchema,
    PortFieldDefinitionSchema,
    ConstraintSchema,
    ObjectiveContributionSchema,
    SetSchema,
)
from gems_craft.model.resolve_library import resolve_library

flow = PortTypeSchema(id="flow", fields=[FieldSchema(id="flow")])

bus_model = ModelSchema(
    id="bus",
    parameters=[
        ParameterSchema(id="spillage_cost"),
        ParameterSchema(id="unsupplied_energy_cost"),
    ],
    variables=[
        VariableSchema(id="spillage", lower_bound="0"),
        VariableSchema(id="unsupplied_energy", lower_bound="0"),
    ],
    ports=[ModelPortSchema(id="balance_port", type="flow")],
    binding_constraints=[
        ConstraintSchema(
            id="balance",
            expression="sum_connections(balance_port.flow) = spillage - unsupplied_energy",
        ),
    ],
    objective_contributions=[
        ObjectiveContributionSchema(
            id="objective",
            expression="sum(spillage_cost * spillage + unsupplied_energy_cost * unsupplied_energy)",
        ),
    ],
)

load_model = ModelSchema(
    id="load",
    parameters=[ParameterSchema(id="load", time_dependent=True, scenario_dependent=True)],
    ports=[ModelPortSchema(id="balance_port", type="flow")],
    port_field_definitions=[
        PortFieldDefinitionSchema(port="balance_port", field="flow", definition="-load"),
    ],
)

generator_model = ModelSchema(
    id="generator",
    sets=[SetSchema(id="tier")],  # local set: instantiated by each component
    parameters=[
        ParameterSchema(id="p_min", time_dependent=True, scenario_dependent=True),
        ParameterSchema(
            id="p_max", time_dependent=True, scenario_dependent=True, indexed_by=["fuel", "tier"]
        ),
        ParameterSchema(id="generation_cost", indexed_by=["fuel", "tier"]),
    ],
    variables=[
        VariableSchema(
            id="generation", lower_bound="p_min", upper_bound="p_max", indexed_by=["fuel", "tier"]
        ),
        VariableSchema(id="num_units_on", lower_bound="0", variable_type="integer"),
    ],
    ports=[ModelPortSchema(id="balance_port", type="flow")],
    port_field_definitions=[
        PortFieldDefinitionSchema(
            port="balance_port",
            field="flow",
            definition="sum_over(fuel, sum_over(tier, generation))",
        ),
    ],
    objective_contributions=[
        ObjectiveContributionSchema(
            id="objective",
            expression="sum(sum_over(fuel, sum_over(tier, generation_cost * generation)))",
        ),
    ],
)

library = LibrarySchema(
    id="simple_library",
    port_types=[flow],
    sets=[SetSchema(id="fuel")],  # global set: instantiated once on the system
    models=[bus_model, load_model, generator_model],
)
~~~

`generator` also shows custom sets: `p_max`, `generation_cost` and `generation`
are indexed by the global set `fuel` and by the model's own local set `tier`.
The port field sums `generation` over both sets, as a local set cannot cross a
port.

## Defining a ComponentSchema

The syntax to build components with the GemsPy API is the following:

~~~ python
from gems_craft.study.parsing import (
    ComponentSchema,
    ComponentParameterSchema,
    IntegerStrategy,
    IntegerStrategyId,
    HeuristicId,
    SetInstanceSchema,
)

components = []

components.append(
    ComponentSchema(
        id="bus_de",
        model="simple_library.bus",
        parameters=[
            ComponentParameterSchema(
                id="ens_cost",
                time_dependent=False,
                scenario_dependent=False,
                value=40000  # €/MWh
            ),
            ComponentParameterSchema(
                id="spillage_cost",
                time_dependent=False,
                scenario_dependent=False,
                value=3000  # €/MWh
            ),
        ],
    )
)

components.append(
    ComponentSchema(
        id="load_de",
        model="simple_library.load",
        parameters=[
            ComponentParameterSchema(
                id="load",
                time_dependent=True,
                scenario_dependent=True,
                value="load_ts.txt"),
        ],
    )
)

components.append(
    ComponentSchema(
        id="gen_de",
        model="simple_library.generator",
        integer_strategy=IntegerStrategy(
            id=IntegerStrategyId.HEURISTIC,
            heuristic_id=HeuristicId.FAST,
        ),
        parameters=[
            ComponentParameterSchema(
                id="generation_cost",
                time_dependent=False,
                scenario_dependent=False,
                indexed_by=["fuel", "tier"],
                value="generation_cost",  # tidy CSV, see "Loading set-indexed data"
            ),
            ComponentParameterSchema(
                id="p_max",
                time_dependent=False,
                scenario_dependent=False,
                indexed_by=["fuel", "tier"],
                value="p_max",  # tidy CSV, MWh
            ),
        ],
        sets=[SetInstanceSchema(id="tier", elements=["low", "high"])],
    )
)
~~~

Components of one model may instantiate a local set (here `tier`) with
different elements and sizes.
An explicit position (`X[tier=2]`) must exist in every component's instantiation,
so it is limited by the smallest one; use `X[tier]` or `sum_over` otherwise.

A component may also set `integer_strategy` to relax or heuristically process
its model's integer/binary variables — see
[Optimisation configuration](optim-config.md#heuristics-integer-strategy-and-thermal-heuristics).
Here, `gen_de` uses the `FAST` heuristic instead of solving `generator`'s
integer variable (`num_units_on`) exactly.

## Defining a PortConnectionsSchema

The syntax to build connections between components with the GemsPy API is the following:

~~~ python
from gems_craft.study.parsing import PortConnectionsSchema

connections = []

connections.append(
    PortConnectionsSchema(
        component1="bus_de",
        port1="balance_port",
        component2="gen_de",
        port2="balance_port",
    )
)

connections.append(
    PortConnectionsSchema(
        component1="bus_de",
        port1="balance_port",
        component2="load_de",
        port2="balance_port",
    )
)
~~~

## Defining a SystemSchema

~~~ python
from gems_craft.study.parsing import SystemSchema

input_system = SystemSchema(
    sets=[SetInstanceSchema(id="fuel", elements=["coal", "gas"])],
    components=components,
    connections=connections,
)
~~~

The `input_system` variable can then be used in the same way as when it was created using the [parse_yaml_system](inputs.md) method.
