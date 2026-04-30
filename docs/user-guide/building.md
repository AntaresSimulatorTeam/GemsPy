# Building systems with the Python API

Instead of reading a `.yml` file, one can also build a system with GemsPy by using the API of the package.

The Pydantic schema classes used to describe systems programmatically follow the `*Schema` naming convention.


## Defining a ComponentSchema

The syntax to build components with the GemsPy API is the following:

~~~ python
from gems.study.parsing import ComponentSchema, ComponentParameterSchema

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
        parameters=[
            ComponentParameterSchema(
                id="marginal_cost",
                time_dependent=False,
                scenario_dependent=False,
                value=70  # €/MWh
            ),
            ComponentParameterSchema(
                id="pmax",
                time_dependent=False,
                scenario_dependent=False,
                value=700  # MWh
            ),
        ],
    )
)
~~~

## Defining a PortConnectionsSchema

The syntax to build connections between components with the GemsPy API is the following:

~~~ python
from gems.study.parsing import PortConnectionsSchema

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
from gems.study.parsing import SystemSchema

input_system = SystemSchema(
    components=components,
    connections=connections,
)
~~~

The `input_system` variable can then be used in the same way as when it was created using the [parse_yaml_components](inputs.md) method.
