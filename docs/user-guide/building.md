# Building systems with the Python API

Instead of reading a .yml file, one can build a system with GemsPy by using the API of the package.


## Defining an InputComponent

The syntax to build components with the GemsPy API is the following:

~~~ python
components = []

components.append(
    InputComponent(
        id="bus_de",
        model="simple_library.bus",
        parameters=[
            InputComponentParameter(
                id="ens_cost",
                time_dependent=False,
                scenario_dependent=False,
                value=40000 #€/MWh
            ),
            InputComponentParameter(
                id="ens_cost",
                time_dependent=False,
                scenario_dependent=False,
                value=3000 #€/MWh
            ),
        ],
    )
)

components.append(
    InputComponent(
        id="load_de",
        model="simple_library.load",
        parameters=[
            InputComponentParameter(
                id="load",
                time_dependent=True,
                scenario_dependent=True,
                value="load_ts.txt"),
        ],
    )
)

components.append(
    InputComponent(
        id="gen_de",
        model="simple_library.generator",
        parameters=[
            InputComponentParameter(
                id="marginal_cost",
                time_dependent=False,
                scenario_dependent=False,
                value=70 #€/MWh
            ),
            InputComponentParameter(
                id="pmax",
                time_dependent=False,
                scenario_dependent=False,
                value=700 #MWh
            ),
        ],
    )
)


~~~

## Defining an InputPortConnection

The syntax to build connections between components with the GemsPy API is the following:


~~~ python
connections = []

connections.append(
    InputPortConnections(
        component1="bus_de",
        port1="balance_port",
        component2="gen_de",
        port2="balance_port",
    )
)

connections.append(
    InputPortConnections(
        component1="bus_de",
        port1="balance_port",
        component2="load_de",
        port2="balance_port",
    )
)
~~~

## Defining an InputSystem

~~~ python
input_system = InputSystem(
            components=components,
            connections=connections,
        )
~~~

Then, the input_system variable can be used in the same way as when it was created using the [parse_yaml_components](inputs.md) method.