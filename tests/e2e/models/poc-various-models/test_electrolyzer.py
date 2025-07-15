# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

from libs.standard import CONSTANT, TIME_AND_SCENARIO_FREE

from gems.expression import literal, param, var
from gems.expression.expression import port_field
from gems.model import (
    Constraint,
    ModelPort,
    PortField,
    PortType,
    float_parameter,
    float_variable,
    model,
)
from gems.model.port import PortFieldDefinition, PortFieldId
from gems.simulation import TimeBlock, build_problem
from gems.study import ConstantData, DataBase, Network, Node, PortRef, create_component

ELECTRICAL_PORT = PortType(id="electrical_port", fields=[PortField("flow")])

ELECTRICAL_NODE_MODEL = model(
    id="ELECTRICAL_NODE_MODEL",
    ports=[ModelPort(port_type=ELECTRICAL_PORT, port_name="electrical_port")],
    binding_constraints=[
        Constraint(
            name="Balance",
            expression=port_field("electrical_port", "flow").sum_connections()
            == literal(0),
        )
    ],
)

ELECTRICAL_GENERATOR_MODEL = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),
        float_parameter("cost", CONSTANT),
    ],
    variables=[
        float_variable("generation", lower_bound=literal(0), upper_bound=param("p_max"))
    ],
    ports=[ModelPort(port_type=ELECTRICAL_PORT, port_name="electrical_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("electrical_port", "flow"),
            definition=var("generation"),
        )
    ],
    objective_operational_contribution=(param("cost") * var("generation"))
    .time_sum()
    .expec(),
)

H2_PORT = PortType(id="h2_port", fields=[PortField("flow")])

H2_NODE_MODEL = model(
    id="H2_NODE_MODEL",
    ports=[ModelPort(port_type=H2_PORT, port_name="h2_port")],
    binding_constraints=[
        Constraint(
            name="Balance",
            expression=port_field("h2_port", "flow").sum_connections() == literal(0),
        )
    ],
)

H2_DEMAND = model(
    id="H2_DEMAND",
    parameters=[
        float_parameter("demand", TIME_AND_SCENARIO_FREE),
    ],
    ports=[ModelPort(port_type=H2_PORT, port_name="h2_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("h2_port", "flow"),
            definition=-param("demand"),
        )
    ],
)

ELECTROLYZER = model(
    id="ELECTROLYZER",
    parameters=[
        float_parameter("efficiency", TIME_AND_SCENARIO_FREE),
    ],
    variables=[
        float_variable("electrical_input"),
        float_variable("h2_output"),
    ],
    ports=[
        ModelPort(port_type=ELECTRICAL_PORT, port_name="electrical_port"),
        ModelPort(port_type=H2_PORT, port_name="h2_port"),
    ],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("electrical_port", "flow"),
            definition=-var("electrical_input"),
        ),
        PortFieldDefinition(
            port_field=PortFieldId("h2_port", "flow"),
            definition=var("h2_output"),
        ),
    ],
    constraints=[
        Constraint(
            name="Conversion",
            expression=var("h2_output")
            == var("electrical_input") * param("efficiency"),
        )
    ],
)


def test_electrolyzer() -> None:
    elec_node = Node(model=ELECTRICAL_NODE_MODEL, id="1")
    h2_node = Node(model=H2_NODE_MODEL, id="2")

    electric_gen = create_component(
        model=ELECTRICAL_GENERATOR_MODEL,
        id="G",
    )

    demand_h2 = create_component(
        model=H2_DEMAND,
        id="D",
    )

    electrolyzer = create_component(
        model=ELECTROLYZER,
        id="E",
    )

    database = DataBase()
    database.add_data("D", "demand", ConstantData(70))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(30))
    database.add_data("E", "efficiency", ConstantData(0.7))

    network = Network("test")
    network.add_node(elec_node)
    network.add_node(h2_node)
    network.add_component(demand_h2)
    network.add_component(electric_gen)
    network.add_component(electrolyzer)
    network.connect(PortRef(demand_h2, "h2_port"), PortRef(h2_node, "h2_port"))
    network.connect(PortRef(h2_node, "h2_port"), PortRef(electrolyzer, "h2_port"))
    network.connect(
        PortRef(elec_node, "electrical_port"), PortRef(electrolyzer, "electrical_port")
    )
    network.connect(
        PortRef(elec_node, "electrical_port"),
        PortRef(electric_gen, "electrical_port"),
    )

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()
    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 3000
