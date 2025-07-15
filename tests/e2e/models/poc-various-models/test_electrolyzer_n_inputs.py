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

import math

from libs.standard import DEMAND_MODEL, GENERATOR_MODEL, NODE_BALANCE_MODEL
from libs.standard_sc import (
    CONVERTOR_MODEL,
    CONVERTOR_RECEIVE_IN,
    DECOMPOSE_1_FLOW_INTO_2_FLOW,
    NODE_BALANCE_MODEL_MOD,
    TWO_INPUTS_CONVERTOR_MODEL,
)

from gems.simulation import OutputValues, TimeBlock, build_problem
from gems.study import ConstantData, DataBase, Network, Node, PortRef, create_component

"""
This file tests various modellings for an electrolyser with multiple inputs. The models are created in Python directly.

For every following test we have two electrical productions with an electrolyzer converting to a gaz flow
we always have:
    first electric production:
        - p_max = 70
        - cost = 10
    second electric production:
        - p_max = 80
        - cost = 20
    gaz production:
        - p_max = 30
        - cost = 15
    first conversion rate:
        - alpha = 0.7
    second conversion rate:
        - alpha = 0.5
    for a gaz demand of 100
"""


def test_electrolyzer_n_inputs_1() -> None:
    """
    Test with an electrolyzer for each input

    ep1 = electric production 1
    ep2 = electric production 2
    ez1 = electrolyzer 1
    ez2 = electrolyzer 2
    gp = gaz production

    total gaz production = flow_ep1 * alpha_ez1 + flow_ep2 * alpha_ez2 + flow_gp

    """
    elec_node_1 = Node(model=NODE_BALANCE_MODEL, id="e1")
    electric_prod_1 = create_component(model=GENERATOR_MODEL, id="ep1")
    electrolyzer1 = create_component(model=CONVERTOR_MODEL, id="ez1")

    elec_node_2 = Node(model=NODE_BALANCE_MODEL, id="e2")
    electric_prod_2 = create_component(model=GENERATOR_MODEL, id="ep2")
    electrolyzer2 = create_component(model=CONVERTOR_MODEL, id="ez2")

    gaz_node = Node(model=NODE_BALANCE_MODEL, id="g")
    gaz_prod = create_component(model=GENERATOR_MODEL, id="gp")
    gaz_demand = create_component(model=DEMAND_MODEL, id="gd")

    database = DataBase()

    database.add_data("ep1", "p_max", ConstantData(70))
    database.add_data("ep1", "cost", ConstantData(10))
    database.add_data("ez1", "alpha", ConstantData(0.7))

    database.add_data("ep2", "p_max", ConstantData(80))
    database.add_data("ep2", "cost", ConstantData(20))
    database.add_data("ez2", "alpha", ConstantData(0.5))

    database.add_data("gd", "demand", ConstantData(100))
    database.add_data("gp", "p_max", ConstantData(30))
    database.add_data("gp", "cost", ConstantData(15))

    network = Network("test")
    network.add_node(elec_node_1)
    network.add_component(electric_prod_1)
    network.add_component(electrolyzer1)
    network.add_node(elec_node_2)
    network.add_component(electric_prod_2)
    network.add_component(electrolyzer2)
    network.add_node(gaz_node)
    network.add_component(gaz_prod)
    network.add_component(gaz_demand)

    network.connect(
        PortRef(electric_prod_1, "balance_port"), PortRef(elec_node_1, "balance_port")
    )
    network.connect(
        PortRef(elec_node_1, "balance_port"), PortRef(electrolyzer1, "FlowDI")
    )
    network.connect(PortRef(electrolyzer1, "FlowDO"), PortRef(gaz_node, "balance_port"))
    network.connect(
        PortRef(electric_prod_2, "balance_port"), PortRef(elec_node_2, "balance_port")
    )
    network.connect(
        PortRef(elec_node_2, "balance_port"), PortRef(electrolyzer2, "FlowDI")
    )
    network.connect(PortRef(electrolyzer2, "FlowDO"), PortRef(gaz_node, "balance_port"))
    network.connect(
        PortRef(gaz_node, "balance_port"), PortRef(gaz_demand, "balance_port")
    )
    network.connect(
        PortRef(gaz_prod, "balance_port"), PortRef(gaz_node, "balance_port")
    )

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    output = OutputValues(problem)
    ep1_gen = output.component("ep1").var("generation").value
    ep2_gen = output.component("ep2").var("generation").value
    gp_gen = output.component("gp").var("generation").value
    print(ep1_gen)
    print(ep2_gen)
    print(gp_gen)

    assert math.isclose(ep1_gen, 70)  # type:ignore
    assert math.isclose(ep2_gen, 42)  # type:ignore
    assert math.isclose(gp_gen, 30)  # type:ignore

    assert status == problem.solver.OPTIMAL
    assert math.isclose(problem.solver.Objective().Value(), 1990)


def test_electrolyzer_n_inputs_2() -> None:
    """
    Test with one electrolyzer that has two inputs

    ep1 = electric production 1
    ep2 = electric production 2
    ez = electrolyzer
    gp = gaz production

    total gaz production = flow_ep1 * alpha1_ez + flow_ep2 * alpha2_ez + flow_gp
    """

    elec_node_1 = Node(model=NODE_BALANCE_MODEL, id="e1")
    elec_node_2 = Node(model=NODE_BALANCE_MODEL, id="e2")
    gaz_node = Node(model=NODE_BALANCE_MODEL, id="g")

    electric_prod_1 = create_component(model=GENERATOR_MODEL, id="ep1")
    electric_prod_2 = create_component(model=GENERATOR_MODEL, id="ep2")

    gaz_prod = create_component(model=GENERATOR_MODEL, id="gp")
    gaz_demand = create_component(model=DEMAND_MODEL, id="gd")

    electrolyzer = create_component(model=TWO_INPUTS_CONVERTOR_MODEL, id="ez")

    database = DataBase()

    database.add_data("ez", "alpha1", ConstantData(0.7))
    database.add_data("ez", "alpha2", ConstantData(0.5))

    database.add_data("ep1", "p_max", ConstantData(70))
    database.add_data("ep1", "cost", ConstantData(10))

    database.add_data("ep2", "p_max", ConstantData(80))
    database.add_data("ep2", "cost", ConstantData(20))

    database.add_data("gd", "demand", ConstantData(100))
    database.add_data("gp", "p_max", ConstantData(30))
    database.add_data("gp", "cost", ConstantData(15))

    network = Network("test")
    network.add_node(elec_node_1)
    network.add_node(elec_node_2)
    network.add_node(gaz_node)
    network.add_component(electric_prod_1)
    network.add_component(electric_prod_2)
    network.add_component(gaz_prod)
    network.add_component(gaz_demand)
    network.add_component(electrolyzer)

    network.connect(
        PortRef(electric_prod_1, "balance_port"), PortRef(elec_node_1, "balance_port")
    )
    network.connect(
        PortRef(elec_node_1, "balance_port"), PortRef(electrolyzer, "FlowDI1")
    )
    network.connect(
        PortRef(electric_prod_2, "balance_port"), PortRef(elec_node_2, "balance_port")
    )
    network.connect(
        PortRef(elec_node_2, "balance_port"), PortRef(electrolyzer, "FlowDI2")
    )
    network.connect(PortRef(electrolyzer, "FlowDO"), PortRef(gaz_node, "balance_port"))
    network.connect(
        PortRef(gaz_node, "balance_port"), PortRef(gaz_demand, "balance_port")
    )
    network.connect(
        PortRef(gaz_prod, "balance_port"), PortRef(gaz_node, "balance_port")
    )

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    output = OutputValues(problem)
    ep1_gen = output.component("ep1").var("generation").value
    ep2_gen = output.component("ep2").var("generation").value
    gp_gen = output.component("gp").var("generation").value
    print(ep1_gen)
    print(ep2_gen)
    print(gp_gen)

    assert math.isclose(ep1_gen, 70)  # type:ignore
    assert math.isclose(ep2_gen, 42)  # type:ignore
    assert math.isclose(gp_gen, 30)  # type:ignore

    assert status == problem.solver.OPTIMAL
    assert math.isclose(problem.solver.Objective().Value(), 1990)


def test_electrolyzer_n_inputs_3() -> None:
    """
    Test with a consumption_electrolyzer with two inputs

    ep1 = electric production 1
    ep2 = electric production 2
    ez = electrolyzer
    gp = gaz production

    total gaz production = (flow_ep1 + flow_ep2) * alpha_ez + flow_gp

    The result is different since we only have one alpha at 0.7
    """
    elec_node_1 = Node(model=NODE_BALANCE_MODEL, id="e1")
    elec_node_2 = Node(model=NODE_BALANCE_MODEL, id="e2")
    gaz_node = Node(model=NODE_BALANCE_MODEL, id="g")

    electric_prod_1 = create_component(model=GENERATOR_MODEL, id="ep1")
    electric_prod_2 = create_component(model=GENERATOR_MODEL, id="ep2")

    gaz_prod = create_component(model=GENERATOR_MODEL, id="gp")
    gaz_demand = create_component(model=DEMAND_MODEL, id="gd")

    electrolyzer = create_component(model=CONVERTOR_MODEL, id="ez")
    consumption_electrolyzer = create_component(
        model=DECOMPOSE_1_FLOW_INTO_2_FLOW, id="ce"
    )

    database = DataBase()

    database.add_data("ez", "alpha", ConstantData(0.7))

    database.add_data("ep1", "p_max", ConstantData(70))
    database.add_data("ep1", "cost", ConstantData(10))

    database.add_data("ep2", "p_max", ConstantData(80))
    database.add_data("ep2", "cost", ConstantData(20))

    database.add_data("gd", "demand", ConstantData(100))
    database.add_data("gp", "p_max", ConstantData(30))
    database.add_data("gp", "cost", ConstantData(15))

    network = Network("test")
    network.add_node(elec_node_1)
    network.add_node(elec_node_2)
    network.add_node(gaz_node)
    network.add_component(electric_prod_1)
    network.add_component(electric_prod_2)
    network.add_component(gaz_prod)
    network.add_component(gaz_demand)
    network.add_component(electrolyzer)
    network.add_component(consumption_electrolyzer)

    network.connect(
        PortRef(electric_prod_1, "balance_port"), PortRef(elec_node_1, "balance_port")
    )
    network.connect(
        PortRef(elec_node_1, "balance_port"),
        PortRef(consumption_electrolyzer, "FlowDI1"),
    )
    network.connect(
        PortRef(electric_prod_2, "balance_port"), PortRef(elec_node_2, "balance_port")
    )
    network.connect(
        PortRef(elec_node_2, "balance_port"),
        PortRef(consumption_electrolyzer, "FlowDI2"),
    )
    network.connect(
        PortRef(consumption_electrolyzer, "FlowDO"), PortRef(electrolyzer, "FlowDI")
    )
    network.connect(PortRef(electrolyzer, "FlowDO"), PortRef(gaz_node, "balance_port"))
    network.connect(
        PortRef(gaz_node, "balance_port"), PortRef(gaz_demand, "balance_port")
    )
    network.connect(
        PortRef(gaz_prod, "balance_port"), PortRef(gaz_node, "balance_port")
    )

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    output = OutputValues(problem)
    ep1_gen = output.component("ep1").var("generation").value
    ep2_gen = output.component("ep2").var("generation").value
    gp_gen = output.component("gp").var("generation").value

    assert math.isclose(ep1_gen, 70)  # type:ignore
    assert math.isclose(ep2_gen, 30)  # type:ignore
    assert math.isclose(gp_gen, 30)  # type:ignore

    assert status == problem.solver.OPTIMAL
    assert math.isclose(problem.solver.Objective().Value(), 1750)


def test_electrolyzer_n_inputs_4() -> None:
    """
    Test with one electrolyzer with one input that takes every inputs

    ep1 = electric production 1
    ep2 = electric production 2
    ez = electrolyzer
    gp = gaz production

    total gaz production = (flow_ep1 + flow_ep2) * alpha_ez + flow_gp

    same as test 3, the result is different than the first two since we only have one alpha at 0.7
    """
    elec_node_1 = Node(model=NODE_BALANCE_MODEL_MOD, id="e1")
    elec_node_2 = Node(model=NODE_BALANCE_MODEL_MOD, id="e2")
    gaz_node = Node(model=NODE_BALANCE_MODEL, id="g")

    electric_prod_1 = create_component(model=GENERATOR_MODEL, id="ep1")
    electric_prod_2 = create_component(model=GENERATOR_MODEL, id="ep2")

    gaz_prod = create_component(model=GENERATOR_MODEL, id="gp")
    gaz_demand = create_component(model=DEMAND_MODEL, id="gd")

    electrolyzer = create_component(model=CONVERTOR_RECEIVE_IN, id="ez")

    database = DataBase()

    database.add_data("ez", "alpha", ConstantData(0.7))

    database.add_data("ep1", "p_max", ConstantData(70))
    database.add_data("ep1", "cost", ConstantData(10))

    database.add_data("ep2", "p_max", ConstantData(80))
    database.add_data("ep2", "cost", ConstantData(20))

    database.add_data("gd", "demand", ConstantData(100))
    database.add_data("gp", "p_max", ConstantData(30))
    database.add_data("gp", "cost", ConstantData(15))

    network = Network("test")
    network.add_node(elec_node_1)
    network.add_node(elec_node_2)
    network.add_node(gaz_node)
    network.add_component(electric_prod_1)
    network.add_component(electric_prod_2)
    network.add_component(gaz_prod)
    network.add_component(gaz_demand)
    network.add_component(electrolyzer)

    network.connect(
        PortRef(electric_prod_1, "balance_port"), PortRef(elec_node_1, "balance_port_n")
    )
    network.connect(
        PortRef(elec_node_1, "balance_port_e"), PortRef(electrolyzer, "FlowDI")
    )
    network.connect(
        PortRef(electric_prod_2, "balance_port"), PortRef(elec_node_2, "balance_port_n")
    )
    network.connect(
        PortRef(elec_node_2, "balance_port_e"), PortRef(electrolyzer, "FlowDI")
    )
    network.connect(PortRef(electrolyzer, "FlowDO"), PortRef(gaz_node, "balance_port"))
    network.connect(
        PortRef(gaz_node, "balance_port"), PortRef(gaz_demand, "balance_port")
    )
    network.connect(
        PortRef(gaz_prod, "balance_port"), PortRef(gaz_node, "balance_port")
    )

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL

    output = OutputValues(problem)
    ep1_gen = output.component("ep1").var("generation").value
    ep2_gen = output.component("ep2").var("generation").value
    gp_gen = output.component("gp").var("generation").value

    assert math.isclose(ep1_gen, 70)  # type:ignore
    assert math.isclose(ep2_gen, 30)  # type:ignore
    assert math.isclose(gp_gen, 30)  # type:ignore

    assert status == problem.solver.OPTIMAL
    assert math.isclose(problem.solver.Objective().Value(), 1750)
