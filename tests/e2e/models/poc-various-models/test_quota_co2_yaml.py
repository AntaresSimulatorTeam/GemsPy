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

"""
This file tests the model of CO2 quota. The models are parsed from a YAML model library.
"""

import math

from gems.model.library import Library
from gems.simulation import OutputValues, TimeBlock, build_problem
from gems.study import ConstantData, DataBase, Network, Node, PortRef, create_component


def test_quota_co2(
    lib_dict: dict[str, Library], lib_dict_sc: dict[str, Library]
) -> None:
    """
    Builds the quota CO² test system.

    N1 -----N2----Demand         ^
    |       |
    Oil1    Coal1
    |       |
    ---------
        |
    QuotaCO2

    Test of a generation of energy and co2 with a quota to limit the emission"""
    gen_model = lib_dict_sc["basic"].models["generator_with_co2"]
    node_model = lib_dict["basic"].models["node"]
    quota_co2_model = lib_dict_sc["basic"].models["quota_co2"]
    demand_model = lib_dict["basic"].models["demand"]
    link_model = lib_dict_sc["basic"].models["link"]

    n1 = Node(model=node_model, id="N1")
    n2 = Node(model=node_model, id="N2")
    oil1 = create_component(model=gen_model, id="Oil1")
    coal1 = create_component(model=gen_model, id="Coal1")
    l12 = create_component(model=link_model, id="L12")
    demand = create_component(model=demand_model, id="Demand")
    monQuotaCO2 = create_component(model=quota_co2_model, id="QuotaCO2")

    network = Network("test")
    network.add_node(n1)
    network.add_node(n2)
    network.add_component(oil1)
    network.add_component(coal1)
    network.add_component(l12)
    network.add_component(demand)
    network.add_component(monQuotaCO2)

    network.connect(PortRef(demand, "injection_port"), PortRef(n2, "injection_port"))
    network.connect(PortRef(n2, "injection_port"), PortRef(l12, "injection_port_from"))
    network.connect(PortRef(l12, "injection_port_to"), PortRef(n1, "injection_port"))
    network.connect(PortRef(n1, "injection_port"), PortRef(oil1, "injection_port"))
    network.connect(PortRef(n2, "injection_port"), PortRef(coal1, "injection_port"))
    network.connect(PortRef(oil1, "co2_port"), PortRef(monQuotaCO2, "emission_port"))
    network.connect(PortRef(coal1, "co2_port"), PortRef(monQuotaCO2, "emission_port"))

    database = DataBase()
    database.add_data("Demand", "demand", ConstantData(100))
    database.add_data("Coal1", "pmin", ConstantData(0))
    database.add_data("Oil1", "pmin", ConstantData(0))
    database.add_data("Coal1", "pmax", ConstantData(100))
    database.add_data("Oil1", "pmax", ConstantData(100))
    database.add_data("Coal1", "emission_rate", ConstantData(2))
    database.add_data("Oil1", "emission_rate", ConstantData(1))
    database.add_data("Coal1", "cost", ConstantData(10))
    database.add_data("Oil1", "cost", ConstantData(100))
    database.add_data("L12", "f_max", ConstantData(100))
    database.add_data("QuotaCO2", "quota", ConstantData(150))

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    output = OutputValues(problem)
    oil1_p = output.component("Oil1").var("p").value
    coal1_p = output.component("Coal1").var("p").value
    l12_flow = output.component("L12").var("flow").value

    assert status == problem.solver.OPTIMAL
    assert math.isclose(problem.solver.Objective().Value(), 5500)
    assert math.isclose(oil1_p, 50)  # type:ignore
    assert math.isclose(coal1_p, 50)  # type:ignore
    assert math.isclose(l12_flow, -50)  # type:ignore
