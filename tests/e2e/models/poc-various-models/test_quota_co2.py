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
This file tests the model of CO2 quota. The models are created in Python directly.
"""

import math

from libs.standard import DEMAND_MODEL, LINK_MODEL, NODE_BALANCE_MODEL
from libs.standard_sc import C02_POWER_MODEL, QUOTA_CO2_MODEL

from gems.simulation import OutputValues, TimeBlock, build_problem
from gems.study import ConstantData, DataBase, Network, Node, PortRef, create_component


def test_quota_co2() -> None:
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
    n1 = Node(model=NODE_BALANCE_MODEL, id="N1")
    n2 = Node(model=NODE_BALANCE_MODEL, id="N2")
    oil1 = create_component(model=C02_POWER_MODEL, id="Oil1")
    coal1 = create_component(model=C02_POWER_MODEL, id="Coal1")
    l12 = create_component(model=LINK_MODEL, id="L12")
    demand = create_component(model=DEMAND_MODEL, id="Demand")
    monQuotaCO2 = create_component(model=QUOTA_CO2_MODEL, id="QuotaCO2")

    network = Network("test")
    network.add_node(n1)
    network.add_node(n2)
    network.add_component(oil1)
    network.add_component(coal1)
    network.add_component(l12)
    network.add_component(demand)
    network.add_component(monQuotaCO2)

    network.connect(PortRef(demand, "balance_port"), PortRef(n2, "balance_port"))
    network.connect(PortRef(n2, "balance_port"), PortRef(l12, "balance_port_from"))
    network.connect(PortRef(l12, "balance_port_to"), PortRef(n1, "balance_port"))
    network.connect(PortRef(n1, "balance_port"), PortRef(oil1, "FlowP"))
    network.connect(PortRef(n2, "balance_port"), PortRef(coal1, "FlowP"))
    network.connect(PortRef(oil1, "OutCO2"), PortRef(monQuotaCO2, "emissionCO2"))
    network.connect(PortRef(coal1, "OutCO2"), PortRef(monQuotaCO2, "emissionCO2"))

    database = DataBase()
    database.add_data("Demand", "demand", ConstantData(100))
    database.add_data("Coal1", "p_min", ConstantData(0))
    database.add_data("Oil1", "p_min", ConstantData(0))
    database.add_data("Coal1", "p_max", ConstantData(100))
    database.add_data("Oil1", "p_max", ConstantData(100))
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
