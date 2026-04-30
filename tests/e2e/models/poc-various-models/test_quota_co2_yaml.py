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
from gems.simulation import TimeBlock, build_problem
from gems.simulation.simulation_table import SimulationTableBuilder
from gems.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    Study,
    System,
    create_component,
)


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
    gen_model = lib_dict_sc["basic"].models["basic.generator_with_co2"]
    node_model = lib_dict["basic"].models["basic.node"]
    quota_co2_model = lib_dict_sc["basic"].models["basic.quota_co2"]
    demand_model = lib_dict["basic"].models["basic.demand"]
    link_model = lib_dict_sc["basic"].models["basic.link"]

    n1 = Component(model=node_model, id="N1")
    n2 = Component(model=node_model, id="N2")
    oil1 = create_component(model=gen_model, id="Oil1")
    coal1 = create_component(model=gen_model, id="Coal1")
    l12 = create_component(model=link_model, id="L12")
    demand = create_component(model=demand_model, id="Demand")
    monQuotaCO2 = create_component(model=quota_co2_model, id="QuotaCO2")

    system = System("test")
    system.add_component(n1)
    system.add_component(n2)
    system.add_component(oil1)
    system.add_component(coal1)
    system.add_component(l12)
    system.add_component(demand)
    system.add_component(monQuotaCO2)

    system.connect(PortRef(demand, "injection_port"), PortRef(n2, "injection_port"))
    system.connect(PortRef(n2, "injection_port"), PortRef(l12, "injection_port_from"))
    system.connect(PortRef(l12, "injection_port_to"), PortRef(n1, "injection_port"))
    system.connect(PortRef(n1, "injection_port"), PortRef(oil1, "injection_port"))
    system.connect(PortRef(n2, "injection_port"), PortRef(coal1, "injection_port"))
    system.connect(PortRef(oil1, "co2_port"), PortRef(monQuotaCO2, "emission_port"))
    system.connect(PortRef(coal1, "co2_port"), PortRef(monQuotaCO2, "emission_port"))

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
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, 5500)

    df = SimulationTableBuilder().build(problem)
    assert math.isclose(
        df.component("Oil1").output("p").value(time_index=0, scenario_index=0), 50
    )
    assert math.isclose(
        df.component("Coal1").output("p").value(time_index=0, scenario_index=0), 50
    )
    assert math.isclose(
        df.component("L12").output("flow").value(time_index=0, scenario_index=0), -50
    )
