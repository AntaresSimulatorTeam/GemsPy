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

"""
This file tests various modellings for an electrolyser with multiple inputs. The models are read from a YAML model file.

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


def test_electrolyzer_n_inputs_1(
    lib_dict: dict[str, Library], lib_dict_sc: dict[str, Library]
) -> None:
    """
    Test with an electrolyzer for each input

    ep1 = electric production 1
    ep2 = electric production 2
    ez1 = electrolyzer 1
    ez2 = electrolyzer 2
    gp = gaz production

    total gaz production = flow_ep1 * alpha_ez1 + flow_ep2 * alpha_ez2 + flow_gp

    """

    gen_model = lib_dict["basic"].models["basic.generator"]
    node_model = lib_dict["basic"].models["basic.node"]
    convertor_model = lib_dict_sc["basic"].models["basic.convertor"]
    demand_model = lib_dict["basic"].models["basic.demand"]

    elec_node_1 = Component(model=node_model, id="e1")
    electric_prod_1 = create_component(model=gen_model, id="ep1")
    electrolyzer1 = create_component(model=convertor_model, id="ez1")

    elec_node_2 = Component(model=node_model, id="e2")
    electric_prod_2 = create_component(model=gen_model, id="ep2")
    electrolyzer2 = create_component(model=convertor_model, id="ez2")

    gaz_node = Component(model=node_model, id="g")
    gaz_prod = create_component(model=gen_model, id="gp")
    gaz_demand = create_component(model=demand_model, id="gd")

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

    system = System("test")
    system.add_component(elec_node_1)
    system.add_component(electric_prod_1)
    system.add_component(electrolyzer1)
    system.add_component(elec_node_2)
    system.add_component(electric_prod_2)
    system.add_component(electrolyzer2)
    system.add_component(gaz_node)
    system.add_component(gaz_prod)
    system.add_component(gaz_demand)

    system.connect(
        PortRef(electric_prod_1, "injection_port"),
        PortRef(elec_node_1, "injection_port"),
    )
    system.connect(
        PortRef(elec_node_1, "injection_port"), PortRef(electrolyzer1, "input_port")
    )
    system.connect(
        PortRef(electrolyzer1, "output_port"), PortRef(gaz_node, "injection_port")
    )
    system.connect(
        PortRef(electric_prod_2, "injection_port"),
        PortRef(elec_node_2, "injection_port"),
    )
    system.connect(
        PortRef(elec_node_2, "injection_port"), PortRef(electrolyzer2, "input_port")
    )
    system.connect(
        PortRef(electrolyzer2, "output_port"), PortRef(gaz_node, "injection_port")
    )
    system.connect(
        PortRef(gaz_node, "injection_port"), PortRef(gaz_demand, "injection_port")
    )
    system.connect(
        PortRef(gaz_prod, "injection_port"), PortRef(gaz_node, "injection_port")
    )

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, 1990)

    df = SimulationTableBuilder().build(problem)
    assert math.isclose(
        df.component("ep1").output("generation").value(time_index=0, scenario_index=0),
        70,
    )
    assert math.isclose(
        df.component("ep2").output("generation").value(time_index=0, scenario_index=0),
        42,
    )
    assert math.isclose(
        df.component("gp").output("generation").value(time_index=0, scenario_index=0),
        30,
    )


def test_electrolyzer_n_inputs_2(
    lib_dict: dict[str, Library], lib_dict_sc: dict[str, Library]
) -> None:
    """
    Test with one electrolyzer that has two inputs

    ep1 = electric production 1
    ep2 = electric production 2
    ez = electrolyzer
    gp = gaz production

    total gaz production = flow_ep1 * alpha1_ez + flow_ep2 * alpha2_ez + flow_gp
    """

    gen_model = lib_dict["basic"].models["basic.generator"]
    node_model = lib_dict["basic"].models["basic.node"]
    convertor_model = lib_dict_sc["basic"].models["basic.two_input_convertor"]
    demand_model = lib_dict["basic"].models["basic.demand"]

    elec_node_1 = Component(model=node_model, id="e1")
    elec_node_2 = Component(model=node_model, id="e2")
    gaz_node = Component(model=node_model, id="g")

    electric_prod_1 = create_component(model=gen_model, id="ep1")
    electric_prod_2 = create_component(model=gen_model, id="ep2")

    gaz_prod = create_component(model=gen_model, id="gp")
    gaz_demand = create_component(model=demand_model, id="gd")

    electrolyzer = create_component(model=convertor_model, id="ez")

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

    system = System("test")
    system.add_component(elec_node_1)
    system.add_component(elec_node_2)
    system.add_component(gaz_node)
    system.add_component(electric_prod_1)
    system.add_component(electric_prod_2)
    system.add_component(gaz_prod)
    system.add_component(gaz_demand)
    system.add_component(electrolyzer)

    system.connect(
        PortRef(electric_prod_1, "injection_port"),
        PortRef(elec_node_1, "injection_port"),
    )
    system.connect(
        PortRef(elec_node_1, "injection_port"), PortRef(electrolyzer, "input_port1")
    )
    system.connect(
        PortRef(electric_prod_2, "injection_port"),
        PortRef(elec_node_2, "injection_port"),
    )
    system.connect(
        PortRef(elec_node_2, "injection_port"), PortRef(electrolyzer, "input_port2")
    )
    system.connect(
        PortRef(electrolyzer, "output_port"), PortRef(gaz_node, "injection_port")
    )
    system.connect(
        PortRef(gaz_node, "injection_port"), PortRef(gaz_demand, "injection_port")
    )
    system.connect(
        PortRef(gaz_prod, "injection_port"), PortRef(gaz_node, "injection_port")
    )

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, 1990)

    df = SimulationTableBuilder().build(problem)
    assert math.isclose(
        df.component("ep1").output("generation").value(time_index=0, scenario_index=0),
        70,
    )
    assert math.isclose(
        df.component("ep2").output("generation").value(time_index=0, scenario_index=0),
        42,
    )
    assert math.isclose(
        df.component("gp").output("generation").value(time_index=0, scenario_index=0),
        30,
    )


def test_electrolyzer_n_inputs_3(
    lib_dict: dict[str, Library], lib_dict_sc: dict[str, Library]
) -> None:
    """
    Test with a consumption_electrolyzer with two inputs

    ep1 = electric production 1
    ep2 = electric production 2
    ez = electrolyzer
    gp = gaz production

    total gaz production = (flow_ep1 + flow_ep2) * alpha_ez + flow_gp

    The result is different since we only have one alpha at 0.7
    """

    gen_model = lib_dict["basic"].models["basic.generator"]
    node_model = lib_dict["basic"].models["basic.node"]
    convertor_model = lib_dict_sc["basic"].models["basic.convertor"]
    demand_model = lib_dict["basic"].models["basic.demand"]
    decompose_flow_model = lib_dict_sc["basic"].models[
        "basic.decompose_1_flow_into_2_flow"
    ]

    elec_node_1 = Component(model=node_model, id="e1")
    elec_node_2 = Component(model=node_model, id="e2")
    gaz_node = Component(model=node_model, id="g")

    electric_prod_1 = create_component(model=gen_model, id="ep1")
    electric_prod_2 = create_component(model=gen_model, id="ep2")

    gaz_prod = create_component(model=gen_model, id="gp")
    gaz_demand = create_component(model=demand_model, id="gd")

    electrolyzer = create_component(model=convertor_model, id="ez")
    consumption_electrolyzer = create_component(model=decompose_flow_model, id="ce")

    database = DataBase()

    database.add_data("ez", "alpha", ConstantData(0.7))

    database.add_data("ep1", "p_max", ConstantData(70))
    database.add_data("ep1", "cost", ConstantData(10))

    database.add_data("ep2", "p_max", ConstantData(80))
    database.add_data("ep2", "cost", ConstantData(20))

    database.add_data("gd", "demand", ConstantData(100))
    database.add_data("gp", "p_max", ConstantData(30))
    database.add_data("gp", "cost", ConstantData(15))

    system = System("test")
    system.add_component(elec_node_1)
    system.add_component(elec_node_2)
    system.add_component(gaz_node)
    system.add_component(electric_prod_1)
    system.add_component(electric_prod_2)
    system.add_component(gaz_prod)
    system.add_component(gaz_demand)
    system.add_component(electrolyzer)
    system.add_component(consumption_electrolyzer)

    system.connect(
        PortRef(electric_prod_1, "injection_port"),
        PortRef(elec_node_1, "injection_port"),
    )
    system.connect(
        PortRef(elec_node_1, "injection_port"),
        PortRef(consumption_electrolyzer, "input_port1"),
    )
    system.connect(
        PortRef(electric_prod_2, "injection_port"),
        PortRef(elec_node_2, "injection_port"),
    )
    system.connect(
        PortRef(elec_node_2, "injection_port"),
        PortRef(consumption_electrolyzer, "input_port2"),
    )
    system.connect(
        PortRef(consumption_electrolyzer, "output_port"),
        PortRef(electrolyzer, "input_port"),
    )
    system.connect(
        PortRef(electrolyzer, "output_port"), PortRef(gaz_node, "injection_port")
    )
    system.connect(
        PortRef(gaz_node, "injection_port"), PortRef(gaz_demand, "injection_port")
    )
    system.connect(
        PortRef(gaz_prod, "injection_port"), PortRef(gaz_node, "injection_port")
    )

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, 1750)

    df = SimulationTableBuilder().build(problem)
    assert math.isclose(
        df.component("ep1").output("generation").value(time_index=0, scenario_index=0),
        70,
    )
    assert math.isclose(
        df.component("ep2").output("generation").value(time_index=0, scenario_index=0),
        30,
    )
    assert math.isclose(
        df.component("gp").output("generation").value(time_index=0, scenario_index=0),
        30,
    )


def test_electrolyzer_n_inputs_4(
    lib_dict: dict[str, Library], lib_dict_sc: dict[str, Library]
) -> None:
    """
    Test with one electrolyzer with one input that takes every inputs

    ep1 = electric production 1
    ep2 = electric production 2
    ez = electrolyzer
    gp = gaz production

    total gaz production = (flow_ep1 + flow_ep2) * alpha_ez + flow_gp

    same as test 3, the result is different than the first two since we only have one alpha at 0.7
    """

    gen_model = lib_dict["basic"].models["basic.generator"]
    node_model = lib_dict["basic"].models["basic.node"]
    node_mod_model = lib_dict_sc["basic"].models["basic.node_mod"]
    convertor_model = lib_dict_sc["basic"].models["basic.convertor_receive_in"]
    demand_model = lib_dict["basic"].models["basic.demand"]

    elec_node_1 = Component(model=node_mod_model, id="e1")
    elec_node_2 = Component(model=node_mod_model, id="e2")
    gaz_node = Component(model=node_model, id="g")

    electric_prod_1 = create_component(model=gen_model, id="ep1")
    electric_prod_2 = create_component(model=gen_model, id="ep2")

    gaz_prod = create_component(model=gen_model, id="gp")
    gaz_demand = create_component(model=demand_model, id="gd")

    electrolyzer = create_component(model=convertor_model, id="ez")

    database = DataBase()

    database.add_data("ez", "alpha", ConstantData(0.7))

    database.add_data("ep1", "p_max", ConstantData(70))
    database.add_data("ep1", "cost", ConstantData(10))

    database.add_data("ep2", "p_max", ConstantData(80))
    database.add_data("ep2", "cost", ConstantData(20))

    database.add_data("gd", "demand", ConstantData(100))
    database.add_data("gp", "p_max", ConstantData(30))
    database.add_data("gp", "cost", ConstantData(15))

    system = System("test")
    system.add_component(elec_node_1)
    system.add_component(elec_node_2)
    system.add_component(gaz_node)
    system.add_component(electric_prod_1)
    system.add_component(electric_prod_2)
    system.add_component(gaz_prod)
    system.add_component(gaz_demand)
    system.add_component(electrolyzer)

    system.connect(
        PortRef(electric_prod_1, "injection_port"),
        PortRef(elec_node_1, "injection_port_n"),
    )
    system.connect(
        PortRef(elec_node_1, "injection_port_e"), PortRef(electrolyzer, "input_port")
    )
    system.connect(
        PortRef(electric_prod_2, "injection_port"),
        PortRef(elec_node_2, "injection_port_n"),
    )
    system.connect(
        PortRef(elec_node_2, "injection_port_e"), PortRef(electrolyzer, "input_port")
    )
    system.connect(
        PortRef(electrolyzer, "output_port"), PortRef(gaz_node, "injection_port")
    )
    system.connect(
        PortRef(gaz_node, "injection_port"), PortRef(gaz_demand, "injection_port")
    )
    system.connect(
        PortRef(gaz_prod, "injection_port"), PortRef(gaz_node, "injection_port")
    )

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"

    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, 1750)

    df = SimulationTableBuilder().build(problem)
    assert math.isclose(
        df.component("ep1").output("generation").value(time_index=0, scenario_index=0),
        70,
    )
    assert math.isclose(
        df.component("ep2").output("generation").value(time_index=0, scenario_index=0),
        30,
    )
    assert math.isclose(
        df.component("gp").output("generation").value(time_index=0, scenario_index=0),
        30,
    )
