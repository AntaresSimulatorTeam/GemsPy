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
This module contains end-to-end functional tests for systems built by:
- Reading the model library from a YAML file,
- Building the network objet directly in Python.

The tests validate various scenarios involving energy balance, generation, spillage, and demand across nodes and networks.

Tests included:
1. `test_basic_balance`: Verifies energy balance on a single node with fixed demand and generation for one timestep.
2. `test_link`: Tests energy balance across two nodes connected by a link with fixed demand and generation for one timestep.
3. `test_stacking_generation`: Validates energy balance on a single node with fixed demand and two generators having different costs for one timestep.
4. `test_spillage`: Ensures proper handling of spillage when generation exceeds demand on a single node for one timestep.
5. `test_min_up_down_times`: Simulates a scenario with minimum up/down times for a thermal generator over three timesteps, ensuring constraints are satisfied.
6. `test_changing_demand`: Tests energy balance on a single node with changing demand over three timesteps.
7. `test_min_up_down_times_2`: Similar to `test_min_up_down_times`, but with different minimum up/down time constraints for a thermal generator over three timesteps.

Each test builds a network of nodes and components, defines a database of
parameters, and solves the problem. Assertions are made to ensure the solver's results meet expected outcomes.
"""

import pandas as pd
import pytest

from gems.model.library import Library
from gems.simulation import (
    BlockBorderManagement,
    OutputValues,
    TimeBlock,
    build_problem,
)
from gems.study import (
    ConstantData,
    DataBase,
    Network,
    Node,
    PortRef,
    TimeScenarioSeriesData,
    create_component,
)

# TODO : Use fixtures for models and components used several times to simplify this test file


def test_basic_balance(lib_dict: dict[str, Library]) -> None:
    """
    Balance on one node with one fixed demand and one generation, on 1 timestep.
    """

    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(30))

    node_model = lib_dict["basic"].models["node"]
    demand_model = lib_dict["basic"].models["demand"]
    production_model = lib_dict["basic"].models["production"]

    node = Node(model=node_model, id="N")
    demand = create_component(
        model=demand_model,
        id="D",
    )

    gen = create_component(
        model=production_model,
        id="G",
    )

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(gen)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(gen, "balance_port"), PortRef(node, "balance_port"))

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 3000


def test_link(lib_dict: dict[str, Library]) -> None:
    """
    Balance on one node with one fixed demand and one generation, on 1 timestep.
    """

    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(35))

    database.add_data("L", "f_max", ConstantData(150))

    node_model = lib_dict["basic"].models["node"]
    demand_model = lib_dict["basic"].models["demand"]
    production_model = lib_dict["basic"].models["production"]
    link_model = lib_dict["basic"].models["link"]

    node1 = Node(model=node_model, id="1")
    node2 = Node(model=node_model, id="2")
    demand = create_component(
        model=demand_model,
        id="D",
    )
    gen = create_component(
        model=production_model,
        id="G",
    )
    link = create_component(
        model=link_model,
        id="L",
    )

    network = Network("test")
    network.add_node(node1)
    network.add_node(node2)
    network.add_component(demand)
    network.add_component(gen)
    network.add_component(link)
    network.connect(PortRef(demand, "balance_port"), PortRef(node1, "balance_port"))
    network.connect(PortRef(gen, "balance_port"), PortRef(node2, "balance_port"))
    network.connect(PortRef(link, "in_port"), PortRef(node1, "balance_port"))
    network.connect(PortRef(link, "out_port"), PortRef(node2, "balance_port"))

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 3500

    for variable in problem.solver.variables():
        if "balance_port_from" in variable.name():
            assert variable.solution_value() == 100
        if "balance_port_to" in variable.name():
            assert variable.solution_value() == -100


def test_stacking_generation(lib_dict: dict[str, Library]) -> None:
    """
    Balance on one node with one fixed demand and 2 generations with different costs, on 1 timestep.
    """

    database = DataBase()
    database.add_data("D", "demand", ConstantData(150))

    database.add_data("G1", "p_max", ConstantData(100))
    database.add_data("G1", "cost", ConstantData(30))

    database.add_data("G2", "p_max", ConstantData(100))
    database.add_data("G2", "cost", ConstantData(50))

    node_model = lib_dict["basic"].models["node"]
    demand_model = lib_dict["basic"].models["demand"]
    production_model = lib_dict["basic"].models["production"]

    node1 = Node(model=node_model, id="1")

    demand = create_component(
        model=demand_model,
        id="D",
    )

    gen1 = create_component(
        model=production_model,
        id="G1",
    )

    gen2 = create_component(
        model=production_model,
        id="G2",
    )

    network = Network("test")
    network.add_node(node1)
    network.add_component(demand)
    network.add_component(gen1)
    network.add_component(gen2)
    network.connect(PortRef(demand, "balance_port"), PortRef(node1, "balance_port"))
    network.connect(PortRef(gen1, "balance_port"), PortRef(node1, "balance_port"))
    network.connect(PortRef(gen2, "balance_port"), PortRef(node1, "balance_port"))

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 30 * 100 + 50 * 50


def test_spillage(lib_dict: dict[str, Library]) -> None:
    """
    Balance on one node with one fixed demand and 1 generation higher than demand and 1 timestep .
    """

    database = DataBase()
    database.add_data("D", "demand", ConstantData(150))
    database.add_data("S", "cost", ConstantData(10))

    database.add_data("G1", "p_max", ConstantData(300))
    database.add_data("G1", "p_min", ConstantData(200))
    database.add_data("G1", "cost", ConstantData(30))

    node_model = lib_dict["basic"].models["node"]
    demand_model = lib_dict["basic"].models["demand"]
    production_with_min_model = lib_dict["basic"].models["production_with_min"]
    spillage_model = lib_dict["basic"].models["spillage"]

    node = Node(model=node_model, id="1")
    spillage = create_component(model=spillage_model, id="S")
    demand = create_component(model=demand_model, id="D")

    gen1 = create_component(model=production_with_min_model, id="G1")

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(gen1)
    network.add_component(spillage)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(gen1, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(spillage, "balance_port"), PortRef(node, "balance_port"))

    problem = build_problem(network, database, TimeBlock(0, [1]), 1)
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 30 * 200 + 50 * 10


def test_min_up_down_times(lib_dict: dict[str, Library]) -> None:
    """
    Model on 3 time steps with one thermal generation and one demand on a single node.
        - Demand is the following time series : [500 MW, 0, 0]
        - Thermal generation is characterized with:
            - P_min = 100 MW
            - P_max = 500 MW
            - Min up/down time = 3
            - Generation cost = 100€ / MWh
        - Unsupplied energy = 3000 €/MWh
        - Spillage = 10 €/MWh

    The optimal solution consists is turning on the thermal plant, which must then stay on for the 3 timesteps and producing [500, 100, 100] to satisfy P_min constraints.

    The optimal cost is then :
          500 x 100 (prod step 1)
        + 100 x 100 (prod step 2)
        + 100 x 100 (prod step 3)
        + 100 x 10 (spillage step 2)
        + 100 x 10 (spillage step 3)
        = 72 000
    """

    database = DataBase()

    database.add_data("G", "p_max", ConstantData(500))
    database.add_data("G", "p_min", ConstantData(100))
    database.add_data("G", "cost", ConstantData(100))
    database.add_data("G", "d_min_up", ConstantData(3))
    database.add_data("G", "d_min_down", ConstantData(3))
    database.add_data("G", "nb_units_max", ConstantData(1))
    database.add_data("G", "nb_failures", ConstantData(0))

    database.add_data("U", "cost", ConstantData(3000))
    database.add_data("S", "cost", ConstantData(10))

    demand_data = pd.DataFrame(
        [
            [500],
            [0],
            [0],
        ],
        index=[0, 1, 2],
        columns=[0],
    )
    demand_time_scenario_series = TimeScenarioSeriesData(demand_data)
    database.add_data("D", "demand", demand_time_scenario_series)

    time_block = TimeBlock(1, [0, 1, 2])
    scenarios = 1

    node_model = lib_dict["basic"].models["node"]
    demand_model = lib_dict["basic"].models["demand"]
    spillage_model = lib_dict["basic"].models["spillage"]
    unsuplied_model = lib_dict["basic"].models["unsuplied"]
    thermal_cluster = lib_dict["basic"].models["thermal_cluster"]

    node = Node(model=node_model, id="1")
    demand = create_component(model=demand_model, id="D")

    gen = create_component(model=thermal_cluster, id="G")

    spillage = create_component(model=spillage_model, id="S")

    unsupplied_energy = create_component(model=unsuplied_model, id="U")

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(gen)
    network.add_component(spillage)
    network.add_component(unsupplied_energy)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(gen, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(spillage, "balance_port"), PortRef(node, "balance_port"))
    network.connect(
        PortRef(unsupplied_energy, "balance_port"), PortRef(node, "balance_port")
    )

    problem = build_problem(
        network,
        database,
        time_block,
        scenarios,
        border_management=BlockBorderManagement.CYCLE,
    )
    status = problem.solver.Solve()

    print(OutputValues(problem).component("G").var("nb_units_on").value)

    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == pytest.approx(72000, abs=0.01)


def test_changing_demand(lib_dict: dict[str, Library]) -> None:
    """
    Model on 3 time steps simple production, demand
        - P_max = 500 MW
        - Generation cost = 100€ / MWh
    """

    database = DataBase()

    database.add_data("G", "p_max", ConstantData(500))
    database.add_data("G", "cost", ConstantData(100))

    demand_data = pd.DataFrame(
        [
            [300],
            [100],
            [0],
        ],
        index=[0, 1, 2],
        columns=[0],
    )
    demand_time_scenario_series = TimeScenarioSeriesData(demand_data)
    database.add_data("D", "demand", demand_time_scenario_series)

    time_block = TimeBlock(1, [0, 1, 2])
    scenarios = 1

    node_model = lib_dict["basic"].models["node"]
    demand_model = lib_dict["basic"].models["demand"]
    production_model = lib_dict["basic"].models["production"]

    node = Node(model=node_model, id="1")
    demand = create_component(model=demand_model, id="D")

    prod = create_component(model=production_model, id="G")

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(prod)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(prod, "balance_port"), PortRef(node, "balance_port"))

    problem = build_problem(
        network,
        database,
        time_block,
        scenarios,
        border_management=BlockBorderManagement.CYCLE,
    )
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 40000


def test_min_up_down_times_2(lib_dict: dict[str, Library]) -> None:
    """
    Model on 3 time steps with one thermal generation and one demand on a single node.
        - Demand is the following time series : [500 MW, 0, 0]
        - Thermal generation is characterized with:
            - P_min = 100 MW
            - P_max = 500 MW
            - Min up/down time = 2
            - Generation cost = 100€ / MWh
        - Unsupplied energy = 3000 €/MWh
        - Spillage = 10 €/MWh

    The optimal solution consists is turning on the thermal plant, which must then stay on for the 3 timesteps and producing [500, 100, 100] to satisfy P_min constraints.

    The optimal cost is then :
          500 x 100 (prod step 1)
        + 100 x 100 (prod step 2)
        + 0   x 100 (prod step 3)
        + 100 x 10  (spillage step 2)
        + 0   x 10  (spillage step 3)
        = 61 000
    """

    database = DataBase()

    database.add_data("G", "p_max", ConstantData(500))
    database.add_data("G", "p_min", ConstantData(100))
    database.add_data("G", "cost", ConstantData(100))
    database.add_data("G", "d_min_up", ConstantData(2))
    database.add_data("G", "d_min_down", ConstantData(1))
    database.add_data("G", "nb_units_max", ConstantData(1))
    database.add_data("G", "nb_failures", ConstantData(0))

    database.add_data("U", "cost", ConstantData(3000))
    database.add_data("S", "cost", ConstantData(10))

    demand_data = pd.DataFrame(
        [
            [500],
            [0],
            [0],
        ],
        index=[0, 1, 2],
        columns=[0],
    )
    demand_time_scenario_series = TimeScenarioSeriesData(demand_data)
    database.add_data("D", "demand", demand_time_scenario_series)

    time_block = TimeBlock(1, [0, 1, 2])
    scenarios = 1

    node_model = lib_dict["basic"].models["node"]
    demand_model = lib_dict["basic"].models["demand"]
    spillage_model = lib_dict["basic"].models["spillage"]
    unsuplied_model = lib_dict["basic"].models["unsuplied"]
    thermal_cluster = lib_dict["basic"].models["thermal_cluster"]

    node = Node(model=node_model, id="1")
    demand = create_component(model=demand_model, id="D")

    gen = create_component(model=thermal_cluster, id="G")

    spillage = create_component(model=spillage_model, id="S")

    unsupplied_energy = create_component(model=unsuplied_model, id="U")

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(gen)
    network.add_component(spillage)
    network.add_component(unsupplied_energy)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(gen, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(spillage, "balance_port"), PortRef(node, "balance_port"))
    network.connect(
        PortRef(unsupplied_energy, "balance_port"), PortRef(node, "balance_port")
    )

    problem = build_problem(
        network,
        database,
        time_block,
        scenarios,
        border_management=BlockBorderManagement.CYCLE,
    )
    status = problem.solver.Solve()

    print(problem.solver.ExportModelAsMpsFormat(fixed_format=False, obfuscated=False))
    print(OutputValues(problem).component("G").var("nb_units_on").value)

    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == pytest.approx(61000)
