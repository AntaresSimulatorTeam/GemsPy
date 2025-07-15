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

from typing import List

import pandas as pd
import pytest

from gems.simulation import TimeBlock, build_problem
from gems.study import ConstantData, DataBase, Network, Node, PortRef, create_component
from gems.study.data import TimeScenarioSeriesData
from tests.e2e.functional.libs.standard import (
    DEMAND_MODEL,
    NODE_BALANCE_MODEL,
    THERMAL_CLUSTER_MODEL_DHD,
    THERMAL_CLUSTER_MODEL_HD,
)


def generate_scalar_matrix_data(
    value: float, horizon: int, scenarios: int
) -> TimeScenarioSeriesData:
    data = pd.DataFrame(index=range(horizon), columns=range(scenarios))

    data.fillna(value, inplace=True)

    return TimeScenarioSeriesData(time_scenario_series=data)


@pytest.fixture
def horizon() -> int:
    return 2


@pytest.fixture
def scenarios() -> int:
    return 2


@pytest.fixture
def database(horizon: int, scenarios: int) -> DataBase:
    database = DataBase()

    database.add_data(
        "D", "demand", generate_scalar_matrix_data(500, horizon, scenarios)
    )

    database.add_data(
        "BASE", "nb_failures", generate_scalar_matrix_data(1, horizon, scenarios)
    )
    database.add_data(
        "SEMIBASE", "nb_failures", generate_scalar_matrix_data(1, horizon, scenarios)
    )
    database.add_data(
        "PEAK", "nb_failures", generate_scalar_matrix_data(1, horizon, scenarios)
    )

    database.add_data("BASE", "p_max", ConstantData(250))
    database.add_data("BASE", "p_min", ConstantData(100))
    database.add_data("BASE", "cost", ConstantData(30))
    database.add_data("BASE", "d_min_up", ConstantData(5))
    database.add_data("BASE", "d_min_down", ConstantData(5))
    database.add_data("BASE", "nb_units_max", ConstantData(1))

    database.add_data("SEMIBASE", "p_max", ConstantData(250))
    database.add_data("SEMIBASE", "p_min", ConstantData(100))
    database.add_data("SEMIBASE", "cost", ConstantData(50))
    database.add_data("SEMIBASE", "d_min_up", ConstantData(3))
    database.add_data("SEMIBASE", "d_min_down", ConstantData(3))
    database.add_data("SEMIBASE", "nb_units_max", ConstantData(1))

    database.add_data("PEAK", "p_max", ConstantData(100))
    database.add_data("PEAK", "p_min", ConstantData(0))
    database.add_data("PEAK", "cost", ConstantData(100))
    database.add_data("PEAK", "d_min_up", ConstantData(1))
    database.add_data("PEAK", "d_min_down", ConstantData(1))
    database.add_data("PEAK", "nb_units_max", ConstantData(1))

    return database


@pytest.fixture
def time_blocks(horizon: int) -> List[TimeBlock]:
    time_blocks = [TimeBlock(1, list(range(horizon)))]
    return time_blocks


def test_stochastic_model_with_HD_for_thermal_startup(
    horizon: int,
    scenarios: int,
    time_blocks: List[TimeBlock],
    database: DataBase,
) -> None:
    """
    Small stochastic model with one node, thermal generation, storage and demand. The start-up decisions for thermal plants are in Hazard-Decision. All other decisions are also in Hazard-Decision.

    Randomness only comes from the availability of thermal plants, demand is fixed.
    """

    node = Node(model=NODE_BALANCE_MODEL, id="N")
    demand = create_component(model=DEMAND_MODEL, id="D")

    base = create_component(model=THERMAL_CLUSTER_MODEL_HD, id="BASE")

    semibase = create_component(model=THERMAL_CLUSTER_MODEL_HD, id="SEMIBASE")

    peak = create_component(model=THERMAL_CLUSTER_MODEL_HD, id="PEAK")

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(base)
    network.add_component(semibase)
    network.add_component(peak)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(base, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(semibase, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(peak, "balance_port"), PortRef(node, "balance_port"))

    for block in time_blocks:  # TODO : To manage blocks simply for now
        problem = build_problem(network, database, block, scenarios)
        status = problem.solver.Solve()

        assert (
            status == problem.solver.OPTIMAL
        )  # Tester qu'on trouve bien la solution optimale

        # Generation, nb_on, nb_start, nb_stop for each of 3 thermal clusters
        nb_anticipative_time_varying_var = 4 * 3

        # No non anticipative variable
        nb_non_anticipative_time_varying_var = 0

        # None
        nb_scenario_constant_constraint = 0

        # Balance constraint + For each 3 thermal clusters : Max generation, Min generation, NODU balance, Min up time, "Min down time
        nb_scenario_varying_constraint = 1 + 5 * 3

        # TODO this test should pass with the next port implementation
        assert (
            problem.solver.NumVariables()
            == nb_anticipative_time_varying_var * horizon * scenarios
            + nb_non_anticipative_time_varying_var * horizon
        )
        assert (
            problem.solver.NumConstraints()
            == nb_scenario_constant_constraint * horizon
            + nb_scenario_varying_constraint * horizon * scenarios
        )


def test_stochastic_model_with_DH_for_thermal_startup(
    horizon: int,
    scenarios: int,
    database: DataBase,
) -> None:
    """
    Small stochastic model with one node, thermal generation, storage and demand. The start-up decisions for thermal plants are in Decision-Hazard. All other decisions are in Hazard-Decision.

    Randomness only comes from the availability of thermal plants, demand is fixed.
    """

    time_blocks = [TimeBlock(1, list(range(horizon)))]

    node = Node(model=NODE_BALANCE_MODEL, id="N")
    demand = create_component(model=DEMAND_MODEL, id="D")

    base = create_component(model=THERMAL_CLUSTER_MODEL_DHD, id="BASE")

    semibase = create_component(model=THERMAL_CLUSTER_MODEL_DHD, id="SEMIBASE")

    peak = create_component(model=THERMAL_CLUSTER_MODEL_DHD, id="PEAK")

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(base)
    network.add_component(semibase)
    network.add_component(peak)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(base, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(semibase, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(peak, "balance_port"), PortRef(node, "balance_port"))

    for block in time_blocks:  # TODO : To manage blocks simply for now
        problem = build_problem(network, database, block, scenarios)
        status = problem.solver.Solve()

        assert (
            status == problem.solver.OPTIMAL
        )  # Tester qu'on trouve bien la solution optimale

        # Generation for each of 3 thermal clusters
        nb_anticipative_time_varying_var = 3

        # For each 3 thermal clusters : nb_on, nb_start, nb_stop
        nb_non_anticipative_time_varying_var = 3 * 3

        # For each 3 thermal clusters : NODU balance, Min up time, "Min down time
        nb_scenario_constant_constraint = 3 * 3

        # Balance constraint + For each 3 thermal clusters : Max generation, Min generation
        nb_scenario_varying_constraint = 1 + 2 * 3

        assert (
            problem.solver.NumVariables()
            == nb_anticipative_time_varying_var * horizon * scenarios
            + nb_non_anticipative_time_varying_var * horizon
        )
        assert (
            problem.solver.NumConstraints()
            == nb_scenario_constant_constraint * horizon
            + nb_scenario_varying_constraint * horizon * scenarios
        )
