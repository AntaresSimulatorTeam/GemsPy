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

from typing import Dict, List

import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pandas as pd
import pytest

from andromede.libs.standard import (
    DEMAND_MODEL,
    NODE_BALANCE_MODEL,
    SPILLAGE_MODEL,
    UNSUPPLIED_ENERGY_MODEL,
)
from andromede.simulation import BlockBorderManagement, TimeBlock, build_problem
from andromede.simulation.optimization import OptimizationProblem
from andromede.study import (
    ConstantData,
    DataBase,
    Network,
    Node,
    PortRef,
    TimeScenarioSeriesData,
    create_component,
)
from andromede.study.data import AbstractDataStructure
from tests.functional.libs.lib_thermal_heuristic import THERMAL_CLUSTER_MODEL_MILP


def set_solver_parameters_to_antares_config() -> pywraplp.MPSolverParameters:
    parameters = pywraplp.MPSolverParameters()
    parameters.SetIntegerParam(parameters.PRESOLVE, parameters.PRESOLVE_OFF)
    parameters.SetIntegerParam(parameters.SCALING, 0)
    parameters.SetDoubleParam(parameters.PRIMAL_TOLERANCE, 1e-7)
    parameters.SetDoubleParam(parameters.DUAL_TOLERANCE, 1e-7)
    parameters.SetDoubleParam(parameters.RELATIVE_MIP_GAP, 0.0001)
    return parameters


number_hours = 168
weeks = 2
scenarios = [0, 1]
expected_cost = [[78933742, 102109698], [17472101, 17424769]]
expected_cost_more_restrictive_parameters = [
    [78933742, 102103588],
    [17472101, 17424769],
]

""" This test compares the POC and Antares with MILP solver and we expect to have the same result. 
Weekly problems are solved scenario per scenario and for all scnearios at the same time."""


def test_one_problem_per_scenario() -> None:
    """Resolve a simple problem with milp solver and with the same parameters as Antares. If the problem is solved scenario per scenario the result is the same as Antares."""

    solver = pywraplp.Solver.CreateSolver("XPRESS")
    if solver:
        for scenario in scenarios:
            for week in range(weeks):
                problem = create_complex_problem(week=week, scenarios=[scenario])

                parameters = set_solver_parameters_to_antares_config()

                status = problem.solver.Solve(parameters)
                assert status == problem.solver.OPTIMAL

                assert problem.solver.Objective().Value() == pytest.approx(
                    expected_cost[scenario][week]
                )


def test_one_problem_per_scenario_with_different_parameters() -> None:
    """Resolve the same problem as above with more restrictive parameters. If the problem is solved scenario per scenario the result is better than above."""

    solver = pywraplp.Solver.CreateSolver("XPRESS")
    if solver:
        for scenario in scenarios:
            for week in range(weeks):
                problem = create_complex_problem(week=week, scenarios=[scenario])

                parameters = set_solver_parameters_to_antares_config()
                parameters.SetDoubleParam(parameters.RELATIVE_MIP_GAP, 0.000001)

                status = problem.solver.Solve(parameters)
                assert status == problem.solver.OPTIMAL

                assert problem.solver.Objective().Value() == pytest.approx(
                    expected_cost_more_restrictive_parameters[scenario][week]
                )


def test_one_problem_for_all_scenarios() -> None:
    """Resolve the same problem as above with same parameters as Antares. If the problem is solved for all scenarios at the same time, the result is the same as solving the problem one by one."""

    solver = pywraplp.Solver.CreateSolver("XPRESS")
    if solver:
        for week in range(weeks):
            problem = create_complex_problem(week=week, scenarios=scenarios)

            parameters = set_solver_parameters_to_antares_config()

            status = problem.solver.Solve(parameters)
            assert status == problem.solver.OPTIMAL

            assert problem.solver.Objective().Value() == pytest.approx(
                sum([expected_cost[s][week] for s in scenarios]) / len(scenarios)
            )


def test_one_problem_for_all_scenarios_with_different_parameters() -> None:
    """Resolve the same problem as above with more restrictive parameters. If the problem is solved for all scenarios at the same time, the result is the same than solving the problem one by one. All those tests show that solver parameters and solving scenario at one or one by one are important factors to take into account."""

    solver = pywraplp.Solver.CreateSolver("XPRESS")
    if solver:
        for week in range(weeks):
            problem = create_complex_problem(week=week, scenarios=scenarios)

            parameters = set_solver_parameters_to_antares_config()
            parameters.SetDoubleParam(parameters.RELATIVE_MIP_GAP, 0.000001)

            status = problem.solver.Solve(parameters)
            assert status == problem.solver.OPTIMAL

            assert problem.solver.Objective().Value() == pytest.approx(
                sum(
                    [
                        expected_cost_more_restrictive_parameters[s][week]
                        for s in scenarios
                    ]
                )
                / len(scenarios)
            )


def create_complex_problem(
    week: int,
    scenarios: List[int],
) -> OptimizationProblem:
    database = generate_database(week=week, scenarios=scenarios)

    time_block = TimeBlock(1, [i for i in range(number_hours)])

    node = Node(model=NODE_BALANCE_MODEL, id="1")
    demand = create_component(model=DEMAND_MODEL, id="D")

    gen1 = create_component(model=THERMAL_CLUSTER_MODEL_MILP, id="G1")
    gen2 = create_component(model=THERMAL_CLUSTER_MODEL_MILP, id="G2")
    gen3 = create_component(model=THERMAL_CLUSTER_MODEL_MILP, id="G3")

    spillage = create_component(model=SPILLAGE_MODEL, id="S")

    unsupplied_energy = create_component(model=UNSUPPLIED_ENERGY_MODEL, id="U")

    network = Network("test")
    network.add_node(node)
    network.add_component(demand)
    network.add_component(gen1)
    network.add_component(gen2)
    network.add_component(gen3)
    network.add_component(spillage)
    network.add_component(unsupplied_energy)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(gen1, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(gen2, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(gen3, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(spillage, "balance_port"), PortRef(node, "balance_port"))
    network.connect(
        PortRef(unsupplied_energy, "balance_port"), PortRef(node, "balance_port")
    )

    problem = build_problem(
        network,
        database,
        time_block,
        len(scenarios),
        border_management=BlockBorderManagement.CYCLE,
        solver_id="XPRESS",
    )

    return problem


def generate_database(
    week: int,
    scenarios: List[int],
) -> DataBase:
    database = DataBase()

    failures_1 = pd.DataFrame(
        np.transpose(
            [
                np.repeat(get_failures_for_cluster1(week, scenario), 24)
                for scenario in scenarios
            ]
        ),
        index=[i for i in range(number_hours)],
        columns=list(range(len(scenarios))),
    )

    database.add_data("G1", "p_max", ConstantData(410))
    database.add_data("G1", "p_min", ConstantData(180))
    database.add_data("G1", "cost", ConstantData(96))
    database.add_data("G1", "startup_cost", ConstantData(100500))
    database.add_data("G1", "fixed_cost", ConstantData(1))
    database.add_data("G1", "d_min_up", ConstantData(8))
    database.add_data("G1", "d_min_down", ConstantData(8))
    database.add_data("G1", "nb_units_min", ConstantData(0))
    database.add_data("G1", "nb_units_max", ConstantData(1))
    database.add_data("G1", "nb_units_max_min_down_time", ConstantData(1))
    database.add_data("G1", "max_generating", TimeScenarioSeriesData(failures_1))
    database.add_data("G1", "min_generating", ConstantData(0))

    database.add_data("G2", "p_max", ConstantData(90))
    database.add_data("G2", "p_min", ConstantData(60))
    database.add_data("G2", "cost", ConstantData(137))
    database.add_data("G2", "startup_cost", ConstantData(24500))
    database.add_data("G2", "fixed_cost", ConstantData(1))
    database.add_data("G2", "d_min_up", ConstantData(11))
    database.add_data("G2", "d_min_down", ConstantData(11))
    database.add_data("G2", "nb_units_min", ConstantData(0))
    database.add_data("G2", "nb_units_max", ConstantData(3))
    database.add_data("G2", "nb_units_max_min_down_time", ConstantData(3))
    database.add_data("G2", "max_generating", ConstantData(270))
    database.add_data("G2", "min_generating", ConstantData(0))

    failures_3 = pd.DataFrame(
        np.transpose(
            [
                np.repeat(get_failures_for_cluster3(week, scenario), 24)
                for scenario in scenarios
            ]
        ),
        index=[i for i in range(number_hours)],
        columns=list(range(len(scenarios))),
    )

    database.add_data("G3", "p_max", ConstantData(275))
    database.add_data("G3", "p_min", ConstantData(150))
    database.add_data("G3", "cost", ConstantData(107))
    database.add_data("G3", "startup_cost", ConstantData(69500))
    database.add_data("G3", "fixed_cost", ConstantData(1))
    database.add_data("G3", "d_min_up", ConstantData(9))
    database.add_data("G3", "d_min_down", ConstantData(9))
    database.add_data("G3", "nb_units_min", ConstantData(0))
    database.add_data("G3", "nb_units_max", ConstantData(4))
    database.add_data("G3", "nb_units_max_min_down_time", ConstantData(4))
    database.add_data("G3", "max_generating", TimeScenarioSeriesData(failures_3))
    database.add_data("G3", "min_generating", ConstantData(0))

    database.add_data("U", "cost", ConstantData(10000))
    database.add_data("S", "cost", ConstantData(1))

    for g in ["G1", "G2", "G3"]:
        database.add_data(g, "max_failure", ConstantData(0))

    output = {}
    for scenario in scenarios:
        output_file = open(
            "tests/functional/data/thermal_heuristic_three_clusters/milp/"
            + str(scenario)
            + "/values-hourly.txt",
            "r",
        )
        output[scenario] = output_file.readlines()

    demand_data = pd.DataFrame(
        data=np.transpose(
            [
                [
                    float(line.strip().split("\t")[10]) - 300
                    for line in output[scenario][168 * week + 7 : 168 * week + 7 + 168]
                ]
                for scenario in scenarios
            ]
        ),
        index=[i for i in range(number_hours)],
        columns=list(range(len(scenarios))),
    )

    demand_time_scenario_series = TimeScenarioSeriesData(demand_data)
    database.add_data("D", "demand", demand_time_scenario_series)
    return database


def get_failures_for_cluster3(week: int, scenario: int) -> List:
    if scenario == 0:
        if week == 0:
            failures_3 = [1100, 1100, 0, 1100, 1100, 1100, 1100]
        elif week == 1:
            failures_3 = [1100, 1100, 1100, 1100, 1100, 0, 1100]
    elif scenario == 1:
        failures_3 = [1100, 1100, 1100, 1100, 1100, 1100, 1100]

    return failures_3


def get_failures_for_cluster1(week: int, scenario: int) -> List:
    if scenario == 0:
        failures = [410, 410, 410, 410, 410, 410, 410]
    elif scenario == 1:
        failures = [410, 410, 410, 410, 0, 410, 410]

    return failures
