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

from math import ceil
from pathlib import Path
from typing import List

import ortools.linear_solver.pywraplp as pywraplp
import pytest

from andromede.libs.standard import BALANCE_PORT_TYPE
from andromede.simulation import OutputValues
from andromede.study.parsing import InputComponents
from andromede.thermal_heuristic.cluster_parameter import compute_slot_length
from andromede.thermal_heuristic.model import (
    AccurateModelBuilder,
    FastModelBuilder,
    HeuristicAccurateModelBuilder,
    HeuristicFastModelBuilder,
    Model,
)
from andromede.thermal_heuristic.problem import (
    BlockScenarioIndex,
    ThermalProblemBuilder,
    TimeScenarioHourParameter,
    get_database,
    get_network,
)
from tests.functional.conftest import ExpectedOutput, ExpectedOutputIndexes
from tests.functional.libs.lib_thermal_heuristic import THERMAL_CLUSTER_MODEL_MILP


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent.parent / "data/thermal_heuristic_two_clusters_low_load"


def test_milp_version(
    solver_parameters: pywraplp.MPSolverParameters,
    models: list[Model],
    data_path: Path,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """Solve weekly problem with two clusters and low residual load."""
    output_indexes = ExpectedOutputIndexes(
        idx_generation=4, idx_nodu=8, idx_spillage=10, idx_unsupplied=9
    )
    network = get_network(
        input_components,
        port_types=[BALANCE_PORT_TYPE],
        models=[THERMAL_CLUSTER_MODEL_MILP] + models,
    )
    database = get_database(
        input_components,
        data_path,
        fast=False,
        cluster=heuristic_components,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    thermal_problem_builder = ThermalProblemBuilder(
        network=network,
        database=database,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    for scenario in range(
        thermal_problem_builder.time_scenario_hour_parameter.scenario
    ):
        for week in range(thermal_problem_builder.time_scenario_hour_parameter.week):
            week_scenario_index = BlockScenarioIndex(week, scenario)
            resolution_step = thermal_problem_builder.main_resolution_step(
                week_scenario_index,
            )

            status = resolution_step.solver.Solve(solver_parameters)
            assert status == pywraplp.Solver.OPTIMAL

            expected_output = ExpectedOutput(
                mode="milp",
                index=week_scenario_index,
                dir_path=data_path,
                list_cluster=heuristic_components,
                output_idx=output_indexes,
            )
            expected_output.check_output_values(OutputValues(resolution_step))

            expected_cost = [[36036414]]
            assert resolution_step.solver.Objective().Value() == pytest.approx(
                expected_cost[scenario][week]
            )
            assert sum(
                OutputValues(resolution_step)
                .component("S")
                .var("spillage")
                .value[0]  # type:ignore
            ) == pytest.approx(15884)


def test_accurate_heuristic(
    solver_parameters: pywraplp.MPSolverParameters,
    models: list[Model],
    data_path: Path,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """
    Solve the same problem as before with the heuristic accurate of Antares. Spillage is bigger.
    """

    output_indexes = ExpectedOutputIndexes(
        idx_generation=4, idx_nodu=8, idx_spillage=11, idx_unsupplied=10
    )
    network = get_network(
        input_components,
        port_types=[BALANCE_PORT_TYPE],
        models=[AccurateModelBuilder(THERMAL_CLUSTER_MODEL_MILP).model] + models,
    )
    database = get_database(
        input_components,
        data_path,
        fast=False,
        cluster=heuristic_components,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    thermal_problem_builder = ThermalProblemBuilder(
        network=network,
        database=database,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    for scenario in range(
        thermal_problem_builder.time_scenario_hour_parameter.scenario
    ):
        for week in range(thermal_problem_builder.time_scenario_hour_parameter.week):
            week_scenario_index = BlockScenarioIndex(week, scenario)
            # First optimization
            resolution_step_1 = thermal_problem_builder.main_resolution_step(
                week_scenario_index,
            )

            status = resolution_step_1.solver.Solve(solver_parameters)
            assert status == pywraplp.Solver.OPTIMAL

            thermal_problem_builder.update_database_heuristic(
                OutputValues(resolution_step_1),
                week_scenario_index,
                heuristic_components,
                param_to_update="nb_units_min",
                var_to_read="nb_on",
                fn_to_apply=lambda x: ceil(round(x, 12)),
            )

            for g in heuristic_components:
                # Solve heuristic problem
                resolution_step_accurate_heuristic = (
                    thermal_problem_builder.heuristic_resolution_step(
                        week_scenario_index,
                        id_component=g,
                        model=HeuristicAccurateModelBuilder(
                            THERMAL_CLUSTER_MODEL_MILP
                        ).model,
                    )
                )

                status = resolution_step_accurate_heuristic.solver.Solve(
                    solver_parameters
                )
                assert status == pywraplp.Solver.OPTIMAL

                thermal_problem_builder.update_database_heuristic(
                    OutputValues(resolution_step_accurate_heuristic),
                    week_scenario_index,
                    [g],
                    param_to_update="nb_units_min",
                    var_to_read="nb_on",
                    fn_to_apply=lambda x: ceil(round(x, 12)),
                )

            # Second optimization with lower bound modified
            resolution_step_2 = thermal_problem_builder.main_resolution_step(
                week_scenario_index,
            )

            status = resolution_step_2.solver.Solve(solver_parameters)
            assert status == pywraplp.Solver.OPTIMAL

            expected_output = ExpectedOutput(
                mode="accurate",
                index=week_scenario_index,
                dir_path=data_path,
                list_cluster=heuristic_components,
                output_idx=output_indexes,
            )
            expected_output.check_output_values(OutputValues(resolution_step_2))

            expected_cost = [[36060641]]
            assert resolution_step_2.solver.Objective().Value() == pytest.approx(
                expected_cost[scenario][week]
            )
            assert sum(
                OutputValues(resolution_step_2)
                .component("S")
                .var("spillage")
                .value[0]  # type:ignore
            ) == pytest.approx(22191)


def test_fast_heuristic(
    solver_parameters: pywraplp.MPSolverParameters,
    models: list[Model],
    data_path: Path,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """
    Solve the same problem as before with the heuristic fast of Antares. Spillage is bigger.
    """
    output_indexes = ExpectedOutputIndexes(
        idx_generation=4, idx_nodu=8, idx_spillage=11, idx_unsupplied=10
    )
    network = get_network(
        input_components,
        port_types=[BALANCE_PORT_TYPE],
        models=[FastModelBuilder(THERMAL_CLUSTER_MODEL_MILP).model] + models,
    )
    database = get_database(
        input_components,
        data_path,
        fast=True,
        cluster=heuristic_components,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    thermal_problem_builder = ThermalProblemBuilder(
        network=network,
        database=database,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    for scenario in range(
        thermal_problem_builder.time_scenario_hour_parameter.scenario
    ):
        for week in range(thermal_problem_builder.time_scenario_hour_parameter.week):
            week_scenario_index = BlockScenarioIndex(week, scenario)
            # First optimization
            resolution_step_1 = thermal_problem_builder.main_resolution_step(
                week_scenario_index,
            )

            status = resolution_step_1.solver.Solve(solver_parameters)
            assert status == pywraplp.Solver.OPTIMAL

            thermal_problem_builder.update_database_heuristic(
                OutputValues(resolution_step_1),
                week_scenario_index,
                list_cluster_id=heuristic_components,
                var_to_read="generation",
                param_to_update="n_guide",
                fn_to_apply=lambda x, y: ceil(round(x / y, 12)),
                param_needed_to_compute=["p_max"],
            )

            for g in heuristic_components:  #
                resolution_step_heuristic = (
                    thermal_problem_builder.heuristic_resolution_step(
                        id_component=g,
                        index=week_scenario_index,
                        model=HeuristicFastModelBuilder(
                            thermal_problem_builder.time_scenario_hour_parameter.hour,
                            slot_length=compute_slot_length(
                                g, thermal_problem_builder.database
                            ),
                        ).model,
                    )
                )
                status = resolution_step_heuristic.solver.Solve()
                assert status == pywraplp.Solver.OPTIMAL

                thermal_problem_builder.update_database_heuristic(
                    OutputValues(resolution_step_heuristic),
                    week_scenario_index,
                    [g],
                    var_to_read="n",
                    param_to_update="min_generating",
                    fn_to_apply=lambda x, y, z: min(x * y, z),
                    param_needed_to_compute=["p_min", "max_generating"],
                )

            # Second optimization with lower bound modified
            resolution_step_2 = thermal_problem_builder.main_resolution_step(
                week_scenario_index
            )

            status = resolution_step_2.solver.Solve(solver_parameters)
            assert status == pywraplp.Solver.OPTIMAL

            expected_output = ExpectedOutput(
                mode="fast",
                index=week_scenario_index,
                dir_path=data_path,
                list_cluster=heuristic_components,
                output_idx=output_indexes,
            )
            expected_output.check_output_values(
                OutputValues(resolution_step_2),
            )

            expected_cost = [[35774633]]
            assert resolution_step_2.solver.Objective().Value() == pytest.approx(
                expected_cost[scenario][week]
            )
            assert sum(
                OutputValues(resolution_step_2)
                .component("S")
                .var("spillage")
                .value[0]  # type:ignore
            ) == pytest.approx(255873)
