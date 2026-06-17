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

from pathlib import Path
from typing import List

import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pandas as pd
import pytest

from andromede.simulation import OutputValues
from andromede.study import TimeScenarioSeriesData
from andromede.study.data import ComponentParameterIndex
from andromede.study.parsing import InputComponents
from andromede.thermal_heuristic.cluster_parameter import compute_slot_length
from andromede.thermal_heuristic.model import (
    FastModelBuilder,
    HeuristicFastModelBuilder,
)
from andromede.thermal_heuristic.problem import (
    BlockScenarioIndex,
    ThermalProblemBuilder,
    TimeScenarioHourParameter,
    get_database,
    get_network,
)
from tests.functional.libs.lib_thermal_heuristic import THERMAL_CLUSTER_MODEL_MILP


@pytest.fixture
def data_path() -> Path:
    return (
        Path(__file__).parent.parent
        / "data/thermal_heuristic_fast_min_down_not_respected"
    )


def test_fast_heuristic(
    data_path: Path,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
    week_scenario_index: BlockScenarioIndex,
) -> None:
    """
    Solve a weekly problem with fast heuristic. The thermal cluster has long d_min_up and d_min_down. The fast heuristic doesn't respect the d_min constraints.
    """
    number_hours = 168

    network = get_network(
        input_components,
        port_types=[],
        models=[FastModelBuilder(THERMAL_CLUSTER_MODEL_MILP).model],
    )
    database = get_database(
        input_components,
        data_path,
        fast=True,
        cluster=heuristic_components,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    thermal_problem_builder = ThermalProblemBuilder(
        database=database,
        network=network,
        time_scenario_hour_parameter=time_scenario_parameters,
    )

    pmax = thermal_problem_builder.database.get_value(
        ComponentParameterIndex(heuristic_components[0], "p_max"), 0, 0
    )
    input_data = np.loadtxt(data_path / "itr1_fast_cluster.txt")
    nb_on_1 = pd.DataFrame(
        np.ceil(np.round(input_data / pmax, 12)),  # type: ignore
        index=list(range(number_hours)),
        columns=[week_scenario_index.scenario],
    )

    thermal_problem_builder.database.add_data(
        heuristic_components[0], "n_guide", TimeScenarioSeriesData(nb_on_1)
    )

    # Solve heuristic problem
    resolution_step_heuristic = thermal_problem_builder.heuristic_resolution_step(
        id_component=heuristic_components[0],
        index=week_scenario_index,
        model=HeuristicFastModelBuilder(
            number_hours,
            slot_length=compute_slot_length(
                heuristic_components[0], thermal_problem_builder.database
            ),
        ).model,
    )
    status = resolution_step_heuristic.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    thermal_problem_builder.update_database_heuristic(
        OutputValues(resolution_step_heuristic),
        week_scenario_index,
        heuristic_components,
        var_to_read="n",
        param_to_update="min_generating",
        fn_to_apply=lambda x, y, z: min(x * y, z),
        param_needed_to_compute=["p_min", "max_generating"],
    )

    expected_output = np.loadtxt(data_path / "itr2_fast_cluster.txt")
    for t in range(number_hours):
        assert thermal_problem_builder.database.get_value(
            ComponentParameterIndex(heuristic_components[0], "min_generating"),
            t,
            week_scenario_index.scenario,
        ) == pytest.approx(expected_output[t])
