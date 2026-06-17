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
from andromede.study.data import ComponentParameterIndex, ConstantData, DataBase
from andromede.study.parsing import InputComponents
from andromede.thermal_heuristic.model import (
    AccurateModelBuilder,
    HeuristicAccurateModelBuilder,
    Model,
)
from andromede.thermal_heuristic.problem import (
    BlockScenarioIndex,
    ThermalProblemBuilder,
    TimeScenarioHourParameter,
    get_database,
    get_network,
)
from andromede.thermal_heuristic.time_scenario_parameter import timesteps
from tests.functional.libs.lib_thermal_heuristic import (
    DEMAND_WITH_RESERVE_MODEL,
    NODE_WITH_RESERVE_MODEL,
    THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP,
)


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent.parent / "data/thermal_heuristic_day_ahead_reserve"


@pytest.fixture
def models() -> List[Model]:
    return [NODE_WITH_RESERVE_MODEL, DEMAND_WITH_RESERVE_MODEL]


@pytest.fixture
def demand_day_ahead() -> float:
    demand_day_ahead = 23.0
    return demand_day_ahead


def shift_load(
    week_scenario_index: BlockScenarioIndex,
    time_scenario_parameters: TimeScenarioHourParameter,
    demand_shift: float,
    database: DataBase,
) -> None:
    for timestep in timesteps(week_scenario_index, time_scenario_parameters):
        initial_data = database.get_value(
            ComponentParameterIndex("D", "demand_energy"),
            timestep,
            week_scenario_index.scenario,
        )
        database.set_value(
            ComponentParameterIndex("D", "demand_energy"),
            initial_data + demand_shift,
            timestep,
            week_scenario_index.scenario,
        )


def test_milp_with_day_ahead_reserve(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
    demand_day_ahead: float,
) -> None:
    """ """
    network = get_network(
        input_components,
        port_types=[BALANCE_PORT_TYPE],
        models=[THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP] + models,
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

    shift_load(
        week_scenario_index,
        time_scenario_parameters,
        demand_day_ahead,
        thermal_problem_builder.database,
    )

    main_resolution_step_1 = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )

    status = main_resolution_step_1.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    shift_load(
        week_scenario_index,
        time_scenario_parameters,
        -demand_day_ahead,
        thermal_problem_builder.database,
    )

    thermal_problem_builder.update_database_heuristic(
        OutputValues(main_resolution_step_1),
        week_scenario_index,
        heuristic_components,
        param_to_update="nb_units_min",
        var_to_read="nb_on",
        fn_to_apply=lambda x: ceil(round(x, 12)),
    )

    main_resolution_step_2 = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )

    status = main_resolution_step_2.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    output = OutputValues(main_resolution_step_2)
    assert sum(
        output.component("N").var("spillage_energy").value[0]  # type:ignore
    ) == pytest.approx(358)
    assert main_resolution_step_2.solver.Objective().Value() == pytest.approx(7935769)


def test_milp_without_day_ahead_reserve(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """ """
    network = get_network(
        input_components,
        port_types=[BALANCE_PORT_TYPE],
        models=[THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP] + models,
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

    main_resolution_step = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )

    status = main_resolution_step.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    output = OutputValues(main_resolution_step)
    assert sum(
        output.component("N").var("spillage_energy").value[0]  # type:ignore
    ) == pytest.approx(0, abs=1e-10)
    assert main_resolution_step.solver.Objective().Value() == pytest.approx(2512645)


def test_accurate_heuristic_with_day_ahead_reserve(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
    demand_day_ahead: float,
) -> None:
    """ """

    network = get_network(
        input_components,
        port_types=[BALANCE_PORT_TYPE],
        models=[AccurateModelBuilder(THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP).model]
        + models,
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

    shift_load(
        week_scenario_index,
        time_scenario_parameters,
        demand_day_ahead,
        thermal_problem_builder.database,
    )

    # First optimization
    resolution_step_1 = thermal_problem_builder.main_resolution_step(
        week_scenario_index,
    )

    status = resolution_step_1.solver.Solve()
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
                    THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP
                ).model,
            )
        )

        status = resolution_step_accurate_heuristic.solver.Solve()
        assert status == pywraplp.Solver.OPTIMAL

        thermal_problem_builder.update_database_heuristic(
            OutputValues(resolution_step_accurate_heuristic),
            week_scenario_index,
            [g],
            param_to_update="nb_units_min",
            var_to_read="nb_on",
            fn_to_apply=lambda x: ceil(round(x, 12)),
        )

    shift_load(
        week_scenario_index,
        time_scenario_parameters,
        -demand_day_ahead,
        thermal_problem_builder.database,
    )

    # Second optimization with lower bound modified
    resolution_step_2 = thermal_problem_builder.main_resolution_step(
        week_scenario_index,
    )

    status = resolution_step_2.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    output = OutputValues(resolution_step_2)
    assert resolution_step_2.solver.Objective().Value() == pytest.approx(12600136)
    assert sum(
        output.component("N").var("spillage_energy").value[0]  # type:ignore
    ) == pytest.approx(667)


def test_accurate_heuristic_without_day_ahead_reserve(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """ """

    network = get_network(
        input_components,
        port_types=[BALANCE_PORT_TYPE],
        models=[AccurateModelBuilder(THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP).model]
        + models,
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

    # First optimization
    resolution_step_1 = thermal_problem_builder.main_resolution_step(
        week_scenario_index,
    )

    status = resolution_step_1.solver.Solve()
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
                    THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP
                ).model,
            )
        )

        status = resolution_step_accurate_heuristic.solver.Solve()
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

    status = resolution_step_2.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL
    output = OutputValues(resolution_step_2)
    assert resolution_step_2.solver.Objective().Value() == pytest.approx(3415137)
    assert sum(
        output.component("N").var("spillage_energy").value[0]  # type:ignore
    ) == pytest.approx(60)
