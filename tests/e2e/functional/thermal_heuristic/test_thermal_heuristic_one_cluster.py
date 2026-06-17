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
from andromede.study.data import ComponentParameterIndex
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
    return Path(__file__).parent.parent / "data/thermal_heuristic_one_cluster"


def test_milp_version(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """
    Model on 168 time steps with one thermal generation and demand on a single node.
        - Demand is constant to 2000 MW except for the 13th hour for which it is 2050 MW
        - Thermal generation is characterized with:
            - P_min = 700 MW
            - P_max = 1000 MW
            - Min up time = 3
            - Min down time = 10
            - Generation cost = 50€ / MWh
            - Startup cost = 50
            - Fixed cost = 1 /h
            - Number of unit = 3
        - Unsupplied energy = 1000 €/MWh
        - Spillage = 0 €/MWh

    The optimal milp solution consists in turning on two thermal plants at the begining, turning on a third thermal plant at the 13th hour and turning off the first thermal plant at the 14th hour, the other two thermal plants stay on for the rest of the week producing 1000MW each. At the 13th hour, the production is [700,700,700] to satisfy Pmin constraints.

    The optimal cost is then :
          50 x 2 x 1000 x 167 (prod step 1-12 and 14-168)
        + 50 x 3 x 700 (prod step 13)
        + 50 (start up step 13)
        + 2 x 1 x 167 (fixed cost step 1-12 and 14-168)
        + 3 x 1 (fixed cost step 13)
        = 16 805 387
    """
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

    main_resolution_step = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )
    status = main_resolution_step.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    assert main_resolution_step.solver.Objective().Value() == 16805387

    expected_output = ExpectedOutput(
        mode="milp",
        index=week_scenario_index,
        dir_path=data_path,
        list_cluster=heuristic_components,
        output_idx=ExpectedOutputIndexes(
            idx_generation=4, idx_nodu=6, idx_spillage=29, idx_unsupplied=25
        ),
    )
    expected_output.check_output_values(OutputValues(main_resolution_step))


def test_lp_version(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """
    Model on 168 time steps with one thermal generation and one demand on a single node.
        - Demand is constant to 2000 MW except for the 13th hour for which it is 2050 MW
        - Thermal generation is characterized with:
            - P_min = 700 MW
            - P_max = 1000 MW
            - Min up time = 3
            - Min down time = 10
            - Generation cost = 50€ / MWh
            - Startup cost = 50
            - Fixed cost = 1 /h
            - Number of unit = 3
        - Unsupplied energy = 1000 €/MWh
        - Spillage = 0 €/MWh

    The optimal solution of the linear relaxation consists in producing exactly the demand at each hour. The number of on units is equal to the production divided by P_max.

    The optimal cost is then :
          50 x 2000 x 167 (prod step 1-12 and 14-168)
        + 50 x 2050 (prod step 13)
        + 2 x 1 x 168 (fixed cost step 1-12 and 14-168)
        + 2050/1000 x 1 (fixed cost step 13)
        + 0,05 x 50 (start up cost step 13)
        = 16 802 840,55
    """
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

    main_resolution_step = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )
    status = main_resolution_step.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    assert main_resolution_step.solver.Objective().Value() == pytest.approx(16802840.55)

    expected_output = ExpectedOutput(
        mode="lp",
        index=week_scenario_index,
        dir_path=data_path,
        list_cluster=heuristic_components,
        output_idx=ExpectedOutputIndexes(
            idx_generation=4, idx_nodu=6, idx_spillage=29, idx_unsupplied=25
        ),
    )
    expected_output.check_output_values(OutputValues(main_resolution_step))


def test_accurate_heuristic(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """
    Solve the same problem as before with the heuristic accurate of Antares. The accurate heuristic is able to retrieve the milp optimal solution because when the number of on units found in the linear relaxation is ceiled, we found the optimal number of on units which is already feasible.
    """

    number_hours = 168
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

    # First optimization
    resolution_step_1 = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )
    status = resolution_step_1.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    # Get number of on units and round it to integer
    thermal_problem_builder.update_database_heuristic(
        OutputValues(resolution_step_1),
        week_scenario_index,
        heuristic_components,
        param_to_update="nb_units_min",
        var_to_read="nb_on",
        fn_to_apply=lambda x: ceil(round(x, 12)),
    )
    for time_step in range(number_hours):
        assert (
            thermal_problem_builder.database.get_value(
                ComponentParameterIndex(heuristic_components[0], "nb_units_min"),
                time_step,
                0,
            )
            == 2
            if time_step != 12
            else 3
        )

    # Solve heuristic problem
    resolution_step_accurate_heuristic = (
        thermal_problem_builder.heuristic_resolution_step(
            week_scenario_index,
            id_component=heuristic_components[0],
            model=HeuristicAccurateModelBuilder(THERMAL_CLUSTER_MODEL_MILP).model,
        )
    )
    status = resolution_step_accurate_heuristic.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL

    thermal_problem_builder.update_database_heuristic(
        OutputValues(resolution_step_accurate_heuristic),
        week_scenario_index,
        heuristic_components,
        param_to_update="nb_units_min",
        var_to_read="nb_on",
        fn_to_apply=lambda x: ceil(round(x, 12)),
    )

    for time_step in range(number_hours):
        assert (
            thermal_problem_builder.database.get_value(
                ComponentParameterIndex(heuristic_components[0], "nb_units_min"),
                time_step,
                0,
            )
            == 2
            if time_step != 12
            else 3
        )

    # Second optimization with lower bound modified
    resolution_step_2 = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )
    status = resolution_step_2.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL
    assert resolution_step_2.solver.Objective().Value() == 16805387

    expected_output = ExpectedOutput(
        mode="accurate",
        index=week_scenario_index,
        dir_path=data_path,
        list_cluster=heuristic_components,
        output_idx=ExpectedOutputIndexes(
            idx_generation=4, idx_nodu=6, idx_spillage=33, idx_unsupplied=29
        ),
    )
    expected_output.check_output_values(OutputValues(resolution_step_2))


def test_fast_heuristic(
    data_path: Path,
    models: list[Model],
    week_scenario_index: BlockScenarioIndex,
    input_components: InputComponents,
    heuristic_components: List[str],
    time_scenario_parameters: TimeScenarioHourParameter,
) -> None:
    """
    Solve the same problem as before with the heuristic fast of Antares
    Model on 168 time steps with one thermal generation and one demand on a single node.
        - Demand is constant to 2000 MW except for the 13th hour for which it is 2050 MW
        - Thermal generation is characterized with:
            - P_min = 700 MW
            - P_max = 1000 MW
            - Min up time = 3
            - Min down time = 10
            - Generation cost = 50€ / MWh
            - Startup cost = 50
            - Fixed cost = 1 /h
            - Number of unit = 3
        - Unsupplied energy = 1000 €/MWh
        - Spillage = 0 €/MWh

    The optimal solution consists in having 3 units turned on between time steps 10 and 19 with production equal to 2100 to respect pmin and 2 the rest of the time. Fast heuristic turns on 3 units for 10 timesteps because min down time is equal to 10.

    The optimal cost is then :
          50 x 2000 x 158 (prod step 1-9 and 20-168)
        + 50 x 2100 x 10 (prod step 10-19)
        = 16 850 000
    """

    number_hours = 168
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

    # First optimization
    resolution_step_1 = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )
    status = resolution_step_1.solver.Solve()
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

    for time_step in range(number_hours):
        assert (
            thermal_problem_builder.database.get_value(
                ComponentParameterIndex(heuristic_components[0], "min_generating"),
                time_step,
                0,
            )
            == 3 * 700
            if time_step in [t for t in range(10, 20)]
            else 2 * 700
        )

    # Second optimization with lower bound modified
    resolution_step_2 = thermal_problem_builder.main_resolution_step(
        week_scenario_index
    )
    status = resolution_step_2.solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL
    assert resolution_step_2.solver.Objective().Value() == pytest.approx(16850000)

    expected_output = ExpectedOutput(
        mode="fast",
        index=week_scenario_index,
        dir_path=data_path,
        list_cluster=heuristic_components,
        output_idx=ExpectedOutputIndexes(
            idx_generation=4, idx_nodu=6, idx_spillage=33, idx_unsupplied=29
        ),
    )
    expected_output.check_output_values(OutputValues(resolution_step_2))
