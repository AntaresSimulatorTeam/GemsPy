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
- Reading the network from a YAML file.

Several cases are tested:

1. **Basic balance using YAML inputs**:
    - **Function**: `test_basic_balance_using_yaml`
    - **Description**: Verifies that the system can achieve an optimal balance between supply and demand using basic YAML inputs for the model and network. The test ensures that the solver reaches an optimal solution with the expected objective value.

2. **Basic balance with time-only series**:
    - **Function**: `test_basic_balance_time_only_series`
    - **Description**: Tests the system's behavior when time-dependent series are provided, ensuring correct optimization over multiple time steps. The test validates that the solver achieves an optimal solution with the expected objective value for time-only series.

3. **Basic balance with scenario-only series**:
    - **Function**: `test_basic_balance_scenario_only_series`
    - **Description**: Evaluates the system's ability to handle scenario-dependent series, ensuring proper optimization across different scenarios. The test confirms that the solver computes the expected weighted objective value for multiple scenarios.

4. **Short-term storage behavior with YAML inputs**:
    - **Function**: `test_short_term_storage_base_with_yaml`
    - **Description**: Checks the functionality of short-term storage components, ensuring they operate correctly to satisfy load without spillage or unsupplied energy. The test validates that the solver achieves an optimal solution with no energy spillage or unmet demand, while satisfying storage constraints.
"""

from pathlib import Path
from typing import Callable, Tuple

import pytest

from gems.model.parsing import InputLibrary, parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import TimeBlock, build_problem
from gems.simulation.optimization import BlockBorderManagement
from gems.study.data import DataBase
from gems.study.network import Network
from gems.study.parsing import InputSystem, parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    build_network,
    consistency_check,
    resolve_system,
)


def test_basic_balance_using_yaml(
    input_system: InputSystem, input_library: InputLibrary
) -> None:
    result_lib = resolve_library([input_library])
    components_input = resolve_system(input_system, result_lib)
    consistency_check(components_input.components, result_lib["basic"].models)

    database = build_data_base(input_system, None)
    network = build_network(components_input)

    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()
    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 3000


@pytest.fixture
def setup_test(
    libs_dir: Path, systems_dir: Path, series_dir: Path
) -> Callable[[], Tuple[Network, DataBase]]:
    def _setup_test(study_file_name: str):
        study_file = systems_dir / study_file_name
        lib_file = libs_dir / "lib_unittest.yml"
        with lib_file.open() as lib:
            input_library = parse_yaml_library(lib)

        with study_file.open() as c:
            input_study = parse_yaml_components(c)
        lib_dict = resolve_library([input_library])
        network_components = resolve_system(input_study, lib_dict)
        consistency_check(network_components.components, lib_dict["basic"].models)

        database = build_data_base(input_study, series_dir)
        network = build_network(network_components)
        return network, database

    return _setup_test


def test_basic_balance_time_only_series(
    setup_test: Callable[[], Tuple[Network, DataBase]],
) -> None:
    network, database = setup_test("study_time_only_series.yml")
    scenarios = 1
    problem = build_problem(network, database, TimeBlock(1, [0, 1]), scenarios)
    status = problem.solver.Solve()
    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 10000


def test_basic_balance_scenario_only_series(
    setup_test: Callable[[], Tuple[Network, DataBase]],
) -> None:
    network, database = setup_test("study_scenario_only_series.yml")
    scenarios = 2
    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios)
    status = problem.solver.Solve()
    assert status == problem.solver.OPTIMAL
    assert problem.solver.Objective().Value() == 0.5 * 5000 + 0.5 * 10000


def test_short_term_storage_base_with_yaml(
    setup_test: Callable[[], Tuple[Network, DataBase]],
) -> None:
    network, database = setup_test("components_for_short_term_storage.yml")
    # 18 produced in the 1st time-step, then consumed 2 * efficiency in the rest
    scenarios = 1
    horizon = 10
    time_blocks = [TimeBlock(0, list(range(horizon)))]

    problem = build_problem(
        network,
        database,
        time_blocks[0],
        scenarios,
        border_management=BlockBorderManagement.CYCLE,
    )
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL

    # The short-term storage should satisfy the load
    # No spillage / unsupplied energy is expected
    assert problem.solver.Objective().Value() == 0

    count_variables = 0
    for variable in problem.solver.variables():
        if "injection" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 100
        elif "withdrawal" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 50
        elif "level" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 1000
    assert count_variables == 3 * horizon
