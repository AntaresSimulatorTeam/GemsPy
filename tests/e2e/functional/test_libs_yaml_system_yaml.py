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
- Reading the system from a YAML file.

Several cases are tested:

1. **Basic balance using YAML inputs**:
    - **Function**: `test_basic_balance_using_yaml`
    - **Description**: Verifies that the system can achieve an optimal balance between supply and demand using basic YAML inputs for the model and system. The test ensures that the solver reaches an optimal solution with the expected objective value.

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

from gems.model.parsing import LibrarySchema, parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import TimeBlock, build_problem
from gems.study.data import DataBase
from gems.study.parsing import SystemSchema, parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)
from gems.study.study import Study
from gems.study.system import System


def test_basic_balance_using_yaml(
    input_system: SystemSchema, input_library: LibrarySchema
) -> None:
    result_lib = resolve_library([input_library])
    system = resolve_system(input_system, result_lib)
    consistency_check(system, result_lib["basic"].models)

    database = build_data_base(input_system, None)

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == 3000


@pytest.fixture
def setup_test(
    libs_dir: Path, systems_dir: Path, series_dir: Path
) -> Callable[[str], Study]:
    def _setup_test(study_file_name: str) -> Study:
        study_file = systems_dir / study_file_name
        lib_file = libs_dir / "lib_unittest.yml"
        with lib_file.open() as lib:
            input_library = parse_yaml_library(lib)

        with study_file.open() as c:
            input_system = parse_yaml_components(c)
        lib_dict = resolve_library([input_library])
        system = resolve_system(input_system, lib_dict)
        consistency_check(system, lib_dict["basic"].models)

        database = build_data_base(input_system, series_dir)
        return Study(system, database)

    return _setup_test


def test_basic_balance_time_only_series(
    setup_test: Callable[[str], Study],
) -> None:
    study = setup_test("study_time_only_series.yml")
    scenarios = 1
    problem = build_problem(study, TimeBlock(1, [0, 1]), list(range(scenarios)))
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == 10000


def test_basic_balance_scenario_only_series(
    setup_test: Callable[[str], Study],
) -> None:
    study = setup_test("study_scenario_only_series.yml")
    scenarios = 2
    problem = build_problem(study, TimeBlock(1, [0]), list(range(scenarios)))
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == 0.5 * 5000 + 0.5 * 10000


def test_short_term_storage_base_with_yaml(
    setup_test: Callable[[str], Study],
) -> None:
    study = setup_test("components_for_short_term_storage.yml")
    # 18 produced in the 1st time-step, then consumed 2 * efficiency in the rest
    scenarios = 1
    horizon = 10
    time_blocks = [TimeBlock(0, list(range(horizon)))]

    problem = build_problem(
        study,
        time_blocks[0],
        list(range(scenarios)),
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"

    # The short-term storage should satisfy the load
    # No spillage / unsupplied energy is expected
    assert problem.objective_value == 0

    # TODO: update variable access


def test_varying_down_time(
    setup_test: Callable[[str], Study],
) -> None:
    """
    Two thermal clusters with different min-down-times actually start and stop,
    proving the per-component aggregation constraint is binding.

    Setup:
      G_0: cost=10, p_min=p_max=50, d_min_down=3
      G_1: cost=15, p_min=p_max=50, d_min_down=5
      G_2: cost=50, p_max=100 (backup generator, no p_min)
      Demand: [100,100,100, 0,0,0, 100,100,100,100]

    Both G_0 and G_1 have p_min=50, so they MUST be off when demand=0 (t=3,4,5).
    They stop at t=3 and cannot restart until their min-down-time is met:
      G_0 (d_min_down=3): off at t=3,4,5 → restarts at t=6
      G_1 (d_min_down=5): off at t=3,4,5,6,7 → restarts at t=8

    At t=6 and t=7, demand=100 but G_1 is still locked out → G_2 covers 50.

    The fact that objective=12250 (not 9250 which would be achieved if G_1
    could restart at t=6) proves d_min_down=5 is binding.

    Expected objective:
      G_0: 7 steps × 50 × 10 =  3500
      G_1: 5 steps × 50 × 15 =  3750
      G_2: 2 steps × 50 × 50 =  5000
      Total                   = 12250
    """
    study = setup_test("system_with_varying_down_time.yml")
    scenarios = 1
    horizon = 10

    problem = build_problem(
        study,
        TimeBlock(0, list(range(horizon))),
        list(range(scenarios)),
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(12250.0)
