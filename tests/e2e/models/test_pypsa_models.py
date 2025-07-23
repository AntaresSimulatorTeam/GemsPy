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
from pathlib import Path

import pytest

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation.optimization import build_problem
from gems.simulation.time_block import TimeBlock
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import build_data_base, build_network, resolve_system


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture
def systems_dir(data_dir: Path) -> Path:
    return data_dir / "systems"


@pytest.fixture
def series_dir(data_dir: Path) -> Path:
    return data_dir / "series"


@pytest.mark.parametrize(
    "system_file, timespan, target_value, relative_accuracy",
    [
        (
            "pypsa_basic_system.yml",
            2,
            7500,
            1e-6,
        ),
    ],
)
def test_model_behaviour(
    system_file: str,
    systems_dir: Path,
    series_dir: Path,
    timespan: float,
    target_value: float,
    relative_accuracy: float,
) -> None:
    scenarios = 1

    with open(systems_dir / system_file) as compo_file:
        input_component = parse_yaml_components(compo_file)

    with open("src/gems/libs/pypsa_models/pypsa_models.yml") as lib_file1:
        input_libraries = [parse_yaml_library(lib_file1)]

    result_lib = resolve_library(input_libraries)
    components_input = resolve_system(input_component, result_lib)
    database = build_data_base(input_component, Path(series_dir))
    network = build_network(components_input)
    problem = build_problem(
        network,
        database,
        TimeBlock(1, [i for i in range(0, timespan)]),
        scenarios,
    )
    status = problem.solver.Solve()
    print(problem.solver.Objective().Value())
    assert status == problem.solver.OPTIMAL
    assert math.isclose(
        target_value,
        problem.solver.Objective().Value(),
        rel_tol=relative_accuracy,
    )
