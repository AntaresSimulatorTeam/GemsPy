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

import pytest

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import build_problem
from gems.simulation.time_block import TimeBlock
from gems.study import Study
from gems.study.data import DataBase
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)
from gems.study.scenario_builder import ScenarioBuilder


@pytest.fixture
def scenario_builder(series_dir: Path) -> ScenarioBuilder:
    return ScenarioBuilder.load(series_dir / "modeler-scenariobuilder.dat")


@pytest.fixture
def database(
    series_dir: Path, systems_dir: Path, scenario_builder: ScenarioBuilder
) -> DataBase:
    system_path = systems_dir / "with_scenarization.yml"
    with system_path.open() as components:
        return build_data_base(
            parse_yaml_components(components), series_dir, scenario_builder
        )


def test_system_with_scenarization(
    libs_dir: Path, systems_dir: Path, database: DataBase
) -> None:
    library_path = libs_dir / "lib_unittest.yml"
    with library_path.open("r") as file:
        yaml_lib = parse_yaml_library(file)
        lib_dict = resolve_library([yaml_lib])

    components_path = systems_dir / "with_scenarization.yml"
    with components_path.open("r") as file:
        yaml_comp = parse_yaml_components(file)
        components = resolve_system(yaml_comp, lib_dict)

    consistency_check(components, lib_dict["basic"].models)

    timeblock = TimeBlock(1, list(range(2)))
    problem = build_problem(Study(components, database), timeblock, list(range(3)))

    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(40000 / 3, abs=0.001)
