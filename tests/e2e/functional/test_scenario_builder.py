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

import pandas as pd
import pytest

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation.optimization import build_problem
from gems.simulation.time_block import TimeBlock
from gems.study.data import DataBase
from gems.study.parsing import parse_scenario_builder, parse_yaml_components
from gems.study.resolve_components import (
    build_network,
    build_scenarized_data_base,
    consistency_check,
    resolve_system,
)


@pytest.fixture
def scenario_builder(series_dir: Path) -> pd.DataFrame:
    buider_path = series_dir / "scenario_builder.csv"
    return parse_scenario_builder(buider_path)


@pytest.fixture
def database(
    series_dir: Path, systems_dir: Path, scenario_builder: pd.DataFrame
) -> DataBase:
    system_path = systems_dir / "with_scenarization.yml"
    with system_path.open() as components:
        return build_scenarized_data_base(
            parse_yaml_components(components), scenario_builder, series_dir
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

    consistency_check(components.components, lib_dict["basic"].models)
    network = build_network(components)

    timeblock = TimeBlock(1, list(range(2)))
    problem = build_problem(network, database, timeblock, 3)

    status = problem.solver.Solve()
    cost = problem.solver.Objective().Value()

    assert status == 0
    assert cost == pytest.approx(40000 / 3, abs=0.001)
