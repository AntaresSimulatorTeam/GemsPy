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

from gems.study import DataBase
from gems.study.data import ComponentParameterIndex
from gems.study.parsing import parse_scenario_builder, parse_yaml_components
from gems.study.resolve_components import build_scenarized_data_base


@pytest.fixture(scope="session")
def series_dir() -> Path:
    return Path(__file__).parent / "series"


@pytest.fixture
def scenario_builder(series_dir: Path) -> pd.DataFrame:
    buider_path = series_dir / "scenario_builder.csv"
    return parse_scenario_builder(buider_path)


@pytest.fixture
def database(series_dir: Path, scenario_builder: pd.DataFrame) -> DataBase:
    system_path = Path(__file__).parent / "systems/with_scenarization.yml"
    with system_path.open() as components:
        return build_scenarized_data_base(
            parse_yaml_components(components), scenario_builder, series_dir
        )


def test_parser(scenario_builder: pd.DataFrame) -> None:
    builder = pd.DataFrame(
        {
            "name": [
                "load",
                "load",
                "load",
                "load",
                "cost-group",
                "cost-group",
                "cost-group",
                "cost-group",
            ],
            "year": [0, 1, 2, 3, 0, 1, 2, 3],
            "scenario": [0, 1, 0, 1, 0, 0, 1, 1],
        }
    )

    assert builder.equals(scenario_builder)


# cost-group group isnt use in following test because sum can't take time dependant parameters
def test_scenarized_data_base(database: DataBase) -> None:
    load_index = ComponentParameterIndex("D", "demand")
    assert database.get_value(load_index, 0, 0) == 50
    assert database.get_value(load_index, 0, 1) == 100
    assert database.get_value(load_index, 0, 2) == 50
    assert database.get_value(load_index, 0, 3) == 100
