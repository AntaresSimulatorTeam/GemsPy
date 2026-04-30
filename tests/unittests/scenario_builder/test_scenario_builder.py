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

import numpy as np
import pytest

from gems.study import DataBase
from gems.study.data import ComponentParameterIndex
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import build_data_base
from gems.study.scenario_builder import ScenarioBuilder


@pytest.fixture(scope="session")
def series_dir() -> Path:
    return Path(__file__).parent / "series"


@pytest.fixture(scope="session")
def scenario_builder(series_dir: Path) -> ScenarioBuilder:
    return ScenarioBuilder.load(series_dir / "modeler-scenariobuilder.dat")


@pytest.fixture
def database(series_dir: Path, scenario_builder: ScenarioBuilder) -> DataBase:
    system_path = Path(__file__).parent / "systems/with_scenarization.yml"
    with system_path.open() as components:
        return build_data_base(
            parse_yaml_components(components), series_dir, scenario_builder
        )


def test_scenario_builder_load(scenario_builder: ScenarioBuilder) -> None:
    """ScenarioBuilder.load() parses the .dat file into correct 0-based col_idx arrays."""
    mc = np.array([0, 1, 2, 3], dtype=int)
    assert list(scenario_builder.resolve_vectorized("load", mc)) == [0, 1, 0, 1]
    assert list(scenario_builder.resolve_vectorized("cost-group", mc)) == [0, 0, 1, 1]


def test_data_base_with_scenario_builder(database: DataBase) -> None:
    load_index = ComponentParameterIndex("D", "demand")
    # MC scenario 0 → col 0 (value 50), MC scenario 1 → col 1 (value 100), etc.
    assert database.get_value(load_index, 0, 0) == 50
    assert database.get_value(load_index, 0, 1) == 100
    assert database.get_value(load_index, 0, 2) == 50
    assert database.get_value(load_index, 0, 3) == 100


def test_resolve_vectorized_subset_playlist(scenario_builder: ScenarioBuilder) -> None:
    """A subset (playlist) of MC scenarios resolves to the correct columns."""
    mc = np.array([0, 2], dtype=int)  # skip scenarios 1 and 3
    assert list(scenario_builder.resolve_vectorized("load", mc)) == [0, 0]
    assert list(scenario_builder.resolve_vectorized("cost-group", mc)) == [0, 1]


def test_resolve_vectorized_out_of_bounds_raises(
    scenario_builder: ScenarioBuilder,
) -> None:
    """Requesting an MC scenario index beyond the defined range raises ValueError."""
    mc = np.array([0, 99], dtype=int)
    with pytest.raises(ValueError, match="not defined for group"):
        scenario_builder.resolve_vectorized("load", mc)


def test_resolve_vectorized_unknown_group_raises(
    scenario_builder: ScenarioBuilder,
) -> None:
    """Requesting an unknown scenario group raises ValueError."""
    mc = np.array([0, 1], dtype=int)
    with pytest.raises(ValueError, match="not defined in the scenario builder"):
        scenario_builder.resolve_vectorized("nonexistent-group", mc)
