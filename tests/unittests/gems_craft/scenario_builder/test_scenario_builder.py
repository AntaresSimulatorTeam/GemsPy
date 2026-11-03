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

from gems_craft.study import DataBase
from gems_craft.study.data import ComponentParameterIndex
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import build_data_base
from gems_craft.study.scenario_builder import ScenarioBuilder


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
            parse_yaml_system(components), series_dir, scenario_builder
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


def test_resolve_vectorized_scenario_subset(scenario_builder: ScenarioBuilder) -> None:
    """A subset of MC scenarios resolves to the correct columns."""
    mc = np.array([0, 2], dtype=int)  # skip scenarios 1 and 3
    assert list(scenario_builder.resolve_vectorized("load", mc)) == [0, 0]
    assert list(scenario_builder.resolve_vectorized("cost-group", mc)) == [0, 1]


def test_resolve_vectorized_after_the_last_line_uses_the_default(
    scenario_builder: ScenarioBuilder,
) -> None:
    """A scenario after the group's last line uses its own column."""
    mc = np.array([0, 99], dtype=int)
    assert list(scenario_builder.resolve_vectorized("load", mc)) == [0, 99]


def test_resolve_vectorized_unknown_group_raises(
    scenario_builder: ScenarioBuilder,
) -> None:
    """Requesting an unknown scenario group raises ValueError."""
    mc = np.array([0, 1], dtype=int)
    with pytest.raises(ValueError, match="not defined in the scenario builder"):
        scenario_builder.resolve_vectorized("nonexistent-group", mc)


@pytest.mark.parametrize("number", ["0", "-1"])
def test_load_refuses_time_series_number_below_one(tmp_path: Path, number: str) -> None:
    """Time series numbers are 1-based: 0 would read the last column."""
    path = tmp_path / "modeler-scenariobuilder.dat"
    path.write_text(f"wind, 0 = 1\nwind, 1 = {number}\n")

    with pytest.raises(ValueError, match="time series numbers start at 1"):
        ScenarioBuilder.load(path)


def _load(tmp_path: Path, text: str) -> ScenarioBuilder:
    path = tmp_path / "modeler-scenariobuilder.dat"
    path.write_text(text)
    return ScenarioBuilder.load(path)


def test_gap_uses_the_default(tmp_path: Path) -> None:
    sb = _load(tmp_path, "wind, 0 = 4\nwind, 1 = 4\nwind, 3 = 4\n")

    mc = np.arange(5)
    assert sb.resolve_vectorized("wind", mc).tolist() == [3, 3, 2, 3, 4]


def test_lines_only_after_the_first_scenarios(tmp_path: Path) -> None:
    sb = _load(tmp_path, "wind, 5 = 1\n")

    assert sb.resolve_vectorized("wind", np.arange(7)).tolist() == [0, 1, 2, 3, 4, 0, 6]


def test_resolve_vectorized_is_the_same_in_any_order(tmp_path: Path) -> None:
    """The columns cached for the first call do not change later calls."""
    sb = _load(tmp_path, "wind, 1 = 3\n")

    assert sb.resolve_vectorized("wind", np.array([0, 1])).tolist() == [0, 2]
    assert sb.resolve_vectorized("wind", np.array([9, 1, 4])).tolist() == [9, 2, 4]
    assert sb.resolve_vectorized("wind", np.array([1])).tolist() == [2]
    assert sb.resolve_vectorized("wind", np.array([], dtype=int)).tolist() == []


def test_is_from_file(tmp_path: Path) -> None:
    sb = _load(tmp_path, "wind, 0 = 1\nwind, 2 = 1\n")

    assert sb.is_from_file("wind", np.arange(5)).tolist() == [
        True,
        False,
        True,
        False,
        False,
    ]


def test_scenarios_without_line(tmp_path: Path) -> None:
    sb = _load(tmp_path, "wind, 0 = 1\nwind, 2 = 1\nload, 0 = 1\nload, 1 = 1\n")

    assert sb.scenarios_without_line([0, 1, 2]) == {"wind": [1], "load": [2]}
    assert sb.scenarios_without_line([0]) == {}


def test_load_refuses_a_negative_scenario(tmp_path: Path) -> None:
    """It used to overwrite the line of the last scenario."""
    with pytest.raises(ValueError, match="MC scenarios start at 0, got -1"):
        _load(tmp_path, "wind, 0 = 1\nwind, 1 = 2\nwind, -1 = 4\n")


def test_validate_mc_scenarios_is_deprecated(
    scenario_builder: ScenarioBuilder,
) -> None:
    with pytest.warns(DeprecationWarning, match="scenarios_without_line"):
        assert scenario_builder.validate_mc_scenarios([0, 99]) == []


def test_cache_covers_only_the_scenarios_asked_for(tmp_path: Path) -> None:
    """A line with a very large scenario (a typo) does not grow the cache."""
    sb = _load(tmp_path, "wind, 1 = 3\nwind, 10000000 = 1\n")

    assert sb.resolve_vectorized("wind", np.arange(4)).tolist() == [0, 2, 2, 3]
    assert len(sb._resolved["wind"]) == 4
    assert sb.resolve_vectorized("wind", np.array([10000000])).tolist() == [0]
