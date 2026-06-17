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

from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import build_data_base
from gems.study.scenario_builder import ScenarioBuilder


@pytest.fixture(scope="session")
def dispatch_series_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Series directory with 3-column loads and a modeler-scenariobuilder.dat."""
    d = tmp_path_factory.mktemp("dispatch_series")
    # 1 row × 3 columns: col 0 = 10, col 1 = 20, col 2 = 30
    (d / "loads.txt").write_text("10  20  30\n")
    # MC scenario 0 → column 3 (1-based) → col_idx 2 → value 30
    # MC scenario 1 → column 1 (1-based) → col_idx 0 → value 10
    # MC scenario 2 → column 2 (1-based) → col_idx 1 → value 20
    (d / "modeler-scenariobuilder.dat").write_text(
        "load, 0 = 3\n" "load, 1 = 1\n" "load, 2 = 2\n"
    )
    return d


@pytest.fixture(scope="session")
def dispatch_system_yml() -> str:
    return """\
system:
  model-libraries: basic
  components:
    - id: D
      model: basic.demand
      scenario-group: load
      parameters:
        - id: demand
          scenario-dependent: true
          value: loads
"""


def test_scenario_builder_load(dispatch_series_dir: Path) -> None:
    """ScenarioBuilder.load() parses the .dat file into correct 0-based col_idx arrays."""
    sb = ScenarioBuilder.load(dispatch_series_dir / "modeler-scenariobuilder.dat")
    mc = np.array([0, 1, 2], dtype=int)
    cols = sb.resolve_vectorized("load", mc)
    assert list(cols) == [2, 0, 1]


def test_dispatch_mc_scenarios_to_columns(
    dispatch_series_dir: Path, dispatch_system_yml: str
) -> None:
    """DataBase.get_values() dispatches each MC scenario to the correct data column."""
    sb = ScenarioBuilder.load(dispatch_series_dir / "modeler-scenariobuilder.dat")

    import io

    db = build_data_base(
        parse_yaml_components(io.StringIO(dispatch_system_yml)),
        dispatch_series_dir,
        scenario_builder=sb,
    )

    result = db.get_values("D", "demand", timesteps=None, mc_scenarios=[0, 1, 2])
    # MC scenario 0 → col 2 → 30, scenario 1 → col 0 → 10, scenario 2 → col 1 → 20
    assert list(result) == [30, 10, 20]
