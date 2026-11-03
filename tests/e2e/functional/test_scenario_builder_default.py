# Copyright (c) 2026, RTE (https://www.rte-france.com)
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
E2E test: a modeler-scenariobuilder.dat that lists only some scenarios.

The scenarios without a line use the default (scenario n → time series n+1),
both in a gap and after the last line, and the run warns about them once.
"""

from pathlib import Path

import pytest

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession

_LIBRARY = """\
library:
  id: lib
  models:
    - id: m
      parameters:
        - id: demand
          time-dependent: true
          scenario-dependent: true
      variables:
        - id: x
          lower-bound: 0
      constraints:
        - id: meet_demand
          expression: x >= demand
      objective-contributions:
        - id: objective
          expression: expec(sum(x))
"""

_SYSTEM = """\
system:
  id: s
  components:
    - id: wind_farm
      model: lib.m
      scenario-group: wind
      parameters:
        - id: demand
          time-dependent: true
          scenario-dependent: true
          value: wind_data
"""


def _study(folder: Path, scenario_builder: str, last_scenario: int) -> Path:
    (folder / "input" / "model-libraries").mkdir(parents=True)
    (folder / "input" / "data-series").mkdir()
    (folder / "input" / "model-libraries" / "lib.yml").write_text(_LIBRARY)
    (folder / "input" / "system.yml").write_text(_SYSTEM)
    # Time series n (1-based) holds 10 * n at every time step.
    (folder / "input" / "data-series" / "wind_data.tsv").write_text(
        "10\t20\t30\t40\t50\n10\t20\t30\t40\t50\n"
    )
    (folder / "input" / "data-series" / "modeler-scenariobuilder.dat").write_text(
        scenario_builder
    )
    (folder / "input" / "optim-config.yml").write_text(
        "time-scope:\n  first-time-step: 0\n  last-time-step: 1\n"
        "solver-options:\n  name: highs\n  logs: false\n"
        f"scenario-scope:\n  include: ['0-{last_scenario}']\n"
    )
    return folder


def test_scenarios_without_line_use_the_default(tmp_path: Path) -> None:
    # Lines for scenarios 0 and 2 only: 1 is a gap, 3 and 4 are after the end.
    study = _study(tmp_path, "wind, 0 = 3\nwind, 2 = 5\n", last_scenario=4)

    with pytest.warns(UserWarning) as record:
        results = SimulationSession(
            load_study(study), load_optim_config(study / "input" / "optim-config.yml")
        ).run()

    x = results.component("wind_farm").output("x")
    assert [x.value(time_index=0, scenario_index=s) for s in range(5)] == [
        pytest.approx(30),  # line: time series 3
        pytest.approx(20),  # gap: default, time series 2
        pytest.approx(50),  # line: time series 5
        pytest.approx(40),  # after the end: default, time series 4
        pytest.approx(50),  # after the end: default, time series 5
    ]
    messages = [str(w.message) for w in record if "Scenario group" in str(w.message)]
    assert messages == [
        "Scenario group 'wind': scenarios 1, 3, 4 have no line in "
        "modeler-scenariobuilder.dat and use the default "
        "(scenario n uses time series n+1)."
    ]


def test_default_beyond_the_data_is_reported_before_the_run(tmp_path: Path) -> None:
    study = _study(tmp_path, "wind, 0 = 1\n", last_scenario=6)

    with pytest.warns(UserWarning), pytest.raises(ValueError) as error:
        SimulationSession(
            load_study(study), load_optim_config(study / "input" / "optim-config.yml")
        ).run()
    assert (
        "scenarios 5, 6: time series 6, 7 do not exist in 'wind_data', which has 5 "
        "time series (default: scenario n uses time series n+1)."
    ) in str(error.value)
