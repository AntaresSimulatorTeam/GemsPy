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
E2E test: a model parameter that omits ``time-dependent`` / ``scenario-dependent``
is time- and scenario-dependent, as in Antares Simulator. A component may give it
a constant value or a time series.
"""

import textwrap
from pathlib import Path

import pytest

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession

_LIBRARY = textwrap.dedent("""\
    library:
      id: lib
      models:
        - id: plant
          parameters:
            - id: demand
            - id: cost
          variables:
            - id: generation
              lower-bound: 0
          constraints:
            - id: meet_demand
              expression: generation = demand
          objective-contributions:
            - id: cost
              expression: expec(sum(cost * generation))
    """)

_CONSTANT_DEMAND = """\
            - id: demand
              value: 40
"""

_TIME_SERIES_DEMAND = """\
            - id: demand
              time-dependent: true
              scenario-dependent: false
              value: demand_ts
"""


def _make_study(tmp_path: Path, demand: str) -> Path:
    study_dir = tmp_path / "study"
    (study_dir / "input" / "model-libraries").mkdir(parents=True)
    (study_dir / "input" / "data-series").mkdir()
    (study_dir / "input" / "model-libraries" / "lib.yml").write_text(_LIBRARY)
    (study_dir / "input" / "system.yml").write_text(
        "system:\n"
        "  id: s\n"
        "  components:\n"
        "    - id: p\n"
        "      model: lib.plant\n"
        "      parameters:\n" + demand + "            - id: cost\n"
        "              value: 2\n"
    )
    (study_dir / "input" / "data-series" / "demand_ts.tsv").write_text("10\n20\n30\n")
    (study_dir / "input" / "optim-config.yml").write_text(
        "time-scope:\n  first-time-step: 0\n  last-time-step: 2\n"
        "solver-options:\n  name: highs\n  logs: false\n"
    )
    return study_dir


def _generation(study_dir: Path) -> list:
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    table = SimulationSession(load_study(study_dir), optim_config).run().data
    rows = table[table["output"] == "generation"].sort_values("absolute_time_index")
    return rows["value"].tolist()


def test_parameter_without_flags_accepts_a_constant(tmp_path: Path) -> None:
    assert _generation(_make_study(tmp_path, _CONSTANT_DEMAND)) == pytest.approx(
        [40, 40, 40]
    )


def test_parameter_without_flags_accepts_a_time_series(tmp_path: Path) -> None:
    """Rejected before the parameter defaults were aligned with Antares, as
    a time series for a constant parameter."""
    assert _generation(_make_study(tmp_path, _TIME_SERIES_DEMAND)) == pytest.approx(
        [10, 20, 30]
    )
