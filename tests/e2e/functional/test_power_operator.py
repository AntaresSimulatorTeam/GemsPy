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
E2E test: the power operator ``^`` in a YAML study.

In constraints and objectives, ``^`` may only be applied to parameters and
literals; in extra outputs, it may also be applied to variables. ``^`` binds
tighter than unary minus: ``-k^2`` is ``-(k^2)``, also inside a time shift.
"""

from pathlib import Path
from typing import Dict, List

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
        - id: d
          time-dependent: true
          scenario-dependent: false
        - id: k
          time-dependent: false
          scenario-dependent: false
      variables:
        - id: x
        - id: y
        - id: z
        - id: s
      constraints:
{constraints}
      extra-outputs:
        - id: square_root
          expression: x^0.5
        - id: inverse
          expression: d^-1
        - id: root_k
          expression: y^(1/k)
      objective-contributions:
        - id: objective
          expression: sum(x + y + z + s)
"""

_SYSTEM = """\
system:
  id: s
  components:
    - id: m1
      model: lib.m
      parameters:
        - id: d
          time-dependent: true
          scenario-dependent: false
          value: d_ts
        - id: k
          time-dependent: false
          scenario-dependent: false
          value: 2
"""

_OPTIM_CONFIG = """\
time-scope:
  first-time-step: 0
  last-time-step: 3
solver-options:
  name: highs
  logs: false
"""


def _make_study(tmp_path: Path, constraints: List[str]) -> Path:
    study_dir = tmp_path / "study"
    (study_dir / "input" / "model-libraries").mkdir(parents=True)
    (study_dir / "input" / "data-series").mkdir()
    (study_dir / "input" / "model-libraries" / "lib.yml").write_text(
        _LIBRARY.format(
            constraints="".join(
                f"        - id: c{i}\n          expression: {c}\n"
                for i, c in enumerate(constraints)
            )
        )
    )
    (study_dir / "input" / "system.yml").write_text(_SYSTEM)
    (study_dir / "input" / "data-series" / "d_ts.tsv").write_text("1\n2\n3\n4\n")
    (study_dir / "input" / "optim-config.yml").write_text(_OPTIM_CONFIG)
    return study_dir


def _solve(study_dir: Path) -> Dict[str, List[float]]:
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    table = SimulationSession(load_study(study_dir), optim_config).run().data
    return {
        output: rows.sort_values("absolute_time_index")["value"].round(6).tolist()
        for output, rows in table.groupby("output")
    }


def test_power_of_parameters_in_constraints_and_of_variables_in_extra_outputs(
    tmp_path: Path,
) -> None:
    study_dir = _make_study(
        tmp_path,
        constraints=[
            "x = d^k",
            "y = k^d",
            "z = -k^2 + 10",
            # -k^2/4 = -(k^2)/4 = -1: x one time step before, cyclic in the block.
            "s = x[t-k^2/4]",
        ],
    )

    results = _solve(study_dir)

    # d = [1, 2, 3, 4], k = 2
    assert results["x"] == [1, 4, 9, 16]
    assert results["y"] == [2, 4, 8, 16]
    assert results["z"] == [6, 6, 6, 6]
    assert results["s"] == [16, 1, 4, 9]
    assert results["square_root"] == [1, 2, 3, 4]
    assert results["inverse"] == [1, 0.5, 0.333333, 0.25]
    assert results["root_k"] == [1.414214, 2, 2.828427, 4]


@pytest.mark.parametrize("constraint", ["x^2 = d", "x^k = d", "k^x = d"])
def test_power_of_variable_in_constraint_is_rejected(
    tmp_path: Path, constraint: str
) -> None:
    study_dir = _make_study(tmp_path, constraints=[constraint])

    with pytest.raises(ValueError, match="Non-linear expression is not allowed"):
        load_study(study_dir)


@pytest.mark.parametrize("constraint", ["x = (-d)^0.5", "x = 0^(-k)"])
def test_power_without_finite_real_value_in_constraint_raises(
    tmp_path: Path, constraint: str
) -> None:
    study_dir = _make_study(tmp_path, constraints=[constraint])

    with pytest.raises(ValueError, match="no finite real value"):
        _solve(study_dir)
