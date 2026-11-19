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
from typing import Dict, List, Tuple

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
          lower-bound: 0
        - id: z
          lower-bound: 0
        - id: s
          lower-bound: 0
        - id: w
          lower-bound: 0
      constraints:
{constraints}
      extra-outputs:
        - id: square_root
          expression: x^0.5
        - id: exponential
          expression: k^x
      objective-contributions:
        - id: objective
          expression: sum(x + z + s + w)
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


def _make_study(
    tmp_path: Path, constraints: List[str], dropped_constraints: Tuple[int, ...] = ()
) -> Path:
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
    optim_config = _OPTIM_CONFIG
    if dropped_constraints:
        optim_config += (
            "models:\n  - id: lib.m\n    out-of-bounds-processing:\n"
            "      constraints:\n"
            + "".join(
                f"        - id: c{i}\n          mode: drop\n"
                for i in dropped_constraints
            )
        )
    (study_dir / "input" / "optim-config.yml").write_text(optim_config)
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
            "z = -k^2 + 10",
            # -(k^2)/4 = -1: x one time step before, cyclic in the block.
            "s = x[t-(k^2)/4]",
            "w = sum(t-(k^2)/4 .. t, x)",
        ],
    )

    results = _solve(study_dir)

    # d = [1, 2, 3, 4], k = 2
    assert results["x"] == [1, 4, 9, 16]
    assert results["z"] == [6, 6, 6, 6]
    assert results["s"] == [16, 1, 4, 9]
    assert results["w"] == [17, 5, 13, 25]
    assert results["square_root"] == [1, 2, 3, 4]
    assert results["exponential"] == [2, 16, 512, 65536]


def test_power_with_time_shift_in_drop_mode(tmp_path: Path) -> None:
    study_dir = _make_study(
        tmp_path,
        constraints=[
            "x = d",
            # k^2/4 = 1: x read one time step before, out of the block at t = 0.
            "s = x[t-k^2/4]",
            "w = sum(t-k^2/4 .. t, x)",
        ],
        dropped_constraints=(1, 2),
    )

    results = _solve(study_dir)

    # d = [1, 2, 3, 4], k = 2; at t = 0, s and w are free and minimized to 0.
    assert results["s"] == [0, 1, 2, 3]
    assert results["w"] == [0, 3, 5, 7]
