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

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from gems_craft.study.folder import load_study
from gems_runner.simulation import TimeBlock, build_problem

STUDY_DIR = Path(__file__).parent / "studies" / "custom_sets"

# Expected solution of the joint LP (checked with scipy), `nan` where a set dimension
# is padded beyond the component's own size (`seg`: 3, `slot`: 2).
EXPECTED = {
    ("global-plant", "gen", "G"): [2, 0],
    ("global-plant", "gen", "G2"): [0, 9],
    ("grid", "x", "P"): [[0, 1], [1, 0], [np.nan, np.nan]],
    ("grid", "x", "Q"): [[1, np.nan], [0, np.nan], [0, np.nan]],
}


def test_custom_sets_study() -> None:
    """Ragged local sets (`seg` and `slot` together), a global set and an
    `indexed-by` override (G2), all sharing one node and one demand.

    The cyclic ramp `x[seg+1] - x <= 1` of `grid` binds for P and for Q (through
    the wrap); `gen[fuel=0] <= 2` binds for G.
    """
    problem = build_problem(load_study(STUDY_DIR), TimeBlock(1, [0]), [0])
    problem.solve()
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(25.4)

    for (model, var, component), expected in EXPECTED.items():
        solution = problem.get_variable_solution(f"custom_sets.{model}", var)
        np.testing.assert_allclose(
            solution.sel(component=component).values.squeeze(),
            expected,
            equal_nan=True,
            err_msg=f"{component}.{var}",
        )


def test_positive_lower_bound_on_ragged_sets(tmp_path: Path) -> None:
    """A lower bound above the padded upper bound (0) is not a bound violation."""
    study_dir = Path(shutil.copytree(STUDY_DIR, tmp_path / "study"))
    library = study_dir / "input" / "model-libraries" / "custom_sets.yml"
    text, count = re.subn(
        r"lower-bound: 0(\r?\n\s+upper-bound: cap)",
        r"lower-bound: 0.5\1",
        library.read_text(),
    )
    assert count == 1
    library.write_text(text)

    problem = build_problem(load_study(study_dir), TimeBlock(1, [0]), [0])
    problem.solve()
    assert problem.termination_condition == "optimal"
    x = problem.get_variable_solution("custom_sets.grid", "x").values
    assert np.all(x[~np.isnan(x)] >= 0.5 - 1e-9)
