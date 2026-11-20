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
E2E test: time indices and time shifts given by a time-dependent parameter.
"""

from pathlib import Path
from typing import List

import pytest

from tests.e2e.functional.test_time_sum_bounds import (
    _make_study,
    _out_of_bounds,
    _solve,
)


@pytest.mark.parametrize(
    "mode, expression, expected",
    [
        # d = [1, 2, 3, 0]: x[1], x[2], x[3], x[0]
        pytest.param("cyclic", "x[d[t]]", [2, 3, 0, 1], id="index"),
        # d[t-1] = [0, 1, 2, 3], cyclic
        pytest.param("cyclic", "x[d[t-1]]", [1, 2, 3, 0], id="shifted-index"),
        # t + k = [-1, 2, 1, 4], cyclic: x[3], x[2], x[1], x[0]
        pytest.param("cyclic", "x[t + k[t]]", [0, 3, 2, 1], id="shift"),
        # skipped at t = 0 and t = 3, where a is then 0
        pytest.param("drop", "x[t + k[t]]", [0, 3, 2, 0], id="shift-drop"),
        # d is a parameter, read modulo the block: as in cyclic mode
        pytest.param("drop", "x[d[t-1]]", [1, 2, 3, 0], id="shifted-index-drop"),
        # d + 2 = [3, 4, 5, 2]: skipped at t = 1 and t = 2, where x is read out of the block
        pytest.param("drop", "x[d[t] + 2]", [0, 0, 0, 3], id="index-drop"),
    ],
)
def test_time_operator_given_by_a_time_dependent_parameter(
    tmp_path: Path, mode: str, expression: str, expected: List[float]
) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", f"a = {expression}"],
        objective="sum(x) + sum(a)",
        data={"d": [1, 2, 3, 0], "k": [-1, 1, -1, 1]},
        constants={},
        nb_time_steps=4,
        resolution=_out_of_bounds(mode, ["c1"]),
    )

    assert _solve(study_dir)["a"] == expected


def test_drop_skips_a_time_independent_constraint(tmp_path: Path) -> None:
    """x[5] >= c reads x outside a block of 4 time steps."""
    study_dir = _make_study(
        tmp_path,
        variables=["x"],
        constraints=["x[5] >= c"],
        objective="sum(x)",
        data={},
        constants={"c": 3},
        nb_time_steps=4,
        resolution=_out_of_bounds("drop", ["c0"]),
    )

    assert _solve(study_dir)["x"] == [0] * 4
