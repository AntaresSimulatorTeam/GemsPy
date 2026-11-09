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

from tests.e2e.functional.test_time_sum_bounds import _make_study, _solve


@pytest.mark.parametrize(
    "expression, expected",
    [
        # d = [1, 2, 3, 0]: x[1], x[2], x[3], x[0]
        pytest.param("x[d[t]]", [2, 3, 0, 1], id="index"),
        # d[t-1] = [0, 1, 2, 3], cyclic
        pytest.param("x[d[t-1]]", [1, 2, 3, 0], id="shifted-index"),
        # t + k = [1, 2, 1, 2]
        pytest.param("x[t + k[t]]", [2, 3, 2, 3], id="shift"),
    ],
)
def test_time_operator_given_by_a_time_dependent_parameter(
    tmp_path: Path, expression: str, expected: List[float]
) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", f"a = {expression}"],
        objective="sum(x)",
        data={"d": [1, 2, 3, 0], "k": [1, 1, -1, -1]},
        constants={},
        nb_time_steps=4,
    )

    assert _solve(study_dir)["a"] == expected
