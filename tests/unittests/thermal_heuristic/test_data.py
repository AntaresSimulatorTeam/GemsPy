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

import numpy as np
import pandas as pd

from gems.thermal_heuristic.data import (
    get_max_failures,
    get_max_unit,
    get_max_unit_for_min_down_time,
    shift_df,
)


def test_get_max_unit() -> None:
    max_generating = pd.DataFrame([0, 10, 10, 15, 20, 5, 25])

    max_unit = get_max_unit(pmax=10, units=2, max_generating=max_generating)

    assert list(max_unit.values) == [0, 1, 1, 2, 2, 1, 2]


def test_get_max_failures() -> None:
    max_unit = pd.DataFrame(
        np.transpose([[0, 1, 1, 2, 2, 1, 2], [1, 1, 1, 2, 2, 1, 0]])
    )

    max_failures = get_max_failures(max_unit, 7)

    assert list(max_failures.values[:, 0]) == [2, 0, 0, 0, 0, 1, 0]
    assert list(max_failures.values[:, 1]) == [0, 0, 0, 0, 0, 1, 1]


def test_get_max_unit_for_min_down_time() -> None:
    max_unit = pd.DataFrame(
        np.transpose([[0, 1, 1, 2, 2, 1, 2], [1, 1, 1, 2, 2, 1, 0]])
    )

    max_unit_for_min_down_time = get_max_unit_for_min_down_time(2, max_unit, 7)

    assert list(max_unit_for_min_down_time.values[:, 0]) == [2, 3, 1, 2, 2, 2, 3]
    assert list(max_unit_for_min_down_time.values[:, 1]) == [2, 1, 1, 2, 2, 2, 2]


def test_shift_df() -> None:
    A = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    df = pd.DataFrame(A.transpose())

    shifted = shift_df(df, 1, 3)

    assert list(shifted.values[:, 0]) == [3, 1, 2, 6, 4, 5]
    assert list(shifted.values[:, 1]) == [9, 7, 8, 12, 10, 11]
