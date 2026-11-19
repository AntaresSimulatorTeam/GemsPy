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

import math

import pytest
import xarray as xr

from gems_runner.expression.power import real_power


@pytest.mark.parametrize(
    "base, exponent, expected",
    [
        (-2.0, 3.0, -8.0),
        (4.0, 0.5, 2.0),
        (2.0, -1.0, 0.5),
        # Non-finite operands are not an error.
        (math.inf, 2.0, math.inf),
        (math.nan, 2.0, math.nan),
    ],
)
def test_real_power(base: float, exponent: float, expected: float) -> None:
    assert real_power(base, exponent) == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize(
    "base, exponent, expected",
    [(-2.0, 0.5, math.nan), (0.0, -1.0, math.inf), (10.0, 1000.0, math.inf)],
    ids=["complex", "division_by_zero", "overflow"],
)
def test_real_power_without_finite_real_value_is_nan_or_inf_per_cell(
    base: float, exponent: float, expected: float
) -> None:
    # Like `/`: only the undefined cell is affected, nothing raises.
    result = real_power(xr.DataArray([4.0, base]), xr.DataArray([2.0, exponent]))
    xr.testing.assert_equal(result, xr.DataArray([16.0, expected]))
