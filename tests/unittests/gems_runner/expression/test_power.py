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
    "base, exponent",
    [(-2.0, 0.5), (0.0, -1.0), (10.0, 1000.0)],
    ids=["complex", "division_by_zero", "overflow"],
)
def test_real_power_without_finite_real_value_raises(
    base: float, exponent: float
) -> None:
    # A single undefined element of an array is enough.
    with pytest.raises(ValueError, match="no finite real value"):
        real_power(xr.DataArray([4.0, base]), xr.DataArray([2.0, exponent]))
