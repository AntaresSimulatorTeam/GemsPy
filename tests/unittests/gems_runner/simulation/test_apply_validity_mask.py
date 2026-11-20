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

"""Unit tests for _apply_validity_mask (OutOfBoundsMode.DROP)."""

import numpy as np
import xarray as xr

from gems_runner.simulation.optimization import _apply_validity_mask

# Valid everywhere but at (c1, t=0) and (c2, t=0..1).
MASK = xr.DataArray(
    [[False, True, True], [False, False, True]],
    dims=["component", "time"],
    coords={"component": ["c1", "c2"], "time": [0, 1, 2]},
)


def test_bound_is_masked_as_the_other_side() -> None:
    # A time-only bound is masked per component.
    bound = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": [0, 1, 2]})
    result = _apply_validity_mask(bound, MASK)
    expected = [[np.nan, 2.0, 3.0], [np.nan, np.nan, 3.0]]
    np.testing.assert_array_equal(
        result.transpose("component", "time").values, expected  # type: ignore[union-attr]
    )
    # A time-independent bound is left as is: the other side carries the mask.
    constant = xr.DataArray([1.0, 2.0], dims="component", coords=MASK.component.coords)
    assert _apply_validity_mask(constant, MASK) is constant
