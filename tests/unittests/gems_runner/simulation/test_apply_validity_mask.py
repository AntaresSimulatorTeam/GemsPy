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

"""Unit tests for the validity masks of OutOfBoundsMode.DROP."""

import numpy as np
import xarray as xr

from gems_craft.expression.expression import literal, param, var
from gems_craft.model.constraint import Constraint
from gems_runner.simulation.optimization import (
    _apply_validity_mask,
    _constraint_validity_mask,
)
from gems_runner.simulation.vectorized_builder import ShiftValidityVisitor


def test_constraint_mask_includes_the_bounds() -> None:
    # x in [p[t-1], p[t+1]]: p is read out of the block at t = 0 and t = 2.
    p = xr.DataArray([1.0, 2.0, 3.0], dims="time")
    constraint = Constraint(
        "c",
        var("x"),
        lower_bound=param("p").shift(-literal(1)),
        upper_bound=param("p").shift(literal(1)),
    )
    visitor = ShiftValidityVisitor(
        model_id="m", param_arrays={("m", "p"): p}, block_length=3
    )
    mask = _constraint_validity_mask(constraint, visitor)
    assert mask is not None
    assert mask.values.tolist() == [False, True, False]


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
