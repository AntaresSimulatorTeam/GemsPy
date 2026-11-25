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

"""Unit tests for ShiftValidityVisitor."""

import numpy as np
import pytest
import xarray as xr

from gems_craft.expression.expression import (
    literal,
    param,
    port_field,
    relative_time,
    var,
)
from gems_craft.expression.visitor import visit
from gems_runner.simulation.vectorized_builder import ShiftValidityVisitor


def _valid(mask):
    """Mask values, ``True`` when the visitor returns None (all valid)."""
    return np.array(True) if mask is None else mask.values


def _visitor(param_arrays=None, block_length=4):
    return ShiftValidityVisitor(
        model_id="m",
        param_arrays=param_arrays or {},
        block_length=block_length,
    )


@pytest.fixture
def per_comp_lag():
    """Two components with lag values 1 and 2."""
    return {
        ("m", "lag"): xr.DataArray(
            [1.0, 2.0],
            dims=["component"],
            coords={"component": ["c1", "c2"]},
        )
    }


# ---------------------------------------------------------------------------
# Leaf nodes read at t — always valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expr", [literal(3.0), param("lag"), var("x")])
def test_leaf_returns_none(expr, per_comp_lag):
    assert visit(expr, _visitor(per_comp_lag)) is None


# ---------------------------------------------------------------------------
# time_shift
# ---------------------------------------------------------------------------


def test_time_shift_zero_returns_none():
    expr = var("x").shift(0)
    assert visit(expr, _visitor(block_length=3)) is None


def test_nested_shifts_and_combined():
    # shift -1 blocks t=0; shift +1 blocks t=2 (block=3): only t=1 valid
    expr = var("x").shift(-1) + var("x").shift(1)
    mask = visit(expr, _visitor(block_length=3))
    assert mask is not None
    np.testing.assert_array_equal(mask.values, [False, True, False])


@pytest.mark.parametrize(
    "dim, sign, expected",
    [
        # x[t - lag], lag = 1, 2: invalid at t < lag
        pytest.param(
            "component",
            -1,
            [[False, True, True, True], [False, False, True, True]],
            id="per-component",
        ),
        # x[t + lag], lag = 1, 2: invalid at t + lag >= 4
        pytest.param(
            "scenario",
            1,
            [[True, True, True, False], [True, True, False, False]],
            id="per-scenario",
        ),
    ],
)
def test_time_shift_by_a_parameter(dim, sign, expected):
    lag = xr.DataArray([1.0, 2.0], dims=[dim], coords={dim: [0, 1]})
    mask = visit(var("x").shift(sign * param("lag")), _visitor({("m", "lag"): lag}))
    np.testing.assert_array_equal(mask.transpose(dim, "time").values, expected)


# ---------------------------------------------------------------------------
# time_sum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end, expected",
    [
        pytest.param(
            relative_time(-1), relative_time(0), [False, True, True, True], id="t-1"
        ),
        pytest.param(
            relative_time(0), relative_time(1), [True, True, True, False], id="t+1"
        ),
        pytest.param(0, relative_time(1), [True, True, True, False], id="mixed"),
    ],
)
def test_time_sum_reads_out_of_the_block(start, end, expected):
    mask = visit(var("x").time_sum_between(start, end), _visitor())
    np.testing.assert_array_equal(_valid(mask), expected)


@pytest.mark.parametrize(
    "start, end, expected",
    [
        # x[t-1] is out of the block at position 0
        pytest.param(0, 2, [False, False, False, False], id="absolute-reads-0"),
        pytest.param(1, 2, [True, True, True, True], id="absolute-skips-0"),
        pytest.param(0, relative_time(0), [False] * 4, id="mixed-from-0"),
        pytest.param(relative_time(0), 3, [False, True, True, True], id="mixed-to-3"),
        pytest.param(
            relative_time(-1),
            relative_time(0),
            [False, False, True, True],
            id="relative",
        ),
    ],
)
def test_time_sum_shifted_operand_checked_at_summed_positions(start, end, expected):
    expr = var("x").shift(-1).time_sum_between(start, end)
    mask = visit(expr, _visitor(block_length=4))
    np.testing.assert_array_equal(_valid(mask), expected)


def test_time_sum_empty_does_not_read_operand():
    expr = var("x").shift(-1).time_sum_between(2, 1)
    assert visit(expr, _visitor(block_length=4)) is None


def test_time_sum_per_component_param(per_comp_lag):
    # sum from -lag to 0: c1 from=-1, c2 from=-2
    # block=4: c1 invalid at t=0; c2 invalid at t=0 and t=1
    expr = var("x").time_sum(-param("lag"), 0)
    mask = visit(expr, _visitor(per_comp_lag, block_length=4))
    assert mask is not None
    assert "component" in mask.dims
    c1 = mask.sel(component="c1").values
    c2 = mask.sel(component="c2").values
    np.testing.assert_array_equal(c1, [False, True, True, True])
    np.testing.assert_array_equal(c2, [False, False, True, True])


# ---------------------------------------------------------------------------
# Dropped where a variable, parameter or port field is read out of the block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, expected",
    [
        pytest.param(
            param("p").shift(-1), [False, True, True, True], id="shifted-parameter"
        ),
        pytest.param(
            port_field("p", "f").sum_connections().shift(-1),
            [False, True, True, True],
            id="shifted-port-field",
        ),
        pytest.param(var("x").eval(4), False, id="index-out"),
        pytest.param(var("x").eval(param("d")), [True, False], id="index-by-d"),
        pytest.param(var("x").time_sum_between(-1, 1), False, id="absolute-sum"),
        pytest.param(
            var("x").time_sum_between(-1, relative_time(0)), False, id="mixed-sum"
        ),
        # Nested time operators: shifts add up
        pytest.param(var("x").shift(-1).shift(1), True, id="shift-of-shift"),
        pytest.param(var("x").shift(-1).eval(0), False, id="index-of-shift"),
        pytest.param(
            var("x").time_sum(-1, 0).shift(1),
            [True, True, True, False],
            id="shift-of-sum",
        ),
    ],
)
def test_read_out_of_the_block_is_dropped(expr, expected):
    d = xr.DataArray([1.0, 5.0], dims=["component"], coords={"component": ["c1", "c2"]})
    p = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["time"], coords={"time": range(4)})
    mask = visit(expr, _visitor({("m", "d"): d, ("m", "p"): p}))
    np.testing.assert_array_equal(_valid(mask), expected)


def test_time_sum_of_shifted_operand_is_dropped_whole():
    # sum(x[t-1]) reads x[-1]: no time step is valid
    mask = visit(var("x").shift(-1).time_sum(), _visitor())
    np.testing.assert_array_equal(_valid(mask), False)
    assert visit(var("x").time_sum(), _visitor()) is None


def test_expectation_drops_every_scenario_if_one_reads_out_of_the_block():
    # x[t + k]: k = 0 for s1 (valid), k = 1 for s2 (invalid at t = 3)
    k = xr.DataArray([0.0, 1.0], dims=["scenario"], coords={"scenario": [0, 1]})
    mask = visit(var("x").shift(param("k")).expec(), _visitor({("m", "k"): k}))
    assert "scenario" not in mask.dims
    np.testing.assert_array_equal(mask.values, [True, True, True, False])
