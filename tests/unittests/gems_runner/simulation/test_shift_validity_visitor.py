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

from gems_craft.expression.expression import literal, param, relative_time, var
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
# Leaf nodes — must return None (no time-shift constraints)
# ---------------------------------------------------------------------------


def test_literal_returns_none():
    assert visit(literal(3.0), _visitor()) is None


def test_parameter_returns_none(per_comp_lag):
    assert visit(param("lag"), _visitor(per_comp_lag)) is None


def test_variable_returns_none():
    assert visit(var("x"), _visitor()) is None


# ---------------------------------------------------------------------------
# time_shift with compile-time constant shift
# ---------------------------------------------------------------------------


def test_time_shift_negative_literal_blocks_t0():
    # shift = -1: t=0 accesses t-1 = -1, out of bounds
    expr = var("x").shift(-1)
    mask = visit(expr, _visitor(block_length=3))
    assert mask is not None
    assert mask.dims == ("time",)
    np.testing.assert_array_equal(mask.values, [False, True, True])


def test_time_shift_positive_literal_blocks_last_t():
    # shift = +1: t=2 accesses t+1 = 3 >= 3, out of bounds (block=3)
    expr = var("x").shift(1)
    mask = visit(expr, _visitor(block_length=3))
    assert mask is not None
    np.testing.assert_array_equal(mask.values, [True, True, False])


def test_time_shift_zero_returns_none():
    expr = var("x").shift(0)
    assert visit(expr, _visitor(block_length=3)) is None


# ---------------------------------------------------------------------------
# time_shift with per-component parameter shift
# ---------------------------------------------------------------------------


def test_time_shift_per_component_param(per_comp_lag):
    # shift = -lag: c1 lag=1 → shift=-1, c2 lag=2 → shift=-2
    # block=4: c1 invalid at t=0, c2 invalid at t=0 and t=1
    expr = var("x").shift(-param("lag"))
    mask = visit(expr, _visitor(per_comp_lag, block_length=4))
    assert mask is not None
    assert "time" in mask.dims
    assert "component" in mask.dims
    c1_mask = mask.sel(component="c1").values
    c2_mask = mask.sel(component="c2").values
    np.testing.assert_array_equal(c1_mask, [False, True, True, True])
    np.testing.assert_array_equal(c2_mask, [False, False, True, True])


# ---------------------------------------------------------------------------
# time_sum with compile-time constant bounds
# ---------------------------------------------------------------------------


def test_time_sum_negative_from_blocks_t0():
    # sum from -1 to 0: t=0 accesses t-1 = -1, invalid; t=1,2,3 valid
    expr = var("x").time_sum(-1, 0)
    mask = visit(expr, _visitor(block_length=4))
    assert mask is not None
    np.testing.assert_array_equal(mask.values, [False, True, True, True])


def test_time_sum_positive_to_blocks_last_t():
    # sum from 0 to 1: t=3 accesses t+1=4 >= 4, invalid; t=0,1,2 valid
    expr = var("x").time_sum(0, 1)
    mask = visit(expr, _visitor(block_length=4))
    assert mask is not None
    np.testing.assert_array_equal(mask.values, [True, True, True, False])


def test_time_sum_mixed_bounds_blocks_last_t():
    # sum from index 0 to t+1: t=3 accesses 4 >= 4, invalid
    expr = var("x").time_sum_between(0, relative_time(1))
    mask = visit(expr, _visitor(block_length=4))
    assert mask is not None
    np.testing.assert_array_equal(mask.values, [True, True, True, False])


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


# ---------------------------------------------------------------------------
# time_sum with per-component parameter bounds
# ---------------------------------------------------------------------------


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
# Composition — AND of sub-tree masks
# ---------------------------------------------------------------------------


def test_nested_shifts_and_combined():
    # Two time_shift nodes in an addition: shift -1 and shift +1
    # shift -1 blocks t=0; shift +1 blocks t=2 (block=3)
    # combined: t=0 and t=2 invalid, only t=1 valid
    expr = var("x").shift(-1) + var("x").shift(1)
    mask = visit(expr, _visitor(block_length=3))
    assert mask is not None
    np.testing.assert_array_equal(mask.values, [False, True, False])


# ---------------------------------------------------------------------------
# Invalid time amounts — raise ValueError
# ---------------------------------------------------------------------------


def test_variable_in_shift_raises():
    expr = var("x").shift(var("p"))
    with pytest.raises(ValueError, match="cannot contain the variable"):
        visit(expr, _visitor(block_length=4))


def test_time_dependent_param_in_time_sum_bound_raises():
    lag = xr.DataArray([1.0, 2.0, 1.0, 2.0], dims=["time"], coords={"time": range(4)})
    expr = var("x").time_sum(-param("lag"), 0)
    with pytest.raises(ValueError, match="depends on time"):
        visit(expr, _visitor({("m", "lag"): lag}, block_length=4))


# ---------------------------------------------------------------------------
# Shift varying per component, time step or scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lag, expected",
    [
        pytest.param(
            # t + lag = [1, 3, 3, 5] for c1, [2, 2, 4, 4] for c2
            xr.DataArray(
                [[1.0, 2.0, 1.0, 2.0], [2.0, 1.0, 2.0, 1.0]],
                dims=["component", "time"],
                coords={"component": ["c1", "c2"], "time": [0, 1, 2, 3]},
            ),
            [[True, True, True, False], [True, True, False, False]],
            id="time",
        ),
        pytest.param(
            # t + lag = [1, 2, 3, 4] for s1, [2, 3, 4, 5] for s2
            xr.DataArray([1.0, 2.0], dims=["scenario"], coords={"scenario": [0, 1]}),
            [[True, True, True, False], [True, True, False, False]],
            id="scenario",
        ),
    ],
)
def test_shift_by_a_varying_parameter(lag, expected):
    expr = var("x").shift(param("lag"))
    mask = visit(expr, _visitor({("m", "lag"): lag}, block_length=4))
    other = "component" if "component" in lag.dims else "scenario"
    np.testing.assert_array_equal(mask.transpose(other, "time").values, expected)


# ---------------------------------------------------------------------------
# Dropped where a variable is read out of the block; parameters read modulo it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, expected",
    [
        pytest.param(param("d").shift(-1), True, id="shifted-parameter"),
        pytest.param(var("x").eval(4), False, id="index-out"),
        pytest.param(
            var("x").eval(param("d")), [True, False, False, True], id="index-by-d"
        ),
        # d[t-1] = [2, 1, 5, -1]
        pytest.param(
            var("x").eval(param("d").shift(-1)),
            [True, True, False, False],
            id="index-by-d[t-1]",
        ),
        # t + d[t-1] = [2, 2, 7, 2]
        pytest.param(
            var("x").shift(param("d").shift(-1)),
            [True, True, False, True],
            id="shift-by-d[t-1]",
        ),
        pytest.param(var("x").time_sum_between(-1, 1), False, id="absolute-sum"),
        pytest.param(
            var("x").time_sum_between(-1, relative_time(0)), False, id="mixed-sum"
        ),
        # x[s + d[s]] for s in t-1 .. t: out of the block at s = 1 and s = 3
        pytest.param(
            var("x").shift(param("d")).time_sum(-1, 0),
            [True, False, False, False],
            id="sum-of-shift-by-d",
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
def test_variable_read_out_of_the_block(expr, expected):
    d = xr.DataArray([1.0, 5.0, -1.0, 2.0], dims=["time"], coords={"time": range(4)})
    mask = visit(expr, _visitor({("m", "d"): d}))
    np.testing.assert_array_equal(_valid(mask), expected)


def test_time_independent_variable_is_never_dropped():
    visitor = ShiftValidityVisitor(
        model_id="m",
        param_arrays={},
        block_length=4,
        time_independent_variables=frozenset({"x"}),
    )
    assert visit(var("x").shift(-1) + var("x").eval(10), visitor) is None
