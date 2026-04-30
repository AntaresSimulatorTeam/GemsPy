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

"""Unit tests for ShiftValidityVisitor and _ShiftAmountEvaluator."""

import numpy as np
import pytest
import xarray as xr

from gems.expression.expression import literal, param, var
from gems.expression.visitor import visit
from gems.simulation.vectorized_builder import ShiftValidityVisitor


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
# Arithmetic over leaves — None propagates
# ---------------------------------------------------------------------------


def test_addition_of_params_returns_none(per_comp_lag):
    expr = param("lag") + literal(1.0)
    assert visit(expr, _visitor(per_comp_lag)) is None


def test_negation_of_param_returns_none(per_comp_lag):
    expr = -param("lag")
    assert visit(expr, _visitor(per_comp_lag)) is None


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


def test_time_shift_zero_returns_all_true():
    expr = var("x").shift(0)
    mask = visit(expr, _visitor(block_length=3))
    assert mask is not None
    assert bool(mask.all())


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


def test_shift_inside_multiplication():
    # shift -1 inside a multiplication: same validity as bare shift -1
    expr = literal(2.0) * var("x").shift(-1)
    mask = visit(expr, _visitor(block_length=3))
    assert mask is not None
    np.testing.assert_array_equal(mask.values, [False, True, True])


# ---------------------------------------------------------------------------
# Unevaluable shift — raises ValueError
# ---------------------------------------------------------------------------


def test_unevaluable_shift_raises():
    # shift = var("p") is a variable, not a parameter → unevaluable → error
    shift_expr = var("p")
    expr = var("x").shift(shift_expr)
    with pytest.raises(ValueError, match="not evaluable"):
        visit(expr, _visitor(block_length=4))


# ---------------------------------------------------------------------------
# Time- or scenario-dependent parameter in shift — raises specific error
# ---------------------------------------------------------------------------


@pytest.fixture
def time_dependent_lag():
    """Lag parameter that varies over both component and time."""
    return {
        ("m", "lag"): xr.DataArray(
            [[1.0, 2.0, 1.0, 2.0], [2.0, 1.0, 2.0, 1.0]],
            dims=["component", "time"],
            coords={"component": ["c1", "c2"], "time": [0, 1, 2, 3]},
        )
    }


@pytest.fixture
def scenario_dependent_lag():
    """Lag parameter that varies over scenario."""
    return {
        ("m", "lag"): xr.DataArray(
            [1.0, 2.0],
            dims=["scenario"],
            coords={"scenario": ["s1", "s2"]},
        )
    }


def test_time_dependent_param_in_shift_raises(time_dependent_lag):
    expr = var("x").shift(param("lag"))
    with pytest.raises(ValueError, match="depends on time"):
        visit(expr, _visitor(time_dependent_lag, block_length=4))


def test_time_dependent_param_in_time_sum_bound_raises(time_dependent_lag):
    expr = var("x").time_sum(-param("lag"), 0)
    with pytest.raises(ValueError, match="depends on time"):
        visit(expr, _visitor(time_dependent_lag, block_length=4))


def test_scenario_dependent_param_in_shift_raises(scenario_dependent_lag):
    expr = var("x").shift(param("lag"))
    with pytest.raises(ValueError, match="depends on scenario"):
        visit(expr, _visitor(scenario_dependent_lag, block_length=4))
