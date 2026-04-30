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
Nasty unit tests for VectorizedLinearExprBuilder and the _linopy_add helper.

Covers every overridden method, all error paths, boundary conditions, and the
linopy-specific operand-swap quirk in addition.
"""

import linopy
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gems.expression.expression import (
    AllTimeSumNode,
    LiteralNode,
    MaxNode,
    MinNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ScenarioOperatorNode,
    literal,
    maximum,
    minimum,
    param,
    var,
)
from gems.expression.visitor import visit
from gems.model.port import PortFieldId
from gems.simulation.linearize import VectorizedLinearExprBuilder
from gems.simulation.vectorized_builder import VectorizedBuilderBase, _linopy_add

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lp_model() -> linopy.Model:
    """Bare linopy model — each test gets a fresh one."""
    return linopy.Model()


@pytest.fixture
def comp_time_var(lp_model: linopy.Model) -> linopy.Variable:
    """Variable with dims (component, time): 2 components × 3 timesteps."""
    coords = [
        pd.Index(["c1", "c2"], name="component"),
        pd.Index([0, 1, 2], name="time"),
    ]
    return lp_model.add_variables(lower=0, upper=10, coords=coords, name="x")


@pytest.fixture
def param_da() -> xr.DataArray:
    """Parameter DataArray shaped (component=2, time=3)."""
    return xr.DataArray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dims=["component", "time"],
        coords={"component": ["c1", "c2"], "time": [0, 1, 2]},
    )


@pytest.fixture
def builder(
    comp_time_var: linopy.Variable, param_da: xr.DataArray
) -> VectorizedLinearExprBuilder:
    """Builder pre-loaded with one variable and one parameter, block=3, scenarios=2."""
    return VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={("m", "x"): comp_time_var},
        param_arrays={("m", "p"): param_da},
        port_arrays={},
        block_length=3,
    )


@pytest.fixture
def empty_builder() -> VectorizedLinearExprBuilder:
    """Builder with nothing registered — for all KeyError paths."""
    return VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={},
        port_arrays={},
        block_length=3,
    )


# ---------------------------------------------------------------------------
# 1. _linopy_add helper
# ---------------------------------------------------------------------------


def test_linopy_add_da_plus_da_returns_da() -> None:
    """Two DataArrays: no swap, result is DataArray."""
    a = xr.DataArray(3.0)
    b = xr.DataArray(4.0)
    result = _linopy_add(a, b)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(7.0)


def test_linopy_add_da_plus_variable_swaps(
    lp_model: linopy.Model, comp_time_var: linopy.Variable
) -> None:
    """DataArray on left, linopy.Variable on right — must swap so linopy goes left."""
    da = xr.DataArray(1.0)
    result = _linopy_add(da, comp_time_var)
    # xr.DataArray.__add__(Variable) would raise; the swap makes it work
    assert isinstance(result, linopy.LinearExpression)


def test_linopy_add_da_plus_linear_expr_swaps(
    lp_model: linopy.Model, comp_time_var: linopy.Variable
) -> None:
    """DataArray + LinearExpression must swap."""
    da = xr.DataArray(2.0)
    lin_expr = comp_time_var + comp_time_var  # LinearExpression
    result = _linopy_add(da, lin_expr)
    assert isinstance(result, linopy.LinearExpression)


def test_linopy_add_variable_plus_da_no_swap(
    lp_model: linopy.Model, comp_time_var: linopy.Variable
) -> None:
    """Variable on left already — no swap needed."""
    da = xr.DataArray(5.0)
    result = _linopy_add(comp_time_var, da)
    assert isinstance(result, linopy.LinearExpression)


def test_linopy_add_variable_plus_variable(
    lp_model: linopy.Model, comp_time_var: linopy.Variable
) -> None:
    """Two linopy Variables — no DataArray involved, returns LinearExpression."""
    result = _linopy_add(comp_time_var, comp_time_var)
    assert isinstance(result, linopy.LinearExpression)


# ---------------------------------------------------------------------------
# 2. literal()
# ---------------------------------------------------------------------------


def test_literal_scalar(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(42.0), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(42.0)


def test_literal_zero(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(0.0), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(0.0)


def test_literal_negative(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(-99.5), builder)
    assert float(result) == pytest.approx(-99.5)


# ---------------------------------------------------------------------------
# 3. parameter()
# ---------------------------------------------------------------------------


def test_parameter_found(
    builder: VectorizedLinearExprBuilder, param_da: xr.DataArray
) -> None:
    result = visit(param("p"), builder)
    assert isinstance(result, xr.DataArray)
    xr.testing.assert_equal(result, param_da)


def test_parameter_missing_raises_key_error(
    empty_builder: VectorizedLinearExprBuilder,
) -> None:
    with pytest.raises(KeyError, match="MISSING"):
        visit(param("MISSING"), empty_builder)


def test_parameter_wrong_model_id_raises(param_da: xr.DataArray) -> None:
    """Params keyed under model 'A', but builder has model_id 'B' — KeyError."""
    b = VectorizedLinearExprBuilder(
        model_id="B",
        linopy_vars={},
        param_arrays={("A", "p"): param_da},  # wrong model id
        port_arrays={},
        block_length=3,
    )
    with pytest.raises(KeyError):
        visit(param("p"), b)


# ---------------------------------------------------------------------------
# 4. variable()
# ---------------------------------------------------------------------------


def test_variable_found(
    builder: VectorizedLinearExprBuilder, comp_time_var: linopy.Variable
) -> None:
    result = visit(var("x"), builder)
    assert isinstance(result, linopy.Variable)


def test_variable_returns_correct_dims(
    builder: VectorizedLinearExprBuilder, comp_time_var: linopy.Variable
) -> None:
    result = visit(var("x"), builder)
    assert set(result.dims) == {"component", "time"}


def test_variable_missing_raises_key_error(
    empty_builder: VectorizedLinearExprBuilder,
) -> None:
    with pytest.raises(KeyError, match="GHOST"):
        visit(var("GHOST"), empty_builder)


def test_variable_key_includes_model_id(
    comp_time_var: linopy.Variable, param_da: xr.DataArray
) -> None:
    """Variable keyed under wrong model id must raise KeyError."""
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={("OTHER_MODEL", "x"): comp_time_var},  # wrong model id
        param_arrays={},
        port_arrays={},
        block_length=3,
    )
    with pytest.raises(KeyError):
        visit(var("x"), b)


# ---------------------------------------------------------------------------
# 5. negation()
# ---------------------------------------------------------------------------


def test_negation_of_literal(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(-literal(3), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(-3.0)


def test_negation_of_variable(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(-var("x"), builder)
    assert isinstance(result, linopy.LinearExpression)


def test_double_negation(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(-(-literal(5)), builder)
    assert float(result) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 6. addition() — overridden with _linopy_add
# ---------------------------------------------------------------------------


def test_addition_two_literals(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(1) + literal(2), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(3.0)


def test_addition_da_plus_variable(builder: VectorizedLinearExprBuilder) -> None:
    """param("p") + var("x"): DataArray first, linopy second — requires swap."""
    result = visit(param("p") + var("x"), builder)
    assert isinstance(result, linopy.LinearExpression)


def test_addition_variable_plus_da(builder: VectorizedLinearExprBuilder) -> None:
    """var("x") + param("p"): linopy first, DataArray second — no swap needed."""
    result = visit(var("x") + param("p"), builder)
    assert isinstance(result, linopy.LinearExpression)


def test_addition_three_operands_da_da_variable(
    builder: VectorizedLinearExprBuilder,
) -> None:
    """literal(1) + literal(2) + var("x"): last linopy operand still works."""
    result = visit(literal(1) + literal(2) + var("x"), builder)
    assert isinstance(result, linopy.LinearExpression)


def test_addition_variable_plus_variable(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(var("x") + var("x"), builder)
    assert isinstance(result, linopy.LinearExpression)


def test_addition_three_operands_variable_da_da(
    builder: VectorizedLinearExprBuilder,
) -> None:
    """var("x") + literal(1) + literal(2): first is linopy, rest are DataArray."""
    result = visit(var("x") + literal(1) + literal(2), builder)
    assert isinstance(result, linopy.LinearExpression)


# ---------------------------------------------------------------------------
# 7. multiplication() and division()
# ---------------------------------------------------------------------------


def test_multiplication_scalar_times_variable(
    builder: VectorizedLinearExprBuilder,
) -> None:
    result = visit(literal(2) * var("x"), builder)
    assert isinstance(result, linopy.LinearExpression)


def test_multiplication_two_literals(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(3) * literal(4), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(12.0)


def test_division_da_by_da(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(param("p") / literal(2), builder)
    assert isinstance(result, xr.DataArray)
    expected = xr.DataArray(
        [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
        dims=["component", "time"],
        coords={"component": ["c1", "c2"], "time": [0, 1, 2]},
    )
    xr.testing.assert_allclose(result, expected)


def test_division_by_zero_produces_inf(builder: VectorizedLinearExprBuilder) -> None:
    """xarray silently produces inf on /0 — document expected behavior."""
    result = visit(literal(5) / literal(0), builder)
    assert not np.isfinite(float(result))


# ---------------------------------------------------------------------------
# 8. comparison() always raises
# ---------------------------------------------------------------------------


def test_comparison_always_raises(builder: VectorizedLinearExprBuilder) -> None:
    with pytest.raises(NotImplementedError, match="ComparisonNode"):
        visit(var("x") <= param("p"), builder)


def test_comparison_equal_also_raises(builder: VectorizedLinearExprBuilder) -> None:
    with pytest.raises(NotImplementedError):
        visit(literal(1) == literal(1), builder)


# ---------------------------------------------------------------------------
# 9. floor() / ceil() — linopy guard
# ---------------------------------------------------------------------------


def test_floor_on_da_works(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(2.7).floor(), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(2.0)


def test_floor_on_variable_raises(builder: VectorizedLinearExprBuilder) -> None:
    with pytest.raises(NotImplementedError, match="floor"):
        visit(var("x").floor(), builder)


def test_floor_on_negative_da(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(-2.1).floor(), builder)
    assert float(result) == pytest.approx(-3.0)


def test_ceil_on_da_works(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(2.1).ceil(), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(3.0)


def test_ceil_on_variable_raises(builder: VectorizedLinearExprBuilder) -> None:
    with pytest.raises(NotImplementedError, match="ceil"):
        visit(var("x").ceil(), builder)


def test_ceil_on_exact_integer_da(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(literal(3.0).ceil(), builder)
    assert float(result) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 10. maximum() / minimum() — linopy guard
# ---------------------------------------------------------------------------


def test_maximum_all_da_works(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(maximum(literal(3), literal(1), literal(4)), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(4.0)


def test_maximum_two_da_returns_larger(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(maximum(literal(7), literal(3)), builder)
    assert float(result) == pytest.approx(7.0)


def test_maximum_with_variable_raises(builder: VectorizedLinearExprBuilder) -> None:
    with pytest.raises(NotImplementedError, match="maximum"):
        visit(maximum(var("x"), literal(5)), builder)


def test_maximum_variable_first_also_raises(
    builder: VectorizedLinearExprBuilder,
) -> None:
    with pytest.raises(NotImplementedError):
        visit(maximum(literal(5), var("x")), builder)


def test_minimum_all_da_works(builder: VectorizedLinearExprBuilder) -> None:
    result = visit(minimum(literal(3), literal(1), literal(4)), builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(1.0)


def test_minimum_with_variable_raises(builder: VectorizedLinearExprBuilder) -> None:
    with pytest.raises(NotImplementedError, match="minimum"):
        visit(minimum(literal(2), var("x")), builder)


# ---------------------------------------------------------------------------
# 11. time_shift()
# ---------------------------------------------------------------------------


def _scalar_time_da(values: list) -> xr.DataArray:
    """Helper: single-component DataArray with time dim."""
    return xr.DataArray(
        [values],
        dims=["component", "time"],
        coords={"component": ["c1"], "time": list(range(len(values)))},
    )


def test_time_shift_by_zero(empty_builder: VectorizedLinearExprBuilder) -> None:
    da = _scalar_time_da([1.0, 2.0, 3.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").shift(0), b)
    xr.testing.assert_allclose(result, da)


def test_time_shift_by_one_cycles(empty_builder: VectorizedLinearExprBuilder) -> None:
    """Shift by +1: [1,2,3] → [2,3,1] (element at position t = original[t+1 % T])."""
    da = _scalar_time_da([1.0, 2.0, 3.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").shift(1), b)
    expected = _scalar_time_da([2.0, 3.0, 1.0])
    xr.testing.assert_allclose(result, expected)


def test_time_shift_by_negative_one_cycles(
    empty_builder: VectorizedLinearExprBuilder,
) -> None:
    """Shift by -1: [1,2,3] → [3,1,2]."""
    da = _scalar_time_da([1.0, 2.0, 3.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").shift(-1), b)
    expected = _scalar_time_da([3.0, 1.0, 2.0])
    xr.testing.assert_allclose(result, expected)


def test_time_shift_by_block_length_is_identity() -> None:
    """Shift by block_length should produce the same array (full cycle)."""
    da = _scalar_time_da([10.0, 20.0, 30.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").shift(3), b)  # shift by block_length
    xr.testing.assert_allclose(result, da)


def test_time_shift_on_no_time_dim_is_noop() -> None:
    """Scalar (no time dim) DataArray: shift is a no-op."""
    scalar = xr.DataArray(99.0)
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): scalar},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").shift(1), b)
    assert float(result) == pytest.approx(99.0)


def test_time_shift_coordinates_are_reassigned() -> None:
    """After a shift, time coords must be [0,1,2] not the shifted positions."""
    da = _scalar_time_da([1.0, 2.0, 3.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").shift(1), b)
    assert list(result.coords["time"].values) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 12. time_eval()
# ---------------------------------------------------------------------------


def test_time_eval_selects_correct_timestep() -> None:
    da = _scalar_time_da([10.0, 20.0, 30.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").eval(1), b)
    # Should select time=1, squeezing the time dim
    assert "time" not in result.dims
    np.testing.assert_allclose(result.values.flatten(), [20.0])


def test_time_eval_wraps_beyond_block() -> None:
    """eval(block_length + 1) == eval(1) via modulo."""
    da = _scalar_time_da([10.0, 20.0, 30.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    r1 = visit(param("s").eval(1), b)
    r4 = visit(param("s").eval(4), b)  # 4 % 3 = 1
    xr.testing.assert_allclose(r1, r4)


def test_time_eval_on_no_time_dim() -> None:
    scalar = xr.DataArray(7.0)
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): scalar},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").eval(2), b)
    assert float(result) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 13. all_time_sum()
# ---------------------------------------------------------------------------


def test_all_time_sum_with_time_dim() -> None:
    da = _scalar_time_da([1.0, 2.0, 3.0])  # sum = 6
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").time_sum(), b)
    assert "time" not in result.dims
    np.testing.assert_allclose(result.values.flatten(), [6.0])


def test_all_time_sum_without_time_dim_multiplies_block_length() -> None:
    scalar = xr.DataArray(5.0)
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): scalar},
        port_arrays={},
        block_length=4,
    )
    result = visit(param("s").time_sum(), b)
    assert float(result) == pytest.approx(20.0)  # 5.0 * 4


# ---------------------------------------------------------------------------
# 14. time_sum(from, to)
# ---------------------------------------------------------------------------


def test_time_sum_same_from_and_to_equals_single_shift() -> None:
    """time_sum(0, 0) should equal shift(0) — the operand itself."""
    da = _scalar_time_da([1.0, 2.0, 3.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").time_sum(0, 0), b)
    xr.testing.assert_allclose(result, da)


def test_time_sum_range_sums_correctly() -> None:
    """time_sum(0, 2) on [1,1,1] = 3 per position (each position accumulates 3 cyclic copies)."""
    da = _scalar_time_da([1.0, 1.0, 1.0])
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=3,
    )
    result = visit(param("s").time_sum(0, 2), b)
    np.testing.assert_allclose(result.values.flatten(), [3.0, 3.0, 3.0])


def test_time_sum_with_variable_returns_linear_expression(
    builder: VectorizedLinearExprBuilder,
) -> None:
    result = visit(var("x").time_sum(0, 1), builder)
    assert isinstance(result, linopy.LinearExpression)


# ---------------------------------------------------------------------------
# 15. scenario_operator()
# ---------------------------------------------------------------------------


def test_expectation_with_scenario_dim() -> None:
    da = xr.DataArray(
        [[10.0, 20.0]],
        dims=["component", "scenario"],
        coords={"component": ["c1"], "scenario": [0, 1]},
    )
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): da},
        port_arrays={},
        block_length=1,
    )
    result = visit(param("s").expec(), b)
    # sum([10, 20]) / 2 = 15
    assert "scenario" not in result.dims
    np.testing.assert_allclose(result.values.flatten(), [15.0])


def test_expectation_without_scenario_dim_is_noop() -> None:
    scalar = xr.DataArray(42.0)
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): scalar},
        port_arrays={},
        block_length=1,
    )
    result = visit(param("s").expec(), b)
    assert float(result) == pytest.approx(42.0)


def test_variance_scenario_operator_raises_at_node_construction() -> None:
    """'Variance' is a recognized operator name but raises ValueError on construction
    because the ScenarioOperatorNode validates names at __post_init__... actually,
    'Variance' IS in the valid list. The builder raises NotImplementedError at visit time.
    """
    scalar = xr.DataArray(1.0)
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={("m", "s"): scalar},
        port_arrays={},
        block_length=1,
    )
    # var("s").variance() builds ScenarioOperatorNode with name="Variance"
    expr = param("s").variance()
    with pytest.raises(NotImplementedError, match="Variance"):
        visit(expr, b)


# ---------------------------------------------------------------------------
# 16. port_field()
# ---------------------------------------------------------------------------


def test_port_field_found() -> None:
    key = PortFieldId("port_a", "flow")
    da = xr.DataArray(99.0)
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={},
        port_arrays={key: da},
        block_length=1,
    )
    from gems.expression.expression import port_field

    result = visit(port_field("port_a", "flow"), b)
    assert float(result) == pytest.approx(99.0)


def test_port_field_missing_raises_key_error(
    empty_builder: VectorizedLinearExprBuilder,
) -> None:
    from gems.expression.expression import port_field

    with pytest.raises(KeyError):
        visit(port_field("no_such_port", "flow"), empty_builder)


# ---------------------------------------------------------------------------
# 17. port_field_aggregator()
# ---------------------------------------------------------------------------


def test_port_sum_with_connection_returns_expression() -> None:
    key = PortFieldId("port_a", "flow")
    da = xr.DataArray(7.0)
    b = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={},
        port_arrays={key: da},
        block_length=1,
    )
    from gems.expression.expression import port_field

    expr = port_field("port_a", "flow").sum_connections()
    result = visit(expr, b)
    assert float(result) == pytest.approx(7.0)


def test_port_sum_no_connection_returns_zero(
    empty_builder: VectorizedLinearExprBuilder,
) -> None:
    """PortFieldAggregatorNode with no matching port returns DataArray(0.0)."""
    from gems.expression.expression import port_field

    expr = port_field("absent_port", "flow").sum_connections()
    result = visit(expr, empty_builder)
    assert isinstance(result, xr.DataArray)
    assert float(result) == pytest.approx(0.0)


def test_unsupported_port_aggregator_raises_at_node_construction() -> None:
    """Only 'PortSum' is valid; other aggregator names raise at node construction."""
    from gems.expression.expression import PortFieldAggregatorNode, port_field

    with pytest.raises(NotImplementedError):
        PortFieldAggregatorNode(operand=port_field("p", "f"), aggregator="PortMax")


# ---------------------------------------------------------------------------
# 18. Private helpers
# ---------------------------------------------------------------------------


def test_eval_int_from_integer_literal() -> None:
    result = VectorizedBuilderBase._eval_int(literal(3))
    assert result == 3
    assert isinstance(result, int)


def test_eval_int_from_float_that_is_integer() -> None:
    result = VectorizedBuilderBase._eval_int(literal(3.0))
    assert result == 3
    assert isinstance(result, int)


def test_eval_int_from_non_integer_float_raises() -> None:
    with pytest.raises(ValueError, match="integer"):
        VectorizedBuilderBase._eval_int(literal(3.5))


def test_da_to_int_from_integer_array() -> None:
    da = xr.DataArray(2.0)
    result = VectorizedBuilderBase._da_to_int(da)
    assert result == 2
    assert isinstance(result, int)


def test_da_to_int_from_non_integer_raises() -> None:
    da = xr.DataArray(2.5)
    with pytest.raises(ValueError, match="integer"):
        VectorizedBuilderBase._da_to_int(da)
