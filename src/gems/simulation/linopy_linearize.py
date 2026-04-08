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
Vectorized linopy expression builder.

Provides :class:`VectorizedLinopyBuilder`, an :class:`ExpressionVisitor` that
traverses a model-level AST once and produces a linopy ``LinearExpression``
covering all components × all time steps × all scenarios simultaneously.
Each AST node type (arithmetic, time operators, port fields, scenario
operators) is handled by a dedicated visitor method that operates on
xarray-backed linopy objects, enabling vectorized constraint generation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import linopy
import numpy as np
import xarray as xr

from gems.expression.evaluate import EvaluationContext, EvaluationVisitor
from gems.expression.expression import (
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
    ComponentParameterNode,
    ComponentVariableNode,
    DivisionNode,
    ExpressionNode,
    FloorNode,
    LiteralNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    VariableNode,
)
from gems.expression.visitor import ExpressionVisitor, visit
from gems.model.port import PortFieldId

# Union of all possible return types from the vectorized visitor.
LinopyExpression = Union[xr.DataArray, linopy.LinearExpression, linopy.Variable]


def _eval_int(node: ExpressionNode) -> int:
    """Evaluate a constant expression node to an integer (e.g. a time shift)."""
    visitor = EvaluationVisitor(EvaluationContext())
    value = visit(node, visitor)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(
        f"Expected an integer constant expression, got {value!r} from {node!r}."
    )


def _da_to_int(da: xr.DataArray) -> int:
    """Extract the first element of a DataArray as an integer (for constant time shifts)."""
    val = float(da.values.flat[0])
    if not val.is_integer():
        raise ValueError(
            f"Expected integer DataArray value for time shift, got {val!r}."
        )
    return int(val)


def _has_dim(operand: LinopyExpression, dim: str) -> bool:
    """Return True if the operand has the named dimension."""
    return dim in operand.dims


def _linopy_add(a: LinopyExpression, b: LinopyExpression) -> LinopyExpression:
    """Add two linopy-compatible expressions, keeping linopy types on the left.

    xarray's DataArray.__add__ does not know about linopy types, so
    ``DataArray + LinearExpression`` fails unless the linopy operand is on the
    left.  For two DataArrays, this reduces to plain ``a + b``.
    """
    if isinstance(a, xr.DataArray) and not isinstance(b, xr.DataArray):
        return b + a  # type: ignore[operator]
    return a + b  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Shared time-operator helpers
#
# These module-level functions encapsulate the cyclic-shift / masked-sum logic
# shared by VectorizedLinopyBuilder and VectorizedExtraOutputBuilder.
# They accept any ExpressionVisitor (via `visitor: Any`) and use _linopy_add
# for accumulation, which degenerates to plain `+` for pure-DataArray visitors.
# ---------------------------------------------------------------------------


def _apply_time_shift(
    operand: LinopyExpression, shift: int, block_length: int
) -> LinopyExpression:
    """Apply a cyclic (modulo block_length) time shift to *operand*.

    Works for both xr.DataArray and linopy Variable/LinearExpression operands.
    Coordinates are reassigned after isel so that subsequent xarray arithmetic
    aligns positionally rather than by the shifted coordinate values.
    """
    if not _has_dim(operand, "time"):
        return operand
    T = block_length
    positions = (np.arange(T) + shift) % T
    indexer = xr.DataArray(positions, dims="time")
    result = operand.isel(time=indexer)  # type: ignore[union-attr]
    if "time" in result.coords:  # type: ignore[operator]
        result = result.assign_coords(time=list(range(T)))  # type: ignore[union-attr]
    return result


def _eval_int_expr(node: ExpressionNode, visitor: Any) -> int:
    """Evaluate a constant integer expression, falling back to visitor if needed.

    Tries the static evaluator first; if that fails (e.g. the expression
    references a model parameter), evaluates it via the visitor and extracts
    the scalar integer.
    """
    try:
        return _eval_int(node)
    except KeyError:
        result = visit(node, visitor)
        if isinstance(result, xr.DataArray):
            return _da_to_int(result)
        raise ValueError(
            f"Expected a constant integer expression for time operation, "
            f"got {result!r} from {node!r}."
        )


def _time_shift(
    node: TimeShiftNode, visitor: Any, block_length: int
) -> LinopyExpression:
    """Evaluate a TimeShiftNode using *visitor* for sub-expression evaluation."""
    operand: LinopyExpression = visit(node.operand, visitor)

    # Fast path: compile-time integer constant.
    try:
        shift = _eval_int(node.time_shift)
        return _apply_time_shift(operand, shift, block_length)
    except (ValueError, KeyError):
        pass

    shift_result: LinopyExpression = visit(node.time_shift, visitor)
    if not isinstance(shift_result, xr.DataArray):
        raise ValueError(
            f"Time shift expression must evaluate to a parameter (DataArray), "
            f"got {type(shift_result).__name__!r}."
        )
    if not shift_result.dims:
        return _apply_time_shift(operand, _da_to_int(shift_result), block_length)

    # Slow path: per-component shift — sum masked contributions.
    shift_int = shift_result.astype(int)
    unique_shifts = np.unique(shift_int.values)
    acc: Optional[LinopyExpression] = None
    for s in unique_shifts:
        mask: xr.DataArray = (shift_int == s).astype(float)
        shifted = _apply_time_shift(operand, int(s), block_length)
        contrib: LinopyExpression = shifted * mask  # type: ignore[operator]
        acc = contrib if acc is None else _linopy_add(acc, contrib)
    return acc  # type: ignore[return-value]


def _time_eval(node: TimeEvalNode, visitor: Any, block_length: int) -> LinopyExpression:
    """Evaluate a TimeEvalNode: select operand at a fixed absolute timestep."""
    timestep = _eval_int_expr(node.eval_time, visitor) % block_length
    operand: LinopyExpression = visit(node.operand, visitor)
    if not _has_dim(operand, "time"):
        return operand
    return operand.isel(time=timestep)  # type: ignore[union-attr]


def _time_sum(node: TimeSumNode, visitor: Any, block_length: int) -> LinopyExpression:
    """Evaluate a TimeSumNode: sum operand over [from_shift, to_shift] (cyclic)."""
    try:
        from_shift_scalar: Optional[int] = _eval_int(node.from_time)
    except (ValueError, KeyError):
        from_shift_scalar = None
    try:
        to_shift_scalar: Optional[int] = _eval_int(node.to_time)
    except (ValueError, KeyError):
        to_shift_scalar = None

    operand: LinopyExpression = visit(node.operand, visitor)

    # Fast path: both bounds are compile-time integer constants.
    if from_shift_scalar is not None and to_shift_scalar is not None:
        result: LinopyExpression = _apply_time_shift(
            operand, from_shift_scalar, block_length
        )
        for shift in range(from_shift_scalar + 1, to_shift_scalar + 1):
            result = _linopy_add(
                result, _apply_time_shift(operand, shift, block_length)
            )
        return result

    # Slow path: at least one bound depends on a parameter (per-component DataArray).
    from_da: LinopyExpression = (
        xr.DataArray(float(from_shift_scalar))
        if from_shift_scalar is not None
        else visit(node.from_time, visitor)
    )
    to_da: LinopyExpression = (
        xr.DataArray(float(to_shift_scalar))
        if to_shift_scalar is not None
        else visit(node.to_time, visitor)
    )
    if not isinstance(from_da, xr.DataArray):
        raise ValueError(
            f"time_sum from_time must be a parameter expression (DataArray), "
            f"got {type(from_da).__name__!r}."
        )
    if not isinstance(to_da, xr.DataArray):
        raise ValueError(
            f"time_sum to_time must be a parameter expression (DataArray), "
            f"got {type(to_da).__name__!r}."
        )
    from_int = from_da.astype(int)
    to_int = to_da.astype(int)
    min_from = int(from_int.values.min())
    max_to = int(to_int.values.max())

    acc: Optional[LinopyExpression] = None
    for shift in range(min_from, max_to + 1):
        shifted = _apply_time_shift(operand, shift, block_length)
        include_from = (
            (from_int <= shift).astype(float)
            if isinstance(from_int, xr.DataArray) and from_int.dims
            else xr.DataArray(1.0)
        )
        include_to = (
            (to_int >= shift).astype(float)
            if isinstance(to_int, xr.DataArray) and to_int.dims
            else xr.DataArray(1.0)
        )
        mask: xr.DataArray = include_from * include_to
        contrib: LinopyExpression = shifted * mask  # type: ignore[operator]
        acc = contrib if acc is None else _linopy_add(acc, contrib)
    return acc  # type: ignore[return-value]


def _all_time_sum(
    node: AllTimeSumNode, visitor: Any, block_length: int
) -> LinopyExpression:
    """Sum over all time steps, or multiply by block_length if time-independent."""
    operand: LinopyExpression = visit(node.operand, visitor)
    if _has_dim(operand, "time"):
        return operand.sum("time")  # type: ignore[union-attr]
    return operand * block_length  # type: ignore[operator]


@dataclass
class VectorizedLinopyBuilder(ExpressionVisitor[LinopyExpression]):
    """
    Builds a linopy LinearExpression from a model-level AST.

    Receives pre-computed linopy variables, parameter DataArrays, and port arrays
    indexed on the [component, time, scenario] dimensions (or a subset thereof).
    Produces a result with the same combined dimensions.

    Parameters
    ----------
    model_key:
        Python id() of the Model object whose AST is being visited.
        Using id() (not model.id string) ensures two distinct Model objects
        with the same .id string (e.g. two "GEN" models) are never confused.
    model_name:
        Human-readable model identifier (model.id) used only for error messages.
    linopy_vars:
        Mapping from (model_key, var_name) to the corresponding linopy Variable,
        with dims at least [component] and optionally [time, scenario].
    param_arrays:
        Mapping from (model_key, param_name) to a DataArray of parameter values,
        with dims in {component, time, scenario} (or a subset).
    port_arrays:
        Pre-computed linopy expressions for each PortFieldId of this model,
        resulting from the incidence-matrix port resolution pass.
        Keyed by PortFieldId(port_name, field_name).
    block_length:
        Number of time steps in the current time block.
    scenarios_count:
        Number of scenarios.
    """

    model_key: int
    model_name: str
    linopy_vars: Dict[Tuple[int, str], linopy.Variable]
    param_arrays: Dict[Tuple[int, str], xr.DataArray]
    port_arrays: Dict[PortFieldId, LinopyExpression]
    block_length: int
    scenarios_count: int

    # ------------------------------------------------------------------ #
    # Leaf nodes                                                            #
    # ------------------------------------------------------------------ #

    def literal(self, node: LiteralNode) -> xr.DataArray:
        return xr.DataArray(node.value)

    def variable(self, node: VariableNode) -> linopy.Variable:
        key = (self.model_key, node.name)
        if key not in self.linopy_vars:
            raise KeyError(
                f"Variable {node.name!r} not found for model {self.model_name!r}. "
                "Ensure all linopy variables are created before building constraints."
            )
        return self.linopy_vars[key]

    def parameter(self, node: ParameterNode) -> xr.DataArray:
        key = (self.model_key, node.name)
        if key not in self.param_arrays:
            raise KeyError(
                f"Parameter {node.name!r} not found for model {self.model_name!r}. "
                "Ensure all parameter arrays are built before visiting constraints."
            )
        return self.param_arrays[key]

    # ------------------------------------------------------------------ #
    # Arithmetic operators                                                  #
    # ------------------------------------------------------------------ #

    def negation(self, node: NegationNode) -> LinopyExpression:
        return -visit(node.operand, self)  # type: ignore[operator]

    def addition(self, node: AdditionNode) -> LinopyExpression:
        operands = [visit(op, self) for op in node.operands]
        result: LinopyExpression = operands[0]
        for op in operands[1:]:
            # xr.DataArray.__add__(linopy_type) fails because xarray does not
            # recognise linopy objects. Swap operands so the linopy side drives
            # the operation (linopy.__radd__ / __add__ handles DataArrays).
            if isinstance(result, xr.DataArray) and isinstance(
                op, (linopy.Variable, linopy.LinearExpression)
            ):
                result = op + result  # type: ignore[operator]
            else:
                result = result + op  # type: ignore[operator]
        return result

    def multiplication(self, node: MultiplicationNode) -> LinopyExpression:
        left = visit(node.left, self)
        right = visit(node.right, self)
        return left * right  # type: ignore[operator]

    def division(self, node: DivisionNode) -> LinopyExpression:
        left = visit(node.left, self)
        right = visit(node.right, self)
        return left / right  # type: ignore[operator]

    def comparison(self, node: ComparisonNode) -> LinopyExpression:
        raise NotImplementedError(
            "ComparisonNode should not appear when visiting with VectorizedLinopyBuilder. "
            "Decompose the constraint into expression + bounds before calling visit()."
        )

    # ------------------------------------------------------------------ #
    # Time operators                                                        #
    # ------------------------------------------------------------------ #

    def time_shift(self, node: TimeShiftNode) -> LinopyExpression:
        return _time_shift(node, self, self.block_length)  # type: ignore[return-value]

    def time_eval(self, node: TimeEvalNode) -> LinopyExpression:
        return _time_eval(node, self, self.block_length)  # type: ignore[return-value]

    def time_sum(self, node: TimeSumNode) -> LinopyExpression:
        return _time_sum(node, self, self.block_length)  # type: ignore[return-value]

    def all_time_sum(self, node: AllTimeSumNode) -> LinopyExpression:
        return _all_time_sum(node, self, self.block_length)  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Scenario operators                                                    #
    # ------------------------------------------------------------------ #

    def scenario_operator(self, node: ScenarioOperatorNode) -> LinopyExpression:
        if node.name != "Expectation":
            raise NotImplementedError(
                f"Scenario operator {node.name!r} is not supported. "
                "Only 'Expectation' is currently implemented."
            )
        operand = visit(node.operand, self)
        if _has_dim(operand, "scenario"):
            return operand.sum("scenario") / self.scenarios_count  # type: ignore[union-attr,operator]
        return operand

    # ------------------------------------------------------------------ #
    # Port operators                                                        #
    # ------------------------------------------------------------------ #

    def port_field(self, node: PortFieldNode) -> LinopyExpression:
        key = PortFieldId(node.port_name, node.field_name)
        if key not in self.port_arrays:
            raise KeyError(
                f"No port array found for {node.port_name}.{node.field_name} "
                f"in model {self.model_name!r}. "
                "Port arrays must be pre-computed before building constraints."
            )
        return self.port_arrays[key]

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> LinopyExpression:
        if node.aggregator != "PortSum":
            raise NotImplementedError(
                f"Port aggregator {node.aggregator!r} is not supported. "
                "Only 'PortSum' is currently implemented."
            )
        if not isinstance(node.operand, PortFieldNode):
            raise ValueError(
                f"PortFieldAggregatorNode operand must be a PortFieldNode, "
                f"got {type(node.operand).__name__!r}."
            )
        port_field_node: PortFieldNode = node.operand
        key = PortFieldId(port_field_node.port_name, port_field_node.field_name)
        if key not in self.port_arrays:
            # No connections: the sum over an empty set is zero
            return xr.DataArray(0.0)
        return self.port_arrays[key]

    # ------------------------------------------------------------------ #
    # Math functions (parameter-only)                                       #
    # ------------------------------------------------------------------ #

    def floor(self, node: FloorNode) -> LinopyExpression:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.floor(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "floor() is only supported for parameter (DataArray) expressions."
        )

    def ceil(self, node: CeilNode) -> LinopyExpression:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.ceil(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "ceil() is only supported for parameter (DataArray) expressions."
        )

    def maximum(self, node: MaxNode) -> LinopyExpression:
        operands = [visit(op, self) for op in node.operands]
        if all(isinstance(op, xr.DataArray) for op in operands):
            result: xr.DataArray = operands[0]  # type: ignore[assignment]
            for op in operands[1:]:
                result = xr.where(result >= op, result, op)  # type: ignore[no-untyped-call]
            return result
        raise NotImplementedError(
            "maximum() is only supported for parameter (DataArray) expressions."
        )

    def minimum(self, node: MinNode) -> LinopyExpression:
        operands = [visit(op, self) for op in node.operands]
        if all(isinstance(op, xr.DataArray) for op in operands):
            result = operands[0]
            for op in operands[1:]:
                result = xr.where(result <= op, result, op)  # type: ignore[no-untyped-call,assignment]
            return result
        raise NotImplementedError(
            "minimum() is only supported for parameter (DataArray) expressions."
        )

    # ------------------------------------------------------------------ #
    # Nodes that should not appear in model-level ASTs                      #
    # ------------------------------------------------------------------ #

    def comp_parameter(self, node: ComponentParameterNode) -> LinopyExpression:
        raise ValueError(
            f"ComponentParameterNode {node!r} should not appear in a model-level AST. "
            "Did you accidentally call add_component_context() before the vectorized pipeline?"
        )

    def comp_variable(self, node: ComponentVariableNode) -> LinopyExpression:
        raise ValueError(
            f"ComponentVariableNode {node!r} should not appear in a model-level AST."
        )
