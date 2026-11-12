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
Shared abstract base for vectorized expression builders.

Provides :data:`VectorizedExpr`, :func:`_linopy_add`, and
:class:`VectorizedBuilderBase`, the abstract parent of both the pre-solve
(:class:`~gems_runner.simulation.linearize.VectorizedLinearExprBuilder`) and
post-solve (:class:`~gems_runner.simulation.extra_output.VectorizedExtraOutputBuilder`)
visitors.

The only axis of variation between the two concrete builders is how a
``VariableNode`` is evaluated:

- Pre-solve: returns a ``linopy.Variable`` (symbolic decision variable).
- Post-solve: returns an ``xr.DataArray`` of optimal solution values.

All 18 other :class:`~gems_craft.expression.visitor.ExpressionVisitor` methods are
implemented here once, with DataArray-friendly semantics as their default.
``VectorizedLinearExprBuilder`` overrides a small subset to add linopy-specific
behaviour (operand-swap in addition, type guards in nonlinear functions).
"""

import functools
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, FrozenSet, Generic, Optional, Tuple, TypeVar, Union

import linopy
import numpy as np
import xarray as xr

from gems_craft.expression.degree import is_constant
from gems_craft.expression.expression import (
    AbsNode,
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
    DivisionNode,
    DualNode,
    ExpressionNode,
    FloorNode,
    LiteralNode,
    LowerBoundNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ReducedCostNode,
    RelativeTimeNode,
    RoundNode,
    ScenarioOperatorNode,
    SetIndexNode,
    SumOverNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    UpperBoundNode,
    VariableNode,
)
from gems_craft.expression.print import print_expr
from gems_craft.expression.visitor import (
    ExpressionVisitor,
    visit,
)
from gems_craft.model.port import PortFieldId
from gems_runner.expression.evaluate import EvaluationContext, EvaluationVisitor

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

VectorizedExpr = Union[xr.DataArray, linopy.LinearExpression, linopy.Variable]
"""Union of all value types that may appear during vectorized expression building."""

# Backward-compatible alias kept for external callers that imported the old name.
LinopyExpression = VectorizedExpr

T_expr = TypeVar("T_expr", bound=VectorizedExpr)
"""Type variable for the concrete expression type used by a builder subclass."""


# ---------------------------------------------------------------------------
# Module-level helper — also used by optimization.py
# ---------------------------------------------------------------------------


def _linopy_add(a: VectorizedExpr, b: VectorizedExpr) -> VectorizedExpr:
    """Add two linopy-compatible expressions, keeping linopy types on the left.

    ``xr.DataArray.__add__`` does not recognise linopy objects, so
    ``DataArray + LinearExpression`` raises.  Putting the linopy type on the
    left delegates to ``linopy.__add__`` / ``__radd__``, which handles
    DataArrays correctly.  For two DataArrays this degenerates to ``a + b``.
    """
    if isinstance(a, xr.DataArray) and not isinstance(b, xr.DataArray):
        return b + a  # type: ignore[operator]
    return a + b  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class VectorizedBuilderBase(ExpressionVisitor[VectorizedExpr], Generic[T_expr]):
    """
    Abstract base for vectorized expression builders.

    Implements all :class:`~gems_craft.expression.visitor.ExpressionVisitor` methods
    with DataArray-friendly defaults.  The sole abstract method is
    :meth:`variable` — subclasses differ only in how they resolve a
    ``VariableNode`` (symbolic linopy variable vs. solved DataArray value).

    Parameters
    ----------
    model_id:
        The ``model.id`` string of the ``Model`` object whose AST is being
        visited.  Used as the first element of dict lookup keys and in error
        messages.
    param_arrays:
        Mapping from ``(model_id, param_name)`` to a DataArray of parameter
        values, with dims in ``{component, time, scenario}`` (or a subset).
    port_arrays:
        Pre-computed expressions for each ``PortFieldId`` of this model.
        Keyed by ``PortFieldId(port_name, field_name)``.
    block_length:
        Number of time steps in the current time block.
    """

    model_id: str
    param_arrays: Dict[Tuple[str, str], xr.DataArray]
    port_arrays: Dict[PortFieldId, T_expr]
    block_length: int

    # ------------------------------------------------------------------ #
    # Abstract                                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def variable(self, node: VariableNode) -> T_expr: ...

    # ------------------------------------------------------------------ #
    # Leaf nodes                                                            #
    # ------------------------------------------------------------------ #

    def literal(self, node: LiteralNode) -> xr.DataArray:
        return xr.DataArray(node.value)

    def parameter(self, node: ParameterNode) -> xr.DataArray:
        key = (self.model_id, node.name)
        if key not in self.param_arrays:
            raise KeyError(
                f"Parameter {node.name!r} not found for model {self.model_id!r}."
            )
        return self.param_arrays[key]

    # ------------------------------------------------------------------ #
    # Arithmetic operators                                                  #
    # ------------------------------------------------------------------ #

    def negation(self, node: NegationNode) -> VectorizedExpr:
        return -visit(node.operand, self)  # type: ignore[operator,return-value]

    def addition(self, node: AdditionNode) -> VectorizedExpr:
        """Simple left-to-right addition (DataArray default).

        Overridden in
        :class:`~gems_runner.simulation.linearize.VectorizedLinearExprBuilder`
        to handle mixed DataArray / linopy-type operands via :func:`_linopy_add`.
        """
        operands = [visit(op, self) for op in node.operands]
        result = operands[0]
        for op in operands[1:]:
            result = result + op  # type: ignore[operator]
        return result  # type: ignore[return-value]

    def multiplication(self, node: MultiplicationNode) -> VectorizedExpr:
        left = visit(node.left, self)
        right = visit(node.right, self)
        return left * right  # type: ignore[operator,return-value]

    def division(self, node: DivisionNode) -> VectorizedExpr:
        left = visit(node.left, self)
        right = visit(node.right, self)
        return left / right  # type: ignore[operator,return-value]

    def comparison(self, node: ComparisonNode) -> VectorizedExpr:
        raise NotImplementedError(
            f"ComparisonNode is not supported by {type(self).__name__}. "
            "Decompose comparisons into expressions before visiting."
        )

    # ------------------------------------------------------------------ #
    # Time operators                                                        #
    # ------------------------------------------------------------------ #

    def time_shift(self, node: TimeShiftNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        shift = self._time_value(node.time_shift, "Time shift")
        return self._apply_time_shift(operand, shift)  # type: ignore[return-value]

    def time_eval(self, node: TimeEvalNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        if not self._has_dim(operand, "time"):
            return operand  # type: ignore[return-value]
        index = self._time_value(node.eval_time, "Time index")
        return self._take_time(operand, index)  # type: ignore[return-value]

    def time_sum(self, node: TimeSumNode) -> VectorizedExpr:
        if not isinstance(node.from_time, RelativeTimeNode) or not isinstance(
            node.to_time, RelativeTimeNode
        ):
            return self._time_sum_with_absolute_bound(node)
        from_int = self._bound_value(node.from_time.offset)
        to_int = self._bound_value(node.to_time.offset)

        operand = visit(node.operand, self)

        # Fast path: both bounds are the same for every component and scenario.
        if not from_int.dims and not to_int.dims:
            start, end = int(from_int), int(to_int)
            if end < start:  # empty sum
                return operand * 0.0  # type: ignore[operator,return-value]
            result = self._apply_time_shift(operand, start)
            for shift in range(start + 1, end + 1):
                result = _linopy_add(result, self._apply_time_shift(operand, shift))
            return result  # type: ignore[return-value]

        # Slow path: at least one bound varies per component or scenario.
        min_from = int(from_int.values.min())
        max_to = int(to_int.values.max())
        if max_to < min_from:  # empty sum for every component
            return operand * 0.0  # type: ignore[operator,return-value]

        acc: Optional[Any] = None
        for shift in range(min_from, max_to + 1):
            shifted = self._apply_time_shift(operand, shift)
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
            contrib = shifted * mask  # type: ignore[operator]
            acc = contrib if acc is None else _linopy_add(acc, contrib)
        return acc  # type: ignore[return-value]

    def _bound_value(self, node: ExpressionNode) -> xr.DataArray:
        """Integer value of a time sum bound (an absolute bound or a relative
        offset). It is time-independent: this is checked when the model is
        created."""
        result = self._time_value(node, "Time sum bound")
        if isinstance(result, int):
            return xr.DataArray(result)
        if "time" in result.dims:  # cannot happen in a model that passed the check
            raise ValueError(
                f"Model '{self.model_id}': the time sum bound '{print_expr(node)}' "
                "depends on time, but must be time-independent."
            )
        # A time operator such as d[0] leaves a scalar time coordinate.
        return result.drop_vars("time", errors="ignore")

    def _time_sum_bound(self, bound: ExpressionNode, t: xr.DataArray) -> xr.DataArray:
        """Time index of a time sum bound at each time step: ``t + offset`` for a
        relative bound, the bound itself for an absolute one."""
        relative = isinstance(bound, RelativeTimeNode)
        value_node = bound.offset if isinstance(bound, RelativeTimeNode) else bound
        value = self._bound_value(value_node)
        return value + t if relative else value

    def _time_sum_with_absolute_bound(self, node: TimeSumNode) -> VectorizedExpr:
        """Time sum with at least one absolute bound (a time index of the block).

        At time step ``t``, the operand is summed over the positions from the
        start to the end bound, wrapped cyclically into the block, as ``x[N]``.
        With two absolute bounds the result is time-independent. A start after
        the end gives an empty sum.
        """
        T = self.block_length
        t = xr.DataArray(np.arange(T), dims="time")
        start = self._time_sum_bound(node.from_time, t)
        end = self._time_sum_bound(node.to_time, t)
        operand = visit(node.operand, self)

        first, last = int(start.min()), int(end.max())
        if last < first:  # empty sum at every time step
            return operand * 0.0  # type: ignore[operator,return-value]
        if not self._has_dim(operand, "time"):
            # The same value at every position: multiply by their number.
            count = (end - start + 1).clip(min=0).astype(float)
            return operand * count  # type: ignore[operator,return-value]

        # Time steps of the block reached by the positions first..last.
        sources = np.unique(np.arange(first, last + 1) % T)
        source = xr.DataArray(sources, dims="time_src", coords={"time_src": sources})
        # Number of positions start..end that wrap onto each source time step
        # (0 outside the range, more than 1 if the range wraps past it).
        weight = ((end - source) // T - (start - 1 - source) // T).clip(min=0)
        selected = operand.isel(time=sources).rename(time="time_src")  # type: ignore[union-attr]
        # One vectorized product instead of one operation per position.
        return (selected * weight.astype(float)).sum("time_src")  # type: ignore[no-any-return]

    def all_time_sum(self, node: AllTimeSumNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        if self._has_dim(operand, "time"):
            return operand.sum("time")  # type: ignore[union-attr,attr-defined,return-value]
        return operand * self.block_length  # type: ignore[operator,return-value]

    # ------------------------------------------------------------------ #
    # Custom-set operators (not yet supported)                            #
    # ------------------------------------------------------------------ #

    def set_index(self, node: SetIndexNode) -> VectorizedExpr:
        raise NotImplementedError(
            "Custom-set indexing is not yet supported by the vectorized builder."
        )

    def sum_over(self, node: SumOverNode) -> VectorizedExpr:
        raise NotImplementedError(
            "sum_over() is not yet supported by the vectorized builder."
        )

    # ------------------------------------------------------------------ #
    # Scenario operators                                                    #
    # ------------------------------------------------------------------ #

    def scenario_operator(self, node: ScenarioOperatorNode) -> VectorizedExpr:
        if node.name != "Expectation":
            raise NotImplementedError(
                f"Scenario operator {node.name!r} is not supported. "
                "Only 'Expectation' is currently implemented."
            )
        operand = visit(node.operand, self)
        if self._has_dim(operand, "scenario"):
            return operand.sum("scenario") / operand.sizes["scenario"]  # type: ignore[union-attr,attr-defined,operator,return-value]
        return operand  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Port operators                                                        #
    # ------------------------------------------------------------------ #

    def port_field(self, node: PortFieldNode) -> VectorizedExpr:
        key = PortFieldId(node.port_name, node.field_name)
        if key not in self.port_arrays:
            raise KeyError(
                f"No port array found for {node.port_name}.{node.field_name} "
                f"in model {self.model_id!r}."
            )
        return self.port_arrays[key]

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> VectorizedExpr:
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
            # No connections: the sum over an empty set is zero.
            return xr.DataArray(0.0)  # type: ignore[return-value]
        return self.port_arrays[key]

    # ------------------------------------------------------------------ #
    # Math functions (DataArray default — no guard)                         #
    # ------------------------------------------------------------------ #
    # Overridden in VectorizedLinearExprBuilder to raise when operands contain
    # linopy types: these operations cannot be expressed as linear constraints.

    def floor(self, node: FloorNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        return np.floor(operand)  # type: ignore[return-value,arg-type,call-overload]

    def ceil(self, node: CeilNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        return np.ceil(operand)  # type: ignore[return-value,arg-type,call-overload]

    def abs(self, node: AbsNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        return np.abs(operand)  # type: ignore[return-value,arg-type,call-overload]

    def round(self, node: RoundNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        return np.round(operand)  # type: ignore[return-value,arg-type,call-overload]

    def maximum(self, node: MaxNode) -> VectorizedExpr:
        operands = [visit(op, self) for op in node.operands]
        result = operands[0]
        for op in operands[1:]:
            result = xr.where(result >= op, result, op)  # type: ignore[no-untyped-call,assignment,operator]
        return result  # type: ignore[return-value]

    def minimum(self, node: MinNode) -> VectorizedExpr:
        operands = [visit(op, self) for op in node.operands]
        result = operands[0]
        for op in operands[1:]:
            result = xr.where(result <= op, result, op)  # type: ignore[no-untyped-call,assignment,operator]
        return result  # type: ignore[return-value]

    def dual(self, node: DualNode) -> VectorizedExpr:
        raise NotImplementedError(
            f"dual() is only available in the extra-output builder, "
            f"not in {type(self).__name__}."
        )

    def reduced_cost(self, node: ReducedCostNode) -> VectorizedExpr:
        raise NotImplementedError(
            f"reduced_cost() is only available in the extra-output builder, "
            f"not in {type(self).__name__}."
        )

    def lower_bound(self, node: LowerBoundNode) -> VectorizedExpr:
        raise NotImplementedError(
            f"lower_bound() is only available in the extra-output builder, "
            f"not in {type(self).__name__}."
        )

    def upper_bound(self, node: UpperBoundNode) -> VectorizedExpr:
        raise NotImplementedError(
            f"upper_bound() is only available in the extra-output builder, "
            f"not in {type(self).__name__}."
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _apply_time_shift(self, operand: Any, shift: Union[int, xr.DataArray]) -> Any:
        """Apply a cyclic (modulo ``self.block_length``) time shift to *operand*.

        *shift* may vary per component, time step or scenario."""
        t = xr.DataArray(np.arange(self.block_length), dims="time")
        return self._take_time(operand, t + shift)

    def _take_time(self, operand: Any, positions: Any) -> Any:
        """*operand* read at the time steps *positions*, wrapped into the block.

        *positions* is an integer or an integer DataArray: along its dimensions
        (component, time, scenario), each element reads its own time step.
        Works for both ``xr.DataArray`` and linopy Variable/LinearExpression.
        """
        if not self._has_dim(operand, "time"):
            return operand
        T = self.block_length
        if not isinstance(positions, xr.DataArray):
            return operand.isel(time=positions % T)
        # The indexer's time coordinate would conflict with the operand's.
        indexer = (positions % T).drop_vars("time", errors="ignore")
        if not indexer.dims:
            return operand.isel(time=int(indexer))
        if "time" in indexer.dims:
            # Align the result on the positions, not on the time steps read.
            result = operand.isel(time=indexer)
            return result.assign_coords(time=list(range(T)))
        # Time-independent positions: drop the time steps read, left as a
        # coordinate (linopy variables have no drop_vars).
        if isinstance(operand, linopy.Variable):
            operand = operand.to_linexpr()
        return operand.isel(time=indexer).drop_vars("time")

    def _time_value(self, node: ExpressionNode, what: str) -> Union[int, xr.DataArray]:
        """Integer value of a time shift, time index or time sum bound: an int
        when it is a constant, else an integer DataArray computed from the data.
        A non-integer value raises, whether it is a constant or not."""
        try:
            return self._eval_int(node)
        except (KeyError, NotImplementedError):  # depends on the data
            return self._time_amount(node, what)

    def _time_amount(self, node: ExpressionNode, what: str) -> xr.DataArray:
        """Integer values of a time shift, time index or time sum bound that
        depends on the data."""
        value = visit(node, self)
        if not isinstance(value, xr.DataArray):
            raise ValueError(
                f"{what} expression must evaluate to a parameter (DataArray), "
                f"got {type(value).__name__!r}."
            )
        values = value.values
        if np.isnan(values).any():
            raise ValueError(
                f"{what} '{print_expr(node)}' in model '{self.model_id}' has missing "
                "values (NaN)."
            )
        rounded = np.round(values)
        # A tolerance absorbs rounding errors, e.g. d / 3 * 3.
        wrong = values[~np.isclose(values, rounded, rtol=0.0, atol=_INTEGER_TOLERANCE)]
        if wrong.size:
            raise ValueError(
                f"{what} '{print_expr(node)}' in model '{self.model_id}' must take "
                f"integer values, got {_preview(wrong)}."
            )
        return value.copy(data=rounded.astype(int))

    @staticmethod
    def _eval_int(node: ExpressionNode) -> int:
        """Evaluate a constant expression node to an integer (e.g. a time shift)."""
        visitor = EvaluationVisitor(EvaluationContext())
        value = visit(node, visitor)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and abs(value - round(value)) <= _INTEGER_TOLERANCE:
            return int(round(value))
        raise ValueError(
            f"Expected an integer constant expression, got {value!r} from "
            f"'{print_expr(node)}'."
        )

    @staticmethod
    def _has_dim(operand: Any, dim: str) -> bool:
        """Return True if *operand* has a dimension named *dim*."""
        return dim in operand.dims


# Gap to the nearest integer accepted for a time shift, time index or sum bound.
_INTEGER_TOLERANCE = 1e-9


def _preview(values: np.ndarray, count: int = 5) -> str:
    """The first *count* distinct *values*, for an error message."""
    unique = np.unique(values)
    shown = ", ".join(f"{v:g}" for v in unique[:count])
    return f"{shown}, ..." if unique.size > count else shown


# ---------------------------------------------------------------------------
# Shift validity helpers — used for OutOfBoundsMode.DROP
# ---------------------------------------------------------------------------


def _and_mask(
    a: Optional[xr.DataArray], b: Optional[xr.DataArray]
) -> Optional[xr.DataArray]:
    """AND two optional boolean DataArray masks; None means 'all valid'."""
    if a is None:
        return b
    if b is None:
        return a
    return a & b  # type: ignore[operator]


@dataclass(kw_only=True)
class _TimeAmountEvaluator(VectorizedBuilderBase[xr.DataArray]):
    """Evaluates time shifts, time indices and time sum bounds with the
    component data, as the builders do."""

    def variable(self, node: VariableNode) -> xr.DataArray:
        raise ValueError(
            f"A time shift, time index or time sum bound cannot contain the "
            f"variable {node.name!r}."
        )


@dataclass(kw_only=True)
class ShiftValidityVisitor(ExpressionVisitor[Optional[xr.DataArray]]):
    """Validity mask of an expression for ``OutOfBoundsMode.DROP``.

    An instance (component, time step, scenario) is valid iff the expression
    reads no variable at a time step outside ``[0, block_length)``. Time
    operators compose: ``(x[t-1])[t+1]`` reads ``x[t]``. Parameters, including
    those of shift amounts and time indices, are read modulo the block, as in
    cyclic mode.

    Returns ``None`` when every instance is valid, else a boolean DataArray
    where ``True`` marks the valid instances.
    """

    model_id: str
    param_arrays: Dict[Tuple[str, str], xr.DataArray]
    block_length: int
    time_independent_variables: FrozenSet[str] = frozenset()
    # Time steps at which the expression is read, not wrapped into the block;
    # None for t itself.
    positions: Optional[xr.DataArray] = None

    def _evaluator(self) -> _TimeAmountEvaluator:
        return _TimeAmountEvaluator(
            model_id=self.model_id,
            param_arrays=self.param_arrays,
            port_arrays={},
            block_length=self.block_length,
        )

    def _positions(self) -> xr.DataArray:
        if self.positions is None:
            return xr.DataArray(np.arange(self.block_length), dims="time")
        return self.positions

    def _read_at(
        self, operand: ExpressionNode, positions: xr.DataArray
    ) -> Optional[xr.DataArray]:
        return visit(operand, replace(self, positions=positions))

    def _at_positions(self, value: Union[int, xr.DataArray]) -> Any:
        """*value*, computed over the block, at the positions read (modulo the
        block, as parameters are read)."""
        if isinstance(value, int):
            return value
        return self._evaluator()._take_time(value, self._positions())

    def _in_block(self) -> Optional[xr.DataArray]:
        positions = self._positions()
        mask = (positions >= 0) & (positions < self.block_length)
        return None if bool(mask.all()) else mask

    # ------------------------------------------------------------------ #
    # Time operators — move the positions read                              #
    # ------------------------------------------------------------------ #

    def time_shift(self, node: TimeShiftNode) -> Optional[xr.DataArray]:
        if is_constant(node.operand):  # reads no variable: nothing to drop
            return None
        amount = self._evaluator()._time_value(node.time_shift, "Time shift")
        return self._read_at(
            node.operand, self._positions() + self._at_positions(amount)
        )

    def time_eval(self, node: TimeEvalNode) -> Optional[xr.DataArray]:
        if is_constant(node.operand):  # reads no variable: nothing to drop
            return None
        index = self._evaluator()._time_value(node.eval_time, "Time index")
        return self._read_at(node.operand, xr.DataArray(self._at_positions(index)))

    def time_sum(self, node: TimeSumNode) -> Optional[xr.DataArray]:
        start = self._time_sum_position(node.from_time)
        end = self._time_sum_position(node.to_time)
        first, last = int(start.min()), int(end.max())
        if last < first:  # empty sum: the operand is never read
            return None

        # The operand is read at every position first..last, along its own dim.
        dim = f"summed_{id(node)}"
        summed = xr.DataArray(np.arange(first, last + 1), dims=dim)
        operand_mask = self._read_at(node.operand, summed)
        if operand_mask is None:
            return None
        operand_mask, _ = xr.broadcast(operand_mask, summed)

        # Running count of the invalid positions: the count over start..end is
        # the difference of two entries, 0 when every position is valid.
        invalid = (~operand_mask).astype(int)  # type: ignore[operator]
        counted = xr.concat(
            [invalid.isel({dim: [0]}) * 0, invalid.cumsum(dim)], dim=dim
        )
        size = last - first + 1
        upper = (end - first + 1).clip(0, size)
        lower = (start - first).clip(0, size)
        valid = counted.isel({dim: upper}) - counted.isel({dim: lower}) <= 0
        return None if bool(valid.all()) else valid

    def _time_sum_position(self, bound: ExpressionNode) -> xr.DataArray:
        """Position of a time sum bound: offset from the positions read for a
        relative bound, the bound itself for an absolute one."""
        if isinstance(bound, RelativeTimeNode):
            return self._positions() + self._evaluator()._bound_value(bound.offset)
        return self._evaluator()._bound_value(bound)

    # ------------------------------------------------------------------ #
    # Structural nodes — AND-propagate children                             #
    # ------------------------------------------------------------------ #

    def negation(self, node: NegationNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def addition(self, node: AdditionNode) -> Optional[xr.DataArray]:
        return functools.reduce(
            _and_mask, (visit(op, self) for op in node.operands), None
        )

    def multiplication(self, node: MultiplicationNode) -> Optional[xr.DataArray]:
        return _and_mask(visit(node.left, self), visit(node.right, self))

    def division(self, node: DivisionNode) -> Optional[xr.DataArray]:
        return _and_mask(visit(node.left, self), visit(node.right, self))

    def comparison(self, node: ComparisonNode) -> Optional[xr.DataArray]:
        return _and_mask(visit(node.left, self), visit(node.right, self))

    def floor(self, node: FloorNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def ceil(self, node: CeilNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def abs(self, node: AbsNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def round(self, node: RoundNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def maximum(self, node: MaxNode) -> Optional[xr.DataArray]:
        return functools.reduce(
            _and_mask, (visit(op, self) for op in node.operands), None
        )

    def minimum(self, node: MinNode) -> Optional[xr.DataArray]:
        return functools.reduce(
            _and_mask, (visit(op, self) for op in node.operands), None
        )

    def all_time_sum(self, node: AllTimeSumNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def set_index(self, node: SetIndexNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def sum_over(self, node: SumOverNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def port_field_aggregator(
        self, node: PortFieldAggregatorNode
    ) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    # ------------------------------------------------------------------ #
    # Leaf nodes — variables must be read in the block                      #
    # ------------------------------------------------------------------ #

    def literal(self, node: LiteralNode) -> Optional[xr.DataArray]:
        return None

    def parameter(self, node: ParameterNode) -> Optional[xr.DataArray]:
        return None

    def variable(self, node: VariableNode) -> Optional[xr.DataArray]:
        if node.name in self.time_independent_variables:
            return None
        return self._in_block()

    def port_field(self, node: PortFieldNode) -> Optional[xr.DataArray]:
        # Port fields are never checked: they are read as in cyclic mode.
        return None

    def dual(self, node: DualNode) -> Optional[xr.DataArray]:
        return None

    def reduced_cost(self, node: ReducedCostNode) -> Optional[xr.DataArray]:
        return None

    def lower_bound(self, node: LowerBoundNode) -> Optional[xr.DataArray]:
        return None

    def upper_bound(self, node: UpperBoundNode) -> Optional[xr.DataArray]:
        return None
