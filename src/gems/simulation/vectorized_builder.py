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
(:class:`~gems.simulation.linearize.VectorizedLinearExprBuilder`) and
post-solve (:class:`~gems.simulation.extra_output.VectorizedExtraOutputBuilder`)
visitors.

The only axis of variation between the two concrete builders is how a
``VariableNode`` is evaluated:

- Pre-solve: returns a ``linopy.Variable`` (symbolic decision variable).
- Post-solve: returns an ``xr.DataArray`` of optimal solution values.

All 18 other :class:`~gems.expression.visitor.ExpressionVisitor` methods are
implemented here once, with DataArray-friendly semantics as their default.
``VectorizedLinearExprBuilder`` overrides a small subset to add linopy-specific
behaviour (operand-swap in addition, type guards in nonlinear functions).
"""

import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import linopy
import numpy as np
import xarray as xr

from gems.expression.evaluate import EvaluationContext, EvaluationVisitor
from gems.expression.expression import (
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
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
from gems.expression.visitor import (
    ExpressionVisitor,
    ExpressionVisitorOperations,
    visit,
)
from gems.model.port import PortFieldId

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

    Implements all :class:`~gems.expression.visitor.ExpressionVisitor` methods
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
        :class:`~gems.simulation.linearize.VectorizedLinearExprBuilder`
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

        # Fast path: compile-time integer constant.
        try:
            shift = self._eval_int(node.time_shift)
            return self._apply_time_shift(operand, shift)  # type: ignore[return-value]
        except (ValueError, KeyError):
            pass

        shift_result = visit(node.time_shift, self)
        if not isinstance(shift_result, xr.DataArray):
            raise ValueError(
                f"Time shift expression must evaluate to a parameter (DataArray), "
                f"got {type(shift_result).__name__!r}."
            )
        if not shift_result.dims:
            return self._apply_time_shift(  # type: ignore[return-value]
                operand, self._da_to_int(shift_result)
            )

        # Slow path: per-component shift — sum masked contributions.
        shift_int = shift_result.astype(int)
        unique_shifts = np.unique(shift_int.values)
        acc: Optional[Any] = None
        for s in unique_shifts:
            mask: xr.DataArray = (shift_int == s).astype(float)
            shifted = self._apply_time_shift(operand, int(s))
            contrib = shifted * mask  # type: ignore[operator]
            acc = contrib if acc is None else _linopy_add(acc, contrib)
        return acc  # type: ignore[return-value]

    def time_eval(self, node: TimeEvalNode) -> VectorizedExpr:
        timestep = self._eval_int_expr(node.eval_time) % self.block_length
        operand = visit(node.operand, self)
        if not self._has_dim(operand, "time"):
            return operand  # type: ignore[return-value]
        return operand.isel(time=timestep)  # type: ignore[union-attr,attr-defined,return-value]

    def time_sum(self, node: TimeSumNode) -> VectorizedExpr:
        try:
            from_shift_scalar: Optional[int] = self._eval_int(node.from_time)
        except (ValueError, KeyError):
            from_shift_scalar = None
        try:
            to_shift_scalar: Optional[int] = self._eval_int(node.to_time)
        except (ValueError, KeyError):
            to_shift_scalar = None

        operand = visit(node.operand, self)

        # Fast path: both bounds are compile-time integer constants.
        if from_shift_scalar is not None and to_shift_scalar is not None:
            result = self._apply_time_shift(operand, from_shift_scalar)
            for shift in range(from_shift_scalar + 1, to_shift_scalar + 1):
                result = _linopy_add(result, self._apply_time_shift(operand, shift))
            return result  # type: ignore[return-value]

        # Slow path: at least one bound depends on a parameter (per-component).
        from_da = (
            xr.DataArray(float(from_shift_scalar))
            if from_shift_scalar is not None
            else visit(node.from_time, self)
        )
        to_da = (
            xr.DataArray(float(to_shift_scalar))
            if to_shift_scalar is not None
            else visit(node.to_time, self)
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

    def all_time_sum(self, node: AllTimeSumNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        if self._has_dim(operand, "time"):
            return operand.sum("time")  # type: ignore[union-attr,attr-defined,return-value]
        return operand * self.block_length  # type: ignore[operator,return-value]

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

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _apply_time_shift(self, operand: Any, shift: int) -> Any:
        """Apply a cyclic (modulo ``self.block_length``) time shift to *operand*.

        Works for both ``xr.DataArray`` and linopy Variable/LinearExpression.
        Coordinates are reassigned after ``isel`` so that subsequent xarray
        arithmetic aligns positionally rather than by the shifted values.
        """
        if not self._has_dim(operand, "time"):
            return operand
        T = self.block_length
        positions = (np.arange(T) + shift) % T
        indexer = xr.DataArray(positions, dims="time")
        result = operand.isel(time=indexer)  # type: ignore[union-attr]
        if "time" in result.coords:  # type: ignore[operator]
            result = result.assign_coords(time=list(range(T)))  # type: ignore[union-attr]
        return result

    @staticmethod
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

    @staticmethod
    def _da_to_int(da: xr.DataArray) -> int:
        """Extract the first element of a DataArray as an integer."""
        val = float(da.values.flat[0])
        if not val.is_integer():
            raise ValueError(
                f"Expected integer DataArray value for time shift, got {val!r}."
            )
        return int(val)

    def _eval_int_expr(self, node: ExpressionNode) -> int:
        """Evaluate a constant integer expression, falling back to ``self`` if needed.

        Tries the static evaluator first; if that fails (e.g. the expression
        references a model parameter), evaluates it via ``self`` and extracts
        the scalar integer.
        """
        try:
            return self._eval_int(node)
        except KeyError:
            result = visit(node, self)
            if isinstance(result, xr.DataArray):
                return self._da_to_int(result)
            raise ValueError(
                f"Expected a constant integer expression for time operation, "
                f"got {result!r} from {node!r}."
            )

    @staticmethod
    def _has_dim(operand: Any, dim: str) -> bool:
        """Return True if *operand* has a dimension named *dim*."""
        return dim in operand.dims


# ---------------------------------------------------------------------------
# Shift validity helpers — used for OutOfBoundsMode.DROP
# ---------------------------------------------------------------------------


class _TimeDependentShiftError(ValueError):
    """Raised when a shift amount resolves to a time-dependent parameter."""


class _ScenarioDependentShiftError(ValueError):
    """Raised when a shift amount resolves to a scenario-dependent parameter."""


@dataclass(kw_only=True)
class _ShiftAmountEvaluator(ExpressionVisitorOperations[xr.DataArray]):
    """Evaluate a shift-amount sub-expression to an xr.DataArray.

    Handles literals, parameter references, and arithmetic over them.
    All time/scenario/port/variable nodes raise so that callers can catch
    and treat the shift as unevaluable.
    """

    model_id: str
    param_arrays: Dict[Tuple[str, str], xr.DataArray]

    def literal(self, node: LiteralNode) -> xr.DataArray:
        return xr.DataArray(node.value)

    def parameter(self, node: ParameterNode) -> xr.DataArray:
        da = self.param_arrays[(self.model_id, node.name)]
        if "time" in da.dims:
            raise _TimeDependentShiftError(
                f"Parameter '{node.name}' depends on time and cannot be used as a"
                " shift amount."
            )
        if "scenario" in da.dims:
            raise _ScenarioDependentShiftError(
                f"Parameter '{node.name}' depends on scenario and cannot be used as a"
                " shift amount."
            )
        return da

    def variable(self, node: VariableNode) -> xr.DataArray:
        raise ValueError("VariableNode cannot appear in a time-shift amount")

    def comparison(self, node: ComparisonNode) -> xr.DataArray:
        raise NotImplementedError

    def time_shift(self, node: TimeShiftNode) -> xr.DataArray:
        raise NotImplementedError

    def time_eval(self, node: TimeEvalNode) -> xr.DataArray:
        raise NotImplementedError

    def time_sum(self, node: TimeSumNode) -> xr.DataArray:
        raise NotImplementedError

    def all_time_sum(self, node: AllTimeSumNode) -> xr.DataArray:
        raise NotImplementedError

    def scenario_operator(self, node: ScenarioOperatorNode) -> xr.DataArray:
        raise NotImplementedError

    def port_field(self, node: PortFieldNode) -> xr.DataArray:
        raise NotImplementedError

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> xr.DataArray:
        raise NotImplementedError

    def floor(self, node: FloorNode) -> xr.DataArray:
        return np.floor(visit(node.operand, self))  # type: ignore[return-value]

    def ceil(self, node: CeilNode) -> xr.DataArray:
        return np.ceil(visit(node.operand, self))  # type: ignore[return-value]

    def maximum(self, node: MaxNode) -> xr.DataArray:
        ops = [visit(op, self) for op in node.operands]
        return functools.reduce(
            lambda a, b: xr.where(a >= b, a, b), ops  # type: ignore[return-value,no-untyped-call]
        )

    def minimum(self, node: MinNode) -> xr.DataArray:
        ops = [visit(op, self) for op in node.operands]
        return functools.reduce(
            lambda a, b: xr.where(a <= b, a, b), ops  # type: ignore[return-value,no-untyped-call]
        )


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
class ShiftValidityVisitor(ExpressionVisitor[Optional[xr.DataArray]]):
    """Walk an expression AST and compute a boolean [component, time] validity mask.

    Returns ``None`` when no time-shifting nodes are encountered (all instances
    are valid).  Otherwise returns a boolean DataArray with at least a ``time``
    dimension (and a ``component`` dimension when shift amounts vary per
    component) where ``True`` marks valid constraint instances.

    Used for ``OutOfBoundsMode.DROP``: a ``(component, time)`` entry is
    ``True`` iff every time-shifted access in the expression falls within
    ``[0, block_length)``.
    """

    model_id: str
    param_arrays: Dict[Tuple[str, str], xr.DataArray]
    block_length: int

    def _eval_as_da(self, expr: ExpressionNode) -> Optional[xr.DataArray]:
        """Evaluate *expr* as a DataArray shift amount.

        Tries a pure-constant evaluation first; falls back to a parameter-
        aware DataArray evaluation.  Returns ``None`` if neither succeeds so
        that the caller can skip the validity check for that shift.
        """
        try:
            val = visit(expr, EvaluationVisitor(EvaluationContext()))
            if isinstance(val, (int, float)):
                return xr.DataArray(float(val))
        except (KeyError, NotImplementedError):
            pass
        try:
            return visit(
                expr,
                _ShiftAmountEvaluator(
                    model_id=self.model_id, param_arrays=self.param_arrays
                ),
            )
        except (_TimeDependentShiftError, _ScenarioDependentShiftError):
            raise
        except (KeyError, ValueError, NotImplementedError):
            return None

    # ------------------------------------------------------------------ #
    # Time operators — produce validity conditions                          #
    # ------------------------------------------------------------------ #

    def time_shift(self, node: TimeShiftNode) -> Optional[xr.DataArray]:
        operand_mask = visit(node.operand, self)
        shift_da = self._eval_as_da(node.time_shift)
        if shift_da is None:
            raise ValueError(
                f"Time-shift amount is not evaluable to a literal or parameter: "
                f"{node.time_shift!r}. Only literals and parameter references are "
                f"supported as shift amounts in OutOfBoundsMode.DROP constraints."
            )
        T = self.block_length
        t = xr.DataArray(np.arange(T), dims="time")
        own_mask = (t + shift_da >= 0) & (t + shift_da < T)
        return _and_mask(operand_mask, own_mask)

    def time_sum(self, node: TimeSumNode) -> Optional[xr.DataArray]:
        operand_mask = visit(node.operand, self)
        from_da = self._eval_as_da(node.from_time)
        to_da = self._eval_as_da(node.to_time)
        if from_da is None or to_da is None:
            unevaluable = node.from_time if from_da is None else node.to_time
            raise ValueError(
                f"Time-sum bound is not evaluable to a literal or parameter: "
                f"{unevaluable!r}. Only literals and parameter references are "
                f"supported as bounds in OutOfBoundsMode.DROP constraints."
            )
        T = self.block_length
        t = xr.DataArray(np.arange(T), dims="time")
        # Every offset in [from, to] must be in bounds; the extreme values are binding.
        own_mask = (t + from_da >= 0) & (t + to_da < T)
        return _and_mask(operand_mask, own_mask)

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

    def time_eval(self, node: TimeEvalNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    def port_field_aggregator(
        self, node: PortFieldAggregatorNode
    ) -> Optional[xr.DataArray]:
        return visit(node.operand, self)

    # ------------------------------------------------------------------ #
    # Leaf nodes — no time-shift constraints                                #
    # ------------------------------------------------------------------ #

    def literal(self, node: LiteralNode) -> Optional[xr.DataArray]:
        return None

    def parameter(self, node: ParameterNode) -> Optional[xr.DataArray]:
        return None

    def variable(self, node: VariableNode) -> Optional[xr.DataArray]:
        return None

    def port_field(self, node: PortFieldNode) -> Optional[xr.DataArray]:
        return None
