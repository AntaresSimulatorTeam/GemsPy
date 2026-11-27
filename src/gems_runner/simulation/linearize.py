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

Provides :class:`VectorizedLinearExprBuilder`, a concrete subclass of
:class:`~gems_runner.simulation.vectorized_builder.VectorizedBuilderBase` that
resolves ``VariableNode`` to a pre-solve ``linopy.Variable`` and overrides
arithmetic / nonlinear methods with linopy-specific behaviour.

Also re-exports :data:`~gems_runner.simulation.vectorized_builder.VectorizedExpr`
and :func:`~gems_runner.simulation.vectorized_builder._linopy_add` for backward
compatibility with callers that import them from this module.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import linopy
import numpy as np
import xarray as xr

from gems_craft.expression.expression import (
    AbsNode,
    AdditionNode,
    CeilNode,
    DivisionNode,
    FloorNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    PowerNode,
    RoundNode,
    VariableNode,
)
from gems_craft.expression.visitor import visit
from gems_craft.model.port import PortFieldId
from gems_runner.expression.power import real_power
from gems_runner.simulation.vectorized_builder import (
    VectorizedBuilderBase,
    VectorizedExpr,
    _linopy_add,
    _raise_if_invalid,
)


@dataclass(kw_only=True)
class VectorizedLinearExprBuilder(VectorizedBuilderBase[VectorizedExpr]):
    """
    Builds a linopy LinearExpression from a model-level AST.

    Receives pre-computed linopy variables, parameter DataArrays, and port arrays
    indexed on the [component, time, scenario] dimensions (or a subset thereof).
    Produces a result with the same combined dimensions.

    Parameters
    ----------
    model_id:
        The model.id string of the Model object whose AST is being visited.
    linopy_vars:
        Mapping from (model_id, var_name) to the corresponding linopy Variable,
        with dims at least [component] and optionally [time, scenario].
    param_arrays:
        Mapping from (model_id, param_name) to a DataArray of parameter values,
        with dims in {component, time, scenario} (or a subset).
    port_arrays:
        Pre-computed linopy expressions for each PortFieldId of this model,
        resulting from the incidence-matrix port resolution pass.
        Keyed by PortFieldId(port_name, field_name).
    block_length:
        Number of time steps in the current time block.
    """

    linopy_vars: Dict[Tuple[str, str], linopy.Variable]

    # ------------------------------------------------------------------ #
    # Abstract method implementation                                        #
    # ------------------------------------------------------------------ #

    def variable(self, node: VariableNode) -> linopy.Variable:
        key = (self.model_id, node.name)
        if key not in self.linopy_vars:
            raise KeyError(
                f"Variable {node.name!r} not found for model {self.model_id!r}. "
                "Ensure all linopy variables are created before building constraints."
            )
        return self.linopy_vars[key]

    # ------------------------------------------------------------------ #
    # Overrides: arithmetic                                                 #
    # ------------------------------------------------------------------ #

    def addition(self, node: AdditionNode) -> VectorizedExpr:
        """Left-to-right addition with linopy-aware operand swapping.

        ``xr.DataArray.__add__(linopy_type)`` fails because xarray does not
        recognise linopy objects.  :func:`_linopy_add` puts the linopy type on
        the left so linopy's ``__add__`` / ``__radd__`` handles DataArrays.
        """
        operands = [visit(op, self) for op in node.operands]
        result: VectorizedExpr = operands[0]
        for op in operands[1:]:
            # An infinite constant is kept: whether it is valid depends on the
            # constraint sense, checked by the problem builder.
            self._check_constant_operand(result, op, allow_inf=True)
            self._check_constant_operand(op, result, allow_inf=True)
            result = _linopy_add(result, op)
        return result

    def multiplication(self, node: MultiplicationNode) -> VectorizedExpr:
        left = visit(node.left, self)
        right = visit(node.right, self)
        self._check_constant_operand(left, right, allow_inf=False)
        self._check_constant_operand(right, left, allow_inf=False)
        return left * right  # type: ignore[operator,return-value]

    def division(self, node: DivisionNode) -> VectorizedExpr:
        left = visit(node.left, self)
        right = visit(node.right, self)
        if isinstance(right, xr.DataArray) and not isinstance(left, xr.DataArray):
            # Check the resulting coefficient: a zero divisor gives inf.
            with np.errstate(divide="ignore", invalid="ignore"):
                _raise_if_invalid(self.model_id, 1 / right)
        return left / right  # type: ignore[operator,return-value]

    def _check_constant_operand(
        self, operand: VectorizedExpr, other: VectorizedExpr, allow_inf: bool
    ) -> None:
        """Raise if *operand*, a constant combined with a linopy object, is NaN (or inf).

        linopy replaces NaN with 0 in such operations, so the check is made here.
        """
        if isinstance(operand, xr.DataArray) and not isinstance(other, xr.DataArray):
            invalid = operand.isnull() if allow_inf else None
            _raise_if_invalid(self.model_id, operand, invalid)

    def power(self, node: PowerNode) -> VectorizedExpr:
        base = visit(node.left, self)
        exponent = visit(node.right, self)
        if isinstance(base, xr.DataArray) and isinstance(exponent, xr.DataArray):
            return real_power(base, exponent)
        raise NotImplementedError(
            "^ only accepts parameters and literals as operands; "
            "it cannot be applied to decision variables in a linear problem."
        )

    # ------------------------------------------------------------------ #
    # Overrides: nonlinear math functions (guard — variables not allowed)   #
    # ------------------------------------------------------------------ #

    def floor(self, node: FloorNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.floor(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "floor() is only supported for parameter (DataArray) expressions; "
            "it cannot be used with decision variables in a linear programme."
        )

    def ceil(self, node: CeilNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.ceil(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "ceil() is only supported for parameter (DataArray) expressions; "
            "it cannot be used with decision variables in a linear programme."
        )

    def abs(self, node: AbsNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.abs(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "abs() is only supported for parameter (DataArray) expressions; "
            "it cannot be used with decision variables in a linear programme."
        )

    def round(self, node: RoundNode) -> VectorizedExpr:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.round(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "round() is only supported for parameter (DataArray) expressions; "
            "it cannot be used with decision variables in a linear programme."
        )

    def maximum(self, node: MaxNode) -> VectorizedExpr:
        operands = [visit(op, self) for op in node.operands]
        if all(isinstance(op, xr.DataArray) for op in operands):
            result: xr.DataArray = operands[0]  # type: ignore[assignment]
            for op in operands[1:]:
                result = xr.where(result >= op, result, op)  # type: ignore[no-untyped-call]
            return result
        raise NotImplementedError(
            "maximum() is only supported for parameter (DataArray) expressions; "
            "it cannot be used with decision variables in a linear programme."
        )

    def minimum(self, node: MinNode) -> VectorizedExpr:
        operands = [visit(op, self) for op in node.operands]
        if all(isinstance(op, xr.DataArray) for op in operands):
            result = operands[0]
            for op in operands[1:]:
                result = xr.where(result <= op, result, op)  # type: ignore[no-untyped-call,assignment]
            return result
        raise NotImplementedError(
            "minimum() is only supported for parameter (DataArray) expressions; "
            "it cannot be used with decision variables in a linear programme."
        )
