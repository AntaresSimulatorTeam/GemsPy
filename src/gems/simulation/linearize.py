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
:class:`~gems.simulation.vectorized_builder.VectorizedBuilderBase` that
resolves ``VariableNode`` to a pre-solve ``linopy.Variable`` and overrides
arithmetic / nonlinear methods with linopy-specific behaviour.

Also re-exports :data:`~gems.simulation.vectorized_builder.VectorizedExpr`
and :func:`~gems.simulation.vectorized_builder._linopy_add` for backward
compatibility with callers that import them from this module.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import linopy
import numpy as np
import xarray as xr

from gems.expression.expression import (
    AdditionNode,
    CeilNode,
    FloorNode,
    MaxNode,
    MinNode,
    VariableNode,
)
from gems.expression.visitor import visit
from gems.model.port import PortFieldId
from gems.simulation.vectorized_builder import (
    VectorizedBuilderBase,
    VectorizedExpr,
    _linopy_add,
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
            result = _linopy_add(result, op)
        return result

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
