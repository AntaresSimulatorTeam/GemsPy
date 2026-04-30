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
Extra output storage and vectorized evaluation.

Provides :class:`ExtraOutput` for storing post-solve expression values and
:class:`VectorizedExtraOutputBuilder`, a concrete subclass of
:class:`~gems.simulation.vectorized_builder.VectorizedBuilderBase` that
resolves ``VariableNode`` to an ``xr.DataArray`` of solved values, enabling
nonlinear operations (products of variables, floor, ceil, min, max) that
are not permitted during pre-solve constraint building.

Port arrays for ``sum_connections`` support are built by calling
:func:`~gems.simulation.optimization.build_port_arrays` with a
:class:`VectorizedExtraOutputBuilder` factory.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from gems.expression.expression import VariableNode
from gems.model.port import PortFieldId
from gems.simulation.vectorized_builder import VectorizedBuilderBase
from gems.study.system import Component


@dataclass
class ExtraOutput:
    """
    Stores a post-solve extra output expression as a vectorized xr.DataArray
    with dims ⊆ {component, time, scenario}.
    """

    _name: str
    _data: Optional[xr.DataArray] = field(init=False, default=None)
    ignore: bool = field(default=False, init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtraOutput):
            return NotImplemented
        return self.is_close(other, rel_tol=0.0, abs_tol=0.0)

    def is_close(
        self,
        other: "ExtraOutput",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        if not isinstance(other, ExtraOutput):
            return NotImplemented  # type: ignore[return-value]
        if self.ignore or other.ignore:
            return True
        if (self._data is None) != (other._data is None):
            return False
        if self._data is None:
            return True
        assert other._data is not None  # narrowing: both non-None by checks above
        try:
            lhs, rhs = xr.align(self._data, other._data, join="exact")
        except ValueError:
            return False
        return bool(np.allclose(lhs.values, rhs.values, rtol=rel_tol, atol=abs_tol))

    def __str__(self) -> str:
        return f"{self._name} : {self._data!r} {'(ignored)' if self.ignore else ''}"


@dataclass(kw_only=True)
class VectorizedExtraOutputBuilder(VectorizedBuilderBase[xr.DataArray]):
    """
    Evaluates a model-level extra output expression as a vectorized xr.DataArray.

    A concrete subclass of :class:`VectorizedBuilderBase` that resolves
    decision variables to their post-solve optimal values (``xr.DataArray``),
    enabling nonlinear operations such as products of variables, floor, ceil,
    min, and max that are forbidden during pre-solve constraint building.

    Parameters
    ----------
    model_id:
        The model.id string of the Model object whose AST is being visited.
    param_arrays:
        Mapping from (model_id, param_name) to a DataArray of parameter values,
        with dims in {component, time, scenario} (or a subset).
    var_solution_arrays:
        Mapping from (model_id, var_name) to a DataArray of solution values,
        with dims in {component, time, scenario} (or a subset).
    port_arrays:
        Pre-computed xr.DataArray for each PortFieldId of this model.
        Keyed by PortFieldId(port_name, field_name).
    block_length:
        Number of time steps in the current time block.
    """

    var_solution_arrays: Dict[Tuple[str, str], xr.DataArray]

    def variable(self, node: VariableNode) -> xr.DataArray:
        key = (self.model_id, node.name)
        if key not in self.var_solution_arrays:
            raise KeyError(
                f"Variable {node.name!r} not found in solution for model "
                f"{self.model_id!r}."
            )
        return self.var_solution_arrays[key]
