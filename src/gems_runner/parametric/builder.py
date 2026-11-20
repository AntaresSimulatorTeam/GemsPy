# Copyright (c) 2026, RTE (https://www.rte-france.com)
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
Builds one problem shared by all time blocks and scenarios, plus its entries
that depend on them.

The ``scenario`` dimension of parameters indexes *realizations*: the
(block, scenario) pairs, flattened.  Parameters that depend on time or on the
scenario get this dimension, while variables are created without it.  xarray
broadcasting then carries it into every coefficient, right-hand side, bound or
objective term that depends on such a parameter, whatever the expression.
Each such element is added to the linopy model for the first realization
(first block, first scenario: the nominal problem), and its entries that
differ between realizations are recorded in :class:`ParametricEntries`.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import linopy
import linopy.io
import numpy as np
import pandas as pd
import xarray as xr

from gems_craft.expression.expression import ScenarioOperatorNode
from gems_craft.expression.visitor import visit
from gems_craft.model.parameter import Parameter
from gems_craft.model.port import PortFieldId
from gems_craft.model.variable import Variable
from gems_craft.study.study import Study
from gems_craft.study.system import Component
from gems_runner.simulation.linearize import (
    VectorizedExpr,
    VectorizedLinearExprBuilder,
)
from gems_runner.simulation.optimization import (
    DecompositionFilter,
    OutOfBoundsFilter,
    _OptimizationProblemBuilder,
)
from gems_runner.simulation.time_block import TimeBlock

SCENARIO = "scenario"
TERM = "_term"

# Label of a missing row or column (e.g. an objective entry has no row).
_NO_LABEL = -1

# Columns of the parameters table and their types.
_COLUMNS: Dict[str, Any] = {
    "kind": str,
    "row": str,
    "col": str,
    "block": "int64",
    "scenario": "int64",
    "value": float,
}


class ParametricEntries:
    """Entries of a problem that depend on the realization, keyed by linopy
    labels.

    Each chunk holds *n* entries of one kind (``rhs``, ``coef``, ``lower``,
    ``upper``, ``obj`` or ``obj_const``): their row (constraint) label, their
    column (variable) label, and their value in each realization, shaped
    ``(n, number of realizations)``.
    """

    def __init__(self) -> None:
        self._chunks: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    def add(
        self,
        kind: str,
        values: np.ndarray,
        rows: Optional[np.ndarray] = None,
        cols: Optional[np.ndarray] = None,
    ) -> None:
        """Add the entries *values*; *rows* or *cols* are omitted when the
        entries have no row or column."""
        if not len(values):
            return
        no_label = np.full(len(values), _NO_LABEL)
        rows = no_label if rows is None else rows
        cols = no_label if cols is None else cols
        self._chunks.append((kind, rows, cols, values))

    def to_frame(
        self, linopy_model: linopy.Model, realizations: List[Tuple[int, int]]
    ) -> pd.DataFrame:
        """Return the entries in long format (one row per entry and
        realization), rows and columns named as in the MPS file written with
        ``explicit_coordinate_names=True``.

        *realizations* lists the (block id, scenario) pair of each realization.
        """
        if not self._chunks:
            return pd.DataFrame(columns=list(_COLUMNS)).astype(_COLUMNS)

        print_variables, print_constraints = linopy.io.get_printers_scalar(
            linopy_model, explicit_coordinate_names=True
        )
        n = len(realizations)
        block_ids = [block for block, _ in realizations]
        scenario_ids = [scenario for _, scenario in realizations]
        frames = []
        for kind, rows, cols, values in self._chunks:
            frames.append(
                pd.DataFrame(
                    {
                        "kind": kind,
                        "row": np.repeat(_names(rows, print_constraints), n),
                        "col": np.repeat(_names(cols, print_variables), n),
                        "block": np.tile(block_ids, len(values)),
                        "scenario": np.tile(scenario_ids, len(values)),
                        "value": values.reshape(-1),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def _names(
    labels: np.ndarray, printer: Callable[[np.ndarray], List[str]]
) -> np.ndarray:
    """Names of *labels*, ``None`` for :data:`_NO_LABEL`."""
    names = np.full(len(labels), None, dtype=object)
    present = labels != _NO_LABEL
    if present.any():
        unique, inverse = np.unique(labels[present], return_inverse=True)
        names[present] = np.array(printer(unique), dtype=object)[inverse]
    return names


def _has_scenario(value: object) -> bool:
    """Whether *value* (DataArray, linopy object, number) depends on the
    realization."""
    return SCENARIO in getattr(value, "dims", ())


def _varying(values: np.ndarray) -> np.ndarray:
    """Mask of the rows of *values* that differ between realizations (columns)."""
    return ~np.all(values == values[:, :1], axis=1)


def _as_array(
    da: xr.DataArray, dims: Sequence[Hashable], sizes: Mapping[Hashable, int]
) -> np.ndarray:
    """Values of *da* broadcast and transposed to *dims*."""
    missing = {d: sizes[d] for d in dims if d not in da.dims}
    if missing:
        da = da.expand_dims(missing)
    return np.asarray(da.transpose(*dims).values)


def _drop_scenario(da: xr.DataArray) -> xr.DataArray:
    """*da* without its scenario dimension (values are equal across it)."""
    return da.isel({SCENARIO: 0}) if SCENARIO in da.dims else da


class _ParametricLinearExprBuilder(VectorizedLinearExprBuilder):
    """Linear builder for a problem holding a single realization: the
    expectation over scenarios is the identity, applied to each realization
    separately."""

    def scenario_operator(self, node: ScenarioOperatorNode) -> VectorizedExpr:
        if node.name != "Expectation":
            return super().scenario_operator(node)
        return visit(node.operand, self)


class _ParametricProblemBuilder(_OptimizationProblemBuilder):
    """Builds the nominal problem of *blocks*, which must have the same length,
    and records in :attr:`entries` its entries that depend on the realization
    (see the module docstring)."""

    def __init__(
        self,
        name: str,
        study: Study,
        blocks: List[TimeBlock],
        scenario_ids: List[int],
        location_filter: Optional[DecompositionFilter] = None,
        oob_filter: Optional[OutOfBoundsFilter] = None,
    ) -> None:
        if len({len(block.timesteps) for block in blocks}) != 1:
            raise ValueError("All time blocks must have the same length.")
        super().__init__(
            name=name,
            study=study,
            block=blocks[0],
            scenario_ids=scenario_ids,
            location_filter=location_filter,
            oob_filter=oob_filter,
        )
        self.blocks = blocks
        # The (block id, scenario) pair of each realization, block-major.
        self.realizations = [(b.id, s) for b in blocks for s in scenario_ids]
        self.local_scenario_coord = list(range(len(self.realizations)))
        self.entries = ParametricEntries()
        # Bounds of the variable being added, in each realization.
        self._pending_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _param_array(
        self, param: Parameter, components: List[Component]
    ) -> xr.DataArray:
        """Values of *param* in each realization when it depends on time or on
        the scenario: the time steps are those of the block."""
        use_time, use_scenario = param.structure.time, param.structure.scenario
        if not (use_time or use_scenario):
            return super()._param_array(param, components)

        n_blocks, n_scenarios = len(self.blocks), len(self.scenario_ids)
        shape = (len(components), self.block_length, n_blocks, n_scenarios)
        data = np.zeros(shape if use_time else (shape[0], n_blocks, n_scenarios))
        for i, component in enumerate(components):
            for b, block in enumerate(self.blocks):
                values = self.study.database.get_values(
                    component.id,
                    param.name,
                    block.timesteps if use_time else None,
                    self.scenario_ids if use_scenario else None,
                )
                if use_time:
                    if not use_scenario and np.ndim(values) == 1:
                        values = np.asarray(values)[:, None]  # (T,) → (T, 1)
                    data[i, :, b, :] = values
                else:
                    data[i, b, :] = values

        dims = ["component", "time", SCENARIO] if use_time else ["component", SCENARIO]
        coords: Dict[str, object] = {
            "component": [c.id for c in components],
            SCENARIO: self.local_scenario_coord,
        }
        if use_time:
            coords["time"] = self.time_coord
        # (block, scenario) flattened block-major, as in :attr:`realizations`.
        data = data.reshape(*data.shape[:-2], n_blocks * n_scenarios)
        return xr.DataArray(data, dims=dims, coords=coords)

    def _new_expr_builder(
        self,
        model_id: str,
        linopy_vars: Dict[Tuple[str, str], linopy.Variable],
        param_arrays: Dict[Tuple[str, str], xr.DataArray],
        port_arrays: Dict[PortFieldId, VectorizedExpr],
    ) -> VectorizedLinearExprBuilder:
        return _ParametricLinearExprBuilder(
            model_id=model_id,
            linopy_vars=linopy_vars,
            param_arrays=param_arrays,
            port_arrays=port_arrays,
            block_length=self.block_length,
        )

    @property
    def _n_realizations(self) -> int:
        return len(self.local_scenario_coord)

    # ------------------------------------------------------------------
    # Recording entries
    # ------------------------------------------------------------------

    def _add_varying(
        self,
        kind: str,
        values: np.ndarray,
        rows: Optional[np.ndarray] = None,
        cols: Optional[np.ndarray] = None,
    ) -> None:
        """Record the entries of *values* (one row per entry, one column per
        realization) that differ between realizations and are in the model."""
        keep = _varying(values)
        if rows is not None:
            keep &= rows != _NO_LABEL
        if cols is not None:
            keep &= cols != _NO_LABEL
        self.entries.add(
            kind,
            values[keep],
            rows=None if rows is None else rows[keep],
            cols=None if cols is None else cols[keep],
        )

    def _add_coefficients(
        self, kind: str, rows: np.ndarray, cols: np.ndarray, values: np.ndarray
    ) -> None:
        """Sum the coefficients of each (row, variable) pair, then record the
        pairs whose coefficient differs between realizations.

        *rows* and *cols* are shaped ``(N,)``, *values* ``(N, S)``.
        """
        keep = cols != _NO_LABEL
        summed = pd.DataFrame(values[keep]).groupby([rows[keep], cols[keep]]).sum()
        summed_values = summed.to_numpy()
        varying = _varying(summed_values)
        index = summed.index[varying]
        self.entries.add(
            kind,
            summed_values[varying],
            rows=index.get_level_values(0).to_numpy(),
            cols=index.get_level_values(1).to_numpy(),
        )

    # ------------------------------------------------------------------
    # Variables: shared by all realizations, bounds may depend on them
    # ------------------------------------------------------------------

    def _variable_coords(
        self, var: Variable, comp_ids: List[str]
    ) -> Tuple[Dict[str, object], List[str]]:
        coords, dims = super()._variable_coords(var, comp_ids)
        coords.pop(SCENARIO, None)
        return coords, [d for d in dims if d != SCENARIO]

    def _resolve_bounds(
        self,
        var: Variable,
        comp_ids: List[str],
        var_shape: Tuple[int, ...],
        dims: List[str],
        lower_value: Optional[object],
        upper_value: Optional[object],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not (_has_scenario(lower_value) or _has_scenario(upper_value)):
            return super()._resolve_bounds(
                var, comp_ids, var_shape, dims, lower_value, upper_value
            )

        def in_scenario(value: Optional[object], s: int) -> Optional[object]:
            if _has_scenario(value):
                return value.isel({SCENARIO: s})  # type: ignore[union-attr]
            return value

        # Checked in every realization, not only the nominal one.
        per_realization = [
            super(_ParametricProblemBuilder, self)._resolve_bounds(
                var,
                comp_ids,
                var_shape,
                dims,
                in_scenario(lower_value, s),
                in_scenario(upper_value, s),
            )
            for s in range(self._n_realizations)
        ]
        self._pending_bounds = (
            np.stack([lower for lower, _ in per_realization], axis=-1),
            np.stack([upper for _, upper in per_realization], axis=-1),
        )
        return per_realization[0]

    def _on_variable_added(self, linopy_var: linopy.Variable) -> None:
        if self._pending_bounds is None:
            return
        labels = np.asarray(linopy_var.labels.values).reshape(-1)
        lower, upper = self._pending_bounds
        self._add_varying("lower", lower.reshape(len(labels), -1), cols=labels)
        self._add_varying("upper", upper.reshape(len(labels), -1), cols=labels)
        self._pending_bounds = None

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _add_constraint(self, constraint: linopy.Constraint, name: str) -> None:
        if not _has_scenario(constraint):
            super()._add_constraint(constraint, name)
            return

        super()._add_constraint(constraint.isel({SCENARIO: 0}), name)
        row_labels = self.linopy_model.constraints[name].labels

        n = self._n_realizations
        sizes = {**constraint.coeffs.sizes, **constraint.rhs.sizes, SCENARIO: n}
        row_dims: List[Hashable] = list(row_labels.dims)
        rows = _as_array(row_labels, row_dims, sizes).reshape(-1)  # (R,)
        cols = _as_array(
            _drop_scenario(constraint.vars), row_dims + [TERM], sizes
        ).reshape(
            len(rows), -1
        )  # (R, T)
        coeffs = _as_array(
            constraint.coeffs, row_dims + [TERM, SCENARIO], sizes
        ).reshape(
            -1, n
        )  # (R * T, S)
        rhs = _as_array(constraint.rhs, row_dims + [SCENARIO], sizes).reshape(-1, n)

        # One entry per term; terms of rows absent from the model are dropped.
        term_rows = np.repeat(rows, cols.shape[1])
        in_model = term_rows != _NO_LABEL
        self._add_coefficients(
            "coef", term_rows[in_model], cols.reshape(-1)[in_model], coeffs[in_model]
        )
        self._add_varying("rhs", rhs, rows=rows)

    # ------------------------------------------------------------------
    # Objective: one per realization
    # ------------------------------------------------------------------

    def _sum_objective_contribution(
        self, contribution: VectorizedExpr
    ) -> VectorizedExpr:
        if not _has_scenario(contribution):
            return super()._sum_objective_contribution(contribution)
        others = [d for d in contribution.dims if d not in (SCENARIO, TERM)]
        if not others:
            return contribution
        if isinstance(contribution, xr.DataArray):
            return contribution.sum(others)
        return contribution.sum(dim=others)  # type: ignore[union-attr,return-value]

    def _set_objective(self, total_obj: Optional[VectorizedExpr]) -> float:
        n = self._n_realizations
        constants: Optional[np.ndarray] = None

        if isinstance(total_obj, xr.DataArray) and _has_scenario(total_obj):
            constants = _as_array(total_obj, [SCENARIO], {SCENARIO: n})
            total_obj = total_obj.isel({SCENARIO: 0})

        elif isinstance(total_obj, linopy.LinearExpression) and _has_scenario(
            total_obj
        ):
            sizes = {**total_obj.coeffs.sizes, SCENARIO: n}
            cols = _as_array(_drop_scenario(total_obj.vars), [TERM], sizes)
            coeffs = _as_array(total_obj.coeffs, [TERM, SCENARIO], sizes)
            self._add_coefficients("obj", np.full(len(cols), _NO_LABEL), cols, coeffs)
            constants = _as_array(total_obj.const, [SCENARIO], {SCENARIO: n})
            total_obj = total_obj.isel({SCENARIO: 0})

        nominal_constant = super()._set_objective(total_obj)
        if constants is None:
            constants = np.full(n, nominal_constant)
        # linopy keeps the constant out of the model (and of the MPS file):
        # written for every realization whenever it is not zero.
        if np.any(constants != 0):
            self.entries.add("obj_const", constants.reshape(1, -1).astype(float))
        return nominal_constant
