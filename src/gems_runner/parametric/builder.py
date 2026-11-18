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
Builds one problem shared by all scenarios, plus its scenario-dependent entries.

Parameters keep their ``scenario`` dimension while variables are created
without it.  xarray broadcasting then carries the ``scenario`` dimension into
every coefficient, right-hand side, bound or objective term that depends on a
scenario-dependent parameter, whatever the expression.  Each such element is
added to the linopy model for the first scenario (the nominal problem), and
its entries that differ between scenarios are recorded in
:class:`ParametricEntries`.
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
from gems_craft.model.port import PortFieldId
from gems_craft.model.variable import Variable
from gems_runner.simulation.linearize import (
    VectorizedExpr,
    VectorizedLinearExprBuilder,
)
from gems_runner.simulation.optimization import _OptimizationProblemBuilder

SCENARIO = "scenario"
TERM = "_term"

# Label of a missing row or column (e.g. an objective entry has no row).
_NO_LABEL = -1

# Columns of the parameters table and their types.
_COLUMNS: Dict[str, Any] = {
    "kind": str,
    "row": str,
    "col": str,
    "scenario": "int64",
    "value": float,
}


class ParametricEntries:
    """Scenario-dependent entries of a problem, keyed by linopy labels.

    Each chunk holds *n* entries of one kind (``rhs``, ``coef``, ``lower``,
    ``upper``, ``obj`` or ``obj_const``): their row (constraint) label, their
    column (variable) label, and their value in each scenario, shaped
    ``(n, number of scenarios)``.
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
        self, linopy_model: linopy.Model, scenario_ids: List[int]
    ) -> pd.DataFrame:
        """Return the entries in long format (one row per entry and scenario),
        rows and columns named as in the MPS file written with
        ``explicit_coordinate_names=True``."""
        if not self._chunks:
            return pd.DataFrame(columns=list(_COLUMNS)).astype(_COLUMNS)

        print_variables, print_constraints = linopy.io.get_printers_scalar(
            linopy_model, explicit_coordinate_names=True
        )
        n_scenarios = len(scenario_ids)
        frames = []
        for kind, rows, cols, values in self._chunks:
            frames.append(
                pd.DataFrame(
                    {
                        "kind": kind,
                        "row": np.repeat(_names(rows, print_constraints), n_scenarios),
                        "col": np.repeat(_names(cols, print_variables), n_scenarios),
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
    """Whether *value* (DataArray, linopy object, number) depends on the scenario."""
    return SCENARIO in getattr(value, "dims", ())


def _varying(values: np.ndarray) -> np.ndarray:
    """Mask of the rows of *values* that differ between scenarios (columns)."""
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
    """Linear builder for a problem holding a single scenario: the expectation
    over scenarios is the identity, applied to each scenario separately."""

    def scenario_operator(self, node: ScenarioOperatorNode) -> VectorizedExpr:
        if node.name != "Expectation":
            return super().scenario_operator(node)
        return visit(node.operand, self)


class _ParametricProblemBuilder(_OptimizationProblemBuilder):
    """Builds the nominal problem and records its scenario-dependent entries
    in :attr:`entries` (see the module docstring)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.entries = ParametricEntries()
        # Bounds of the variable being added, in each scenario.
        self._pending_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

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
    def _n_scenarios(self) -> int:
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
        scenario) that differ between scenarios and are in the model."""
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
        pairs whose coefficient differs between scenarios.

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
    # Variables: shared by all scenarios, bounds may depend on them
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

        # Checked in every scenario, not only the nominal one.
        per_scenario = [
            super(_ParametricProblemBuilder, self)._resolve_bounds(
                var,
                comp_ids,
                var_shape,
                dims,
                in_scenario(lower_value, s),
                in_scenario(upper_value, s),
            )
            for s in range(self._n_scenarios)
        ]
        self._pending_bounds = (
            np.stack([lower for lower, _ in per_scenario], axis=-1),
            np.stack([upper for _, upper in per_scenario], axis=-1),
        )
        return per_scenario[0]

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

        n = self._n_scenarios
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
    # Objective: one per scenario
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
        n = self._n_scenarios
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
        # written for every scenario whenever it is not zero.
        if np.any(constants != 0):
            self.entries.add("obj_const", constants.reshape(1, -1).astype(float))
        return nominal_constant
