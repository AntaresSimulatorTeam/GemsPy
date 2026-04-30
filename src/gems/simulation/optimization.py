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
Linopy-based optimization problem builder.

Provides :func:`build_problem` and :class:`OptimizationProblem`.
The builder groups system components by model, then constructs the full
optimization problem in four phases:

1. Parameter arrays — convert database values to xarray DataArrays indexed
   on ``[component, time, scenario]``.
2. Decision variables — create one linopy ``Variable`` per model variable,
   covering all components of that model at once.
3. Port arrays — resolve port connections via an incidence matrix so that
   port-field expressions are available as linopy ``LinearExpression`` objects.
4. Constraints and objective — traverse each constraint AST once with
   :class:`~gems.simulation.linearize.VectorizedLinearExprBuilder` to
   produce vectorized linopy constraints added in a single
   ``Model.add_constraints()`` call per constraint type.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import linopy
import numpy as np
import xarray as xr

from gems.expression.expression import is_unbounded
from gems.expression.visitor import visit
from gems.model.common import ValueType
from gems.model.model import Model
from gems.model.port import PortField, PortFieldId
from gems.simulation.linearize import (
    VectorizedExpr,
    VectorizedLinearExprBuilder,
    _linopy_add,
)
from gems.simulation.time_block import TimeBlock
from gems.simulation.vectorized_builder import ShiftValidityVisitor
from gems.study.study import Study
from gems.study.system import Component

if TYPE_CHECKING:
    from gems.optim_config.parsing import ElementLocation, OptimConfig, OutOfBoundsMode

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

LinopyModel = linopy.Model
"""Alias for :class:`linopy.Model`, distinguishing it from :class:`gems.model.Model`."""

# ---------------------------------------------------------------------------
# Decomposition filter
# ---------------------------------------------------------------------------


class DecompositionFilter:
    """Decides which model elements belong to a given problem side (master or subproblems).

    Elements not listed in the config default to ``subproblems``.

    Parameters
    ----------
    config:
        Parsed OptimConfig from optim-config.yml.
    target_locations:
        The set of :class:`~gems.optim_config.parsing.ElementLocation` values
        that should be *included* (not filtered out) by this filter.
    """

    def __init__(
        self, config: "OptimConfig", target_locations: "Set[ElementLocation]"
    ) -> None:
        from gems.optim_config.parsing import ElementLocation as EL

        self._target = target_locations
        self._default = EL.SUBPROBLEMS
        self._vars: Dict[Tuple[str, str], "ElementLocation"] = {}
        self._cons: Dict[Tuple[str, str], "ElementLocation"] = {}
        self._objs: Dict[Tuple[str, str], "ElementLocation"] = {}

        for mc in config.models:
            if mc.model_decomposition is not None:
                for v in mc.model_decomposition.variables:
                    self._vars[(mc.id, v.id)] = v.location
                for c in mc.model_decomposition.constraints:
                    self._cons[(mc.id, c.id)] = c.location
                for o in mc.model_decomposition.objective_contributions:
                    self._objs[(mc.id, o.id)] = o.location

    def include_variable(self, model_id: str, var_name: str) -> bool:
        loc = self._vars.get((model_id, var_name), self._default)
        return loc in self._target

    def include_constraint(self, model_id: str, constraint_name: str) -> bool:
        loc = self._cons.get((model_id, constraint_name), self._default)
        return loc in self._target

    def include_objective(self, model_id: str, obj_id: str) -> bool:
        loc = self._objs.get((model_id, obj_id), self._default)
        return loc in self._target


def _apply_validity_mask(expr: VectorizedExpr, mask: xr.DataArray) -> VectorizedExpr:
    """Filter *expr* to valid (component, time) entries using *mask*.

    When *expr* has both component and time dimensions the mask is applied
    element-wise via ``where``.  When only the time dimension is present the
    intersection across components is used (conservative fallback).
    """
    if not hasattr(expr, "dims"):
        return expr
    dims = expr.dims  # type: ignore[union-attr]
    if "component" in dims and "time" in dims:
        return expr.where(mask)  # type: ignore[union-attr,return-value]
    if "time" in dims:
        valid_times: List[int] = mask.all("component").values.nonzero()[0].tolist()
        return expr.isel(time=valid_times)  # type: ignore[union-attr,return-value]
    return expr


class OutOfBoundsFilter:
    """Maps (model_id, constraint_id) to its :class:`OutOfBoundsMode`.

    Used by :class:`_OptimizationProblemBuilder` to determine whether a
    constraint should be dropped at timesteps where a shifted term falls
    outside the current block.  Constraints not listed in the config default
    to cyclic wrap-around (no masking applied).

    Parameters
    ----------
    config:
        Parsed OptimConfig from optim-config.yml.
    """

    def __init__(self, config: "OptimConfig") -> None:
        self._modes: Dict[Tuple[str, str], "OutOfBoundsMode"] = {}
        for model_config in config.models:
            if model_config.out_of_bounds_processing is not None:
                for constraint in model_config.out_of_bounds_processing.constraints:
                    self._modes[(model_config.id, constraint.id)] = constraint.mode

    def get_mode(
        self, model_id: str, constraint_name: str
    ) -> "Optional[OutOfBoundsMode]":
        return self._modes.get((model_id, constraint_name))


def build_port_arrays(
    model: Model,
    components: List[Component],
    study: Study,
    make_builder: Callable[[str, Model], Any],
) -> Dict[PortFieldId, Any]:
    """Build port arrays for all ports of *model*.

    For each PortFieldId (port_name, field_name):
    - If *model* defines the field (master): evaluate the definition with
      ``make_builder(model.id, model)``.
    - Otherwise (slave): sum contributions from connected master components
      via incidence matrices.

    Parameters
    ----------
    model :
        The model for which to build port arrays.
    components :
        Components of this model.
    study :
        The study, used for component/connection lookup.
    make_builder :
        Factory ``(model_key: str, model: Model) -> builder``.
        Called with an empty port_arrays context for master-field evaluation.
    """
    comp_ids = [c.id for c in components]
    n = len(components)
    port_arrays: Dict[PortFieldId, Any] = {}

    for port_name, model_port in model.ports.items():
        for port_field_obj in model_port.port_type.fields:
            field_name = port_field_obj.name
            pf_id = PortFieldId(port_name, field_name)

            if pf_id in model.port_fields_definitions:
                builder = make_builder(model.id, model)
                defn = model.port_fields_definitions[pf_id].definition
                try:
                    port_arrays[pf_id] = visit(defn, builder)
                except KeyError:
                    # A variable referenced in the port definition is not
                    # available in the current problem (e.g. a subproblem-only
                    # variable when building the master). Treat as zero.
                    port_arrays[pf_id] = xr.DataArray(0.0)
            else:
                port_arrays[pf_id] = _build_slave_port_array(
                    comp_ids,
                    n,
                    port_name,
                    field_name,
                    study,
                    make_builder,
                )

    return port_arrays


def _build_slave_port_array(
    comp_ids: List[str],
    n_components: int,
    port_name: str,
    field_name: str,
    study: Study,
    make_builder: Callable[[str, Model], Any],
) -> Any:
    """Build a slave port array by summing contributions from connected masters.

    Groups connections by (master_model.id, master_port_field_id), builds an
    incidence matrix A[i, j] for each group, and accumulates
    ``sum_j A[i,j] * expr_master[j]`` into the result.
    """
    per_master: Dict[Tuple[str, PortFieldId], List[Tuple[int, Component]]] = (
        defaultdict(list)
    )

    comp_index = {comp_id: i for i, comp_id in enumerate(comp_ids)}
    comp_id_set = set(comp_ids)
    for cnx in study.system.connections:
        for port_ref in [cnx.port1, cnx.port2]:
            if (
                port_ref.port_id != port_name
                or port_ref.component.id not in comp_id_set
            ):
                continue
            i = comp_index[port_ref.component.id]
            master_ref = cnx.master_port.get(PortField(name=field_name))
            if master_ref is None:
                continue
            master_comp = master_ref.component
            master_pf_id = PortFieldId(master_ref.port_id, field_name)
            per_master[(master_comp.model.id, master_pf_id)].append((i, master_comp))

    if not per_master:
        return xr.DataArray(0.0)

    total: Optional[Any] = None

    for (master_mk, master_pf_id), conn_list in per_master.items():
        master_comps = study.model_components[master_mk]
        master_comp_ids = [c.id for c in master_comps]
        n_prime = len(master_comps)

        A_data = np.zeros((n_components, n_prime))
        for i, master_comp in conn_list:
            j = master_comp_ids.index(master_comp.id)
            A_data[i, j] += 1.0

        A = xr.DataArray(
            A_data,
            dims=["component", "component_master"],
            coords={"component": comp_ids, "component_master": master_comp_ids},
        )

        master_model = study.models[master_mk]
        defn = master_model.port_fields_definitions[master_pf_id].definition
        master_builder = make_builder(master_mk, master_model)
        try:
            expr_master = visit(defn, master_builder)
        except KeyError:
            # The connected model has no variables in the current problem
            # (e.g. a subproblem-only model when building the master).
            # Its port contribution is treated as zero.
            continue

        expr_master_r = expr_master.rename({"component": "component_master"})  # type: ignore[union-attr]
        contribution = (A * expr_master_r).sum("component_master")  # type: ignore[operator]

        total = contribution if total is None else _linopy_add(total, contribution)

    return total if total is not None else xr.DataArray(0.0)


class OptimizationProblem:
    """
    Wraps a linopy.Model and provides the high-level API for solving and
    extracting results.
    """

    def __init__(
        self,
        name: str,
        linopy_model: LinopyModel,
        study: Study,
        block: TimeBlock,
        linopy_vars: Dict[Tuple[str, str], linopy.Variable],
        param_arrays: Dict[Tuple[str, str], xr.DataArray],
        objective_constant: float = 0.0,
    ) -> None:
        self.name = name
        self.linopy_model = linopy_model
        self.study = study
        self.block = block
        self._linopy_vars = linopy_vars
        self.param_arrays = param_arrays
        # Constant term of the objective (linopy cannot represent pure-constant objectives).
        self._objective_constant: float = objective_constant

    @property
    def block_length(self) -> int:
        return len(self.block.timesteps)

    def solve(self, solver_name: str = "highs", **kwargs: object) -> None:
        """Solve the problem using the specified solver."""
        # Use io_api="direct" to bypass LP file writing and avoid LP name parsing
        # issues in linopy's set_int_index (e.g. constraint names with spaces or
        # variables with non-standard characters).
        kwargs.setdefault("io_api", "direct")  # type: ignore[call-overload]
        self.linopy_model.solve(solver_name=solver_name, **kwargs)  # type: ignore[arg-type]

    @property
    def status(self) -> str:
        """Solver status: 'ok' (optimal found) or 'warning' (infeasible / other)."""
        return str(self.linopy_model.status)

    @property
    def termination_condition(self) -> str:
        """Termination condition string, e.g., 'optimal', 'infeasible'."""
        return str(self.linopy_model.termination_condition)

    @property
    def objective_value(self) -> float:
        """Objective function value after solving."""
        return float(self.linopy_model.objective.value) + self._objective_constant  # type: ignore[arg-type]

    def export_lp(self, path: Path) -> None:
        """Write the problem to an LP file at *path*."""
        self.linopy_model.to_file(path, explicit_coordinate_names=True)

    def get_variable_labels(
        self, model_id: str, var_name: str
    ) -> Optional[xr.DataArray]:
        """Return the linopy integer label DataArray for *var_name* of *model_id*.

        Each entry in the DataArray is the internal integer ID that linopy
        assigned to the corresponding scalar variable instance.  Returns
        ``None`` if the variable was not built in this problem (e.g. it was
        filtered out by a :class:`DecompositionFilter`).
        """
        lv = self._linopy_vars.get((model_id, var_name))
        return lv.labels if lv is not None else None


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------


class _OptimizationProblemBuilder:
    """
    Builds the linopy problem in 4 phases:
      1. Build parameter DataArrays for all models.
      2. Create all linopy Variables (uses param arrays for bounds).
      3. Build port arrays via incidence matrices.
      4. Add constraints and objectives to the linopy model.
    """

    def __init__(
        self,
        name: str,
        study: Study,
        block: TimeBlock,
        scenario_ids: List[int],
        location_filter: Optional[DecompositionFilter] = None,
        oob_filter: Optional[OutOfBoundsFilter] = None,
        initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
    ) -> None:
        self.name = name
        self.study = study
        self.block = block
        self.scenario_ids = scenario_ids
        self._location_filter = location_filter
        self._oob_filter = oob_filter
        self._initial_values = initial_values or {}

        self.block_length = len(block.timesteps)
        self.time_coord = list(range(self.block_length))
        self.local_scenario_coord = list(range(len(scenario_ids)))

        # Populated during build
        self.linopy_model = linopy.Model()
        self.linopy_vars: Dict[Tuple[str, str], linopy.Variable] = {}
        self.param_arrays: Dict[Tuple[str, str], xr.DataArray] = {}
        self.port_arrays: Dict[str, Dict[PortFieldId, VectorizedExpr]] = {}

    def build(self) -> OptimizationProblem:
        # Phase 1: parameter arrays
        for mk, components in self.study.model_components.items():
            self._build_param_arrays_for_model(self.study.models[mk], components)

        # Phase 2: linopy variables
        for mk, components in self.study.model_components.items():
            self._create_variables_for_model(self.study.models[mk], components)

        # Phase 3: port arrays
        for mk, components in self.study.model_components.items():
            self._build_port_arrays_for_model(self.study.models[mk], components)

        # Phase 4: constraints + objectives
        total_obj: Optional[VectorizedExpr] = None
        for mk, components in self.study.model_components.items():
            model = self.study.models[mk]
            port_arrays_for_model = self.port_arrays.get(mk, {})
            self._create_constraints_for_model(model, port_arrays_for_model)
            total_obj = self._add_objectives_for_model(
                model, port_arrays_for_model, total_obj
            )

        # Phase 5: carry-over constraints (sequential mode only)
        for (mk, var_name), init_val in self._initial_values.items():
            linopy_var = self.linopy_vars.get((mk, var_name))
            if linopy_var is not None and "time" in linopy_var.dims:
                safe = f"{mk}__{var_name}".replace("-", "_")
                self.linopy_model.add_constraints(
                    linopy_var.isel(time=0) == init_val,  # type: ignore[arg-type]
                    name=f"carry_over__{safe}",
                )

        # Extract constant objective contribution (linopy cannot hold pure constants).
        objective_constant = 0.0
        if total_obj is not None and not isinstance(
            total_obj, (xr.DataArray, int, float)
        ):
            self.linopy_model.add_objective(total_obj)  # type: ignore[arg-type]
        elif total_obj is not None:
            if isinstance(total_obj, xr.DataArray):
                objective_constant = float(total_obj.sum())
            else:
                objective_constant = float(total_obj)

        # linopy requires at least one variable to solve; add a fixed dummy if needed.
        if len(self.linopy_model.variables) == 0:
            dummy = self.linopy_model.add_variables(
                lower=xr.DataArray([0.0], dims=["__dummy_dim"]),
                upper=xr.DataArray([0.0], dims=["__dummy_dim"]),
                name="__dummy",
            )
            self.linopy_model.add_objective(0 * dummy)  # type: ignore[operator]

        return OptimizationProblem(
            name=self.name,
            linopy_model=self.linopy_model,
            study=self.study,
            block=self.block,
            linopy_vars=self.linopy_vars,
            param_arrays=self.param_arrays,
            objective_constant=objective_constant,
        )

    # ------------------------------------------------------------------
    # Phase 1 — Parameter arrays
    # ------------------------------------------------------------------

    def _build_param_arrays_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        T = self.block_length
        S = len(self.scenario_ids)
        C = len(components)
        comp_ids = [c.id for c in components]
        abs_timesteps = self.block.timesteps  # mapping: block_t → abs_t

        for param in model.parameters.values():
            use_time = param.structure.time
            use_scenario = param.structure.scenario

            # Determine the minimal shape for this parameter based on its
            # declared structure. Using minimal shapes avoids spurious broadcasting
            # (e.g., invest_cost * p_max should not gain time/scenario dims).
            if use_time and use_scenario:
                data = np.zeros((C, T, S))
                dims = ["component", "time", "scenario"]
                coords: Dict[str, object] = {
                    "component": comp_ids,
                    "time": self.time_coord,
                    "scenario": self.local_scenario_coord,
                }
            elif use_time:
                data = np.zeros((C, T))
                dims = ["component", "time"]
                coords = {"component": comp_ids, "time": self.time_coord}
            elif use_scenario:
                data = np.zeros((C, S))
                dims = ["component", "scenario"]
                coords = {"component": comp_ids, "scenario": self.local_scenario_coord}
            else:
                data = np.zeros((C,))
                dims = ["component"]
                coords = {"component": comp_ids}

            mc_scenarios = self.scenario_ids if use_scenario else None
            for i, c in enumerate(components):
                v = self.study.database.get_values(
                    c.id,
                    param.name,
                    abs_timesteps if use_time else None,
                    mc_scenarios,
                )
                if use_time and use_scenario:
                    data[i, :, :] = v  # (T, S)
                elif use_time:
                    data[i, :] = v  # (T,) or scalar
                elif use_scenario:
                    data[i, :] = v  # (S,) or scalar
                else:
                    data[i] = v  # scalar

            arr = xr.DataArray(data, dims=dims, coords=coords)
            self.param_arrays[(model.id, param.name)] = arr

    # ------------------------------------------------------------------
    # Phase 2 — Variables
    # ------------------------------------------------------------------

    def _create_variables_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        comp_ids = [c.id for c in components]

        for var in model.variables.values():
            if not self._location_filter or self._location_filter.include_variable(
                model.id, var.name
            ):
                # Build coords for this variable
                coords: Dict[str, object] = {"component": comp_ids}
                dims = ["component"]
                if var.structure.time:
                    coords["time"] = self.time_coord
                    dims.append("time")
                if var.structure.scenario:
                    coords["scenario"] = self.local_scenario_coord
                    dims.append("scenario")

                # Shape of this variable (used to broadcast scalar bounds)
                var_shape = tuple(
                    (
                        len(comp_ids)
                        if d == "component"
                        else (
                            len(self.time_coord)
                            if d == "time"
                            else len(self.local_scenario_coord)
                        )
                    )
                    for d in dims
                )

                # Build a minimal builder for bound expressions (no variables needed)
                bound_builder = VectorizedLinearExprBuilder(
                    model_id=model.id,
                    linopy_vars={},
                    param_arrays=self.param_arrays,
                    port_arrays={},
                    block_length=self.block_length,
                )

                lower: object = (
                    np.full(var_shape, -np.inf)
                    if var.lower_bound is None
                    else self._to_bound_array(
                        visit(var.lower_bound, bound_builder), var_shape, dims
                    )
                )
                upper: object = (
                    np.full(var_shape, np.inf)
                    if var.upper_bound is None
                    else self._to_bound_array(
                        visit(var.upper_bound, bound_builder), var_shape, dims
                    )
                )

                # Validate bounds: upper must be >= lower across all timesteps/scenarios
                lower_arr = lower if isinstance(lower, np.ndarray) else np.array(lower)
                upper_arr = upper if isinstance(upper, np.ndarray) else np.array(upper)
                for ci, comp_id in enumerate(comp_ids):
                    lo = np.asarray(lower_arr[ci] if lower_arr.ndim > 0 else lower_arr)
                    up = np.asarray(upper_arr[ci] if upper_arr.ndim > 0 else upper_arr)
                    finite = np.isfinite(lo) & np.isfinite(up)
                    violation = finite & (up < lo)
                    if np.any(violation):
                        idx = int(np.argmax(violation))
                        raise ValueError(
                            f"Upper bound ({float(up.flat[idx]):g}) must be strictly "
                            f"greater than lower bound ({float(lo.flat[idx]):g}) "
                            f"for variable {comp_id}.{var.name}"
                        )

                prefix = model.id.replace("-", "_")
                name = f"{prefix}__{var.name}"
                binary = var.data_type == ValueType.BINARY
                integer = var.data_type == ValueType.INTEGER

                lv = self.linopy_model.add_variables(
                    lower=lower,
                    upper=upper,
                    coords=coords,
                    name=name,
                    binary=binary,
                    integer=integer,
                )
                self.linopy_vars[(model.id, var.name)] = lv

    # ------------------------------------------------------------------
    # Phase 3 — Port arrays
    # ------------------------------------------------------------------

    def _build_port_arrays_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        # If this model has no variables in the current problem (e.g. a
        # subproblem-only model when building the master), its port
        # contributions are zero — skip building port arrays for it.
        has_vars = any(
            (model.id, var_name) in self.linopy_vars for var_name in model.variables
        )
        if not has_vars and model.variables:
            self.port_arrays[model.id] = {}
            return
        self.port_arrays[model.id] = build_port_arrays(
            model,
            components,
            self.study,
            lambda mk_, m: self._make_builder(m, port_arrays={}),
        )

    # ------------------------------------------------------------------
    # Phase 4 — Constraints and Objectives
    # ------------------------------------------------------------------

    def _create_constraints_for_model(
        self,
        model: Model,
        port_arrays_for_model: Dict[PortFieldId, VectorizedExpr],
    ) -> None:
        """Add all constraints for *model* to the linopy model."""
        builder = self._make_builder(model, port_arrays=port_arrays_for_model)

        prefix = model.id.replace("-", "_")
        for constraint in model.get_all_constraints():
            if not self._location_filter or self._location_filter.include_constraint(
                model.id, constraint.name
            ):
                # Compute a per-(component, time) validity mask for drop mode.
                validity_mask: Optional[xr.DataArray] = None
                if self._oob_filter is not None:
                    from gems.optim_config.parsing import OutOfBoundsMode

                    mode = self._oob_filter.get_mode(model.id, constraint.name)
                    if mode == OutOfBoundsMode.DROP:
                        validity_mask = visit(
                            constraint.expression,
                            ShiftValidityVisitor(
                                model_id=model.id,
                                param_arrays=self.param_arrays,
                                block_length=self.block_length,
                            ),
                        )

                lhs = visit(constraint.expression, builder)

                # Skip constraints whose LHS evaluated to a pure DataArray (no
                # decision variables — e.g. an unconnected port aggregation).
                if isinstance(lhs, xr.DataArray):
                    continue

                if validity_mask is not None:
                    lhs = _apply_validity_mask(lhs, validity_mask)

                # Sanitize constraint name for LP format (spaces → underscores)
                safe_name = constraint.name.replace(" ", "_").replace("-", "_")

                # Lower bound constraint: lhs >= lb  (if lb != -inf)
                if not is_unbounded(constraint.lower_bound):
                    lb = visit(constraint.lower_bound, builder)
                    if validity_mask is not None:
                        lb = _apply_validity_mask(lb, validity_mask)
                    name = f"{prefix}__{safe_name}__lb"
                    con_lb = lhs >= lb  # type: ignore[operator]
                    self.linopy_model.add_constraints(con_lb, name=name)  # type: ignore[arg-type]

                # Upper bound constraint: lhs <= ub  (if ub != +inf)
                if not is_unbounded(constraint.upper_bound):
                    ub = visit(constraint.upper_bound, builder)
                    if validity_mask is not None:
                        ub = _apply_validity_mask(ub, validity_mask)
                    name = f"{prefix}__{safe_name}__ub"
                    con_ub = lhs <= ub  # type: ignore[operator]
                    self.linopy_model.add_constraints(con_ub, name=name)  # type: ignore[arg-type]

    def _add_objectives_for_model(
        self,
        model: Model,
        port_arrays_for_model: Dict[PortFieldId, VectorizedExpr],
        total_obj: Optional[VectorizedExpr],
    ) -> Optional[VectorizedExpr]:
        """Accumulate objective contributions from *model*."""
        builder = self._make_builder(model, port_arrays=port_arrays_for_model)

        def _accumulate(
            acc: Optional[VectorizedExpr], contribution: VectorizedExpr
        ) -> VectorizedExpr:
            if isinstance(contribution, xr.DataArray):
                summed: VectorizedExpr = float(contribution.sum().item())  # type: ignore[assignment]
            else:
                summed = contribution.sum()  # type: ignore[union-attr]
            return summed if acc is None else _linopy_add(acc, summed)

        if model.objective_contributions:
            for obj_id, expr in model.objective_contributions.items():
                if (
                    not self._location_filter
                    or self._location_filter.include_objective(model.id, obj_id)
                ) and expr is not None:
                    obj_term = visit(expr, builder)
                    total_obj = _accumulate(total_obj, obj_term)

        return total_obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bound_array(
        val: object,
        var_shape: Tuple[int, ...],
        dims: List[str],
    ) -> np.ndarray:
        """Convert a bound value to a numpy array shaped like *var_shape*.

        Handles scalar DataArrays, partial-dim DataArrays (e.g. a constant
        parameter used as a bound for a time×scenario variable), plain floats,
        and raw numpy arrays.
        """
        if isinstance(val, xr.DataArray):
            if val.dims == ():
                return np.full(var_shape, float(val.item()))
            arr = val.values  # shape may be a subset of var_shape dims
            for ax, d in enumerate(dims):
                if d not in val.dims:
                    arr = np.expand_dims(arr, axis=ax)
            return np.broadcast_to(arr, var_shape).copy()  # type: ignore[return-value]
        if isinstance(val, (int, float)):
            return np.full(var_shape, float(val))
        return val  # type: ignore[return-value]

    def _make_builder(
        self,
        model: Model,
        port_arrays: Dict[PortFieldId, VectorizedExpr],
    ) -> VectorizedLinearExprBuilder:
        return VectorizedLinearExprBuilder(
            model_id=model.id,
            linopy_vars=self.linopy_vars,
            param_arrays=self.param_arrays,
            port_arrays=port_arrays,
            block_length=self.block_length,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_problem(
    study: Study,
    block: TimeBlock,
    scenario_ids: List[int],
    optim_config: "Optional[OptimConfig]" = None,
    problem_name: str = "optimization_problem",
    initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
) -> OptimizationProblem:
    """
    Build and return an OptimizationProblem for the given time block.

    Parameters
    ----------
    study:
        Container holding both the System (components and connections) and
        the DataBase (parameter values for those components).
    block:
        The time block to optimize.
    scenario_ids:
        List of MC scenario indices to include.  Resolution to data-series
        column indices is handled transparently by the DataBase.
    optim_config:
        Optional parsed OptimConfig.  When provided, per-constraint
        out-of-bounds-processing rules (cyclic vs. drop) are applied.
        Constraints not listed default to cyclic.
    problem_name:
        Label for the linopy model.
    initial_values:
        Optional carry-over values keyed by ``(model_id, var_name)``.  For
        each entry a constraint ``var[time=0] == value`` is added, overriding
        the cyclic border condition for the first timestep.
    """
    study.check_consistency()

    oob_filter = OutOfBoundsFilter(optim_config) if optim_config is not None else None
    builder = _OptimizationProblemBuilder(
        name=problem_name,
        study=study,
        block=block,
        oob_filter=oob_filter,
        scenario_ids=scenario_ids,
        initial_values=initial_values,
    )
    return builder.build()


# ---------------------------------------------------------------------------
# Decomposed build — public entry point
# ---------------------------------------------------------------------------


@dataclass
class DecomposedProblems:
    """Holds the results of a decomposed problem build.

    Attributes
    ----------
    subproblem:
        OptimizationProblem containing all elements whose location is
        ``subproblems`` or ``master-and-subproblems``.
    master:
        OptimizationProblem containing all elements whose location is
        ``master`` or ``master-and-subproblems``.  ``None`` when the
        optim-config declares no master-side elements.
    """

    subproblem: OptimizationProblem
    master: Optional[OptimizationProblem]


def build_decomposed_problems(
    study: Study,
    block: TimeBlock,
    scenario_ids: List[int],
    optim_config: "OptimConfig",
    *,
    subproblem_name: str = "subproblem",
    master_name: str = "master",
) -> DecomposedProblems:
    """Build master and subproblem OptimizationProblems according to *optim_config*.

    The subproblem is always built; it contains every element whose declared
    location is ``subproblems`` (the default) or ``master-and-subproblems``.

    The master is built only when at least one element in *optim_config* has
    location ``master`` or ``master-and-subproblems``.

    Per-constraint out-of-bounds-processing rules (cyclic vs. drop) defined
    in the ``out-of-bounds-processing`` section of optim-config are applied
    to both the subproblem and the master.  Constraints not listed default to
    cyclic wrap-around.

    Parameters
    ----------
    study:
        Container holding both the System and the DataBase.
        Same semantics as :func:`build_problem`.
    block, scenario_ids:
        Same semantics as :func:`build_problem`.
    optim_config:
        Parsed ``OptimConfig`` from an ``optim-config.yml`` file.
    subproblem_name, master_name:
        Labels used for the underlying linopy models.
    """
    from gems.optim_config.parsing import ElementLocation

    study.check_consistency()

    oob_filter = OutOfBoundsFilter(optim_config)

    master_locs: Set["ElementLocation"] = {
        ElementLocation.MASTER,
        ElementLocation.MASTER_AND_SUBPROBLEMS,
    }
    sub_locs: Set["ElementLocation"] = {
        ElementLocation.SUBPROBLEMS,
        ElementLocation.MASTER_AND_SUBPROBLEMS,
    }

    oob_filter = OutOfBoundsFilter(optim_config)

    subproblem = _OptimizationProblemBuilder(
        name=subproblem_name,
        study=study,
        block=block,
        scenario_ids=scenario_ids,
        location_filter=DecompositionFilter(optim_config, sub_locs),
        oob_filter=oob_filter,
    ).build()

    master: Optional[OptimizationProblem] = None
    if _has_any_master_element(optim_config):
        master = _OptimizationProblemBuilder(
            name=master_name,
            study=study,
            block=block,
            scenario_ids=scenario_ids,
            location_filter=DecompositionFilter(optim_config, master_locs),
            oob_filter=oob_filter,
        ).build()

    return DecomposedProblems(subproblem=subproblem, master=master)


def _has_any_master_element(config: "OptimConfig") -> bool:
    """Return True if *config* declares at least one master-side element."""
    from gems.optim_config.parsing import ElementLocation

    master_locs = {ElementLocation.MASTER, ElementLocation.MASTER_AND_SUBPROBLEMS}
    for mc in config.models:
        if mc.model_decomposition is not None:
            d = mc.model_decomposition
            if any(v.location in master_locs for v in d.variables):
                return True
            if any(c.location in master_locs for c in d.constraints):
                return True
            if any(o.location in master_locs for o in d.objective_contributions):
                return True
    return False
