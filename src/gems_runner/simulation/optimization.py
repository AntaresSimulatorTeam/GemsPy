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
   :class:`~gems_runner.simulation.linearize.VectorizedLinearExprBuilder` to
   produce vectorized linopy constraints added in a single
   ``Model.add_constraints()`` call per constraint type.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    cast,
)

import linopy
import numpy as np
import xarray as xr

from gems_craft.expression.degree import is_linear
from gems_craft.expression.expression import is_unbounded
from gems_craft.expression.visitor import visit
from gems_craft.model.common import ValueType
from gems_craft.model.model import Model
from gems_craft.model.parameter import Parameter
from gems_craft.model.port import PortField, PortFieldId
from gems_craft.model.variable import Variable
from gems_craft.study.parsing import IntegerStrategyId
from gems_craft.study.study import Study
from gems_craft.study.system import Component
from gems_craft.study.validation import check_data_requirements
from gems_runner.simulation.linearize import (
    VectorizedExpr,
    VectorizedLinearExprBuilder,
    _linopy_add,
)
from gems_runner.simulation.time_block import TimeBlock
from gems_runner.simulation.vectorized_builder import ShiftValidityVisitor

if TYPE_CHECKING:
    from gems_craft.optim_config.parsing import (
        ElementLocation,
        OptimConfig,
        OutOfBoundsMode,
    )

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

LinopyModel = linopy.Model
"""Alias for :class:`linopy.Model`, distinguishing it from :class:`gems_craft.model.Model`."""

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
        The set of :class:`~gems_craft.optim_config.parsing.ElementLocation` values
        that should be *included* (not filtered out) by this filter.
    """

    def __init__(
        self, config: "OptimConfig", target_locations: "Set[ElementLocation]"
    ) -> None:
        from gems_craft.optim_config.parsing import ElementLocation as EL

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
                except NotImplementedError:
                    # Non-linear port-field definitions (dual, reduced_cost, max/min
                    # of variables, …) cannot be evaluated during the LP build phase.
                    # Treat as zero; the extra-output builder handles them post-solve.
                    if is_linear(defn):
                        raise
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
        except NotImplementedError:
            # Non-linear port-field definitions cannot be evaluated during the
            # LP build phase. Skip; the extra-output builder handles them post-solve.
            if is_linear(defn):
                raise
            continue

        expr_master_r = expr_master.rename({"component": "component_master"})  # type: ignore[union-attr]
        contribution = (A * expr_master_r).sum("component_master")  # type: ignore[operator]

        total = contribution if total is None else _linopy_add(total, contribution)

    return total if total is not None else xr.DataArray(0.0)


class _MergedGroupVariable(linopy.Variable):
    """Merged view of a variable split across relaxed/exact strategy groups.

    Detached copy (``xr.concat`` of the real per-group Variables) — safe to
    read and use in expressions, but writing bounds here wouldn't reach the
    solver. ``.lower``/``.upper`` setters raise instead of silently no-oping;
    use :meth:`OptimizationProblem.get_component_variable` to mutate bounds.
    """

    __slots__ = ()

    @linopy.Variable.lower.setter  # type: ignore[attr-defined]
    def lower(self, value: object) -> None:
        raise AttributeError(
            "Cannot set 'lower' on a merged relaxed/exact variable: it is a "
            "detached copy and the write would not reach the solver. Use "
            "OptimizationProblem.get_component_variable(...) instead."
        )

    @linopy.Variable.upper.setter  # type: ignore[attr-defined]
    def upper(self, value: object) -> None:
        raise AttributeError(
            "Cannot set 'upper' on a merged relaxed/exact variable: it is a "
            "detached copy and the write would not reach the solver. Use "
            "OptimizationProblem.get_component_variable(...) instead."
        )


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
        linopy_vars_by_component: Optional[
            Dict[Tuple[str, str, str], linopy.Variable]
        ] = None,
        set_sizes: Optional[Dict[str, Dict[str, xr.DataArray]]] = None,
    ) -> None:
        self.name = name
        # model_id -> set_id -> per-component element count (see VectorizedBuilderBase).
        self.set_sizes = set_sizes or {}
        self.linopy_model = linopy_model
        self.study = study
        self.block = block
        self._linopy_vars = linopy_vars
        self._linopy_vars_by_component = linopy_vars_by_component or {}
        self.param_arrays = param_arrays
        # Constant term of the objective (linopy cannot represent pure-constant objectives).
        self._objective_constant: float = objective_constant

    @property
    def block_length(self) -> int:
        return len(self.block.timesteps)

    # Solvers whose linopy backend implements the direct (in-memory) API.
    # All others fall back to file-based LP/MPS exchange.
    _DIRECT_API_SOLVERS = {"highs", "gurobi"}

    # Native option switching each solver's console output on or off. linopy
    # has no generic logging switch: options are forwarded to the solver as is.
    _LOG_OUTPUT_OPTIONS: Dict[str, Tuple[str, Callable[[bool], object]]] = {
        "highs": ("output_flag", bool),
        "gurobi": ("OutputFlag", int),
        "xpress": ("outputlog", int),
    }

    @classmethod
    def log_output_options(cls, solver_name: str, logs: bool) -> Dict[str, object]:
        """Return the solver option enabling (*logs* True) or silencing solver
        output, or no option for a solver without a known output switch."""
        if solver_name not in cls._LOG_OUTPUT_OPTIONS:
            return {}
        option, convert = cls._LOG_OUTPUT_OPTIONS[solver_name]
        return {option: convert(logs)}

    def solve(self, solver_name: str = "highs", **kwargs: object) -> None:
        """Solve the problem using the specified solver."""
        # Use io_api="direct" to bypass LP file writing and avoid LP name parsing
        # issues in linopy's set_int_index (e.g. constraint names with spaces or
        # variables with non-standard characters).
        if solver_name in self._DIRECT_API_SOLVERS:
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

    def get_component_variable(
        self, model_id: str, var_name: str, component_id: str
    ) -> Optional[linopy.Variable]:
        """Return the actually-registered linopy Variable that *component_id* was
        created on for (model_id, var_name).

        Unlike ``get_variable_labels``/the internal ``_linopy_vars`` dict — which,
        for a model split across relaxed/exact strategy groups, holds a merged
        Variable rebuilt via ``xr.concat`` and detached from the real registered
        objects — this is safe to mutate bounds on (e.g. for heuristics), since
        the mutation reaches what the solver actually reads.
        """
        return self._linopy_vars_by_component.get((model_id, var_name, component_id))

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

    def _reassemble_variable_attr(
        self, model_id: str, var_name: str, attr: str
    ) -> Optional[xr.DataArray]:
        """Reassemble a per-instance ``linopy.Variable`` attribute (``solution``,
        ``lower``, ``upper``) for *var_name* across all its components.

        Unlike reading the attribute off ``self._linopy_vars[(model_id, var_name)]``,
        this is correct even when the variable was split across relaxed/exact
        strategy groups: that merged, detached ``_MergedGroupVariable`` copy
        keeps the ``.name`` of only one of the two really-registered group
        Variables (so indexing the solver's solution Dataset by that name
        silently drops the other group's components), and it is rebuilt once at
        problem-build time so it never reflects later bound mutations (e.g. from
        heuristics). This instead reads the attribute directly off the real
        per-component Variables (``_linopy_vars_by_component``) and reassembles
        them.
        """
        by_name: Dict[str, linopy.Variable] = {}
        for (m, vn, _c), variable in self._linopy_vars_by_component.items():
            if (m, vn) == (model_id, var_name):
                by_name[variable.name] = (
                    variable  # dedupe: components in one group share a Variable
                )
        group_vars = list(by_name.values())
        if not group_vars:
            return None
        if len(group_vars) == 1:
            return cast(xr.DataArray, getattr(group_vars[0], attr))
        return cast(
            xr.DataArray,
            xr.concat([getattr(v, attr) for v in group_vars], dim="component"),
        )

    def get_variable_solution(
        self, model_id: str, var_name: str
    ) -> Optional[xr.DataArray]:
        """Return solved values for *var_name* across all its components.

        See :meth:`_reassemble_variable_attr` for why this must bypass the
        merged ``_linopy_vars`` copy.
        """
        return self._reassemble_variable_attr(model_id, var_name, "solution")

    def get_variable_lower_bound(
        self, model_id: str, var_name: str
    ) -> Optional[xr.DataArray]:
        """Return the current lower bound for *var_name* across all its components.

        Reflects any bound mutation applied after problem construction (e.g. by
        thermal heuristics via :meth:`get_component_variable`). See
        :meth:`_reassemble_variable_attr` for why this must bypass the merged
        ``_linopy_vars`` copy.
        """
        return self._reassemble_variable_attr(model_id, var_name, "lower")

    def get_variable_upper_bound(
        self, model_id: str, var_name: str
    ) -> Optional[xr.DataArray]:
        """Return the current upper bound for *var_name* across all its components.

        Reflects any bound mutation applied after problem construction (e.g. by
        thermal heuristics via :meth:`get_component_variable`). See
        :meth:`_reassemble_variable_attr` for why this must bypass the merged
        ``_linopy_vars`` copy.
        """
        return self._reassemble_variable_attr(model_id, var_name, "upper")


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------


def _validate_initial_values(
    initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]],
) -> Dict[Tuple[str, str], xr.DataArray]:
    """Check the carry-over contract on *initial_values* and return them.

    Every value must carry a ``time`` dimension indexed ``0 .. k-1`` so that it
    aligns with the leading timesteps of the block being built.
    """
    values = initial_values or {}
    for (mk, var_name), init_val in values.items():
        if "time" not in init_val.dims:
            raise ValueError(
                f"initial_values[{mk!r}, {var_name!r}] must carry a 'time' "
                f"dimension indexed 0..k-1; got dims {tuple(init_val.dims)}"
            )
    return values


def _model_set_ids(model: Model) -> List[str]:
    """Sorted ids of the custom sets indexing a parameter or variable of *model*."""
    ids: Set[str] = set()
    for param in model.parameters.values():
        ids |= param.structure.sets
    for var in model.variables.values():
        ids |= var.structure.sets
    return sorted(ids)


def _set_validity(
    sizes: Dict[str, xr.DataArray], dim_sizes: Dict[str, int]
) -> Optional[xr.DataArray]:
    """Boolean mask of the non-padded positions along the sets of *dim_sizes*.

    ``sizes[set_id]`` holds each component's own element count, ``dim_sizes``
    the padded dimension length of each set to check.  Returns ``None`` when
    nothing is padded.
    """
    mask: Optional[xr.DataArray] = None
    for set_id, n in dim_sizes.items():
        if int(sizes[set_id].min()) == n:
            continue
        positions = xr.DataArray(
            np.arange(n), dims=[set_id], coords={set_id: list(range(n))}
        )
        valid = positions < sizes[set_id]
        mask = valid if mask is None else mask & valid
    return mask


class _OptimizationProblemBuilder:
    """
    Builds the linopy problem in 5 phases:
      1. Build parameter DataArrays for all models.
      2. Create all linopy Variables (uses param arrays for bounds).
      3. Build port arrays via incidence matrices.
      4. Add constraints and objectives to the linopy model.
      5. Add the carry-over constraints of *initial_values* (sequential mode):
         each time-dependent variable is *fixed*, over the block's first ``k``
         timesteps, to the value the previous block computed for the same
         absolute timestep.  Time-independent variables are never pinned.
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
        self._initial_values = _validate_initial_values(initial_values)

        self.block_length = len(block.timesteps)
        self.time_coord = list(range(self.block_length))
        self.local_scenario_coord = list(range(len(scenario_ids)))

        # Populated during build
        self.linopy_model = linopy.Model()
        self.linopy_vars: Dict[Tuple[str, str], linopy.Variable] = {}
        # Unlike linopy_vars (which, for a model whose components are split
        # across relaxed/exact groups, holds a *merged* Variable rebuilt via
        # xr.concat — a copy, detached from the actually-registered per-group
        # Variable objects), this maps each individual component to the real
        # Variable it was created on, so bound mutations (heuristics) reach
        # the model.
        self.linopy_vars_by_component: Dict[Tuple[str, str, str], linopy.Variable] = {}
        self.param_arrays: Dict[Tuple[str, str], xr.DataArray] = {}
        self.port_arrays: Dict[str, Dict[PortFieldId, VectorizedExpr]] = {}
        # model_id -> set_id -> per-component element count (sets used by the model).
        self.set_sizes: Dict[str, Dict[str, xr.DataArray]] = {}

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

        # Phase 5: carry-over constraints (sequential mode only).
        # Only time-dependent variables are pinned: time-independent ones
        # (structure.time = False, e.g. an investment capacity) are deliberately
        # left free in every block — see the user guide, `sequential-subproblems`.
        for (mk, var_name), init_val in self._initial_values.items():
            linopy_var = self.linopy_vars.get((mk, var_name))
            if linopy_var is None or "time" not in linopy_var.dims:
                continue
            # Pin the first len(init_val.time) timesteps, clamped to this
            # block's horizon (a truncated final block can be shorter than the
            # carried window).
            pin_length = min(init_val.sizes["time"], self.block_length)
            safe = f"{mk}__{var_name}".replace("-", "_")
            self.linopy_model.add_constraints(
                linopy_var.isel(time=slice(0, pin_length))
                == init_val.isel(time=slice(0, pin_length)),  # type: ignore[arg-type]
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
            linopy_vars_by_component=self.linopy_vars_by_component,
            param_arrays=self.param_arrays,
            objective_constant=objective_constant,
            set_sizes=self.set_sizes,
        )

    # ------------------------------------------------------------------
    # Phase 1 — Parameter arrays
    # ------------------------------------------------------------------

    def _build_param_arrays_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        comp_ids = [c.id for c in components]

        set_ids = _model_set_ids(model)
        if set_ids:
            self.set_sizes[model.id] = {
                set_id: xr.DataArray(
                    [
                        len(self.study.system.set_elements(c, set_id))
                        for c in components
                    ],
                    dims=["component"],
                    coords={"component": comp_ids},
                )
                for set_id in set_ids
            }

        for param in model.parameters.values():
            self.param_arrays[(model.id, param.name)] = self._build_param_array(
                model, param, components
            )

    def _build_param_array(
        self, model: Model, param: Parameter, components: List[Component]
    ) -> xr.DataArray:
        """Array of *param*, dims ``[component][, time][, scenario][, *sets]``.

        Only the dimensions of the parameter's declared structure are kept: using
        minimal shapes avoids spurious broadcasting (e.g., invest_cost * p_max
        should not gain time/scenario dims). Set dimensions are padded (with
        zeros) to the largest instantiation among *components*; dimensions a
        component's data does not vary over (constant, or narrowed ``indexed-by``
        override) are broadcast.
        """
        structure = param.structure
        db = self.study.database

        lead: Dict[str, int] = {}
        if structure.time:
            lead["time"] = self.block_length
        if structure.scenario:
            lead["scenario"] = len(self.scenario_ids)
        dim_sizes = lead | self._set_dim_sizes(model.id, structure.sets)
        dims = list(dim_sizes)
        data = np.zeros((len(components),) + tuple(dim_sizes.values()))

        for i, c in enumerate(components):
            v = np.asarray(
                db.get_values(
                    c.id,
                    param.name,
                    self.block.timesteps if structure.time else None,
                    self.scenario_ids if structure.scenario else None,
                )
            )
            # get_values returns the requested time/scenario axes followed by the
            # data's own set axes (or a scalar for constant data).
            v_dims = (
                []
                if v.ndim == 0
                else list(lead) + list(db.get_data(c.id, param.name).set_dims)
            )
            missing = {d: n for d, n in lead.items() if d not in v_dims}
            missing |= {
                s: len(self.study.system.set_elements(c, s))
                for s in structure.sets
                if s not in v_dims
            }
            da = xr.DataArray(v, dims=v_dims).expand_dims(missing).transpose(*dims)
            data[(i,) + tuple(slice(0, n) for n in da.shape)] = da.values

        coords: Dict[str, object] = {"component": [c.id for c in components]}
        coords.update({d: list(range(n)) for d, n in dim_sizes.items()})
        return xr.DataArray(data, dims=["component"] + dims, coords=coords)

    # ------------------------------------------------------------------
    # Phase 2 — Variables
    # ------------------------------------------------------------------

    def _create_variables_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        RELAXED_STRATEGIES = {IntegerStrategyId.RELAXED, IntegerStrategyId.HEURISTIC}
        comp_ids = [c.id for c in components]

        for var in model.variables.values():
            if not self._location_filter or self._location_filter.include_variable(
                model.id, var.name
            ):
                needs_split = var.data_type in (
                    ValueType.INTEGER,
                    ValueType.BINARY,
                ) and any(
                    c.integer_strategy.id in RELAXED_STRATEGIES for c in components
                )

                if needs_split:
                    groups: Dict[bool, List[Component]] = {}
                    for c in components:
                        groups.setdefault(
                            c.integer_strategy.id in RELAXED_STRATEGIES, []
                        ).append(c)
                    partial = [
                        self._add_variable_for_group(
                            model,
                            var,
                            group,
                            relax=relax,
                            name_suffix="__relaxed" if relax else "",
                        )
                        for relax, group in groups.items()
                    ]
                    for group_var, group in zip(partial, groups.values()):
                        for c in group:
                            self.linopy_vars_by_component[
                                (model.id, var.name, c.id)
                            ] = group_var
                    merged = cast(
                        xr.Dataset,
                        xr.concat([lv.data for lv in partial], dim="component"),
                    ).sel(component=comp_ids)
                    # xr.concat keeps only partial[0]'s attrs; widen label_range
                    # by hand to cover every group's labels.
                    merged.attrs["label_range"] = (
                        min(lv.range[0] for lv in partial),
                        max(lv.range[1] for lv in partial),
                    )
                    prefix = model.id.replace("-", "_")
                    merged_name = f"{prefix}__{var.name}__merged"
                    self.linopy_vars[(model.id, var.name)] = _MergedGroupVariable(
                        merged, partial[0].model, merged_name
                    )
                else:
                    single_var = self._add_variable_for_group(
                        model, var, components, relax=False
                    )
                    self.linopy_vars[(model.id, var.name)] = single_var
                    for c in components:
                        self.linopy_vars_by_component[(model.id, var.name, c.id)] = (
                            single_var
                        )

    def _param_arrays_for_components(
        self, model_id: str, comp_ids: List[str]
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Restrict *model_id*'s parameter arrays to *comp_ids*.

        ``self.param_arrays`` holds one array per (model_id, param_name),
        carrying every component that shares that model. When a model's
        variables are split into per-strategy groups (``_create_variables_for_model``),
        a bound expression evaluated for one group must only see that group's
        own components — otherwise a component-varying parameter's full-model
        array gets force-broadcast into the (smaller) group's shape and fails.
        """
        return {
            key: arr.sel(component=comp_ids) if key[0] == model_id else arr
            for key, arr in self.param_arrays.items()
        }

    def _set_sizes_for_components(
        self, model_id: str, comp_ids: List[str]
    ) -> Dict[str, xr.DataArray]:
        """Restrict *model_id*'s per-component set sizes to *comp_ids*."""
        return {
            set_id: sizes.sel(component=comp_ids)
            for set_id, sizes in self.set_sizes.get(model_id, {}).items()
        }

    def _add_variable_for_group(
        self,
        model: Model,
        var: Variable,
        components: List[Component],
        relax: bool,
        name_suffix: str = "",
    ) -> linopy.Variable:
        comp_ids = [c.id for c in components]
        coords: Dict[str, object] = {"component": comp_ids}
        dims = ["component"]
        if var.structure.time:
            coords["time"] = self.time_coord
            dims.append("time")
        if var.structure.scenario:
            coords["scenario"] = self.local_scenario_coord
            dims.append("scenario")
        # Custom-set dimensions, padded to the model-wide largest instantiation.
        set_ids = sorted(var.structure.sets)
        dim_sizes = self._set_dim_sizes(model.id, set_ids)
        for s in set_ids:
            coords[s] = list(range(dim_sizes[s]))
            dims.append(s)

        var_shape = tuple(
            (
                len(comp_ids)
                if d == "component"
                else (
                    len(self.time_coord)
                    if d == "time"
                    else (
                        len(self.local_scenario_coord)
                        if d == "scenario"
                        else dim_sizes[d]
                    )
                )
            )
            for d in dims
        )

        bound_builder = VectorizedLinearExprBuilder(
            model_id=model.id,
            linopy_vars={},
            param_arrays=self._param_arrays_for_components(model.id, comp_ids),
            port_arrays={},
            block_length=self.block_length,
            set_sizes=self._set_sizes_for_components(model.id, comp_ids),
        )

        is_binary = var.data_type == ValueType.BINARY
        lower: object = (
            np.full(var_shape, 0.0 if is_binary else -np.inf)
            if var.lower_bound is None
            else self._to_bound_array(
                visit(var.lower_bound, bound_builder), var_shape, dims
            )
        )
        upper: object = (
            np.full(var_shape, 1.0 if is_binary else np.inf)
            if var.upper_bound is None
            else self._to_bound_array(
                visit(var.upper_bound, bound_builder), var_shape, dims
            )
        )

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

        mask: Optional[xr.DataArray] = None
        validity = _set_validity(
            self._set_sizes_for_components(model.id, comp_ids), dim_sizes
        )
        if validity is not None:
            full = xr.DataArray(
                np.ones(var_shape, dtype=bool), dims=dims, coords=coords
            )
            mask = (validity & full).transpose(*dims)

        prefix = model.id.replace("-", "_")
        name = f"{prefix}__{var.name}{name_suffix}"
        return self.linopy_model.add_variables(
            lower=lower,
            upper=upper,
            coords=coords,
            name=name,
            mask=mask,
            binary=var.data_type == ValueType.BINARY and not relax,
            integer=var.data_type == ValueType.INTEGER and not relax,
        )

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
                    from gems_craft.optim_config.parsing import OutOfBoundsMode

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

                padding_mask = self._padding_mask(model, lhs)

                if constraint.is_equality:
                    lb = visit(constraint.lower_bound, builder)
                    if validity_mask is not None:
                        lb = _apply_validity_mask(lb, validity_mask)
                    self.linopy_model.add_constraints(lhs == lb, name=f"{prefix}__{safe_name}__eq", mask=padding_mask)  # type: ignore[operator,arg-type]
                else:
                    if not is_unbounded(constraint.lower_bound):
                        lb = visit(constraint.lower_bound, builder)
                        if validity_mask is not None:
                            lb = _apply_validity_mask(lb, validity_mask)
                        self.linopy_model.add_constraints(lhs >= lb, name=f"{prefix}__{safe_name}__lb", mask=padding_mask)  # type: ignore[operator,arg-type]
                    if not is_unbounded(constraint.upper_bound):
                        ub = visit(constraint.upper_bound, builder)
                        if validity_mask is not None:
                            ub = _apply_validity_mask(ub, validity_mask)
                        self.linopy_model.add_constraints(lhs <= ub, name=f"{prefix}__{safe_name}__ub", mask=padding_mask)  # type: ignore[operator,arg-type]

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
            if set(dims) - {"component", "time", "scenario"}:
                # Custom-set dimensions: align by name rather than by position.
                missing = {d: n for d, n in zip(dims, var_shape) if d not in val.dims}
                if missing:
                    val = val.expand_dims(missing)
                return np.broadcast_to(val.transpose(*dims).values, var_shape).copy()  # type: ignore[return-value]
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
            set_sizes=self.set_sizes.get(model.id, {}),
        )

    def _padding_mask(
        self, model: Model, expr: VectorizedExpr
    ) -> Optional[xr.DataArray]:
        """Mask of the non-padded set positions of *expr* (``None`` if nothing is padded)."""
        used = [s for s in self.set_sizes.get(model.id, {}) if s in expr.dims]  # type: ignore[union-attr]
        return _set_validity(
            self.set_sizes.get(model.id, {}), self._set_dim_sizes(model.id, used)
        )

    def _set_dim_sizes(self, model_id: str, set_ids: Iterable[str]) -> Dict[str, int]:
        """Padded dimension length of each of *set_ids* (largest instantiation)."""
        sizes = self.set_sizes.get(model_id, {})
        return {s: int(sizes[s].max()) for s in sorted(set_ids)}


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
        Optional carry-over values keyed by ``(model_id, var_name)``.  Each
        value must be an ``xr.DataArray`` carrying a ``time`` dimension of
        length ``k`` indexed ``0 .. k-1``; constraints
        ``var[time=i] == value[i]`` are then added for the block's first ``k``
        timesteps, overriding the cyclic border condition on that window.  A
        value without a ``time`` dimension raises ``ValueError`` before the
        problem is built.  Entries whose variable is time-independent, or is
        absent from this block, are ignored.
    """
    check_data_requirements(study)

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
    from gems_craft.optim_config.parsing import ElementLocation

    check_data_requirements(study)

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
    from gems_craft.optim_config.parsing import ElementLocation

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
