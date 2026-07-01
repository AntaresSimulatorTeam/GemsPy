from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import xarray as xr

from gems_craft.simulation_table.simulation_table import (
    SimulationTable,
    SimulationColumns,
)


from gems_runner.expression.visitor import visit
from gems_runner.simulation.extra_output import VectorizedExtraOutputBuilder
from gems_runner.simulation.optimization import OptimizationProblem, build_port_arrays


class SimulationTableBuilder:
    """Builds simulation tables directly from a OptimizationProblem."""

    def __init__(self, simulation_id: Optional[str] = None) -> None:
        self.simulation_id: str = simulation_id or datetime.now().strftime(
            "%Y%m%d-%H%M"
        )

    def build(
        self,
        problem: OptimizationProblem,
        absolute_time_offset: Optional[int] = None,
        scenario_ids_remap: Optional[List[int]] = None,
        table_id: str = "",
    ) -> SimulationTable:
        block = problem.block.id
        block_size = problem.block_length

        if absolute_time_offset is None:
            # Use the first element of the block's absolute timestep list so that
            # the offset is correct for all modes, including blocks with overlap and
            # block ids that are not 1-based.
            absolute_time_offset = problem.block.timesteps[0]

        dfs: list[pd.DataFrame] = []
        dfs += self._collect_vars_outputs(
            problem, block, absolute_time_offset, scenario_ids_remap
        )
        dfs += self._collect_extra_outputs(
            problem, block, absolute_time_offset, scenario_ids_remap
        )
        dfs.append(self._collect_objective_value(problem, block))

        return SimulationTable(pd.concat(dfs, ignore_index=True), table_id=table_id)

    # -------------------------------------------------------------------------
    # Solver outputs
    # -------------------------------------------------------------------------

    def _collect_vars_outputs(
        self,
        problem: OptimizationProblem,
        block: int,
        abs_offset: int,
        scenario_ids_remap: Optional[List[int]] = None,
    ) -> list[pd.DataFrame]:
        dfs: list[pd.DataFrame] = []
        solution = problem.linopy_model.solution
        if solution is None:
            return dfs

        for (_, var_name), lv in problem._linopy_vars.items():
            if lv.name not in solution:
                continue

            sol_da: xr.DataArray = solution[lv.name]
            own_components = list(lv.coords["component"].values)
            sol_da = sol_da.sel(component=own_components)

            dfs.append(
                self._da_to_df(
                    sol_da,
                    var_name,
                    block,
                    abs_offset,
                    basis_status=None,
                    scenario_ids_remap=scenario_ids_remap,
                )
            )

        return dfs

    # -------------------------------------------------------------------------
    # Extra outputs
    # -------------------------------------------------------------------------

    def _collect_extra_outputs(
        self,
        problem: OptimizationProblem,
        block: int,
        abs_offset: int,
        scenario_ids_remap: Optional[List[int]] = None,
    ) -> list[pd.DataFrame]:
        dfs: list[pd.DataFrame] = []

        var_solution_arrays: Dict[Tuple[str, str], xr.DataArray] = {}
        solution = problem.linopy_model.solution
        if solution is not None:
            for (mk, vname), lv in problem._linopy_vars.items():
                if lv.name in solution:
                    var_solution_arrays[(mk, vname)] = solution[lv.name]

        constraint_dual_arrays = self._collect_constraint_duals(problem)
        var_reduced_cost_arrays = self._collect_reduced_costs(problem)

        for mk, components in problem.study.model_components.items():
            model = problem.study.models[mk]
            if not model.extra_outputs:
                continue

            port_arrays = build_port_arrays(
                model,
                components,
                problem.study,
                lambda mk_, m: VectorizedExtraOutputBuilder(
                    model_id=mk_,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    constraint_dual_arrays=constraint_dual_arrays,
                    var_reduced_cost_arrays=var_reduced_cost_arrays,
                    port_arrays={},
                    block_length=problem.block_length,
                ),
            )

            for out_id, expr_node in model.extra_outputs.items():
                builder = VectorizedExtraOutputBuilder(
                    model_id=mk,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    constraint_dual_arrays=constraint_dual_arrays,
                    var_reduced_cost_arrays=var_reduced_cost_arrays,
                    port_arrays=port_arrays,
                    block_length=problem.block_length,
                )
                result_da: xr.DataArray = cast(xr.DataArray, visit(expr_node, builder))

                if "component" in result_da.dims:
                    own_ids = [c.id for c in components]
                    present = [
                        c for c in own_ids if c in result_da.coords["component"].values
                    ]
                    result_da = result_da.sel(component=present)

                dfs.append(
                    self._da_to_df(
                        result_da,
                        out_id,
                        block,
                        abs_offset,
                        basis_status=None,
                        scenario_ids_remap=scenario_ids_remap,
                    )
                )

        return dfs

    # -------------------------------------------------------------------------
    # Dual / reduced-cost arrays (helpers for _collect_extra_outputs)
    # -------------------------------------------------------------------------

    @staticmethod
    def _collect_constraint_duals(
        problem: OptimizationProblem,
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Return constraint shadow prices keyed by (model_key, constraint_name)."""
        dual_dataset = problem.linopy_model.dual
        result: Dict[Tuple[str, str], xr.DataArray] = {}
        for mk in problem.study.model_components:
            model = problem.study.models[mk]
            prefix = mk.replace("-", "_")
            all_constraints = {**model.constraints, **model.binding_constraints}
            for cname in all_constraints:
                safe = cname.replace(" ", "_").replace("-", "_")
                dual_val: xr.DataArray = xr.DataArray(0.0)
                eq_name = f"{prefix}__{safe}__eq"
                lb_name = f"{prefix}__{safe}__lb"
                ub_name = f"{prefix}__{safe}__ub"
                if eq_name in dual_dataset:
                    dual_val = dual_val + dual_dataset[eq_name]  # type: ignore[operator]
                if lb_name in dual_dataset:
                    dual_val = dual_val + dual_dataset[lb_name]  # type: ignore[operator]
                if ub_name in dual_dataset:
                    dual_val = dual_val + dual_dataset[ub_name]  # type: ignore[operator]
                result[(mk, cname)] = dual_val
        return result

    @staticmethod
    def _collect_reduced_costs(
        problem: OptimizationProblem,
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Return variable reduced costs keyed by (model_key, var_name).
        Linopy API does not have a way to get reduced cost, so need to fallback to solver-specific API
        """
        solver_model = getattr(problem.linopy_model, "solver_model", None)
        if solver_model is None:
            return {}
        try:
            vlabels = problem.linopy_model.matrices.vlabels
            col_dual_vals: Optional[List[float]] = None

            if hasattr(solver_model, "getLpSol"):
                # Xpress >= 9.8: returns (x, slack, duals, djs).
                # Must be checked before getSolution because xpress.problem also
                # has getSolution (with an incompatible return type).
                _, _, _, dj_list = solver_model.getLpSol()
                col_dual_vals = list(dj_list)
            elif hasattr(solver_model, "getSolution"):
                # HiGHS: col_dual holds reduced costs in column order
                solution = solver_model.getSolution()
                col_dual_vals = list(solution.col_dual)
            elif hasattr(solver_model, "getAttr") and hasattr(solver_model, "getVars"):
                # Gurobi: getAttr("RC", vars) returns reduced costs in column order
                col_dual_vals = list(solver_model.getAttr("RC", solver_model.getVars()))

            if col_dual_vals is None:
                return {}

            rc_array = np.array(col_dual_vals, dtype=float)
            rc_array[np.asarray(vlabels) == -1] = float("nan")
            rc_series = pd.Series(rc_array, index=vlabels, dtype=float)

            result: Dict[Tuple[str, str], xr.DataArray] = {}
            for (mk, vname), lv in problem._linopy_vars.items():
                idx = np.ravel(lv.labels.values)
                rc_vals = rc_series.reindex(idx).to_numpy().reshape(lv.labels.shape)
                result[(mk, vname)] = xr.DataArray(
                    rc_vals, coords=lv.labels.coords, dims=lv.labels.dims
                )
            return result
        except Exception:
            return {}

    # -------------------------------------------------------------------------
    # Objective value
    # -------------------------------------------------------------------------

    def _collect_objective_value(
        self, problem: OptimizationProblem, block: int
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    SimulationColumns.BLOCK.value: block,
                    SimulationColumns.COMPONENT.value: None,
                    SimulationColumns.OUTPUT.value: "objective-value",
                    SimulationColumns.ABSOLUTE_TIME_INDEX.value: None,
                    SimulationColumns.BLOCK_TIME_INDEX.value: None,
                    SimulationColumns.SCENARIO_INDEX.value: None,
                    SimulationColumns.VALUE.value: problem.objective_value,
                    SimulationColumns.BASIS_STATUS.value: None,
                }
            ]
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _da_to_df(
        da: xr.DataArray,
        output_name: str,
        block: int,
        abs_offset: int,
        basis_status: Optional[str],
        scenario_ids_remap: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Vectorize a [component?, time?, scenario?] DataArray into a DataFrame.

        Index columns (absolute-time-index, block-time-index, scenario-index) are
        set to None for dimensions that are absent from the original DataArray,
        signalling that the output is independent of that dimension.
        """
        has_time = "time" in da.dims
        has_scenario = "scenario" in da.dims

        if "component" not in da.dims:
            da = da.expand_dims(component=[None])
        if not has_time:
            da = da.expand_dims(time=[0])
        if not has_scenario:
            da = da.expand_dims(scenario=[0])

        da = da.transpose("component", "time", "scenario")
        comp_vals: List[Any] = list(da.coords["component"].values)
        n_c, n_t, n_s = da.shape

        ci = np.repeat(np.arange(n_c), n_t * n_s)
        ti = np.tile(np.repeat(np.arange(n_t), n_s), n_c)
        raw_si = (
            scenario_ids_remap if scenario_ids_remap is not None else list(range(n_s))
        )
        si = np.tile(raw_si, n_c * n_t)

        return pd.DataFrame(
            {
                SimulationColumns.BLOCK.value: block,
                SimulationColumns.COMPONENT.value: [
                    str(c) if c is not None else None for c in np.array(comp_vals)[ci]
                ],
                SimulationColumns.OUTPUT.value: output_name,
                SimulationColumns.ABSOLUTE_TIME_INDEX.value: (
                    (abs_offset + ti) if has_time else None
                ),
                SimulationColumns.BLOCK_TIME_INDEX.value: ti if has_time else None,
                SimulationColumns.SCENARIO_INDEX.value: si if has_scenario else None,
                SimulationColumns.VALUE.value: da.values.ravel().astype(float),
                SimulationColumns.BASIS_STATUS.value: basis_status,
            }
        )


def merge_simulation_tables(
    tables: List[SimulationTable], table_id: str = ""
) -> SimulationTable:
    """Concatenate multiple SimulationTables into one."""
    return SimulationTable(
        pd.concat([t.data for t in tables], ignore_index=True), table_id=table_id
    )
