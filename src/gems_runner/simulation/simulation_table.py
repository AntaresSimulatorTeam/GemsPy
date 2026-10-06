from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import xarray as xr

_BASE_DIMS = ("component", "time", "scenario")
PARQUET_COMPRESSION: Literal["zstd"] = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 64_000


class OutputView:
    """A Time × Scenario pivot for one (component, output) combination.

    Obtain via ``SimulationTable.component(...).output(...)``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # df: index = absolute_time_index, columns = scenario_index
        self._df = df

    @property
    def data(self) -> pd.DataFrame:
        """Return the underlying Time × Scenario DataFrame (Time × (Scenario, set_index) if set-indexed)."""
        return self._df

    def value(
        self,
        time_index: Optional[int] = None,
        scenario_index: Optional[int] = None,
    ) -> Union[pd.DataFrame, "pd.Series[Any]", float]:
        """Return results filtered by time and/or scenario index.

        Called with no arguments returns the full Time × Scenario DataFrame.
        Called with one argument returns a ``pd.Series``:
        - ``value(scenario_index=s)`` → Series indexed by absolute_time_index
        - ``value(time_index=t)``     → Series indexed by scenario_index
        Called with both arguments returns a scalar ``float``.
        For a set-indexed output the columns are a (scenario_index, set_index)
        MultiIndex: the results are DataFrames/Series keyed by ``set_index`` instead.
        """
        if time_index is None and scenario_index is None:
            return self._df
        if time_index is not None and scenario_index is not None:
            res = self._df.loc[time_index, scenario_index]
            # Series over set_index for a set-indexed output
            return res if isinstance(res, pd.Series) else float(cast(Any, res))
        if time_index is not None:
            return self._df.loc[time_index]  # Series over scenarios
        return self._df[scenario_index]  # Series over time

    def __repr__(self) -> str:
        return repr(self._df)


class ComponentView:
    """Filtered view of simulation results for one component.

    Obtain via ``SimulationTable.component(...)``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def output(self, output_id: str) -> OutputView:
        """Return an OutputView for the given output name.

        For a set-indexed output, the columns are a (scenario_index, set_index)
        MultiIndex.
        """
        col_output = SimulationColumns.OUTPUT.value
        col_time = SimulationColumns.ABSOLUTE_TIME_INDEX.value
        col_scenario = SimulationColumns.SCENARIO_INDEX.value
        col_value = SimulationColumns.VALUE.value

        col_set = SimulationColumns.SET_INDEX.value

        filtered = self._df[self._df[col_output] == output_id].copy()
        # Dimension-independent outputs store None for the missing index.
        # Fill with 0 so the pivot is always well-formed and the accessor
        # API (value(time_index=t, scenario_index=s)) keeps working.
        filtered[col_time] = filtered[col_time].fillna(0)
        filtered[col_scenario] = filtered[col_scenario].fillna(0)
        columns = (
            [col_scenario, col_set] if filtered[col_set].notna().any() else col_scenario
        )
        pivot = filtered.pivot_table(
            index=col_time,
            columns=columns,
            values=col_value,
            aggfunc="first",
        )
        pivot.index.name = col_time
        return OutputView(pivot)


class SimulationTable:
    """Wrapper around the raw simulation results DataFrame.

    Provides a fluent accessor API::

        st = SimulationTableBuilder().build(problem)

        # Full Time × Scenario DataFrame
        st.component("gen_1").output("p").value()

        # Scalar at a specific time and scenario
        st.component("gen_1").output("p").value(time_index=0, scenario_index=0)

        # Time series for scenario 0
        st.component("gen_1").output("p").value(scenario_index=0)

        # Scenario distribution at time step 3
        st.component("gen_1").output("p").value(time_index=3)

    The underlying long-format DataFrame is accessible via the ``data`` property.
    """

    def __init__(self, df: pd.DataFrame, table_id: str = "") -> None:
        self._df = df
        self.table_id = table_id

    @property
    def data(self) -> pd.DataFrame:
        """Return the underlying long-format DataFrame."""
        return self._df

    def component(self, component_id: str) -> ComponentView:
        """Return a ComponentView filtered to the given component ID."""
        mask = self._df[SimulationColumns.COMPONENT.value] == component_id
        return ComponentView(self._df[mask])

    def to_csv(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"simulation_table_{self.table_id}.csv"
        self._df.to_csv(path, index=False)
        return path

    def to_parquet(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"simulation_table_{self.table_id}.parquet"
        self._df.to_parquet(
            path,
            engine="pyarrow",
            index=False,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
        )
        return path

    def to_netcdf(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"simulation_table_{self.table_id}.nc"
        self.to_dataset().to_netcdf(path)
        return path

    def to_dataset(self) -> xr.Dataset:
        """Return simulation results as an xr.Dataset.

        Each output variable becomes a DataArray with dimensions
        (component, absolute_time_index, scenario_index). A set-indexed output
        gets one more dimension named after its ``set_id`` (e.g. ``"fuel|seg"``),
        labelled by ``set_index``.
        Scalar rows without component/time/scenario (e.g. objective-value)
        are stored as zero-dimensional variables.
        """
        df = self._df
        col_comp = SimulationColumns.COMPONENT.value
        col_out = SimulationColumns.OUTPUT.value
        col_time = SimulationColumns.ABSOLUTE_TIME_INDEX.value
        col_scen = SimulationColumns.SCENARIO_INDEX.value
        col_val = SimulationColumns.VALUE.value

        col_set_id = SimulationColumns.SET_ID.value
        col_set_idx = SimulationColumns.SET_INDEX.value

        main = df.dropna(subset=[col_comp, col_time, col_scen])
        plain = main[main[col_set_id].isna()]
        indexed = plain.set_index([col_comp, col_time, col_scen, col_out])[col_val]
        unstacked = indexed.unstack(col_out)
        ds = xr.Dataset.from_dataframe(unstacked)

        for (out, set_id), grp in main[main[col_set_id].notna()].groupby(
            [col_out, col_set_id]
        ):
            series = grp.set_index([col_comp, col_time, col_scen, col_set_idx])[col_val]
            da = xr.DataArray.from_series(series).rename({col_set_idx: set_id})
            ds = xr.merge([ds, da.to_dataset(name=out)])

        scalars = df[df[col_comp].isna() & df[col_time].isna()]
        for _, row in scalars.iterrows():
            ds[row[col_out]] = xr.DataArray(float(row[col_val]))

        return ds


from gems_craft.expression.visitor import visit
from gems_runner.simulation.extra_output import VectorizedExtraOutputBuilder
from gems_runner.simulation.optimization import OptimizationProblem, build_port_arrays


class SimulationColumns(str, Enum):
    BLOCK = "block"
    COMPONENT = "component"
    OUTPUT = "output"
    ABSOLUTE_TIME_INDEX = "absolute_time_index"
    BLOCK_TIME_INDEX = "block_time_index"
    SCENARIO_INDEX = "scenario_index"
    SET_ID = "set_id"
    SET_INDEX = "set_index"
    VALUE = "value"
    BASIS_STATUS = "basis_status"


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

        for (mk, var_name), lv in problem._linopy_vars.items():
            sol_da = problem.get_variable_solution(mk, var_name)
            if sol_da is None:
                continue

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
                    set_elements=self._set_elements_lookup(problem),
                )
            )

        return dfs

    @staticmethod
    def _set_elements_lookup(
        problem: OptimizationProblem,
    ) -> Callable[[Optional[str], str], Sequence[Union[str, int]]]:
        """(component id or None, set id) -> elements of that set for the component."""

        def lookup(
            component_id: Optional[str], set_id: str
        ) -> Sequence[Union[str, int]]:
            system = problem.study.system
            if component_id is None:
                return system.global_sets[set_id]
            return system.set_elements(system.get_component(component_id), set_id)

        return lookup

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
        if problem.linopy_model.solution is not None:
            for (mk, vname), lv in problem._linopy_vars.items():
                sol_da = problem.get_variable_solution(mk, vname)
                if sol_da is None:
                    continue
                if "component" in sol_da.dims:
                    own_components = list(lv.coords["component"].values)
                    sol_da = sol_da.sel(component=own_components)
                var_solution_arrays[(mk, vname)] = sol_da

        constraint_dual_arrays = self._collect_constraint_duals(problem)
        var_reduced_cost_arrays = self._collect_reduced_costs(problem)
        var_lower_bound_arrays = self._collect_lower_bounds(problem)
        var_upper_bound_arrays = self._collect_upper_bounds(problem)

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
                    var_lower_bound_arrays=var_lower_bound_arrays,
                    var_upper_bound_arrays=var_upper_bound_arrays,
                    port_arrays={},
                    block_length=problem.block_length,
                    set_sizes=problem.set_sizes.get(mk_, {}),
                ),
            )

            for out_id, expr_node in model.extra_outputs.items():
                builder = VectorizedExtraOutputBuilder(
                    model_id=mk,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    constraint_dual_arrays=constraint_dual_arrays,
                    var_reduced_cost_arrays=var_reduced_cost_arrays,
                    var_lower_bound_arrays=var_lower_bound_arrays,
                    var_upper_bound_arrays=var_upper_bound_arrays,
                    port_arrays=port_arrays,
                    block_length=problem.block_length,
                    set_sizes=problem.set_sizes.get(mk, {}),
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
                        set_elements=self._set_elements_lookup(problem),
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
        for mk, components in problem.study.model_components.items():
            model = problem.study.models[mk]
            prefix = mk.replace("-", "_")
            own_components = [c.id for c in components]
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
                if "component" in dual_val.dims:
                    dual_val = dual_val.sel(component=own_components)
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

    @staticmethod
    def _collect_lower_bounds(
        problem: OptimizationProblem,
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Return current variable lower bounds keyed by (model_key, var_name)."""
        result: Dict[Tuple[str, str], xr.DataArray] = {}
        for mk, vname in problem._linopy_vars:
            lb = problem.get_variable_lower_bound(mk, vname)
            if lb is not None:
                result[(mk, vname)] = lb
        return result

    @staticmethod
    def _collect_upper_bounds(
        problem: OptimizationProblem,
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Return current variable upper bounds keyed by (model_key, var_name)."""
        result: Dict[Tuple[str, str], xr.DataArray] = {}
        for mk, vname in problem._linopy_vars:
            ub = problem.get_variable_upper_bound(mk, vname)
            if ub is not None:
                result[(mk, vname)] = ub
        return result

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
                    SimulationColumns.SET_ID.value: None,
                    SimulationColumns.SET_INDEX.value: None,
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
        set_elements: Optional[
            Callable[[Optional[str], str], Sequence[Union[str, int]]]
        ] = None,
    ) -> pd.DataFrame:
        """Vectorize a [component?, time?, scenario?, *sets] DataArray into a DataFrame.

        Index columns (absolute_time_index, block_time_index, scenario_index) are
        set to None for dimensions that are absent from the original DataArray,
        signalling that the output is independent of that dimension.

        Custom-set dimensions (sorted by id) are folded into pipe-joined ``set_id`` /
        ``set_index`` columns (element names); rows at padded positions, beyond a
        component's own set size, are dropped.
        """
        has_time = "time" in da.dims
        has_scenario = "scenario" in da.dims
        set_dims = sorted(str(d) for d in da.dims if d not in _BASE_DIMS)

        if "component" not in da.dims:
            da = da.expand_dims(component=[None])
        if not has_time:
            da = da.expand_dims(time=[0])
        if not has_scenario:
            da = da.expand_dims(scenario=[0])

        da = da.transpose("component", "time", "scenario", *set_dims)
        comp_vals: List[Any] = list(da.coords["component"].values)
        n_s = da.shape[2]

        idx = np.indices(da.shape).reshape(da.ndim, -1)
        values = da.values.ravel().astype(float)
        set_index: Optional[List[str]] = None
        if set_dims:
            if set_elements is None:
                raise ValueError("set_elements is required for set-indexed outputs")
            keep = np.ones(values.shape, dtype=bool)
            labels = []
            for k, set_id in enumerate(set_dims):
                # names[component, position]; None marks padded positions
                names = np.full((da.shape[0], da.shape[3 + k]), None, dtype=object)
                for c, comp in enumerate(comp_vals):
                    elems = set_elements(None if comp is None else str(comp), set_id)
                    names[c, : len(elems)] = [str(e) for e in elems]
                labels.append(names[idx[0], idx[3 + k]])
                keep &= labels[-1] != None  # noqa: E711 (elementwise)
            idx, values = idx[:, keep], values[keep]
            set_index = ["|".join(row) for row in zip(*(lab[keep] for lab in labels))]
        ci, ti, pos_s = idx[0], idx[1], idx[2]
        raw_si = np.asarray(
            scenario_ids_remap if scenario_ids_remap is not None else range(n_s)
        )

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
                SimulationColumns.SCENARIO_INDEX.value: (
                    raw_si[pos_s] if has_scenario else None
                ),
                SimulationColumns.SET_ID.value: "|".join(set_dims) or None,
                SimulationColumns.SET_INDEX.value: set_index,
                SimulationColumns.VALUE.value: values,
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
