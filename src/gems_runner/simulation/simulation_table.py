from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

PARQUET_COMPRESSION: Literal["zstd"] = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 64_000


class OutputView:
    """Time × Scenario values of one (component, output).

    Obtain via ``SimulationTable.component(...).output(...)``.

    - An output without a time (or scenario) dimension has a single row (or
      column), labelled ``<NA>``, and ``value()`` accepts any index for that
      dimension.
    - Several blocks can hold a value for the same (time, scenario):
      overlapping sequential blocks, or the per-block rows of an output without
      a time dimension. A call whose requested values include such a
      (time, scenario) raises unless ``block=`` says which block to read.
    """

    def __init__(
        self, rows: pd.DataFrame, name: str, block: Optional[int] = None
    ) -> None:
        # rows: long format, one row per (block, time, scenario).
        self._rows = rows
        self._name = name
        self._block = block
        self._time_missing = bool(rows[_TIME].isna().all())
        self._scenario_missing = bool(rows[_SCENARIO].isna().all())

    @property
    def data(self) -> pd.DataFrame:
        """Return the Time × Scenario DataFrame.

        Raises ``ValueError`` if a (time, scenario) has values from several
        blocks: select one with ``output(name, block=...)``.
        """
        return self._frame(self._select(None, None, self._block))

    def value(
        self,
        time_index: Optional[int] = None,
        scenario_index: Optional[int] = None,
        block: Optional[int] = None,
    ) -> Union[pd.DataFrame, "pd.Series[Any]", float]:
        """Return results filtered by time and/or scenario index.

        Called with no index returns the full Time × Scenario DataFrame.
        Called with one index returns a ``pd.Series``:
        - ``value(scenario_index=s)`` → Series indexed by absolute_time_index
        - ``value(time_index=t)``     → Series indexed by scenario_index
        Called with both indices returns a scalar ``float``.

        An index of a dimension the output does not have is accepted and
        ignored. ``block`` selects the block, and is required when a requested
        (time, scenario) has values from several blocks.
        """
        rows = self._select(
            time_index, scenario_index, self._block if block is None else block
        )
        if time_index is None and scenario_index is None:
            return self._frame(rows)
        if rows.empty:
            raise KeyError(
                f"{self._name}: no value for time_index={time_index}, "
                f"scenario_index={scenario_index}"
            )
        if time_index is not None and scenario_index is not None:
            return float(rows[_VALUE].iloc[0])
        frame = self._frame(rows)
        if time_index is not None:
            return frame.iloc[0]  # Series over scenarios
        return frame.iloc[:, 0]  # Series over time

    def _select(
        self,
        time_index: Optional[int],
        scenario_index: Optional[int],
        block: Optional[int],
    ) -> pd.DataFrame:
        """Rows matching the request; raises if a (time, scenario) of the
        result has values from several blocks."""
        rows = self._rows
        if block is not None:
            rows = rows[_equals(rows[_BLOCK], block)]
        if time_index is not None and not self._time_missing:
            rows = rows[_equals(rows[_TIME], time_index)]
        if scenario_index is not None and not self._scenario_missing:
            rows = rows[_equals(rows[_SCENARIO], scenario_index)]

        shared = rows[rows.duplicated(subset=[_TIME, _SCENARIO], keep=False)]
        if not shared.empty:
            time, scenario = shared[_TIME].iloc[0], shared[_SCENARIO].iloc[0]
            same_key = _same_key(shared[_TIME], time) & _same_key(
                shared[_SCENARIO], scenario
            )
            blocks = sorted(int(b) for b in shared[same_key][_BLOCK].unique())
            raise ValueError(
                f"{self._name}: absolute_time_index={time}, "
                f"scenario_index={scenario} has values from blocks {blocks}; "
                "pass block= to choose one"
            )
        return rows

    @staticmethod
    def _frame(rows: pd.DataFrame) -> pd.DataFrame:
        frame = rows.pivot(index=_TIME, columns=_SCENARIO, values=_VALUE)
        return frame.sort_index(axis=0).sort_index(axis=1)

    def __repr__(self) -> str:
        return repr(self._rows)


class ComponentView:
    """Filtered view of simulation results for one component.

    Obtain via ``SimulationTable.component(...)``.
    """

    def __init__(self, df: pd.DataFrame, component_id: str) -> None:
        self._df = df
        self._component_id = component_id

    def output(self, output_id: str, block: Optional[int] = None) -> OutputView:
        """Return an OutputView for the given output name.

        ``block`` restricts the view to one block, for outputs that have values
        from several blocks for the same (time, scenario).
        """
        rows = self._df[_equals(self._df[_OUTPUT], output_id)]
        return OutputView(
            rows[[_BLOCK, _TIME, _SCENARIO, _VALUE]],
            name=f"{self._component_id}.{output_id}",
            block=block,
        )


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
        mask = _equals(self._df[SimulationColumns.COMPONENT.value], component_id)
        return ComponentView(self._df[mask], component_id)

    def objective_values(self) -> pd.DataFrame:
        """Return the objective value of every solved problem, with columns
        ``block``, ``scenario_index`` and ``value``.

        Sequential/parallel modes give one row per block and scenario. Frontal
        mode gives one row, whose ``scenario_index`` is empty when the run has
        several scenarios.
        """
        rows = self._df[_equals(self._df[_OUTPUT], OBJECTIVE_VALUE_OUTPUT)]
        return rows[[_BLOCK, _SCENARIO, _VALUE]].reset_index(drop=True)

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
        (component, absolute_time_index, scenario_index).
        Scalar rows without component/time/scenario (e.g. objective-value)
        are stored as zero-dimensional variables.
        """
        df = self._df
        col_comp = SimulationColumns.COMPONENT.value
        col_out = SimulationColumns.OUTPUT.value
        col_time = SimulationColumns.ABSOLUTE_TIME_INDEX.value
        col_scen = SimulationColumns.SCENARIO_INDEX.value
        col_val = SimulationColumns.VALUE.value

        main = df.dropna(subset=[col_comp, col_time, col_scen])
        # xarray needs numpy dtypes; these columns have no empty cell left.
        main = main.astype(
            {col_comp: object, col_out: object, col_time: "int64", col_scen: "int64"}
        )
        indexed = main.set_index([col_comp, col_time, col_scen, col_out])[col_val]
        unstacked = indexed.unstack(col_out)
        ds = xr.Dataset.from_dataframe(unstacked)

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
    VALUE = "value"
    BASIS_STATUS = "basis_status"


_BLOCK = SimulationColumns.BLOCK.value
_OUTPUT = SimulationColumns.OUTPUT.value
_TIME = SimulationColumns.ABSOLUTE_TIME_INDEX.value
_SCENARIO = SimulationColumns.SCENARIO_INDEX.value
_VALUE = SimulationColumns.VALUE.value

OBJECTIVE_VALUE_OUTPUT = "objective-value"

# One schema for the simulation table, whatever the content: nullable integer
# indices (empty for a missing dimension), string labels and float values.
_INDEX_COLUMNS = {
    SimulationColumns.BLOCK,
    SimulationColumns.ABSOLUTE_TIME_INDEX,
    SimulationColumns.BLOCK_TIME_INDEX,
    SimulationColumns.SCENARIO_INDEX,
}
_LABEL_COLUMNS = {
    SimulationColumns.COMPONENT,
    SimulationColumns.OUTPUT,
    SimulationColumns.BASIS_STATUS,
}
SIMULATION_TABLE_SCHEMA = pa.schema(
    [
        (
            column.value,
            (
                pa.int64()
                if column in _INDEX_COLUMNS
                else pa.string() if column in _LABEL_COLUMNS else pa.float64()
            ),
        )
        for column in SimulationColumns
    ]
)
# Labels use the python-backed string dtype: with the pyarrow-backed one,
# combining a label comparison and an index comparison with ``&`` raises on
# empty cells (e.g. the component of the objective-value row).
SIMULATION_TABLE_DTYPES: Dict[str, Any] = {
    column.value: (
        "Int64"
        if column in _INDEX_COLUMNS
        else pd.StringDtype("python") if column in _LABEL_COLUMNS else "float64"
    )
    for column in SimulationColumns
}


def _apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Give *df* the columns and dtypes of the simulation table schema."""
    return df[list(SIMULATION_TABLE_DTYPES)].astype(SIMULATION_TABLE_DTYPES)


def _equals(column: "pd.Series[Any]", value: Any) -> "pd.Series[bool]":
    """``column == value`` as a plain boolean mask: empty cells never match."""
    return column.eq(value).fillna(False).astype(bool)


def _same_key(column: "pd.Series[Any]", value: Any) -> "pd.Series[bool]":
    """Like ``_equals``, but an empty *value* matches the empty cells."""
    return column.isna() if pd.isna(value) else _equals(column, value)


def _tag_single_scenario(df: pd.DataFrame, scenario_id: int) -> pd.DataFrame:
    """Tag every row of a problem solved for a single MC scenario with it.

    This is always the case in sequential/parallel modes, and in frontal mode
    with one scenario: every row belongs to that scenario, including
    scenario-independent outputs and the objective value, whose empty scenario
    index would wrongly mean "shared by all scenarios".
    """
    scenario_col = SimulationColumns.SCENARIO_INDEX.value
    column = df[scenario_col]
    # where() + infer_objects() gives an int64 column on every pandas version:
    # fillna() downcasts with a FutureWarning on pandas 2.x, and no longer
    # downcasts on pandas 3.
    df[scenario_col] = column.where(column.notna(), scenario_id).infer_objects()
    return df


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

        df = pd.concat(dfs, ignore_index=True)
        scenario_ids = (
            scenario_ids_remap
            if scenario_ids_remap is not None
            else problem.scenario_ids
        )
        if scenario_ids is not None and len(scenario_ids) == 1:
            # Same label as the rows that have a scenario dimension: the
            # remapped id, or the scenario position 0 without a remap.
            label = scenario_ids_remap[0] if scenario_ids_remap is not None else 0
            df = _tag_single_scenario(df, label)

        return SimulationTable(_apply_schema(df), table_id=table_id)

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
                    SimulationColumns.OUTPUT.value: OBJECTIVE_VALUE_OUTPUT,
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

        Index columns (absolute_time_index, block_time_index, scenario_index) are
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
