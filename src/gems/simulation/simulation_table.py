from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import xarray as xr


class OutputView:
    """A Time × Scenario pivot for one (component, output) combination.

    Obtain via ``SimulationTable.component(...).output(...)``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # df: index = absolute-time-index, columns = scenario-index
        self._df = df

    @property
    def data(self) -> pd.DataFrame:
        """Return the underlying Time × Scenario DataFrame."""
        return self._df

    def value(
        self,
        time_index: Optional[int] = None,
        scenario_index: Optional[int] = None,
    ) -> Union[pd.DataFrame, "pd.Series[Any]", float]:
        """Return results filtered by time and/or scenario index.

        Called with no arguments returns the full Time × Scenario DataFrame.
        Called with one argument returns a ``pd.Series``:
        - ``value(scenario_index=s)`` → Series indexed by absolute-time-index
        - ``value(time_index=t)``     → Series indexed by scenario-index
        Called with both arguments returns a scalar ``float``.
        """
        if time_index is None and scenario_index is None:
            return self._df
        if time_index is not None and scenario_index is not None:
            return float(cast(Any, self._df.loc[time_index, scenario_index]))
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
        """Return an OutputView for the given output name."""
        col_output = SimulationColumns.OUTPUT.value
        col_time = SimulationColumns.ABSOLUTE_TIME_INDEX.value
        col_scenario = SimulationColumns.SCENARIO_INDEX.value
        col_value = SimulationColumns.VALUE.value

        filtered = self._df[self._df[col_output] == output_id].copy()
        # Dimension-independent outputs store None for the missing index.
        # Fill with 0 so the pivot is always well-formed and the accessor
        # API (value(time_index=t, scenario_index=s)) keeps working.
        filtered[col_time] = filtered[col_time].fillna(0)
        filtered[col_scenario] = filtered[col_scenario].fillna(0)
        pivot = filtered.pivot_table(
            index=col_time,
            columns=col_scenario,
            values=col_value,
            aggfunc="first",
        )
        pivot.index.name = col_time
        pivot.columns.name = col_scenario
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
        self._df.to_parquet(path, index=False)
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
        (component, absolute-time-index, scenario-index).
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
        indexed = main.set_index([col_comp, col_time, col_scen, col_out])[col_val]
        unstacked = indexed.unstack(col_out)
        ds = xr.Dataset.from_dataframe(unstacked)

        scalars = df[df[col_comp].isna() & df[col_time].isna()]
        for _, row in scalars.iterrows():
            ds[row[col_out]] = xr.DataArray(float(row[col_val]))

        return ds


from gems.expression.visitor import visit
from gems.simulation.extra_output import VectorizedExtraOutputBuilder
from gems.simulation.optimization import OptimizationProblem, build_port_arrays


class SimulationColumns(str, Enum):
    BLOCK = "block"
    COMPONENT = "component"
    OUTPUT = "output"
    ABSOLUTE_TIME_INDEX = "absolute-time-index"
    BLOCK_TIME_INDEX = "block-time-index"
    SCENARIO_INDEX = "scenario-index"
    VALUE = "value"
    BASIS_STATUS = "basis-status"


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
                    port_arrays={},
                    block_length=problem.block_length,
                ),
            )

            for out_id, expr_node in model.extra_outputs.items():
                builder = VectorizedExtraOutputBuilder(
                    model_id=mk,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
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
