import functools
import warnings
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xarray as xr

from gems_craft.study.parsing import OutputFormat
from gems_runner.simulation.table_folder import _FolderStore, _Manifest, _TableWriter

# The schema names are also re-exported from here, where they used to live.
from gems_runner.simulation.table_schema import (
    _BLOCK,
    _SCENARIO,
    _TIME,
    _VALUE,
    OBJECTIVE_VALUE_OUTPUT,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
    SIMULATION_TABLE_DTYPES,
    SIMULATION_TABLE_SCHEMA,
    SimulationColumns,
    _apply_schema,
    _empty_frame,
    _equals,
    _same_key,
    _with_schema,
)
from gems_runner.simulation.table_store import _MemoryStore, _RowFilter, _TableStore

_VIEW_COLUMNS = [_BLOCK, _TIME, _SCENARIO, _VALUE]


def _filter_values(values: Optional[Sequence[Any]]) -> Optional[frozenset]:
    if values is None:
        return None
    # A single name, not the set of its characters.
    if isinstance(values, str):
        return frozenset({values})
    return frozenset(values)


class OutputView:
    """Time × Scenario values of one (component, output).

    Obtain via ``SimulationTable.component(...).output(...)``.

    - Rows are read from the table's storage on first access, then kept. On a
      table stored on disk, ``value(scenario_index=s)`` before any other access
      reads only the rows of scenario ``s`` and the rows shared by all
      scenarios.
    - An output without a time (or scenario) dimension has a single row (or
      column), labelled ``<NA>``, and ``value()`` accepts any index for that
      dimension.
    - Several blocks can hold a value for the same (time, scenario):
      overlapping sequential blocks, or the per-block rows of an output without
      a time dimension. A call whose requested values include such a
      (time, scenario) raises unless ``block=`` says which block to read.
    """

    def __init__(
        self,
        store: _TableStore,
        row_filter: _RowFilter,
        name: str,
        block: Optional[int] = None,
    ) -> None:
        self._store = store
        self._filter = row_filter
        self._name = name
        self._block = block
        self._scenario_rows: Dict[int, pd.DataFrame] = {}

    @functools.cached_property
    def _rows(self) -> pd.DataFrame:
        """All rows of the view, long format, one row per (block, time, scenario)."""
        return self._store.read(self._filter)[_VIEW_COLUMNS]

    def _rows_for(self, scenario_index: Optional[int]) -> pd.DataFrame:
        """The rows needed for *scenario_index*: all rows if already read,
        otherwise the rows of that scenario and the shared ones, kept per
        scenario.

        Reading one scenario only saves work when the table is stored on disk
        (one file per scenario): in memory, each read starts from all the rows
        of the view's component, so the view reads all its rows once instead."""
        if (
            scenario_index is None
            or self._store.path is None
            or "_rows" in self.__dict__
        ):
            return self._rows
        if scenario_index not in self._scenario_rows:
            narrowed = self._filter.narrow(scenarios=frozenset({scenario_index}))
            self._scenario_rows[scenario_index] = self._store.read(narrowed)[
                _VIEW_COLUMNS
            ]
        return self._scenario_rows[scenario_index]

    @property
    def data(self) -> pd.DataFrame:
        """Return the Time × Scenario DataFrame.

        Raises ``ValueError`` if a (time, scenario) has values from several
        blocks: select one with ``output(name, block=...)``.
        """
        return self._frame(self._select(self._rows, None, None, self._block))

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
            self._rows_for(scenario_index),
            time_index,
            scenario_index,
            self._block if block is None else block,
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
        rows: pd.DataFrame,
        time_index: Optional[int],
        scenario_index: Optional[int],
        block: Optional[int],
    ) -> pd.DataFrame:
        """Rows matching the request; raises if a (time, scenario) of the
        result has values from several blocks."""
        time_missing = bool(rows[_TIME].isna().all())
        scenario_missing = bool(rows[_SCENARIO].isna().all())
        if block is not None:
            rows = rows[_equals(rows[_BLOCK], block)]
        if time_index is not None and not time_missing:
            rows = rows[_equals(rows[_TIME], time_index)]
        if scenario_index is not None and not scenario_missing:
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

    Obtain via ``SimulationTable.component(...)``. It reads nothing itself: its
    outputs read their rows when accessed.
    """

    def __init__(
        self, store: _TableStore, row_filter: _RowFilter, component_id: str
    ) -> None:
        self._store = store
        self._filter = row_filter
        self._component_id = component_id

    def output(self, output_id: str, block: Optional[int] = None) -> OutputView:
        """Return an OutputView for the given output name.

        ``block`` restricts the view to one block, for outputs that have values
        from several blocks for the same (time, scenario).
        """
        return OutputView(
            self._store,
            self._filter.narrow(outputs=frozenset({output_id})),
            name=f"{self._component_id}.{output_id}",
            block=block,
        )


class SimulationTable:
    """The results of a simulation run, over all its scenarios.

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

    The rows are read through a storage: in memory (``SimulationTable(df)``),
    or a folder on disk (``SimulationTable.open(path)``), with the same API. The
    long-format DataFrame is accessible via the ``data`` property (in memory
    only), and parts of it via ``to_pandas()``, ``iter_scenarios()`` and
    ``scan()``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        table_id: str = "",
        scenarios: Optional[Sequence[int]] = None,
    ) -> None:
        """*scenarios* are the scenarios of the run, when known: they are kept
        even if no row names them, and a ``ValueError`` is raised if a row has
        another scenario, or if a scenario has no rows while others do. Without
        them, the scenarios are those of the rows."""
        self._store: _TableStore = _MemoryStore(df, table_id, scenarios)
        self.table_id = table_id

    @classmethod
    def _from_store(cls, store: _TableStore) -> "SimulationTable":
        table = cls.__new__(cls)
        table._store = store
        table.table_id = store.table_id
        return table

    @classmethod
    def open(cls, path: Path) -> "SimulationTable":
        """Open a simulation table folder written by ``save()`` or by a run.

        Raises ``FileNotFoundError`` for a folder without a manifest, e.g. an
        interrupted run. Nothing else is read until rows are asked for.
        """
        # Absolute, so that the table still reads after a chdir or when it is
        # unpickled in another working directory, and so that pyarrow never
        # reads the path as a URI (a relative folder named "s3:x").
        path = Path(path).resolve()
        return cls._from_store(_FolderStore(path, _Manifest.read(path)))

    def save(self, path: Path, format: OutputFormat = OutputFormat.PARQUET) -> Path:
        """Write the table as a simulation table folder at *path*, from either
        storage, and return *path*. *path* is created if missing and must be
        empty otherwise. ``SimulationTable.open(path)`` reads it back. If the
        write fails, the files written so far are removed, so it can be retried.
        """
        path = Path(path)
        writer = _TableWriter(path, self.table_id, format)
        try:
            for scenario, rows in self._store.iter_scenarios(_RowFilter()):
                if scenario is None:
                    writer.write_common(rows)
                else:  # also a scenario without rows, so that it stays listed
                    writer.write_scenario(rows, scenario)
            writer.finalize()
        except BaseException:  # Ctrl-C too
            writer.abort()
            raise
        return path

    @property
    def data(self) -> pd.DataFrame:
        """Return the underlying long-format DataFrame.

        Raises ``ValueError`` on a table stored on disk: use
        ``to_pandas(load_all=True)``, the filters of ``to_pandas()``, or
        ``iter_scenarios()``.
        """
        if self.path is not None:
            raise ValueError(
                f"This simulation table is stored on disk ({self.path}): .data "
                "would load it entirely. Use to_pandas(load_all=True), "
                "to_pandas() with filters, or iter_scenarios()."
            )
        return self._store.read(_RowFilter())

    @property
    def scenarios(self) -> List[int]:
        """The scenarios of the run, sorted."""
        return self._store.scenarios

    @property
    def path(self) -> Optional[Path]:
        """Where the table is stored, None in memory."""
        return self._store.path

    def component(self, component_id: str) -> ComponentView:
        """Return a ComponentView filtered to the given component ID."""
        return ComponentView(
            self._store, _RowFilter(components=frozenset({component_id})), component_id
        )

    def objective_values(self) -> pd.DataFrame:
        """Return the objective value of every solved problem, with columns
        ``block``, ``scenario_index`` and ``value``.

        Sequential/parallel modes give one row per block and scenario. Frontal
        mode gives one row, whose ``scenario_index`` is empty when the run has
        several scenarios.
        """
        rows = self._store.read(_RowFilter(outputs=frozenset({OBJECTIVE_VALUE_OUTPUT})))
        return rows[[_BLOCK, _SCENARIO, _VALUE]].reset_index(drop=True)

    def to_pandas(
        self,
        components: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
        scenarios: Optional[Sequence[int]] = None,
        load_all: bool = False,
    ) -> pd.DataFrame:
        """Return the rows of the given components, outputs and scenarios as a
        new DataFrame, all rows by default.

        A scenario filter keeps the rows shared by all scenarios (empty
        ``scenario_index``), which hold the value of every scenario.

        On a table stored on disk, loading all rows must be asked for with
        ``load_all=True``; it has no effect on a table in memory.
        """
        row_filter = _RowFilter(
            _filter_values(components),
            _filter_values(outputs),
            _filter_values(scenarios),
        )
        if row_filter.is_unrestricted and self.path is not None and not load_all:
            raise ValueError(
                f"This simulation table is stored on disk ({self.path}): "
                "to_pandas() without filter would load it entirely. Pass "
                "load_all=True, or filter by components, outputs or scenarios."
            )
        rows = self._store.read(row_filter)
        # In memory, every row is the store's own frame: copy it. On disk, the
        # rows are a new frame already.
        if row_filter.is_unrestricted and self.path is None:
            return rows.copy()
        return rows

    def iter_scenarios(
        self,
        components: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> Iterator[Tuple[Optional[int], pd.DataFrame]]:
        """Yield the rows of the given components and outputs one scenario at
        a time, as ``(scenario_index, DataFrame)``.

        The rows shared by all scenarios come first, once, as ``(None, rows)``,
        and only if there are any; then every scenario of the run, in order.
        """
        yield from self._store.iter_scenarios(
            _RowFilter(_filter_values(components), _filter_values(outputs))
        )

    def scan(self) -> ds.Dataset:
        """Return the whole table as a ``pyarrow.dataset.Dataset``, e.g. to read
        it with polars or duckdb."""
        return self._store.scan()

    def _deprecated(self, name: str, replacement: str) -> None:
        if self.path is not None:
            raise TypeError(
                f"{name}() is not available on a simulation table stored on disk: "
                "use save() to write it elsewhere, or to_pandas() to read it."
            )
        warnings.warn(
            f"SimulationTable.{name}() is deprecated and will be removed in a "
            f"following release; use {replacement}.",
            DeprecationWarning,
            stacklevel=3,
        )

    def to_csv(self, output_dir: Path) -> Path:
        """Deprecated: use ``save(path, OutputFormat.CSV)``."""
        self._deprecated("to_csv", "save(path, OutputFormat.CSV)")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"simulation_table_{self.table_id}.csv"
        self.data.to_csv(path, index=False)
        return path

    def to_parquet(self, output_dir: Path) -> Path:
        """Deprecated: use ``save(path)``."""
        self._deprecated("to_parquet", "save(path)")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"simulation_table_{self.table_id}.parquet"
        self.data.to_parquet(
            path,
            engine="pyarrow",
            index=False,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
        )
        return path

    def to_netcdf(self, output_dir: Path) -> Path:
        """Deprecated: build the xarray object from ``to_pandas()``."""
        self._deprecated("to_netcdf", "to_pandas() to build an xarray object")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"simulation_table_{self.table_id}.nc"
        self._to_dataset().to_netcdf(path)
        return path

    def to_dataset(self) -> xr.Dataset:
        """Deprecated: build the xarray object from ``to_pandas()``."""
        self._deprecated("to_dataset", "to_pandas() to build an xarray object")
        return self._to_dataset()

    def _to_dataset(self) -> xr.Dataset:
        """Return simulation results as an xr.Dataset.

        Each output variable becomes a DataArray with dimensions
        (component, absolute_time_index, scenario_index).
        Scalar rows without component/time/scenario (e.g. objective-value)
        are stored as zero-dimensional variables.
        """
        df = self.data
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
        return SimulationTable(
            self._build_frame(problem, absolute_time_offset, scenario_ids_remap),
            table_id=table_id,
            scenarios=self._run_scenarios(problem, scenario_ids_remap),
        )

    @staticmethod
    def _run_scenarios(
        problem: OptimizationProblem, scenario_ids_remap: Optional[List[int]]
    ) -> Optional[Sequence[int]]:
        """The scenario labels of the rows: the remapped ids, or the positions."""
        if scenario_ids_remap is not None:
            return scenario_ids_remap
        if problem.scenario_ids is not None:
            return range(len(problem.scenario_ids))
        return None

    def _build_frame(
        self,
        problem: OptimizationProblem,
        absolute_time_offset: Optional[int] = None,
        scenario_ids_remap: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """All the rows of *problem*, with the schema's dtypes."""
        block = problem.block.id
        if absolute_time_offset is None:
            # Use the first element of the block's absolute timestep list so that
            # the offset is correct for all modes, including blocks with overlap and
            # block ids that are not 1-based.
            absolute_time_offset = problem.block.timesteps[0]

        dfs = [
            self._da_to_df(
                da, name, block, absolute_time_offset, None, scenario_ids_remap
            )
            for name, da in self._collect_output_arrays(problem)
        ]
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
        return _apply_schema(df)

    def _iter_scenario_frames(
        self, problem: OptimizationProblem, scenario_ids_remap: List[int]
    ) -> Iterator[Tuple[Optional[int], pd.DataFrame]]:
        """Yield the rows of *problem* one scenario at a time, so that no table
        holding all scenarios is built.

        With several scenarios, the rows shared by all of them (outputs without
        a scenario dimension and the objective value) come first, as
        ``(None, rows)``; then ``(scenario, rows)`` for every scenario, even one
        without rows of its own. With a single scenario, every row belongs to
        it: one frame is yielded, as built by ``_build_frame()``.
        """
        if len(scenario_ids_remap) == 1:
            yield scenario_ids_remap[0], self._build_frame(
                problem, scenario_ids_remap=scenario_ids_remap
            )
            return

        block = problem.block.id
        offset = problem.block.timesteps[0]
        arrays = self._collect_output_arrays(problem)

        shared = [
            self._da_to_df(da, name, block, offset, None)
            for name, da in arrays
            if "scenario" not in da.dims
        ]
        shared.append(self._collect_objective_value(problem, block))
        yield None, _apply_schema(pd.concat(shared, ignore_index=True))

        for position, scenario in enumerate(scenario_ids_remap):
            frames = [
                self._da_to_df(
                    da.isel(scenario=[position]), name, block, offset, None, [scenario]
                )
                for name, da in arrays
                if "scenario" in da.dims
            ]
            rows = _concat_frames(frames)
            del frames  # the pieces are not held while the next scenario is built
            yield scenario, rows
            del rows

    def _collect_output_arrays(
        self, problem: OptimizationProblem
    ) -> List[Tuple[str, xr.DataArray]]:
        """(output name, array) for every variable, then every extra output, in
        table row order. Arrays keep their component, time and scenario
        dimensions; ``_da_to_df`` turns them into rows."""
        return self._collect_vars_outputs(problem) + self._collect_extra_outputs(
            problem
        )

    # -------------------------------------------------------------------------
    # Solver outputs
    # -------------------------------------------------------------------------

    def _collect_vars_outputs(
        self, problem: OptimizationProblem
    ) -> List[Tuple[str, xr.DataArray]]:
        arrays: List[Tuple[str, xr.DataArray]] = []
        solution = problem.linopy_model.solution
        if solution is None:
            return arrays

        for (mk, var_name), lv in problem._linopy_vars.items():
            sol_da = problem.get_variable_solution(mk, var_name)
            if sol_da is None:
                continue

            own_components = list(lv.coords["component"].values)
            arrays.append((var_name, sol_da.sel(component=own_components)))

        return arrays

    # -------------------------------------------------------------------------
    # Extra outputs
    # -------------------------------------------------------------------------

    def _collect_extra_outputs(
        self, problem: OptimizationProblem
    ) -> List[Tuple[str, xr.DataArray]]:
        arrays: List[Tuple[str, xr.DataArray]] = []

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

                arrays.append((out_id, result_da))

        return arrays

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


def _concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate pieces of a run into one DataFrame with the schema's dtypes,
    or an empty one for no piece."""
    frames = list(frames)
    if not frames:
        return _empty_frame()
    return _with_schema(pd.concat(frames, ignore_index=True))
