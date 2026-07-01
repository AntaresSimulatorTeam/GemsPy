from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union, cast

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
    
class SimulationColumns(str, Enum):
    BLOCK = "block"
    COMPONENT = "component"
    OUTPUT = "output"
    ABSOLUTE_TIME_INDEX = "absolute-time-index"
    BLOCK_TIME_INDEX = "block-time-index"
    SCENARIO_INDEX = "scenario-index"
    VALUE = "value"
    BASIS_STATUS = "basis-status"