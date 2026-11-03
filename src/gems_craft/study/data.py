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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gems_craft.study.scenario_builder import ScenarioBuilder


@dataclass(frozen=True)
class TimeScenarioIndex:
    time: int
    scenario: int


@dataclass(frozen=True)
class TimeIndex:
    time: int


@dataclass(frozen=True)
class ScenarioIndex:
    scenario: int


@dataclass(frozen=True)
class AbstractDataStructure(ABC):
    @abstractmethod
    def get_value(
        self,
        timestep: Optional[List[int]],
        scenario: Optional[np.ndarray],
    ) -> Union[float, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def check_requirement(self, time: bool, scenario: bool) -> bool:
        """Check if the data structure meets certain requirements."""
        pass

    def column_count(self) -> Optional[int]:
        """Number of data-series columns (time series), None if the data does
        not depend on the scenario."""
        return None


@dataclass(frozen=True)
class ConstantData(AbstractDataStructure):
    value: float

    def get_value(
        self,
        timestep: Optional[List[int]],
        scenario: Optional[np.ndarray],
    ) -> float:
        return self.value

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, ConstantData):
            raise ValueError("Invalid data type for ConstantData")
        return True


@dataclass(frozen=True)
class TimeSeriesData(AbstractDataStructure):
    """Time-only series: one value per timestep, scenario-independent."""

    time_series: pd.Series

    def get_value(
        self,
        timestep: Optional[List[int]],
        scenario: Optional[np.ndarray],
    ) -> np.ndarray:
        if timestep is None:
            raise KeyError("Time series data requires a time index.")
        result: np.ndarray = np.asarray(self.time_series.values)[np.asarray(timestep)]
        if scenario is not None:
            return np.broadcast_to(
                result[:, np.newaxis], (len(timestep), len(scenario))
            )
        return result

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, TimeSeriesData):
            raise ValueError("Invalid data type for TimeSeriesData")
        return time


@dataclass(frozen=True)
class ScenarioSeriesData(AbstractDataStructure):
    """Scenario-only series: one value per data-series column, time-independent.

    ``scenario_series`` is a 1-D numpy array indexed by 0-based column index.
    """

    scenario_series: np.ndarray

    def get_value(
        self,
        timestep: Optional[List[int]],
        scenario: Optional[np.ndarray],
    ) -> np.ndarray:
        if scenario is None:
            raise KeyError("Scenario series data requires a scenario index.")
        result = self.scenario_series[scenario]  # (S,)
        if timestep is not None:
            return np.broadcast_to(
                result[np.newaxis, :], (len(timestep), len(scenario))
            )
        return result

    def column_count(self) -> int:
        return len(self.scenario_series)

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, ScenarioSeriesData):
            raise ValueError("Invalid data type for ScenarioSeriesData")
        return scenario


@dataclass(frozen=True)
class TimeScenarioSeriesData(AbstractDataStructure):
    """Time × scenario series: values for every (timestep, column) pair."""

    time_scenario_series: pd.DataFrame

    def get_value(
        self,
        timestep: Optional[List[int]],
        scenario: Optional[np.ndarray],
    ) -> np.ndarray:
        if timestep is None:
            raise KeyError("Time scenario data requires a time index.")
        if scenario is None:
            raise KeyError("Time scenario data requires a scenario index.")
        return self.time_scenario_series.values[np.ix_(np.asarray(timestep), scenario)]

    def column_count(self) -> int:
        return self.time_scenario_series.shape[1]

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, TimeScenarioSeriesData):
            raise ValueError("Invalid data type for TimeScenarioSeriesData")
        return time and scenario


def load_ts_from_file(
    timeseries_name: Optional[str], path_to_file: Optional[Path]
) -> pd.DataFrame:
    if path_to_file is None or timeseries_name is None:
        raise FileNotFoundError(f"File '{timeseries_name}' does not exist")

    base_path = path_to_file / timeseries_name
    candidates = [base_path.with_suffix(".txt"), base_path.with_suffix(".tsv")]

    last_exc: Optional[Exception] = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            sep = r"\s+" if candidate.suffix == ".txt" else "\t"
            return pd.read_csv(candidate, header=None, sep=sep)
        except Exception as e:
            last_exc = e
            break

    if last_exc is not None:
        raise Exception(
            f"An error has arrived when processing '{candidate}': {last_exc}"
        )

    raise FileNotFoundError(
        f"File '{timeseries_name}.txt' or '{timeseries_name}.tsv' does not exist"
    )


def dataframe_to_time_series(ts_dataframe: pd.DataFrame) -> pd.Series:
    if ts_dataframe.shape[1] != 1:
        raise ValueError(
            f"Could not convert input data to time series data. Expect data series with exactly one column, got shape {ts_dataframe.shape}"
        )
    return ts_dataframe.iloc[:, 0]


def dataframe_to_scenario_series(ts_dataframe: pd.DataFrame) -> np.ndarray:
    """Return a 1-D numpy array of floats indexed by 0-based column index."""
    if ts_dataframe.shape[0] != 1:
        raise ValueError(
            f"Could not convert input data to scenario series data. Expect data series with exactly one line, got shape {ts_dataframe.shape}"
        )
    return ts_dataframe.iloc[0, :].to_numpy(dtype=float)


@dataclass(frozen=True)
class ComponentParameterIndex:
    component_id: str
    parameter_name: str


class DataBase:
    """Container for component parameter data.

    Resolves MC scenario indices to data-series column indices at use time
    via the optional ``ScenarioBuilder``.  The mapping is vectorized: a
    single numpy array index per call, no Python loop over scenarios.
    """

    def __init__(
        self,
        scenario_builder: Optional["ScenarioBuilder"] = None,
    ) -> None:
        self._data: Dict[ComponentParameterIndex, AbstractDataStructure] = {}
        self._scenario_groups: Dict[ComponentParameterIndex, Optional[str]] = {}
        self._series_names: Dict[ComponentParameterIndex, Optional[str]] = {}
        self._scenario_builder = scenario_builder

    def get_data(self, component_id: str, parameter_name: str) -> AbstractDataStructure:
        return self._data[ComponentParameterIndex(component_id, parameter_name)]

    def add_data(
        self,
        component_id: str,
        parameter_name: str,
        data: AbstractDataStructure,
        scenario_group: Optional[str] = None,
        series_name: Optional[str] = None,
    ) -> None:
        """*series_name* is the data-series file the data was read from, named
        in error messages."""
        idx = ComponentParameterIndex(component_id, parameter_name)
        self._data[idx] = data
        self._scenario_groups[idx] = scenario_group
        self._series_names[idx] = series_name

    def get_values(
        self,
        component_id: str,
        parameter_name: str,
        timesteps: Optional[List[int]],
        mc_scenarios: Optional[List[int]],
    ) -> Union[float, np.ndarray]:
        """Return parameter data for all requested timesteps and MC scenarios.

        MC scenario → col_idx resolution happens here (use-time, vectorized):
        a single numpy array index, no Python loop over S.

        Returns shape ``(T, S)``, ``(T,)``, ``(S,)``, or scalar depending on
        the underlying data type. Raises ``ValueError`` if a scenario needs a
        time series that the data does not have.
        """
        idx = ComponentParameterIndex(component_id, parameter_name)
        raw_data = self._data[idx]
        group = self._scenario_groups.get(idx)

        cols: Optional[np.ndarray] = None
        if mc_scenarios is not None:
            mc_arr = np.asarray(mc_scenarios, dtype=int)
            cols = (
                self._scenario_builder.resolve_vectorized(group, mc_arr)
                if self._scenario_builder
                else mc_arr
            )

        try:
            return raw_data.get_value(timesteps, cols)
        except IndexError:
            # Explained only on failure, so that valid reads cost nothing more.
            # Negative columns do not raise; validate_optim_config() reports them.
            message = (
                None if cols is None else self._missing_time_series(idx, mc_arr, cols)
            )
            if message is None:
                raise
            raise ValueError(message) from None

    def check_time_series(
        self,
        component_id: str,
        parameter_name: str,
        mc_scenarios: Union[List[int], np.ndarray],
    ) -> Optional[str]:
        """Return the error ``get_values()`` would raise for a time series that
        does not exist, None otherwise, so that it can be reported before the
        problem is built.

        Data that is missing, or whose scenarios the scenario builder refuses,
        is left to the other checks."""
        idx = ComponentParameterIndex(component_id, parameter_name)
        if idx not in self._data:
            return None
        mc_arr = np.asarray(mc_scenarios, dtype=int)
        cols = mc_arr
        if self._scenario_builder:
            try:
                cols = self._scenario_builder.resolve_vectorized(
                    self._scenario_groups.get(idx), mc_arr
                )
            except ValueError:
                return None
        return self._missing_time_series(idx, mc_arr, cols)

    def _missing_time_series(
        self, idx: ComponentParameterIndex, mc_scenarios: np.ndarray, cols: np.ndarray
    ) -> Optional[str]:
        """The error for the columns of *cols* that are not in the data of
        *idx*, None if they all are: numpy would raise an ``IndexError``
        without context, or read a negative column from the end."""
        count = self._data[idx].column_count()
        if count is None:
            return None
        missing = (cols < 0) | (cols >= count)
        if not missing.any():
            return None
        scenarios = _sorted_distinct(mc_scenarios[missing])
        # Distinct and sorted: several scenarios can need the same one.
        numbers = _sorted_distinct(cols[missing]) + 1  # 1-based, as in the files
        series = self._series_names.get(idx)
        group = self._scenario_groups.get(idx)
        # A scenario without a line in its group is refused by the scenario
        # builder, so with a group every column comes from the file.
        source = (
            f"set by modeler-scenariobuilder.dat, group '{group}'"
            if group is not None and self._scenario_builder is not None
            else "default: scenario n uses time series n+1"
        )
        what = (
            f"{'scenario' if len(scenarios) == 1 else 'scenarios'} "
            f"{_compact(scenarios, 'scenarios')}: time series "
            f"{_compact(numbers, 'time series')} "
            f"{'does' if len(numbers) == 1 else 'do'} not exist"
        )
        where = f"in '{series}'" if series is not None else "in its data series"
        return (
            f"Component '{idx.component_id}', parameter '{idx.parameter_name}', "
            f"{what} {where}, which has {count} time series ({source})."
        )

    def get_value(
        self, index: ComponentParameterIndex, timestep: int, scenario: int
    ) -> Union[float, np.ndarray]:
        """Scalar convenience wrapper used by tests."""
        result = self.get_values(
            index.component_id,
            index.parameter_name,
            [timestep],
            [scenario],
        )
        if isinstance(result, np.ndarray):
            return result.flat[0]
        return result


def _sorted_distinct(values: np.ndarray) -> np.ndarray:
    """*values* sorted and without duplicates; scenarios usually already are,
    and checking it is much cheaper than ``np.unique``."""
    if len(values) < 2 or bool(np.all(values[1:] > values[:-1])):
        return values
    return np.unique(values)


def _compact(values: np.ndarray, noun: str, limit: int = 10) -> str:
    """*values* (sorted, distinct) for a message: runs of 3 or more as ranges
    (``4–999``), at most *limit* items, then ``…`` and the count of *noun*.
    Loops over the runs, not the values."""
    values = np.asarray(values)
    ends = np.flatnonzero(np.diff(values) != 1)
    starts = np.concatenate(([0], ends + 1))
    stops = np.concatenate((ends, [len(values) - 1]))
    items: List[str] = []
    for start, stop in zip(starts, stops):
        if len(items) > limit:
            break
        if stop - start >= 2:
            items.append(f"{values[start]}–{values[stop]}")
        else:
            items.extend(str(v) for v in values[start : stop + 1])
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", … ({len(values)} {noun})"
