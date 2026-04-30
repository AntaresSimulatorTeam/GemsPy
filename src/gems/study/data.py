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
    from gems.study.scenario_builder import ScenarioBuilder


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
        self._scenario_builder = scenario_builder

    def get_data(self, component_id: str, parameter_name: str) -> AbstractDataStructure:
        return self._data[ComponentParameterIndex(component_id, parameter_name)]

    def add_data(
        self,
        component_id: str,
        parameter_name: str,
        data: AbstractDataStructure,
        scenario_group: Optional[str] = None,
    ) -> None:
        idx = ComponentParameterIndex(component_id, parameter_name)
        self._data[idx] = data
        self._scenario_groups[idx] = scenario_group

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
        the underlying data type.
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

        return raw_data.get_value(timesteps, cols)

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
