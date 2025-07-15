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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd

from gems.study.network import Network


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
class Scenarization:
    _scenarization: Dict[int, int]

    def get_scenario_for_year(self, year: int) -> int:
        return self._scenarization[year]

    def add_year(self, year: int, scenario: int) -> None:
        if year in self._scenarization:
            raise ValueError(f"the year {year} is already defined")
        self._scenarization[year] = scenario


@dataclass(frozen=True)
class AbstractDataStructure(ABC):
    @abstractmethod
    def get_value(
        self, timestep: Optional[int], scenario: Optional[int], node_id: str = ""
    ) -> float:
        raise NotImplementedError()

    @abstractmethod
    def check_requirement(self, time: bool, scenario: bool) -> bool:
        """
        Check if the data structure meets certain requirements.
        Implement this method in subclasses as needed.
        """
        pass


@dataclass(frozen=True)
class ConstantData(AbstractDataStructure):
    value: float

    def get_value(
        self, timestep: Optional[int], scenario: Optional[int], node_id: str = ""
    ) -> float:
        return self.value

    # ConstantData can be used for time varying or constant models
    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, ConstantData):
            raise ValueError("Invalid data type for ConstantData")
        return True


@dataclass(frozen=True)
class TimeSeriesData(AbstractDataStructure):
    """
    Container for identifiable timeseries data.
    When a model is instantiated as a component, property values
    can be defined by referencing one of those timeseries by its ID.
    """

    time_series: Mapping[TimeIndex, float]

    def get_value(
        self, timestep: Optional[int], scenario: Optional[int], node_id: str = ""
    ) -> float:
        if timestep is None:
            raise KeyError("Time series data requires a time index.")
        return self.time_series[TimeIndex(timestep)]

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, TimeSeriesData):
            raise ValueError("Invalid data type for TimeSeriesData")

        return time


@dataclass(frozen=True)
class ScenarioSeriesData(AbstractDataStructure):
    """
    Container for identifiable timeseries data.
    When a model is instantiated as a component, property values
    can be defined by referencing one of those timeseries by its ID.
    """

    scenario_series: Mapping[ScenarioIndex, float]
    scenarization: Optional[Scenarization] = None

    def get_value(
        self, timestep: Optional[int], scenario: Optional[int], node_id: str = ""
    ) -> float:
        if scenario is None:
            raise KeyError("Scenario series data requires a scenario index.")
        if self.scenarization:
            scenario = self.scenarization.get_scenario_for_year(scenario)
        return self.scenario_series[ScenarioIndex(scenario)]

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, ScenarioSeriesData):
            raise ValueError("Invalid data type for TimeSeriesData")

        return scenario


def load_ts_from_txt(
    timeseries_name: Optional[str], path_to_file: Optional[Path]
) -> pd.DataFrame:
    if path_to_file is not None and timeseries_name is not None:
        timeseries_with_extension = timeseries_name + ".txt"
        ts_path = path_to_file / timeseries_with_extension
    try:
        return pd.read_csv(ts_path, header=None, sep=r"\s+")

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{timeseries_name}' does not exist")
    except Exception:
        raise Exception(f"An error has arrived when processing '{ts_path}'")


def dataframe_to_time_series(ts_dataframe: pd.DataFrame) -> Dict[TimeIndex, float]:
    if ts_dataframe.shape[1] != 1:
        raise ValueError(
            f"Could not convert input data to time series data. Expect data series with exactly one column, got shape {ts_dataframe.shape}"
        )
    df_index = ts_dataframe.index.astype(int)  # Only for mypy
    return {
        TimeIndex(index): float(value)
        for index, value in zip(df_index, ts_dataframe.iloc[:, 0].values)
    }


def dataframe_to_scenario_series(
    ts_dataframe: pd.DataFrame,
) -> Dict[ScenarioIndex, float]:
    if ts_dataframe.shape[0] != 1:
        raise ValueError(
            f"Could not convert input data to scenario series data. Expect data series with exactly one line, got shape {ts_dataframe.shape}"
        )

    return {
        ScenarioIndex(col_id): float(value)
        for col_id, value in zip(
            list(range(ts_dataframe.shape[1])), ts_dataframe.iloc[0, :].values
        )
    }


@dataclass(frozen=True)
class TimeScenarioSeriesData(AbstractDataStructure):
    """
    Container for identifiable timeseries data.
    When a model is instantiated as a component, property values
    can be defined by referencing one of those timeseries by its ID.
    """

    time_scenario_series: pd.DataFrame
    scenarization: Optional[Scenarization] = None

    def get_value(
        self, timestep: Optional[int], scenario: Optional[int], node_id: str = ""
    ) -> float:
        if timestep is None:
            raise KeyError("Time scenario data requires a time index.")
        if scenario is None:
            raise KeyError("Time scenario data requires a scenario index.")
        if self.scenarization:
            scenario = self.scenarization.get_scenario_for_year(scenario)
        value = str(self.time_scenario_series.iloc[timestep, scenario])
        return float(value)

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        if not isinstance(self, TimeScenarioSeriesData):
            raise ValueError("Invalid data type for TimeScenarioSeriesData")

        return time and scenario


@dataclass(frozen=True)
class TreeData(AbstractDataStructure):
    data: Mapping[str, AbstractDataStructure]

    def get_value(
        self, timestep: Optional[int], scenario: Optional[int], node_id: str = ""
    ) -> float:
        return self.data[node_id].get_value(timestep, scenario)

    def check_requirement(self, time: bool, scenario: bool) -> bool:
        return all(
            node_data.check_requirement(time, scenario)
            for node_data in self.data.values()
        )


@dataclass(frozen=True)
class ComponentParameterIndex:
    component_id: str
    parameter_name: str


class DataBase:
    """
    Container for identifiable data.
    When a model is instantiated as a component, property values
    can be defined by referencing one of those data by its ID.
    Data can have different structure : constant, varying in time or scenarios.
    """

    _data: Dict[ComponentParameterIndex, AbstractDataStructure]

    def __init__(self) -> None:
        self._data: Dict[ComponentParameterIndex, AbstractDataStructure] = {}

    def get_data(self, component_id: str, parameter_name: str) -> AbstractDataStructure:
        return self._data[ComponentParameterIndex(component_id, parameter_name)]

    def add_data(
        self, component_id: str, parameter_name: str, data: AbstractDataStructure
    ) -> None:
        self._data[ComponentParameterIndex(component_id, parameter_name)] = data

    def get_value(
        self, index: ComponentParameterIndex, timestep: int, scenario: int
    ) -> float:
        if index in self._data:
            return self._data[index].get_value(timestep, scenario)
        else:
            raise KeyError(f"Index {index} not found.")

    def requirements_consistency(self, network: Network) -> None:
        for component in network.components:
            for param in component.model.parameters.values():
                data_structure = self.get_data(component.id, param.name)

                if not data_structure.check_requirement(
                    component.model.parameters[param.name].structure.time,
                    component.model.parameters[param.name].structure.scenario,
                ):
                    raise ValueError(
                        f"Data inconsistency for component: {component.id}, parameter: {param.name}. Requirement not met."
                    )
