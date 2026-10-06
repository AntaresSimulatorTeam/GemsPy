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
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Tuple, Union

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
    @property
    def set_dims(self) -> Tuple[str, ...]:
        """Ids of the custom sets the values vary over (none for legacy structures)."""
        return ()

    @abstractmethod
    def get_value(
        self,
        timestep: Optional[List[int]],
        scenario: Optional[np.ndarray],
    ) -> Union[float, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def check_requirement(
        self, time: bool, scenario: bool, sets: FrozenSet[str] = frozenset()
    ) -> bool:
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

    def check_requirement(
        self, time: bool, scenario: bool, sets: FrozenSet[str] = frozenset()
    ) -> bool:
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

    def check_requirement(
        self, time: bool, scenario: bool, sets: FrozenSet[str] = frozenset()
    ) -> bool:
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

    def check_requirement(
        self, time: bool, scenario: bool, sets: FrozenSet[str] = frozenset()
    ) -> bool:
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

    def check_requirement(
        self, time: bool, scenario: bool, sets: FrozenSet[str] = frozenset()
    ) -> bool:
        if not isinstance(self, TimeScenarioSeriesData):
            raise ValueError("Invalid data type for TimeScenarioSeriesData")
        return time and scenario


@dataclass(frozen=True)
class SetIndexedSeriesData(AbstractDataStructure):
    """Series indexed by custom sets, optionally also by time and/or scenario.

    ``values`` axes follow ``dims``: ``time`` (if present), ``scenario`` (if
    present), then one axis per custom set, with labels given by ``coords``.
    """

    values: np.ndarray
    dims: Tuple[str, ...]
    coords: Dict[str, Tuple[str, ...]]

    @property
    def set_dims(self) -> Tuple[str, ...]:
        return tuple(d for d in self.dims if d not in ("time", "scenario"))

    def get_value(
        self,
        timestep: Optional[List[int]],
        scenario: Optional[np.ndarray],
    ) -> np.ndarray:
        arr = self.values
        if "time" in self.dims:
            if timestep is None:
                raise KeyError("Set-indexed data requires a time index.")
            arr = np.take(arr, np.asarray(timestep), axis=0)
        if "scenario" in self.dims:
            if scenario is None:
                raise KeyError("Set-indexed data requires a scenario index.")
            arr = np.take(arr, np.asarray(scenario), axis=self.dims.index("scenario"))
        if "time" not in self.dims and timestep is not None:
            arr = np.broadcast_to(arr[np.newaxis], (len(timestep),) + arr.shape)
        if "scenario" not in self.dims and scenario is not None:
            pos = 1 if timestep is not None else 0
            arr = np.broadcast_to(
                np.expand_dims(arr, pos),
                arr.shape[:pos] + (len(scenario),) + arr.shape[pos:],
            )
        return arr

    def check_requirement(
        self, time: bool, scenario: bool, sets: FrozenSet[str] = frozenset()
    ) -> bool:
        if "time" in self.dims and not time:
            return False
        if "scenario" in self.dims and not scenario:
            return False
        return set(self.set_dims) <= set(sets)


_SERIES_SEPARATORS = {".txt": r"\s+", ".tsv": "\t", ".csv": ","}


def _read_series_file(
    name: Optional[str], directory: Optional[Path], **read_kwargs: Any
) -> pd.DataFrame:
    """Read ``<name>.txt``, ``.tsv`` or ``.csv`` (first one found) from ``directory``."""
    if directory is None or name is None:
        raise FileNotFoundError(f"File '{name}' does not exist")
    for suffix, sep in _SERIES_SEPARATORS.items():
        candidate = (directory / name).with_suffix(suffix)
        if not candidate.exists():
            continue
        try:
            return pd.read_csv(candidate, sep=sep, **read_kwargs)
        except Exception as e:
            raise Exception(f"An error has arrived when processing '{candidate}': {e}")
    raise FileNotFoundError(
        f"File '{name}' ({', '.join(_SERIES_SEPARATORS)}) does not exist"
    )


def load_ts_from_file(
    timeseries_name: Optional[str], path_to_file: Optional[Path]
) -> pd.DataFrame:
    return _read_series_file(timeseries_name, path_to_file, header=None)


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


def load_tidy_series_from_file(
    series_name: Optional[str], path_to_dir: Optional[Path]
) -> pd.DataFrame:
    """Read a tidy series (header row) as strings."""
    return _read_series_file(series_name, path_to_dir, dtype=str, skipinitialspace=True)


def _positional_codes(column: pd.Series, name: str) -> np.ndarray:
    """Codes of a ``time``/``scenario`` column, which must be exactly 0..n-1."""
    try:
        codes = column.astype(int).to_numpy()
    except ValueError:
        raise ValueError(f"Column '{name}' must contain integers.")
    distinct = sorted(set(codes.tolist()))
    if distinct != list(range(len(distinct))):
        raise ValueError(f"Column '{name}' values must be exactly 0..n-1.")
    return codes


def dataframe_to_set_indexed_series(
    ts_dataframe: pd.DataFrame,
    time_dependent: bool,
    scenario_dependent: bool,
    set_elements: Dict[str, List[Union[str, int]]],
) -> SetIndexedSeriesData:
    """Pivot a tidy dataframe (``[time], [scenario], <set columns>, value``)
    into a ``SetIndexedSeriesData``; columns are matched by name.

    ``set_elements`` maps each indexing set id to its instantiated elements, in
    the order the set axes should take.
    """
    key_cols = (
        (["time"] if time_dependent else [])
        + (["scenario"] if scenario_dependent else [])
        + list(set_elements)
    )
    present = set(ts_dataframe.columns)
    required = set(key_cols) | {"value"}
    if present != required:
        raise ValueError(
            "Tidy series columns do not match the declared dependence: "
            f"missing {sorted(required - present)}, "
            f"unexpected {sorted(present - required)}."
        )

    codes: List[np.ndarray] = []
    shape: List[int] = []
    coords: Dict[str, Tuple[str, ...]] = {}
    for col in key_cols:
        if col in set_elements:
            labels = tuple(str(e) for e in set_elements[col])
            got = set(ts_dataframe[col])
            if got != set(labels):
                raise ValueError(
                    f"Column '{col}' values {sorted(got)} do not match the "
                    f"instantiated elements {list(labels)}."
                )
            coords[col] = labels
            index = {label: i for i, label in enumerate(labels)}
            codes.append(ts_dataframe[col].map(index).to_numpy())
            shape.append(len(labels))
        else:
            col_codes = _positional_codes(ts_dataframe[col], col)
            codes.append(col_codes)
            shape.append(int(col_codes.max()) + 1 if len(col_codes) else 0)

    n_rows = len(ts_dataframe)
    if n_rows != int(np.prod(shape)) or len(set(zip(*codes))) != n_rows:
        raise ValueError(
            "Tidy series must contain exactly one row per combination of "
            f"{key_cols} (found duplicate or missing rows)."
        )
    try:
        value = pd.to_numeric(ts_dataframe["value"]).to_numpy(dtype=float)
    except ValueError:
        raise ValueError("Column 'value' must contain numbers.")
    if np.isnan(value).any():
        raise ValueError("Column 'value' must not contain empty cells.")
    values = np.empty(tuple(shape))
    values[tuple(codes)] = value
    return SetIndexedSeriesData(values=values, dims=tuple(key_cols), coords=coords)


def _flatten_nested(node: Any, set_ids: List[str]) -> List[Dict[str, str]]:
    if not set_ids:
        return [{"value": str(node)}]
    if not isinstance(node, dict):
        raise ValueError(f"Inline values need a mapping over set '{set_ids[0]}'.")
    return [
        {set_ids[0]: str(k), **row}
        for k, sub in node.items()
        for row in _flatten_nested(sub, set_ids[1:])
    ]


def nested_dict_to_set_indexed_series(
    value: Dict[Union[str, int], Any],
    set_elements: Dict[str, List[Union[str, int]]],
) -> SetIndexedSeriesData:
    """Inline set-only values, nested per set in order, as a tidy series."""
    df = pd.DataFrame(
        _flatten_nested(value, list(set_elements)), columns=[*set_elements, "value"]
    )
    return dataframe_to_set_indexed_series(df, False, False, set_elements)


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
