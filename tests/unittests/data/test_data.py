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
"""Unit tests for gems.study.data — data structures, converters, and DataBase."""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import pytest

from gems.study.data import (
    ConstantData,
    DataBase,
    LazyScenarioSeriesData,
    LazyTimeScenarioSeriesData,
    LazyTimeSeriesData,
    ScenarioSeriesData,
    TimeScenarioSeriesData,
    TimeSeriesData,
    dataframe_to_scenario_series,
    dataframe_to_time_series,
    load_ts_from_file,
)
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import _build_data, build_data_base
from gems.study.scenario_builder import ScenarioBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_time_scenario_parquet(path: Path, data: list[list[float]]) -> None:
    """Write a T×S parquet with columns named "0", "1", …"""
    n_cols = len(data[0])
    pl.DataFrame({str(c): [row[c] for row in data] for c in range(n_cols)}).write_parquet(path)


def _write_time_series_parquet(path: Path, values: list[float]) -> None:
    """Write a T×1 parquet with a single column named "0"."""
    pl.DataFrame({"0": values}).write_parquet(path)


def _write_scenario_series_parquet(path: Path, values: list[float], n_rows: int = 3) -> None:
    """Write an S-column parquet where every row holds the same scenario values."""
    pl.DataFrame({str(c): [v] * n_rows for c, v in enumerate(values)}).write_parquet(path)

# ---------------------------------------------------------------------------
# load_ts_from_file
# ---------------------------------------------------------------------------


def test_load_ts_from_file_txt(tmp_path: Path) -> None:
    """Reads a whitespace-separated .txt file into a DataFrame."""
    (tmp_path / "series.txt").write_text("1 2 3\n4 5 6\n")
    df = load_ts_from_file("series", tmp_path)
    assert df.shape == (2, 3)
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 2] == 6


def test_load_ts_from_file_tsv(tmp_path: Path) -> None:
    """Reads a tab-separated .tsv file when no .txt exists."""
    (tmp_path / "series.tsv").write_text("10\t20\t30\n40\t50\t60\n")
    df = load_ts_from_file("series", tmp_path)
    assert df.shape == (2, 3)
    assert df.iloc[0, 1] == 20
    assert df.iloc[1, 2] == 60


def test_load_ts_from_file_not_found_raises(tmp_path: Path) -> None:
    """Raises FileNotFoundError when neither .txt nor .tsv exists."""
    with pytest.raises(FileNotFoundError):
        load_ts_from_file("missing", tmp_path)


def test_load_ts_from_file_none_inputs_raises() -> None:
    """Raises FileNotFoundError when inputs are None."""
    with pytest.raises(FileNotFoundError):
        load_ts_from_file(None, None)


# ---------------------------------------------------------------------------
# dataframe_to_time_series
# ---------------------------------------------------------------------------


def test_dataframe_to_time_series_single_column() -> None:
    df = pd.DataFrame([1.0, 2.0, 3.0])
    series = dataframe_to_time_series(df)
    assert list(series) == [1.0, 2.0, 3.0]


def test_dataframe_to_time_series_multi_column_raises() -> None:
    """Raises ValueError when the DataFrame has more than one column."""
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="exactly one column"):
        dataframe_to_time_series(df)


# ---------------------------------------------------------------------------
# dataframe_to_scenario_series
# ---------------------------------------------------------------------------


def test_dataframe_to_scenario_series_single_row() -> None:
    df = pd.DataFrame([[10.0, 20.0, 30.0]])
    arr = dataframe_to_scenario_series(df)
    assert list(arr) == [10.0, 20.0, 30.0]


def test_dataframe_to_scenario_series_multi_row_raises() -> None:
    """Raises ValueError when the DataFrame has more than one row."""
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="exactly one line"):
        dataframe_to_scenario_series(df)


# ---------------------------------------------------------------------------
# Data structure get_value error paths
# ---------------------------------------------------------------------------


def test_time_series_data_requires_timestep() -> None:
    """TimeSeriesData.get_value raises KeyError when timestep is None."""
    ts = TimeSeriesData(pd.Series([1.0, 2.0, 3.0]))
    with pytest.raises(KeyError):
        ts.get_value(None, None)


def test_scenario_series_data_requires_scenario() -> None:
    """ScenarioSeriesData.get_value raises KeyError when scenario is None."""
    ss = ScenarioSeriesData(np.array([10.0, 20.0]))
    with pytest.raises(KeyError):
        ss.get_value([0], None)


def test_time_scenario_series_data_requires_timestep() -> None:
    """TimeScenarioSeriesData.get_value raises KeyError when timestep is None."""
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    tss = TimeScenarioSeriesData(df)
    with pytest.raises(KeyError):
        tss.get_value(None, np.array([0]))


def test_time_scenario_series_data_requires_scenario() -> None:
    """TimeScenarioSeriesData.get_value raises KeyError when scenario is None."""
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    tss = TimeScenarioSeriesData(df)
    with pytest.raises(KeyError):
        tss.get_value([0], None)


# ---------------------------------------------------------------------------
# _build_data error paths
# ---------------------------------------------------------------------------


def test_build_data_str_not_time_not_scenario_raises(tmp_path: Path) -> None:
    """String param value with neither flag set raises ValueError."""
    (tmp_path / "series.txt").write_text("1 2\n")
    with pytest.raises(ValueError, match="float value is expected"):
        _build_data(
            time_dependent=False,
            scenario_dependent=False,
            param_value="series",
            timeseries_dir=tmp_path,
        )


def test_build_data_float_time_dependent_raises() -> None:
    """Float param value with time_dependent=True raises ValueError."""
    with pytest.raises(ValueError, match="timeseries name is expected"):
        _build_data(
            time_dependent=True,
            scenario_dependent=False,
            param_value=5.0,
            timeseries_dir=None,
        )


def test_build_data_float_scenario_dependent_raises() -> None:
    """Float param value with scenario_dependent=True raises ValueError."""
    with pytest.raises(ValueError, match="timeseries name is expected"):
        _build_data(
            time_dependent=False,
            scenario_dependent=True,
            param_value=5.0,
            timeseries_dir=None,
        )


# ---------------------------------------------------------------------------
# build_data_base — scenario_group priority
# ---------------------------------------------------------------------------

_SYSTEM_WITH_PARAM_GROUP = """\
system:
  model-libraries: basic
  components:
    - id: G
      model: basic.generator
      scenario-group: component-group
      parameters:
        - id: cost
          value: 0
        - id: p_max
          scenario-dependent: true
          scenario-group: param-group
          value: series
"""


def test_build_data_base_param_group_overrides_component_group(
    tmp_path: Path,
) -> None:
    """Parameter-level scenario_group takes precedence over component-level group."""
    (tmp_path / "series.txt").write_text("100 200\n")
    sb = ScenarioBuilder()
    db = build_data_base(
        parse_yaml_components(io.StringIO(_SYSTEM_WITH_PARAM_GROUP)),
        tmp_path,
        scenario_builder=sb,
    )
    from gems.study.data import ComponentParameterIndex

    # p_max has param-group; cost has component-group (constant so group is irrelevant)
    idx_pmax = ComponentParameterIndex("G", "p_max")
    assert db._scenario_groups[idx_pmax] == "param-group"

    idx_cost = ComponentParameterIndex("G", "cost")
    assert db._scenario_groups[idx_cost] == "component-group"


# ---------------------------------------------------------------------------
# DataBase.get_values — no ScenarioBuilder path
# ---------------------------------------------------------------------------


def test_database_get_values_no_builder_passes_mc_through() -> None:
    """Without a ScenarioBuilder mc_scenarios are used as col indices directly."""
    db = DataBase()  # no scenario_builder
    db.add_data("C", "param", ScenarioSeriesData(np.array([10.0, 20.0, 30.0])))
    result = db.get_values("C", "param", timesteps=None, mc_scenarios=[2, 0, 1])
    assert list(result) == [30.0, 10.0, 20.0]


def test_database_get_values_mc_scenarios_none() -> None:
    """mc_scenarios=None keeps cols=None (used for time-only or constant data)."""
    db = DataBase()
    db.add_data("C", "param", ConstantData(42.0))
    result = db.get_values("C", "param", timesteps=None, mc_scenarios=None)
    assert result == 42.0


# ---------------------------------------------------------------------------
# DataBase.get_value — scalar path
# ---------------------------------------------------------------------------


def test_database_get_value_constant_returns_scalar() -> None:
    """get_value on ConstantData returns a plain float, not a numpy array."""
    db = DataBase()
    db.add_data("C", "val", ConstantData(7.5))
    from gems.study.data import ComponentParameterIndex

    result = db.get_value(ComponentParameterIndex("C", "val"), 0, 0)
    assert result == 7.5
    assert not isinstance(result, np.ndarray)


def test_database_get_data_missing_key_raises() -> None:
    """get_data raises KeyError for an unknown (component, parameter) pair."""
    db = DataBase()
    with pytest.raises(KeyError):
        db.get_data("unknown", "param")


# ---------------------------------------------------------------------------
# LazyTimeSeriesData
# ---------------------------------------------------------------------------


def test_lazy_time_series_data_basic(tmp_path: Path) -> None:
    """Reads the requested timesteps from a parquet file."""
    p = tmp_path / "ts.parquet"
    _write_time_series_parquet(p, [10.0, 20.0, 30.0, 40.0])
    lazy = LazyTimeSeriesData(str(p))
    result = lazy.get_value([0, 2], None)
    np.testing.assert_array_equal(result, [10.0, 30.0])


def test_lazy_time_series_data_broadcast_over_scenarios(tmp_path: Path) -> None:
    """Broadcasts time values across scenarios when scenario array is provided."""
    p = tmp_path / "ts.parquet"
    _write_time_series_parquet(p, [1.0, 2.0, 3.0])
    lazy = LazyTimeSeriesData(str(p))
    result = lazy.get_value([0, 1], np.array([0, 1, 2]))
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(result[:, 2], [1.0, 2.0])


def test_lazy_time_series_data_requires_timestep(tmp_path: Path) -> None:
    p = tmp_path / "ts.parquet"
    _write_time_series_parquet(p, [1.0, 2.0])
    with pytest.raises(KeyError):
        LazyTimeSeriesData(str(p)).get_value(None, None)


def test_lazy_time_series_data_check_requirement() -> None:
    lazy = LazyTimeSeriesData("irrelevant.parquet")
    assert lazy.check_requirement(time=True, scenario=False) is True
    assert lazy.check_requirement(time=False, scenario=True) is False


# ---------------------------------------------------------------------------
# LazyScenarioSeriesData
# ---------------------------------------------------------------------------


def test_lazy_scenario_series_data_column_pruning(tmp_path: Path) -> None:
    """Reads only the requested scenario columns."""
    p = tmp_path / "sc.parquet"
    _write_scenario_series_parquet(p, [100.0, 200.0, 300.0])
    lazy = LazyScenarioSeriesData(str(p))
    result = lazy.get_value(None, np.array([2, 0]))
    np.testing.assert_array_equal(result, [300.0, 100.0])


def test_lazy_scenario_series_data_broadcast_over_time(tmp_path: Path) -> None:
    """Broadcasts scenario values across timesteps when timestep list is provided."""
    p = tmp_path / "sc.parquet"
    _write_scenario_series_parquet(p, [5.0, 10.0])
    lazy = LazyScenarioSeriesData(str(p))
    result = lazy.get_value([0, 1, 2], np.array([0, 1]))
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[0], [5.0, 10.0])
    np.testing.assert_array_equal(result[2], [5.0, 10.0])


def test_lazy_scenario_series_data_requires_scenario(tmp_path: Path) -> None:
    p = tmp_path / "sc.parquet"
    _write_scenario_series_parquet(p, [1.0])
    with pytest.raises(KeyError):
        LazyScenarioSeriesData(str(p)).get_value([0], None)


def test_lazy_scenario_series_data_check_requirement() -> None:
    lazy = LazyScenarioSeriesData("irrelevant.parquet")
    assert lazy.check_requirement(time=False, scenario=True) is True
    assert lazy.check_requirement(time=True, scenario=False) is False


# ---------------------------------------------------------------------------
# LazyTimeScenarioSeriesData
# ---------------------------------------------------------------------------


def test_lazy_time_scenario_series_data_column_pruning(tmp_path: Path) -> None:
    """Reads only the requested columns and rows."""
    p = tmp_path / "tss.parquet"
    # 3 timesteps × 4 scenarios
    _write_time_scenario_parquet(p, [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ])
    lazy = LazyTimeScenarioSeriesData(str(p))
    result = lazy.get_value([0, 2], np.array([1, 3]))
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[2.0, 4.0], [10.0, 12.0]])


def test_lazy_time_scenario_series_data_requires_timestep(tmp_path: Path) -> None:
    p = tmp_path / "tss.parquet"
    _write_time_scenario_parquet(p, [[1.0, 2.0]])
    with pytest.raises(KeyError):
        LazyTimeScenarioSeriesData(str(p)).get_value(None, np.array([0]))


def test_lazy_time_scenario_series_data_requires_scenario(tmp_path: Path) -> None:
    p = tmp_path / "tss.parquet"
    _write_time_scenario_parquet(p, [[1.0, 2.0]])
    with pytest.raises(KeyError):
        LazyTimeScenarioSeriesData(str(p)).get_value([0], None)


def test_lazy_time_scenario_series_data_check_requirement() -> None:
    lazy = LazyTimeScenarioSeriesData("irrelevant.parquet")
    assert lazy.check_requirement(time=True, scenario=True) is True
    assert lazy.check_requirement(time=True, scenario=False) is False
    assert lazy.check_requirement(time=False, scenario=True) is False


# ---------------------------------------------------------------------------
# _build_data — parquet format detection and backward compat
# ---------------------------------------------------------------------------


def test_build_data_detects_parquet_time_scenario(tmp_path: Path) -> None:
    """Returns LazyTimeScenarioSeriesData when a .parquet file is present."""
    p = tmp_path / "series.parquet"
    _write_time_scenario_parquet(p, [[1.0, 2.0], [3.0, 4.0]])
    result = _build_data(
        time_dependent=True,
        scenario_dependent=True,
        param_value="series",
        timeseries_dir=tmp_path,
    )
    assert isinstance(result, LazyTimeScenarioSeriesData)
    assert result.uri == str(p)


def test_build_data_detects_parquet_time_only(tmp_path: Path) -> None:
    """Returns LazyTimeSeriesData when a .parquet file is present."""
    p = tmp_path / "series.parquet"
    _write_time_series_parquet(p, [1.0, 2.0, 3.0])
    result = _build_data(
        time_dependent=True,
        scenario_dependent=False,
        param_value="series",
        timeseries_dir=tmp_path,
    )
    assert isinstance(result, LazyTimeSeriesData)
    assert result.uri == str(p)


def test_build_data_detects_parquet_scenario_only(tmp_path: Path) -> None:
    """Returns LazyScenarioSeriesData when a .parquet file is present."""
    p = tmp_path / "series.parquet"
    _write_scenario_series_parquet(p, [10.0, 20.0])
    result = _build_data(
        time_dependent=False,
        scenario_dependent=True,
        param_value="series",
        timeseries_dir=tmp_path,
    )
    assert isinstance(result, LazyScenarioSeriesData)
    assert result.uri == str(p)


def test_build_data_falls_back_to_txt_when_no_parquet(tmp_path: Path) -> None:
    """Falls back to eager TimeScenarioSeriesData when only a .txt file exists."""
    (tmp_path / "series.txt").write_text("1 2\n3 4\n")
    result = _build_data(
        time_dependent=True,
        scenario_dependent=True,
        param_value="series",
        timeseries_dir=tmp_path,
    )
    assert isinstance(result, TimeScenarioSeriesData)


def test_build_data_parquet_takes_precedence_over_txt(tmp_path: Path) -> None:
    """Parquet file wins when both .parquet and .txt exist."""
    (tmp_path / "series.txt").write_text("1 2\n3 4\n")
    p = tmp_path / "series.parquet"
    _write_time_scenario_parquet(p, [[1.0, 2.0], [3.0, 4.0]])
    result = _build_data(
        time_dependent=True,
        scenario_dependent=True,
        param_value="series",
        timeseries_dir=tmp_path,
    )
    assert isinstance(result, LazyTimeScenarioSeriesData)
