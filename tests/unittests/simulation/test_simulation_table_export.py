# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Tests for SimulationTable.to_dataset(), write_parquet(), and write_netcdf()."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gems.simulation.simulation_table import (
    SimulationColumns,
    SimulationTable,
    SimulationTableBuilder,
)

# ---------------------------------------------------------------------------
# Minimal stubs (shared with other simulation table tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeBlock:
    id: int = 1
    timesteps: tuple = (0, 1)


@dataclass
class FakeLinopyVar:
    name: str
    coords: dict


@dataclass
class FakeModel:
    extra_outputs: dict = field(default_factory=dict)


@dataclass
class FakeStudy:
    model_components: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)


@dataclass
class FakeLinopyModel:
    solution: dict


@dataclass
class FakeProblem:
    block: FakeBlock = field(default_factory=FakeBlock)
    block_length: int = 2
    objective_value: float = 99.0
    linopy_model: Optional[FakeLinopyModel] = None
    _linopy_vars: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    model_components: dict = field(default_factory=dict)
    study: FakeStudy = field(default_factory=FakeStudy)
    scenarios: int = 1


def _make_problem(n_scenarios: int = 1) -> FakeProblem:
    """Two time steps, configurable number of scenarios, one component."""
    values = np.arange(1.0 * 2 * n_scenarios).reshape(1, 2, n_scenarios)
    sol_da = xr.DataArray(
        values,
        dims=["component", "time", "scenario"],
        coords={
            "component": ["comp1"],
            "time": [0, 1],
            "scenario": list(range(n_scenarios)),
        },
    )
    fake_var = FakeLinopyVar(
        name="mod__p",
        coords={"component": xr.DataArray(["comp1"])},
    )
    return FakeProblem(
        linopy_model=FakeLinopyModel(solution={"mod__p": sol_da}),
        _linopy_vars={(0, "p"): fake_var},
        models={0: FakeModel()},
        model_components={},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
        scenarios=n_scenarios,
    )


# ---------------------------------------------------------------------------
# Tests: to_dataset()
# ---------------------------------------------------------------------------


def test_to_dataset_returns_xr_dataset() -> None:
    st = SimulationTableBuilder().build(_make_problem())  # type: ignore[arg-type]
    ds = st.to_dataset()
    assert isinstance(ds, xr.Dataset)


def test_to_dataset_contains_expected_variable() -> None:
    st = SimulationTableBuilder().build(_make_problem())  # type: ignore[arg-type]
    ds = st.to_dataset()
    assert "p" in ds.data_vars


def test_to_dataset_values_match_data_single_scenario() -> None:
    st = SimulationTableBuilder().build(_make_problem(n_scenarios=1))  # type: ignore[arg-type]
    ds = st.to_dataset()

    # Check that values in the Dataset match those in the flat DataFrame
    for _, row in st.data.dropna(subset=[SimulationColumns.COMPONENT.value]).iterrows():
        output = row[SimulationColumns.OUTPUT.value]
        comp = row[SimulationColumns.COMPONENT.value]
        t = int(row[SimulationColumns.ABSOLUTE_TIME_INDEX.value])
        s = int(row[SimulationColumns.SCENARIO_INDEX.value])
        expected = float(row[SimulationColumns.VALUE.value])
        actual = float(
            ds[output].sel(
                component=comp, **{"absolute-time-index": t, "scenario-index": s}
            )
        )
        assert actual == pytest.approx(expected)


def test_to_dataset_values_match_data_multi_scenario() -> None:
    st = SimulationTableBuilder().build(_make_problem(n_scenarios=3))  # type: ignore[arg-type]
    ds = st.to_dataset()

    for _, row in st.data.dropna(subset=[SimulationColumns.COMPONENT.value]).iterrows():
        output = row[SimulationColumns.OUTPUT.value]
        comp = row[SimulationColumns.COMPONENT.value]
        t = int(row[SimulationColumns.ABSOLUTE_TIME_INDEX.value])
        s = int(row[SimulationColumns.SCENARIO_INDEX.value])
        expected = float(row[SimulationColumns.VALUE.value])
        actual = float(
            ds[output].sel(
                component=comp, **{"absolute-time-index": t, "scenario-index": s}
            )
        )
        assert actual == pytest.approx(expected)


def test_to_dataset_includes_objective_value_scalar() -> None:
    st = SimulationTableBuilder().build(_make_problem())  # type: ignore[arg-type]
    ds = st.to_dataset()
    assert "objective-value" in ds.data_vars
    assert ds["objective-value"].shape == ()  # scalar (no dims)
    assert float(ds["objective-value"]) == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Tests: write_parquet()
# ---------------------------------------------------------------------------


def test_write_parquet_creates_file(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    st = SimulationTableBuilder().build(_make_problem(), table_id="test")  # type: ignore[arg-type]
    path = st.to_parquet(tmp_path)
    assert path.exists()
    assert path.suffix == ".parquet"


def _to_object_dtype(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast every column to numpy object dtype, normalising all nulls to None."""
    return pd.DataFrame(
        {col: frame[col].to_numpy(dtype=object, na_value=None) for col in frame.columns}
    )


def test_write_parquet_content_matches_original(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    st = SimulationTableBuilder().build(_make_problem(), table_id="test")  # type: ignore[arg-type]
    path = st.to_parquet(tmp_path)

    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(
        _to_object_dtype(loaded.reset_index(drop=True)),
        _to_object_dtype(st.data.reset_index(drop=True)),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Tests: write_netcdf()
# ---------------------------------------------------------------------------


def test_write_netcdf_creates_file(tmp_path: Path) -> None:
    st = SimulationTableBuilder().build(_make_problem(), table_id="test")  # type: ignore[arg-type]
    path = st.to_netcdf(tmp_path)
    assert path.exists()
    assert path.suffix == ".nc"


def test_write_netcdf_readable_as_dataset(tmp_path: Path) -> None:
    st = SimulationTableBuilder().build(_make_problem(), table_id="test")  # type: ignore[arg-type]
    path = st.to_netcdf(tmp_path)

    ds = xr.open_dataset(path)
    assert isinstance(ds, xr.Dataset)
    assert "p" in ds.data_vars
    assert "objective-value" in ds.data_vars
    ds.close()
