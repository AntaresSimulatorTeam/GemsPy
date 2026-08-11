# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from simulation_table_fakes import (
    FakeLinopyModel,
    FakeLinopyVar,
    FakeModel,
    FakeProblem,
    FakeStudy,
    to_object_dtype,
)

from gems_runner.simulation.simulation_table import (
    SimulationColumns,
    SimulationTableBuilder,
)


def test_simulation_table_builder_manual(tmp_path: Path) -> None:
    """Test SimulationTableBuilder with fake data."""
    sol_da = xr.DataArray(
        np.array([[[10.0], [20.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": ["compA"], "time": [0, 1], "scenario": [0]},
    )

    fake_var = FakeLinopyVar(
        name="test_model__p",
        coords={"component": xr.DataArray(["compA"])},
    )

    problem = FakeProblem(
        block_length=3,
        objective_value=42.0,
        linopy_model=FakeLinopyModel(solution={"test_model__p": sol_da}),
        _linopy_vars={(0, "p"): fake_var},
        models={0: FakeModel()},
        model_components={},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
    )

    builder = SimulationTableBuilder(simulation_id="test")
    df = builder.build(problem, table_id="test")  # type: ignore

    expected_rows = [
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 0,
            SimulationColumns.BLOCK_TIME_INDEX: 0,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 10.0,
            SimulationColumns.BASIS_STATUS: None,
        },
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 1,
            SimulationColumns.BLOCK_TIME_INDEX: 1,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 20.0,
            SimulationColumns.BASIS_STATUS: None,
        },
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: None,
            SimulationColumns.OUTPUT: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX: None,
            SimulationColumns.BLOCK_TIME_INDEX: None,
            SimulationColumns.SCENARIO_INDEX: None,
            SimulationColumns.VALUE: 42.0,
            SimulationColumns.BASIS_STATUS: None,
        },
    ]
    expected_df = pd.DataFrame(expected_rows)

    pd.testing.assert_frame_equal(
        to_object_dtype(df.data.reset_index(drop=True)),
        to_object_dtype(expected_df),
        check_dtype=False,
    )

    csv_path = df.to_csv(tmp_path)

    assert csv_path.exists(), "CSV file was not created"

    with csv_path.open("r") as f:
        first_line = f.readline().strip()

    expected_header = ",".join(col.value for col in SimulationColumns)
    assert first_line == expected_header, "CSV header does not match expected columns"

    csv_path.unlink()

    pytest.importorskip("pyarrow")
    parquet_path = df.to_parquet(tmp_path)
    assert parquet_path.exists(), "Parquet file was not created"
    loaded = pd.read_parquet(parquet_path)
    assert list(loaded.columns) == [col.value for col in SimulationColumns]
    parquet_path.unlink()


def _make_problem_with_da(da: xr.DataArray, var_name: str = "p") -> "FakeProblem":
    """Build a FakeProblem whose only variable has the given DataArray as solution."""
    fake_var = FakeLinopyVar(
        name=f"mod__{var_name}",
        coords={"component": xr.DataArray(["compA"])},
    )
    return FakeProblem(
        block_length=3,
        linopy_model=FakeLinopyModel(solution={f"mod__{var_name}": da}),
        _linopy_vars={(0, var_name): fake_var},
        models={0: FakeModel()},
        model_components={},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
    )


def test_time_independent_output_has_none_time_indices() -> None:
    """A var with no time dim produces None for both time index columns."""
    da = xr.DataArray(
        np.array([[5.0, 6.0]]),  # shape [component=1, scenario=2]
        dims=["component", "scenario"],
        coords={"component": ["compA"], "scenario": [0, 1]},
    )
    problem = _make_problem_with_da(da)
    st = SimulationTableBuilder().build(problem)  # type: ignore[arg-type]
    rows = st.data[st.data[SimulationColumns.OUTPUT.value] == "p"]

    assert rows[SimulationColumns.ABSOLUTE_TIME_INDEX.value].isna().all()
    assert rows[SimulationColumns.BLOCK_TIME_INDEX.value].isna().all()
    assert list(rows[SimulationColumns.SCENARIO_INDEX.value]) == [0, 1]
    assert list(rows[SimulationColumns.VALUE.value]) == [5.0, 6.0]


def test_scenario_independent_output_has_none_scenario_index() -> None:
    """A var with no scenario dim produces None for the scenario index column."""
    da = xr.DataArray(
        np.array([[10.0, 20.0, 30.0]]),  # shape [component=1, time=3]
        dims=["component", "time"],
        coords={"component": ["compA"], "time": [0, 1, 2]},
    )
    problem = _make_problem_with_da(da)
    st = SimulationTableBuilder().build(problem)  # type: ignore[arg-type]
    rows = st.data[st.data[SimulationColumns.OUTPUT.value] == "p"]

    assert rows[SimulationColumns.SCENARIO_INDEX.value].isna().all()
    assert list(rows[SimulationColumns.ABSOLUTE_TIME_INDEX.value]) == [0, 1, 2]
    assert list(rows[SimulationColumns.VALUE.value]) == [10.0, 20.0, 30.0]


def test_scalar_output_has_none_time_and_scenario_indices() -> None:
    """A var with no time and no scenario dim produces None for all index columns."""
    da = xr.DataArray(
        np.array([99.0]),  # shape [component=1]
        dims=["component"],
        coords={"component": ["compA"]},
    )
    problem = _make_problem_with_da(da)
    st = SimulationTableBuilder().build(problem)  # type: ignore[arg-type]
    rows = st.data[st.data[SimulationColumns.OUTPUT.value] == "p"]

    assert len(rows) == 1
    assert pd.isna(rows.iloc[0][SimulationColumns.ABSOLUTE_TIME_INDEX.value])
    assert pd.isna(rows.iloc[0][SimulationColumns.BLOCK_TIME_INDEX.value])
    assert pd.isna(rows.iloc[0][SimulationColumns.SCENARIO_INDEX.value])
    assert rows.iloc[0][SimulationColumns.VALUE.value] == 99.0
