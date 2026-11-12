# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Optional

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
    SimulationTable,
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
            SimulationColumns.SET_ID: None,
            SimulationColumns.SET_INDEX: None,
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
            SimulationColumns.SET_ID: None,
            SimulationColumns.SET_INDEX: None,
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
            SimulationColumns.SET_ID: None,
            SimulationColumns.SET_INDEX: None,
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

    with pytest.warns(DeprecationWarning):
        csv_path = df.to_csv(tmp_path)

    assert csv_path.exists(), "CSV file was not created"

    with csv_path.open("r") as f:
        first_line = f.readline().strip()

    expected_header = ",".join(col.value for col in SimulationColumns)
    assert first_line == expected_header, "CSV header does not match expected columns"

    csv_path.unlink()

    pytest.importorskip("pyarrow")
    with pytest.warns(DeprecationWarning):
        parquet_path = df.to_parquet(tmp_path)
    assert parquet_path.exists(), "Parquet file was not created"
    loaded = pd.read_parquet(parquet_path)
    assert list(loaded.columns) == [col.value for col in SimulationColumns]
    parquet_path.unlink()


def _make_problem_with_da(
    da: xr.DataArray, var_name: str = "p", scenario_ids: Optional[list] = None
) -> "FakeProblem":
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
        scenario_ids=scenario_ids,
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


def _set_lookup(elements):  # type: ignore
    return lambda comp, set_id: elements[(comp, set_id)]


def test_da_to_df_two_sets_ragged_drops_padding() -> None:
    """Sets sorted by id, element names pipe-joined, padded positions dropped."""
    da = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]]],
        dims=["component", "seg", "fuel"],
        coords={"component": ["a", "b"], "seg": [0, 1, 2], "fuel": [0, 1]},
    )
    elements = {
        ("a", "seg"): (10, 20, 30),
        ("b", "seg"): (10, 20),
        ("a", "fuel"): ("coal", "gas"),
        ("b", "fuel"): ("coal", "gas"),
    }
    df = SimulationTableBuilder._da_to_df(
        da, "x", 1, 0, None, set_elements=_set_lookup(elements)
    )
    assert set(df[SimulationColumns.SET_ID.value]) == {"fuel|seg"}
    by_comp = {
        c: list(g[SimulationColumns.SET_INDEX.value])
        for c, g in df.groupby(SimulationColumns.COMPONENT.value)
    }
    assert by_comp["a"] == [f"{f}|{g}" for f in ("coal", "gas") for g in (10, 20, 30)]
    assert by_comp["b"] == [f"{f}|{g}" for f in ("coal", "gas") for g in (10, 20)]
    b_values = df[df[SimulationColumns.COMPONENT.value] == "b"][
        SimulationColumns.VALUE.value
    ]
    assert list(b_values) == [7.0, 9.0, 8.0, 10.0]


def test_da_to_df_without_set_has_none_set_columns() -> None:
    da = xr.DataArray([1.0], dims=["component"], coords={"component": ["a"]})
    df = SimulationTableBuilder._da_to_df(da, "x", 1, 0, None)
    assert df[SimulationColumns.SET_ID.value].isna().all()
    assert df[SimulationColumns.SET_INDEX.value].isna().all()


def test_to_dataset_multi_set_outputs_sharing_set_id() -> None:
    """Two outputs indexed by (fuel, seg) share the 'fuel|seg' dimension even
    when components label their local set differently, next to a plain output."""

    def da_of(comps: list, seg: int, offset: float) -> xr.DataArray:  # type: ignore
        return xr.DataArray(
            np.arange(len(comps) * 2 * seg, dtype=float).reshape(
                len(comps), 1, 1, 2, seg
            )
            + offset,
            dims=["component", "time", "scenario", "fuel", "seg"],
            coords={
                "component": comps,
                "time": [0],
                "scenario": [0],
                "fuel": [0, 1],
                "seg": range(seg),
            },
        )

    elements = {
        ("a", "fuel"): ("coal", "gas"),
        ("b", "fuel"): ("coal", "gas"),
        ("a", "seg"): ("s0", "s1"),
        ("b", "seg"): ("t0", "t1"),
    }
    lookup = _set_lookup(elements)
    frames = [
        SimulationTableBuilder._da_to_df(
            da_of([c], 2, off), out, 1, 0, None, set_elements=lookup
        )
        for out, off in (("g1", 0.0), ("g2", 100.0))
        for c in ("a", "b")
    ]
    plain = xr.DataArray(
        [[[5.0]]],
        dims=["component", "time", "scenario"],
        coords={"component": ["a"], "time": [0], "scenario": [0]},
    )
    frames.append(SimulationTableBuilder._da_to_df(plain, "p", 1, 0, None))
    ds = SimulationTable(pd.concat(frames, ignore_index=True)).to_dataset()

    assert ds["g1"].dims == ds["g2"].dims
    assert ds["g1"].dims == (
        "component",
        "absolute_time_index",
        "scenario_index",
        "fuel|seg",
    )
    assert ds["g1"].sel(component="a", **{"fuel|seg": "gas|s1"}).item() == 3.0
    assert ds["g2"].sel(component="b", **{"fuel|seg": "gas|t0"}).item() == 102.0
    assert "fuel|seg" not in ds["p"].dims
    assert ds["p"].sel(component="a").item() == 5.0


# ---------------------------------------------------------------------------
# Tests: scenario_index of single-scenario solves
# ---------------------------------------------------------------------------

SCENARIO_COL = SimulationColumns.SCENARIO_INDEX.value


def _scalar_da() -> xr.DataArray:
    """Solution of a variable without time and scenario dimensions."""
    return xr.DataArray(
        np.array([99.0]), dims=["component"], coords={"component": ["compA"]}
    )


def test_single_scenario_solve_tags_every_row() -> None:
    """With one scenario, scenario-independent outputs and the objective value
    get that scenario's id too."""
    problem = _make_problem_with_da(_scalar_da())
    data = SimulationTableBuilder().build(problem, scenario_ids_remap=[3]).data  # type: ignore[arg-type]

    assert list(data[SCENARIO_COL]) == [3, 3]  # p, objective-value
    assert data[SCENARIO_COL].dtype == "Int64"


def test_single_scenario_tag_matches_scenario_dependent_rows() -> None:
    """Rows with and without a scenario dimension get the same scenario id."""
    scenario_dependent = xr.DataArray(
        np.array([[5.0]]),
        dims=["component", "scenario"],
        coords={"component": ["compA"], "scenario": [0]},
    )
    problem = _make_problem_with_da(scenario_dependent)
    data = SimulationTableBuilder().build(problem, scenario_ids_remap=[7]).data  # type: ignore[arg-type]

    assert list(data[SCENARIO_COL]) == [7, 7]


def test_several_scenarios_keep_shared_rows_empty() -> None:
    """With several scenarios, scenario-independent outputs and the objective
    value are shared by all scenarios: their scenario index stays empty."""
    problem = _make_problem_with_da(_scalar_da())
    data = SimulationTableBuilder().build(problem, scenario_ids_remap=[0, 1]).data  # type: ignore[arg-type]

    assert data[SCENARIO_COL].isna().all()


def test_single_scenario_problem_is_tagged_without_remap() -> None:
    """build(problem) without scenario_ids_remap uses the problem's own scenario
    list; rows are labelled by scenario position, so the tag is 0."""
    problem = _make_problem_with_da(_scalar_da(), scenario_ids=[4])
    data = SimulationTableBuilder().build(problem).data  # type: ignore[arg-type]

    assert list(data[SCENARIO_COL]) == [0, 0]


def test_several_scenario_problem_keeps_shared_rows_empty_without_remap() -> None:
    problem = _make_problem_with_da(_scalar_da(), scenario_ids=[0, 1])
    data = SimulationTableBuilder().build(problem).data  # type: ignore[arg-type]

    assert data[SCENARIO_COL].isna().all()
