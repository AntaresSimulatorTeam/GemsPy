# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import numpy as np
import pandas as pd
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
from gems_runner.simulation.simulation_table_writer import (
    SIMULATION_TABLE_SCHEMA,
    SimulationTableWriter,
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

    expected_columns = [col.value for col in SimulationColumns]
    for output_format in ("csv", "parquet"):
        paths = SimulationTableWriter(output_format).write(df, tmp_path / output_format)  # type: ignore[arg-type]
        assert paths, f"No {output_format} file was written"
        for path in paths:
            loaded = (
                pd.read_csv(path) if output_format == "csv" else pd.read_parquet(path)
            )
            assert list(loaded.columns) == expected_columns


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


def _make_scenario_independent_problem() -> "FakeProblem":
    """A var with no scenario dim: [component=1, time=2]."""
    da = xr.DataArray(
        np.array([[10.0, 20.0]]),
        dims=["component", "time"],
        coords={"component": ["compA"], "time": [0, 1]},
    )
    return _make_problem_with_da(da)


def test_single_scenario_problem_tags_all_rows_with_its_scenario() -> None:
    """A problem solved for one MC scenario (sequential/parallel modes) owns all
    its rows: scenario-independent outputs and the objective value get its id."""
    st = SimulationTableBuilder().build(
        _make_scenario_independent_problem(), scenario_ids_remap=[3]  # type: ignore[arg-type]
    )
    scenario_col = st.data[SimulationColumns.SCENARIO_INDEX.value]
    outputs = st.data[SimulationColumns.OUTPUT.value]

    assert list(scenario_col[outputs == "p"]) == [3, 3]
    assert list(scenario_col[outputs == "objective-value"]) == [3]
    assert scenario_col.dtype == "int64"


def test_multi_scenario_problem_keeps_shared_rows_without_scenario() -> None:
    """In a problem covering several MC scenarios (frontal mode), scenario-
    independent outputs and the objective value are shared: no scenario index."""
    st = SimulationTableBuilder().build(
        _make_scenario_independent_problem(), scenario_ids_remap=[0, 1]  # type: ignore[arg-type]
    )
    shared_outputs = st.data[SimulationColumns.OUTPUT.value].isin(
        ["p", "objective-value"]
    )

    assert st.data[shared_outputs][SimulationColumns.SCENARIO_INDEX.value].isna().all()


def _make_scenario_dependent_problem() -> "FakeProblem":
    """A var with a scenario dim: [component=1, time=2, scenario=2]."""
    da = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": ["compA"], "time": [0, 1], "scenario": [0, 1]},
    )
    return _make_problem_with_da(da)


def _outputs_and_scenarios(table: SimulationTable) -> list:
    df = table.data
    return sorted(
        zip(
            df[SimulationColumns.OUTPUT.value],
            df[SimulationColumns.SCENARIO_INDEX.value].map(
                lambda s: None if pd.isna(s) else int(s)
            ),
        ),
        key=str,
    )


def test_scenario_tables_split_scenario_rows_from_shared_rows() -> None:
    """Several scenarios: the shared rows come first, then each scenario."""
    common, scenario_5, scenario_7 = SimulationTableBuilder().iter_scenario_tables(
        _make_scenario_dependent_problem(), scenario_ids_remap=[5, 7]  # type: ignore[arg-type]
    )

    assert _outputs_and_scenarios(common) == [("objective-value", None)]
    assert _outputs_and_scenarios(scenario_5) == [("p", 5), ("p", 5)]
    assert _outputs_and_scenarios(scenario_7) == [("p", 7), ("p", 7)]
    assert list(scenario_7.data[SimulationColumns.VALUE.value]) == [2.0, 4.0]


def test_scenario_tables_without_scenario_dependent_outputs() -> None:
    """All outputs are shared by the scenarios: only the shared rows are
    yielded, no scenario has rows of its own, so no scenario file is written."""
    tables = list(
        SimulationTableBuilder().iter_scenario_tables(
            _make_scenario_independent_problem(), scenario_ids_remap=[0, 1]  # type: ignore[arg-type]
        )
    )

    assert len(tables) == 1
    assert _outputs_and_scenarios(tables[0]) == [
        ("objective-value", None),
        ("p", None),
        ("p", None),
    ]


def test_scenario_tables_single_scenario_owns_all_rows() -> None:
    """A single scenario: one table, the same as build()."""
    problem = _make_scenario_independent_problem()
    tables = list(
        SimulationTableBuilder().iter_scenario_tables(problem, scenario_ids_remap=[3])  # type: ignore[arg-type]
    )

    assert len(tables) == 1
    assert _outputs_and_scenarios(tables[0]) == [
        ("objective-value", 3),
        ("p", 3),
        ("p", 3),
    ]
    full = SimulationTableBuilder().build(problem, scenario_ids_remap=[3])  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(tables[0].data, full.data)


def test_builder_columns_match_the_writer_schema() -> None:
    """pa.Table.from_pandas silently drops columns that the schema does not
    list: every column the builder produces must be in the writer schema."""
    problem = _make_scenario_dependent_problem()
    full = SimulationTableBuilder().build(problem, scenario_ids_remap=[0, 1])  # type: ignore[arg-type]
    tables = SimulationTableBuilder().iter_scenario_tables(problem, scenario_ids_remap=[0, 1])  # type: ignore[arg-type]

    for table in (full, *tables):
        assert list(table.data.columns) == SIMULATION_TABLE_SCHEMA.names
