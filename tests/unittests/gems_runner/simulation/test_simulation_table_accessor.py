# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Tests for SimulationTable fluent accessor API (component / output / value)."""

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
)

from gems_runner.simulation.simulation_table import (
    SIMULATION_TABLE_DTYPES,
    SIMULATION_TABLE_SCHEMA,
    ComponentView,
    OutputView,
    SimulationTable,
    SimulationTableBuilder,
    _apply_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_scenario_problem() -> FakeProblem:
    """One component, two time steps, one scenario."""
    sol_da = xr.DataArray(
        np.array([[[10.0], [20.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": ["compA"], "time": [0, 1], "scenario": [0]},
    )
    fake_var = FakeLinopyVar(
        name="test_model__p",
        coords={"component": xr.DataArray(["compA"])},
    )
    return FakeProblem(
        block_length=2,
        linopy_model=FakeLinopyModel(solution={"test_model__p": sol_da}),
        _linopy_vars={(0, "p"): fake_var},
        models={0: FakeModel()},
        model_components={},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
        scenarios=1,
    )


def _make_multi_scenario_problem() -> FakeProblem:
    """One component, two time steps, two scenarios."""
    # values[comp, time, scenario]: compA, t0s0=10, t0s1=11, t1s0=20, t1s1=21
    sol_da = xr.DataArray(
        np.array([[[10.0, 11.0], [20.0, 21.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": ["compA"], "time": [0, 1], "scenario": [0, 1]},
    )
    fake_var = FakeLinopyVar(
        name="test_model__p",
        coords={"component": xr.DataArray(["compA"])},
    )
    return FakeProblem(
        block_length=2,
        linopy_model=FakeLinopyModel(solution={"test_model__p": sol_da}),
        _linopy_vars={(0, "p"): fake_var},
        models={0: FakeModel()},
        model_components={},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
        scenarios=2,
    )


# ---------------------------------------------------------------------------
# Tests: return types
# ---------------------------------------------------------------------------


def test_build_returns_simulation_table() -> None:
    st = SimulationTableBuilder().build(_make_single_scenario_problem())  # type: ignore[arg-type]
    assert isinstance(st, SimulationTable)


def test_data_property_returns_dataframe() -> None:
    st = SimulationTableBuilder().build(_make_single_scenario_problem())  # type: ignore[arg-type]
    assert isinstance(st.data, pd.DataFrame)


def test_component_returns_component_view() -> None:
    st = SimulationTableBuilder().build(_make_single_scenario_problem())  # type: ignore[arg-type]
    assert isinstance(st.component("compA"), ComponentView)


def test_output_returns_output_view() -> None:
    st = SimulationTableBuilder().build(_make_single_scenario_problem())  # type: ignore[arg-type]
    assert isinstance(st.component("compA").output("p"), OutputView)


# ---------------------------------------------------------------------------
# Tests: value() with no arguments → full Time × Scenario DataFrame
# ---------------------------------------------------------------------------


def test_value_no_args_returns_dataframe_single_scenario() -> None:
    st = SimulationTableBuilder().build(_make_single_scenario_problem())  # type: ignore[arg-type]
    result = st.component("compA").output("p").value()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)  # 2 time steps × 1 scenario
    assert list(result.index) == [0, 1]
    assert list(result.columns) == [0]


def test_value_no_args_returns_dataframe_multi_scenario() -> None:
    st = SimulationTableBuilder().build(_make_multi_scenario_problem())  # type: ignore[arg-type]
    result = st.component("compA").output("p").value()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)  # 2 time steps × 2 scenarios
    assert list(result.index) == [0, 1]
    assert list(result.columns) == [0, 1]


# ---------------------------------------------------------------------------
# Tests: value(scenario_index=s) → Series over time
# ---------------------------------------------------------------------------


def test_value_scenario_index_returns_series() -> None:
    st = SimulationTableBuilder().build(_make_single_scenario_problem())  # type: ignore[arg-type]
    result = st.component("compA").output("p").value(scenario_index=0)
    assert isinstance(result, pd.Series)
    assert list(result.index) == [0, 1]
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(20.0)


def test_value_scenario_index_multi_scenario() -> None:
    st = SimulationTableBuilder().build(_make_multi_scenario_problem())  # type: ignore[arg-type]
    s0 = st.component("compA").output("p").value(scenario_index=0)
    s1 = st.component("compA").output("p").value(scenario_index=1)
    assert s0.iloc[0] == pytest.approx(10.0)
    assert s0.iloc[1] == pytest.approx(20.0)
    assert s1.iloc[0] == pytest.approx(11.0)
    assert s1.iloc[1] == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# Tests: value(time_index=t) → Series over scenarios
# ---------------------------------------------------------------------------


def test_value_time_index_returns_series() -> None:
    st = SimulationTableBuilder().build(_make_multi_scenario_problem())  # type: ignore[arg-type]
    result = st.component("compA").output("p").value(time_index=0)
    assert isinstance(result, pd.Series)
    assert list(result.index) == [0, 1]
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# Tests: value(time_index=t, scenario_index=s) → scalar float
# ---------------------------------------------------------------------------


def test_value_both_indices_returns_float() -> None:
    st = SimulationTableBuilder().build(_make_multi_scenario_problem())  # type: ignore[arg-type]
    view = st.component("compA").output("p")
    assert view.value(time_index=0, scenario_index=0) == pytest.approx(10.0)
    assert view.value(time_index=0, scenario_index=1) == pytest.approx(11.0)
    assert view.value(time_index=1, scenario_index=0) == pytest.approx(20.0)
    assert view.value(time_index=1, scenario_index=1) == pytest.approx(21.0)


def test_value_both_indices_single_scenario() -> None:
    st = SimulationTableBuilder().build(_make_single_scenario_problem())  # type: ignore[arg-type]
    val = st.component("compA").output("p").value(time_index=0, scenario_index=0)
    assert isinstance(val, float)
    assert val == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: OutputView.data exposes the pivot DataFrame
# ---------------------------------------------------------------------------


def test_output_view_data_property() -> None:
    st = SimulationTableBuilder().build(_make_multi_scenario_problem())  # type: ignore[arg-type]
    view = st.component("compA").output("p")
    df = view.data
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, 0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: dimension-independent outputs are accessible via the fluent API
# ---------------------------------------------------------------------------


def _make_scenario_independent_problem() -> FakeProblem:
    """One component, two time steps, NO scenario dimension."""
    sol_da = xr.DataArray(
        np.array([[10.0, 20.0]]),
        dims=["component", "time"],
        coords={"component": ["compA"], "time": [0, 1]},
    )
    fake_var = FakeLinopyVar(
        name="test_model__p",
        coords={"component": xr.DataArray(["compA"])},
    )
    return FakeProblem(
        block_length=2,
        linopy_model=FakeLinopyModel(solution={"test_model__p": sol_da}),
        _linopy_vars={(0, "p"): fake_var},
        models={0: FakeModel()},
        model_components={},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
        scenarios=1,
    )


def _make_scalar_output_problem() -> FakeProblem:
    """One component, NO time dimension, NO scenario dimension."""
    sol_da = xr.DataArray(
        np.array([99.0]),
        dims=["component"],
        coords={"component": ["compA"]},
    )
    fake_var = FakeLinopyVar(
        name="test_model__p",
        coords={"component": xr.DataArray(["compA"])},
    )
    return FakeProblem(
        block_length=1,
        linopy_model=FakeLinopyModel(solution={"test_model__p": sol_da}),
        _linopy_vars={(0, "p"): fake_var},
        models={0: FakeModel()},
        model_components={},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
        scenarios=1,
    )


def test_scenario_independent_value_accessible_by_scenario_index() -> None:
    """value(time_index=t, scenario_index=0) works even without a scenario dim."""
    st = SimulationTableBuilder().build(_make_scenario_independent_problem())  # type: ignore[arg-type]
    assert st.component("compA").output("p").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(10.0)
    assert st.component("compA").output("p").value(
        time_index=1, scenario_index=0
    ) == pytest.approx(20.0)


def test_scalar_output_accessible_via_fluent_api() -> None:
    """value(time_index=0, scenario_index=0) works for a fully scalar output."""
    st = SimulationTableBuilder().build(_make_scalar_output_problem())  # type: ignore[arg-type]
    assert st.component("compA").output("p").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Tests: hand-built tables (missing dimensions, several blocks, objective)
# ---------------------------------------------------------------------------


def _table(rows: list) -> SimulationTable:
    """SimulationTable from (block, component, output, time, scenario, value)."""
    df = pd.DataFrame(
        [
            {
                "block": b,
                "component": c,
                "output": o,
                "absolute_time_index": t,
                "block_time_index": None if t is None else t,
                "scenario_index": s,
                "value": v,
                "basis_status": None,
            }
            for b, c, o, t, s, v in rows
        ]
    )
    return SimulationTable(_apply_schema(df))


def test_schema_dtypes_do_not_depend_on_content() -> None:
    st = SimulationTableBuilder().build(_make_scalar_output_problem())  # type: ignore[arg-type]
    assert st.data.dtypes.to_dict() == SIMULATION_TABLE_DTYPES
    assert [f.name for f in SIMULATION_TABLE_SCHEMA] == list(st.data.columns)


def test_scenario_independent_output_accepts_any_scenario_index() -> None:
    st = SimulationTableBuilder().build(_make_scenario_independent_problem())  # type: ignore[arg-type]
    view = st.component("compA").output("p")

    assert view.value(time_index=1, scenario_index=5) == pytest.approx(20.0)
    assert list(view.value(scenario_index=3)) == pytest.approx([10.0, 20.0])
    # One column, labelled <NA>: the value is not repeated per scenario.
    assert view.data.shape == (2, 1)
    assert pd.isna(view.data.columns[0])


def test_time_independent_output_accepts_any_time_index() -> None:
    st = _table([(0, "g", "p_max", None, 0, 7.0), (0, "g", "p_max", None, 1, 9.0)])
    view = st.component("g").output("p_max")

    assert view.value(time_index=42, scenario_index=1) == pytest.approx(9.0)
    assert list(view.value(time_index=3)) == pytest.approx([7.0, 9.0])
    assert view.data.shape == (1, 2)
    assert pd.isna(view.data.index[0])


def test_scalar_output_accepts_any_index() -> None:
    st = SimulationTableBuilder().build(_make_scalar_output_problem())  # type: ignore[arg-type]
    view = st.component("compA").output("p")

    assert view.value(time_index=8, scenario_index=3) == pytest.approx(99.0)
    assert view.data.shape == (1, 1)


def test_missing_value_still_raises_key_error() -> None:
    st = SimulationTableBuilder().build(_make_multi_scenario_problem())  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        st.component("compA").output("p").value(time_index=5, scenario_index=0)


# Overlapping blocks: time 1 is solved in block 0 and in block 1.
_OVERLAP = [
    (0, "g", "p", 0, 0, 1.0),
    (0, "g", "p", 1, 0, 2.0),
    (1, "g", "p", 1, 0, 3.0),
    (1, "g", "p", 2, 0, 4.0),
]


def test_several_blocks_raise_without_block() -> None:
    view = _table(_OVERLAP).component("g").output("p")

    with pytest.raises(ValueError, match=r"blocks \[0, 1\]"):
        view.value(time_index=1, scenario_index=0)
    with pytest.raises(ValueError, match="block="):
        view.data


def test_block_selects_among_several_blocks() -> None:
    view = _table(_OVERLAP).component("g").output("p")

    assert view.value(time_index=1, scenario_index=0, block=0) == pytest.approx(2.0)
    assert view.value(time_index=1, scenario_index=0, block=1) == pytest.approx(3.0)
    assert list(_table(_OVERLAP).component("g").output("p", block=1).data[0]) == [
        3.0,
        4.0,
    ]


def test_values_with_a_single_block_need_no_block() -> None:
    """The check is made on the requested values, not on the whole view."""
    view = _table(_OVERLAP).component("g").output("p")

    assert view.value(time_index=0, scenario_index=0) == pytest.approx(1.0)
    assert view.value(time_index=2, scenario_index=0) == pytest.approx(4.0)


def test_per_block_rows_of_a_time_independent_output_need_a_block() -> None:
    view = (
        _table([(0, "g", "p_max", None, 0, 100.0), (1, "g", "p_max", None, 0, 80.0)])
        .component("g")
        .output("p_max")
    )

    with pytest.raises(ValueError, match="block="):
        view.value(time_index=3, scenario_index=0)
    assert view.value(time_index=3, scenario_index=0, block=1) == pytest.approx(80.0)


def test_objective_values() -> None:
    st = _table(
        [
            (0, "g", "p", 0, 0, 1.0),
            (0, None, "objective-value", None, 0, 10.0),
            (1, None, "objective-value", None, 0, 11.0),
        ]
    )
    objective = st.objective_values()

    assert list(objective.columns) == ["block", "scenario_index", "value"]
    assert objective.values.tolist() == [[0, 0, 10.0], [1, 0, 11.0]]
