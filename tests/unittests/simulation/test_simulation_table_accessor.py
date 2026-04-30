# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Tests for SimulationTable fluent accessor API (component / output / value)."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gems.simulation.simulation_table import (
    ComponentView,
    OutputView,
    SimulationTable,
    SimulationTableBuilder,
)

# ---------------------------------------------------------------------------
# Minimal stubs (reused from test_simulation_table_mock.py)
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
    block_length: int = 3
    objective_value: float = 0.0
    linopy_model: Optional[FakeLinopyModel] = None
    _linopy_vars: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    model_components: dict = field(default_factory=dict)
    study: FakeStudy = field(default_factory=FakeStudy)
    scenarios: int = 1


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
