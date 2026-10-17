# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Tests for the SimulationTable read API and its storage: to_pandas,
iter_scenarios, scan, scenarios, path, and the cache of the lazy views."""

from pathlib import Path
from typing import Any, Iterator, List, Optional, Set, Tuple

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
    SIMULATION_TABLE_SCHEMA,
    SimulationTable,
    SimulationTableBuilder,
    _apply_schema,
    _concat_frames,
)
from gems_runner.simulation.table_store import _MemoryStore, _RowFilter

Row = Tuple[Optional[int], Optional[str], str, Optional[int], Optional[int], float]

# A frontal run with two scenarios: gen.p and load.d depend on the scenario;
# gen.p_max and the objective value are shared by both (empty scenario_index).
_ROWS: List[Row] = [
    (0, "gen", "p", 0, 0, 10.0),
    (0, "gen", "p", 1, 0, 12.0),
    (0, "gen", "p", 0, 1, 11.0),
    (0, "gen", "p", 1, 1, 13.0),
    (0, "gen", "p_max", None, None, 200.0),
    (0, "load", "d", 0, 0, 5.0),
    (0, "load", "d", 0, 1, 6.0),
    (0, None, "objective-value", None, None, 900.0),
]


def _table(
    rows: List[Row] = _ROWS, scenarios: Optional[List[int]] = None
) -> SimulationTable:
    """SimulationTable from (block, component, output, time, scenario, value)."""
    df = pd.DataFrame(
        [
            {
                "block": b,
                "component": c,
                "output": o,
                "absolute_time_index": t,
                "block_time_index": t,
                "scenario_index": s,
                "value": v,
                "basis_status": None,
            }
            for b, c, o, t, s, v in rows
        ]
    )
    return SimulationTable(_apply_schema(df), table_id="run", scenarios=scenarios)


def _keys(df: pd.DataFrame) -> Set[Tuple[Any, ...]]:
    """(component, output, time, scenario) of each row, with None for empty."""
    columns = ["component", "output", "absolute_time_index", "scenario_index"]
    return {
        tuple(None if pd.isna(x) else x for x in row)
        for row in df[columns].itertuples(index=False)
    }


class _CountingStore:
    """A store that counts and records the filters it is read with."""

    def __init__(self, store: _MemoryStore, path: Optional[Path] = None) -> None:
        self._store = store
        self._path = path
        self.table_id = store.table_id
        self.reads: List[_RowFilter] = []

    @property
    def scenarios(self) -> List[int]:
        return self._store.scenarios

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def read(self, row_filter: _RowFilter) -> pd.DataFrame:
        self.reads.append(row_filter)
        return self._store.read(row_filter)

    def iter_scenarios(
        self, row_filter: _RowFilter
    ) -> Iterator[Tuple[Optional[int], pd.DataFrame]]:
        return self._store.iter_scenarios(row_filter)

    def scan(self) -> Any:
        return self._store.scan()


def _counting_table(
    path: Optional[Path] = None,
) -> Tuple[SimulationTable, _CountingStore]:
    """A table whose reads are counted; a *path* makes the store look stored on
    disk."""
    table = _table()
    store = _CountingStore(table._store, path)  # type: ignore[arg-type]
    table._store = store
    return table, store


# ---------------------------------------------------------------------------
# scenarios, path, export
# ---------------------------------------------------------------------------


def test_scenarios_are_sorted_without_the_empty_index() -> None:
    assert _table().scenarios == [0, 1]


def test_path_is_none_in_memory() -> None:
    assert _table().path is None


def test_simulation_table_is_exported_from_the_package() -> None:
    from gems_runner.simulation import SimulationTable as Exported

    assert Exported is SimulationTable


# Only rows shared by all scenarios: they do not say which scenarios exist.
_SHARED_ROWS: List[Row] = [
    (0, "gen", "p_max", None, None, 200.0),
    (0, None, "objective-value", None, None, 900.0),
]


def test_the_run_scenarios_are_kept_when_no_row_names_them() -> None:
    table = _table(_SHARED_ROWS, scenarios=[0, 1, 2])

    assert table.scenarios == [0, 1, 2]
    assert [s for s, _ in table.iter_scenarios()] == [None, 0, 1, 2]


def test_without_run_scenarios_they_come_from_the_rows() -> None:
    assert _table(_SHARED_ROWS).scenarios == []


def test_a_row_outside_the_run_scenarios_raises() -> None:
    with pytest.raises(ValueError, match=r"scenario_index \[1\], which are not"):
        _table(scenarios=[0, 2])


def test_a_scenario_without_rows_raises_when_others_have_rows() -> None:
    with pytest.raises(ValueError, match=r"Scenarios \[2\] of the run have no rows"):
        _table(scenarios=[0, 1, 2])


def test_the_builder_gives_the_table_the_run_scenarios() -> None:
    # Two scenarios, but the only output has no scenario dimension.
    problem = FakeProblem(
        linopy_model=FakeLinopyModel(
            solution={
                "m__p_max": xr.DataArray(
                    [[5.0, 5.0]],
                    dims=["component", "time"],
                    coords={"component": ["c"]},
                )
            }
        ),
        _linopy_vars={
            (0, "p_max"): FakeLinopyVar("m__p_max", {"component": xr.DataArray(["c"])})
        },
        models={0: FakeModel()},
        study=FakeStudy(models={0: FakeModel()}, model_components={}),
        scenarios=2,
    )

    table = SimulationTableBuilder().build(
        problem, scenario_ids_remap=[3, 7]  # type: ignore[arg-type]
    )

    assert table.scenarios == [3, 7]


def test_concatenated_pieces_keep_the_schema() -> None:
    first = _table([(0, "a", "p", 0, 0, 1.0)]).data
    second = _table([(1, "a", "p", 0, 1, 2.0)]).data

    rows = _concat_frames([first, second])

    assert len(rows) == 2
    assert rows.dtypes.to_dict() == first.dtypes.to_dict()
    assert _concat_frames([]).dtypes.to_dict() == first.dtypes.to_dict()


# ---------------------------------------------------------------------------
# to_pandas
# ---------------------------------------------------------------------------


def test_to_pandas_without_filter_returns_a_copy_of_every_row() -> None:
    table = _table()
    df = table.to_pandas()
    pd.testing.assert_frame_equal(df, table.data)
    assert df is not table.data


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param(
            {"components": ["gen"]},
            {
                ("gen", "p", 0, 0),
                ("gen", "p", 1, 0),
                ("gen", "p", 0, 1),
                ("gen", "p", 1, 1),
                ("gen", "p_max", None, None),
            },
            id="components",
        ),
        pytest.param(
            {"outputs": ["d", "objective-value"]},
            {
                ("load", "d", 0, 0),
                ("load", "d", 0, 1),
                (None, "objective-value", None, None),
            },
            id="outputs",
        ),
        pytest.param(
            {"scenarios": [1]},
            {
                ("gen", "p", 0, 1),
                ("gen", "p", 1, 1),
                ("load", "d", 0, 1),
                ("gen", "p_max", None, None),
                (None, "objective-value", None, None),
            },
            id="scenarios-keep-shared-rows",
        ),
        pytest.param(
            {"components": ["gen"], "outputs": ["p", "p_max"], "scenarios": [0]},
            {("gen", "p", 0, 0), ("gen", "p", 1, 0), ("gen", "p_max", None, None)},
            id="combined",
        ),
        pytest.param({"components": ["missing"]}, set(), id="no-match"),
    ],
)
def test_to_pandas_filters(kwargs: Any, expected: Set[Tuple[Any, ...]]) -> None:
    assert _keys(_table().to_pandas(**kwargs)) == expected


@pytest.mark.parametrize(
    "kwargs, as_list",
    [
        pytest.param({"components": "gen"}, {"components": ["gen"]}, id="components"),
        pytest.param({"outputs": "p_max"}, {"outputs": ["p_max"]}, id="outputs"),
    ],
)
def test_a_single_name_filters_like_a_list(kwargs: Any, as_list: Any) -> None:
    table = _table()
    rows = table.to_pandas(**kwargs)

    assert not rows.empty
    assert _keys(rows) == _keys(table.to_pandas(**as_list))


def test_iter_scenarios_accepts_a_single_name() -> None:
    table = _table()
    as_name = [(s, _keys(df)) for s, df in table.iter_scenarios(components="load")]
    as_list = [(s, _keys(df)) for s, df in table.iter_scenarios(components=["load"])]

    assert as_name == as_list
    assert as_name[0][1] == {("load", "d", 0, 0)}


def test_component_filter_keeps_the_rows_and_their_order() -> None:
    # Components are found through an index; the result must match a mask.
    table = _table()
    data = table.data

    rows = table.to_pandas(components=["load", "gen"], outputs=["p", "d"])

    expected = data[
        data["component"].isin(["load", "gen"]) & data["output"].isin(["p", "d"])
    ]
    pd.testing.assert_frame_equal(rows, expected)


def test_to_pandas_load_all_has_no_effect_in_memory() -> None:
    table = _table()
    pd.testing.assert_frame_equal(table.to_pandas(load_all=True), table.data)


# ---------------------------------------------------------------------------
# iter_scenarios
# ---------------------------------------------------------------------------


def test_iter_scenarios_yields_shared_rows_once_first_then_each_scenario() -> None:
    parts = list(_table().iter_scenarios())

    assert [scenario for scenario, _ in parts] == [None, 0, 1]
    assert _keys(parts[0][1]) == {
        ("gen", "p_max", None, None),
        (None, "objective-value", None, None),
    }
    assert _keys(parts[1][1]) == {
        ("gen", "p", 0, 0),
        ("gen", "p", 1, 0),
        ("load", "d", 0, 0),
    }


def test_iter_scenarios_applies_the_filters_and_skips_empty_shared_rows() -> None:
    parts = list(_table().iter_scenarios(components=["load"]))

    assert [scenario for scenario, _ in parts] == [0, 1]
    assert _keys(parts[1][1]) == {("load", "d", 0, 1)}


def test_iter_scenarios_yields_a_scenario_without_matching_rows() -> None:
    table = _table([(0, "a", "p", 0, 0, 1.0), (0, "b", "p", 0, 1, 2.0)])

    parts = dict(table.iter_scenarios(components=["b"]))

    assert list(parts) == [0, 1]
    assert parts[0].empty
    assert _keys(parts[1]) == {("b", "p", 0, 1)}


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def test_scan_returns_a_dataset_with_the_schema_and_every_row() -> None:
    table = _table()
    dataset = table.scan()

    assert dataset.schema.remove_metadata() == SIMULATION_TABLE_SCHEMA
    assert dataset.count_rows() == len(table.data)


# ---------------------------------------------------------------------------
# Lazy views
# ---------------------------------------------------------------------------


def test_views_read_nothing_until_a_value_is_asked() -> None:
    table, store = _counting_table()

    table.component("gen").output("p")

    assert store.reads == []


def test_output_view_reads_its_rows_once() -> None:
    table, store = _counting_table()
    view = table.component("gen").output("p")

    view.data
    view.value(time_index=0, scenario_index=0)
    view.value(scenario_index=1)

    assert len(store.reads) == 1


def test_scenario_value_in_memory_reads_all_rows_once() -> None:
    table, store = _counting_table()
    view = table.component("gen").output("p")

    view.value(time_index=1, scenario_index=1)
    view.value(time_index=0, scenario_index=0)

    assert [f.scenarios for f in store.reads] == [None]


def test_scenario_value_on_disk_reads_only_that_scenario() -> None:
    table, store = _counting_table(path=Path("stored"))
    view = table.component("gen").output("p")

    assert view.value(time_index=1, scenario_index=1) == 13.0
    view.value(time_index=0, scenario_index=1)

    assert [f.scenarios for f in store.reads] == [frozenset({1})]
