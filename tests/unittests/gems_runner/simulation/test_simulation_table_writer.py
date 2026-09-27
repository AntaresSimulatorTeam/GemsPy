# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Tests for SimulationTableWriter (one simulation table file per scenario)."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from simulation_table_fakes import to_object_dtype

from gems_runner.simulation.simulation_table import SimulationColumns, SimulationTable
from gems_runner.simulation.simulation_table_writer import (
    SIMULATION_TABLE_SCHEMA,
    SimulationTableWriter,
)

_SORT_COLS = [
    SimulationColumns.OUTPUT.value,
    SimulationColumns.SCENARIO_INDEX.value,
    SimulationColumns.VALUE.value,
]


def _row(
    component: Optional[str], output: str, scenario: Optional[int], value: float
) -> Dict[str, Any]:
    return {
        SimulationColumns.BLOCK.value: 0,
        SimulationColumns.COMPONENT.value: component,
        SimulationColumns.OUTPUT.value: output,
        SimulationColumns.ABSOLUTE_TIME_INDEX.value: None if output != "p" else 0,
        SimulationColumns.BLOCK_TIME_INDEX.value: None if output != "p" else 0,
        SimulationColumns.SCENARIO_INDEX.value: scenario,
        SimulationColumns.VALUE.value: value,
        SimulationColumns.BASIS_STATUS.value: None,
    }


def _table(rows: List[Dict[str, Any]]) -> SimulationTable:
    return SimulationTable(pd.DataFrame(rows), table_id="run")


def _frontal_like_table() -> SimulationTable:
    """Two scenarios plus rows shared by all scenarios (empty scenario_index)."""
    return _table(
        [
            _row("gen", "p", 0, 1.0),
            _row("gen", "p", 1, 2.0),
            _row("gen", "p_max", None, 100.0),
            _row(None, "objective-value", None, 42.0),
        ]
    )


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows and normalise nulls, which read back as NaN in typed columns."""
    return to_object_dtype(df.sort_values(_SORT_COLS).reset_index(drop=True))


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_split_writes_one_file_per_scenario_and_common(
    tmp_path: Path, output_format: str
) -> None:
    writer = SimulationTableWriter(output_format)  # type: ignore[arg-type]
    paths = writer.write(_frontal_like_table(), tmp_path)

    assert sorted(p.name for p in paths) == [
        f"simulation_table_run_scenario-0.{output_format}",
        f"simulation_table_run_scenario-1.{output_format}",
        f"simulation_table_run_scenario-common.{output_format}",
    ]
    assert all(p.exists() for p in paths)


def test_split_writes_every_row_exactly_once(tmp_path: Path) -> None:
    table = _frontal_like_table()
    paths = SimulationTableWriter("parquet").write(table, tmp_path)

    reloaded = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    pd.testing.assert_frame_equal(
        _sorted(reloaded), _sorted(table.data), check_dtype=False
    )


def test_split_scenario_file_contains_only_its_scenario(tmp_path: Path) -> None:
    SimulationTableWriter("parquet").write(_frontal_like_table(), tmp_path)

    scenario_1 = pd.read_parquet(tmp_path / "simulation_table_run_scenario-1.parquet")
    common = pd.read_parquet(tmp_path / "simulation_table_run_scenario-common.parquet")
    assert list(scenario_1[SimulationColumns.SCENARIO_INDEX.value]) == [1]
    assert common[SimulationColumns.SCENARIO_INDEX.value].isna().all()
    assert sorted(common[SimulationColumns.OUTPUT.value]) == [
        "objective-value",
        "p_max",
    ]


def test_split_without_shared_rows_writes_no_common_file(tmp_path: Path) -> None:
    table = _table([_row("gen", "p", 0, 1.0), _row(None, "objective-value", 0, 5.0)])
    paths = SimulationTableWriter("parquet").write(table, tmp_path)
    assert [p.name for p in paths] == ["simulation_table_run_scenario-0.parquet"]


def test_split_parquet_files_share_schema_and_compression(tmp_path: Path) -> None:
    """The common file only holds empty components here; its schema must still
    match the other files so that the files can be read together."""
    table = _table([_row("gen", "p", 0, 1.0), _row(None, "objective-value", None, 5.0)])
    paths = SimulationTableWriter("parquet").write(table, tmp_path)

    schemas = [pq.read_schema(p).remove_metadata() for p in paths]
    assert all(schema.equals(schemas[0]) for schema in schemas)
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        for column_index in range(metadata.num_columns):
            assert metadata.row_group(0).column(column_index).compression == "ZSTD"


def test_split_parquet_files_use_fixed_schema(tmp_path: Path) -> None:
    paths = SimulationTableWriter("parquet").write(_frontal_like_table(), tmp_path)
    for path in paths:
        assert pq.read_schema(path).remove_metadata().equals(SIMULATION_TABLE_SCHEMA)


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_write_scenario_writes_its_scenario_file(
    tmp_path: Path, output_format: str
) -> None:
    table = _table([_row("gen", "p", 4, 1.0), _row(None, "objective-value", 4, 5.0)])
    writer = SimulationTableWriter(output_format)  # type: ignore[arg-type]
    path = writer.write_scenario(table, tmp_path, scenario_id=4)

    assert path.name == f"simulation_table_run_scenario-4.{output_format}"
    reloaded = (
        pd.read_parquet(path) if output_format == "parquet" else pd.read_csv(path)
    )
    assert len(reloaded) == 2


def test_split_empty_table_writes_nothing(tmp_path: Path) -> None:
    paths = SimulationTableWriter("parquet").write(
        SimulationTable(pd.DataFrame(), table_id="run"), tmp_path
    )
    assert paths == []


def test_unsupported_output_format_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        SimulationTableWriter("xlsx")  # type: ignore[arg-type]


def test_csv_format(tmp_path: Path) -> None:
    """CSV files are written by pyarrow without quotes: integer index columns,
    empty cells for missing values, whole floats without '.0'."""
    table = _table(
        [
            {
                **_row("gen", "p", 1, 1.5),
                "absolute_time_index": 3,
                "block_time_index": 3,
            },
            _row(None, "objective-value", 1, 42.0),
        ]
    )
    path = SimulationTableWriter("csv").write_scenario(table, tmp_path, 1)

    assert path.read_text().splitlines() == [
        "block,component,output,absolute_time_index,block_time_index,"
        "scenario_index,value,basis_status",
        "0,gen,p,3,3,1,1.5,",
        "0,,objective-value,,,1,42,",
    ]


def test_csv_value_with_comma_raises_instead_of_writing_a_broken_row(
    tmp_path: Path,
) -> None:
    table = _table([_row("gen,1", "p", 0, 1.0)])
    with pytest.raises(pa.ArrowInvalid, match="structural characters"):
        SimulationTableWriter("csv").write_scenario(table, tmp_path, 0)


def test_parquet_format(tmp_path: Path) -> None:
    """Parquet files use the fixed schema without pandas metadata, zstd
    compression, 64,000-row groups, and store missing values as nulls."""
    n_rows = 150_000
    rows = pd.DataFrame(
        {
            SimulationColumns.BLOCK.value: 0,
            SimulationColumns.COMPONENT.value: "gen",
            SimulationColumns.OUTPUT.value: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX.value: range(n_rows),
            SimulationColumns.BLOCK_TIME_INDEX.value: range(n_rows),
            SimulationColumns.SCENARIO_INDEX.value: 1,
            SimulationColumns.VALUE.value: 1.5,
            SimulationColumns.BASIS_STATUS.value: None,
        }
    )
    objective = pd.DataFrame([_row(None, "objective-value", 1, 42.0)])
    table = SimulationTable(pd.concat([rows, objective], ignore_index=True), "run")
    path = SimulationTableWriter("parquet").write_scenario(table, tmp_path, 1)

    parquet_file = pq.ParquetFile(path)
    assert parquet_file.schema_arrow.equals(SIMULATION_TABLE_SCHEMA)
    assert parquet_file.schema_arrow.metadata is None  # no pandas metadata
    metadata = parquet_file.metadata
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [
        64_000,
        64_000,
        22_001,
    ]
    for group in range(metadata.num_row_groups):
        for column in range(metadata.num_columns):
            assert metadata.row_group(group).column(column).compression == "ZSTD"
    assert parquet_file.read().slice(n_rows - 1).to_pylist() == [
        {
            "block": 0,
            "component": "gen",
            "output": "p",
            "absolute_time_index": n_rows - 1,
            "block_time_index": n_rows - 1,
            "scenario_index": 1,
            "value": 1.5,
            "basis_status": None,
        },
        {
            "block": 0,
            "component": None,
            "output": "objective-value",
            "absolute_time_index": None,
            "block_time_index": None,
            "scenario_index": 1,
            "value": 42.0,
            "basis_status": None,
        },
    ]
