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

"""
E2E comparison test for the 7_4 study:
  12 components, 11 connections, 76 database entries, 168 timesteps, 1 scenario.

Two time×scenario data series are loaded from TSV files:
  - load_ts_base028  →  component load_base_zone,  parameter load
  - wind_ts_base028  →  component wind_base_zone,  parameter generation

Both tests run the full study twice — once with the original TSV files and
once with Polars-written Parquet equivalents — and assert that every row of
the simulation table is numerically identical.

test_7_4_parquet_coexisting_with_tsv_same_result
    TSV and parquet files coexist in the series directory.
    Parquet takes precedence; TSV is the silent fallback.

test_7_4_parquet_only_same_result_as_tsv
    TSV files are removed after conversion; only parquet remains.
    Verifies that the lazy path works standalone, without a txt/tsv backup.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gems.study.data import LazyTimeScenarioSeriesData
from gems.study.folder import load_study
from gems.study.runner import run_study

_STUDY_SRC = Path(__file__).parent / "studies" / "7_4"

# Pairs (component_id, parameter_id) whose data comes from a TSV file.
_TS_PARAMS = [
    ("load_base_zone", "load"),
    ("wind_base_zone", "generation"),
]


def _convert_tsv_to_parquet(series_dir: Path) -> None:
    """Convert every .tsv file to a .parquet with 0-based string column names.

    Tab-separated files are read by Polars, columns renamed "0", "1", …,
    then written as Parquet.  The original TSV files are left untouched so
    both formats can coexist if desired.
    """
    for tsv in series_dir.glob("*.tsv"):
        df = pl.read_csv(tsv, has_header=False, separator="\t")
        df = df.rename({old: str(i) for i, old in enumerate(df.columns)})
        df.write_parquet(tsv.with_suffix(".parquet"))


def _run_and_get_csv(study_dir: Path) -> Path:
    run_study(study_dir)
    output_files = list((study_dir / "output").glob("**/simulation_table_*.csv"))
    assert len(output_files) == 1, f"Expected 1 output file, got {len(output_files)}"
    return output_files[0]


def _compare_simulation_tables(ref_csv: Path, other_csv: Path) -> None:
    """Assert that two simulation-table CSV files are numerically identical.

    Rows are sorted by (component, output, absolute-time-index, scenario-index)
    so that non-deterministic output ordering does not cause false failures.
    Null-valued index columns (objective-value rows) sort consistently because
    Polars places nulls last in ascending sort by default.
    """
    key_cols = [
        "component",
        "output",
        "absolute-time-index",
        "scenario-index",
    ]
    ref = pl.read_csv(ref_csv).sort(key_cols, nulls_last=True)
    other = pl.read_csv(other_csv).sort(key_cols, nulls_last=True)
    assert_frame_equal(ref, other, check_exact=False, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 1: parquet + tsv coexist — parquet takes precedence
# ---------------------------------------------------------------------------


def test_7_4_parquet_coexisting_with_tsv_same_result(tmp_path: Path) -> None:
    """TSV and parquet coexist; parquet wins and produces identical results.

    Reference run uses the original TSV files (eager TimeScenarioSeriesData).
    Parquet run adds .parquet files alongside each .tsv; the lazy path is
    selected and the original TSV files remain as silent fallbacks.
    Both simulation tables must be numerically identical.
    """
    # --- Reference run (original TSV) ---
    ref_dir = tmp_path / "ref"
    shutil.copytree(_STUDY_SRC, ref_dir)
    ref_csv = _run_and_get_csv(ref_dir)

    # --- Parquet run (parquet + tsv coexist) ---
    pq_dir = tmp_path / "parquet"
    shutil.copytree(_STUDY_SRC, pq_dir)
    _convert_tsv_to_parquet(pq_dir / "input" / "data-series")
    pq_csv = _run_and_get_csv(pq_dir)

    # Verify lazy structures are used (parquet took precedence over tsv)
    study = load_study(pq_dir)
    for comp_id, param_id in _TS_PARAMS:
        ds = study.database.get_data(comp_id, param_id)
        assert isinstance(
            ds, LazyTimeScenarioSeriesData
        ), f"{comp_id}.{param_id}: expected LazyTimeScenarioSeriesData, got {type(ds).__name__}"

    _compare_simulation_tables(ref_csv, pq_csv)


# ---------------------------------------------------------------------------
# Test 2: parquet only — no tsv fallback
# ---------------------------------------------------------------------------


def test_7_4_parquet_only_same_result_as_tsv(tmp_path: Path) -> None:
    """Parquet-only series directory: lazy path works without any TSV fallback.

    Reference run uses the original TSV files.
    Parquet run converts TSV→parquet then removes the TSV files entirely,
    so the only available format is parquet.
    Both simulation tables must be numerically identical.
    """
    # --- Reference run (original TSV) ---
    ref_dir = tmp_path / "ref"
    shutil.copytree(_STUDY_SRC, ref_dir)
    ref_csv = _run_and_get_csv(ref_dir)

    # --- Parquet-only run ---
    pq_dir = tmp_path / "parquet_only"
    shutil.copytree(_STUDY_SRC, pq_dir)
    series_dir = pq_dir / "input" / "data-series"
    _convert_tsv_to_parquet(series_dir)
    for tsv in series_dir.glob("*.tsv"):
        tsv.unlink()

    pq_csv = _run_and_get_csv(pq_dir)

    # Verify lazy structures are used (no tsv to fall back on)
    study = load_study(pq_dir)
    for comp_id, param_id in _TS_PARAMS:
        ds = study.database.get_data(comp_id, param_id)
        assert isinstance(
            ds, LazyTimeScenarioSeriesData
        ), f"{comp_id}.{param_id}: expected LazyTimeScenarioSeriesData, got {type(ds).__name__}"

    _compare_simulation_tables(ref_csv, pq_csv)
