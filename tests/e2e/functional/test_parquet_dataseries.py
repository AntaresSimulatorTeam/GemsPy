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
E2E tests verifying that Polars-backed lazy parquet data structures produce
the same optimization results as the existing txt-based eager structures.

Three lazy data types are exercised:
  - LazyTimeSeriesData         (time-only, mirrors test_basic_balance_time_only_series)
  - LazyScenarioSeriesData     (scenario-only, mirrors test_basic_balance_scenario_only_series)
  - LazyTimeScenarioSeriesData (time × scenario with ScenarioBuilder,
                                mirrors test_system_with_scenarization)

One additional test verifies that a parquet file takes precedence over a
co-located .txt file (backward-compatibility guarantee).

Parquet files are written to tmp_path so no fixture files are polluted.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.optim_config.parsing import OptimConfig
from gems.session.session import MaterializationStrategy, SimulationSession
from gems.simulation import TimeBlock, build_problem
from gems.study.data import (
    LazyScenarioSeriesData,
    LazyTimeScenarioSeriesData,
    LazyTimeSeriesData,
    MaterializedTimeScenarioSeriesData,
)
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)
from gems.study.scenario_builder import ScenarioBuilder
from gems.study.study import Study

_SYSTEMS_DIR = Path(__file__).parent / "systems"
_LIBS_DIR = Path(__file__).parent / "libs"
_SERIES_DIR = Path(__file__).parent / "series"


def _build_study(
    system_file: str,
    series_path: Path,
    scenario_builder: ScenarioBuilder | None = None,
) -> tuple[Study, object]:
    """Parse lib + system YAML and build Study from a given series directory."""
    with (_LIBS_DIR / "lib_unittest.yml").open() as f:
        lib_dict = resolve_library([parse_yaml_library(f)])
    with (_SYSTEMS_DIR / system_file).open() as f:
        input_system = parse_yaml_components(f)
    system = resolve_system(input_system, lib_dict)
    consistency_check(system, lib_dict["basic"].models)
    database = build_data_base(input_system, series_path, scenario_builder)
    return Study(system, database), database


# ---------------------------------------------------------------------------
# Time-only series
# ---------------------------------------------------------------------------


def test_parquet_time_series_same_result_as_txt(tmp_path: Path) -> None:
    """LazyTimeSeriesData gives the same objective as TimeSeriesData from .txt.

    Mirrors test_basic_balance_time_only_series:
      demand = 50 per timestep, generator cost = 100, horizon = 2
      → objective = 100 × 50 × 2 = 10 000.

    Parquet layout: single column "0", two rows [50.0, 50.0].
    """
    pl.DataFrame({"0": [50.0, 50.0]}).write_parquet(tmp_path / "loads-time-only.parquet")

    study, database = _build_study("study_time_only_series.yml", tmp_path)

    assert isinstance(
        database.get_data("D", "demand"), LazyTimeSeriesData
    ), "Expected LazyTimeSeriesData; txt fallback was used instead"

    problem = build_problem(study, TimeBlock(1, [0, 1]), [0])
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(10_000)


# ---------------------------------------------------------------------------
# Scenario-only series
# ---------------------------------------------------------------------------


def test_parquet_scenario_series_same_result_as_txt(tmp_path: Path) -> None:
    """LazyScenarioSeriesData gives the same objective as ScenarioSeriesData.

    Mirrors test_basic_balance_scenario_only_series:
      two scenarios with demand 50 (col 0) and 100 (col 1), cost = 100
      → objective = 0.5 × 5 000 + 0.5 × 10 000 = 7 500.

    Parquet layout: columns "0" and "1", one row each.
    """
    pl.DataFrame({"0": [50.0], "1": [100.0]}).write_parquet(
        tmp_path / "loads-scenario-only.parquet"
    )

    study, database = _build_study("study_scenario_only_series.yml", tmp_path)

    assert isinstance(
        database.get_data("D", "demand"), LazyScenarioSeriesData
    ), "Expected LazyScenarioSeriesData; txt fallback was used instead"

    problem = build_problem(study, TimeBlock(1, [0]), [0, 1])
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(0.5 * 5_000 + 0.5 * 10_000)


# ---------------------------------------------------------------------------
# Time × scenario series — with ScenarioBuilder
# ---------------------------------------------------------------------------


def test_parquet_time_scenario_series_same_result_as_txt(tmp_path: Path) -> None:
    """LazyTimeScenarioSeriesData gives the same objective with ScenarioBuilder.

    Mirrors test_system_with_scenarization (test_scenario_builder.py):
      loads.txt has T=2 rows and S=2 data-series columns:
        col 0 → demand 50, col 1 → demand 100.
      The ScenarioBuilder maps 3 MC scenarios to columns [0, 1, 0], so:
        MC 0 → 50, MC 1 → 100, MC 2 → 50
      Expected objective = (10 000 + 20 000 + 10 000) / 3 = 40 000 / 3.

    Parquet layout: columns "0" and "1", two rows (one per timestep).
    """
    pl.DataFrame({"0": [50.0, 50.0], "1": [100.0, 100.0]}).write_parquet(tmp_path / "loads.parquet")
    shutil.copy(_SERIES_DIR / "modeler-scenariobuilder.dat", tmp_path)
    scenario_builder = ScenarioBuilder.load(tmp_path / "modeler-scenariobuilder.dat")

    study, database = _build_study("with_scenarization.yml", tmp_path, scenario_builder)

    assert isinstance(
        database.get_data("D", "demand"), LazyTimeScenarioSeriesData
    ), "Expected LazyTimeScenarioSeriesData; txt fallback was used instead"

    problem = build_problem(study, TimeBlock(1, [0, 1]), [0, 1, 2])
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(40_000 / 3, abs=0.001)


# ---------------------------------------------------------------------------
# Parquet takes precedence over .txt when both are present
# ---------------------------------------------------------------------------


def test_parquet_takes_precedence_when_both_formats_present(tmp_path: Path) -> None:
    """When both .parquet and .txt exist in the series dir, parquet is selected.

    Copies the existing loads-time-only.txt alongside a parquet file with the
    same content, then verifies that the lazy structure is used and the result
    is unchanged.
    """
    shutil.copy(_SERIES_DIR / "loads-time-only.txt", tmp_path / "loads-time-only.txt")
    pl.DataFrame({"0": [50.0, 50.0]}).write_parquet(tmp_path / "loads-time-only.parquet")

    study, database = _build_study("study_time_only_series.yml", tmp_path)

    assert isinstance(
        database.get_data("D", "demand"), LazyTimeSeriesData
    ), "Parquet file should take precedence over the co-located .txt file"

    problem = build_problem(study, TimeBlock(1, [0, 1]), [0])
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(10_000)


# ---------------------------------------------------------------------------
# materialize_per_worker
# ---------------------------------------------------------------------------


def _make_scenarized_study(tmp_path: Path):
    """Shared setup: 2-column parquet + ScenarioBuilder mapping 3 MC scenarios to 2 columns."""
    pl.DataFrame({"0": [50.0, 50.0], "1": [100.0, 100.0]}).write_parquet(tmp_path / "loads.parquet")
    shutil.copy(_SERIES_DIR / "modeler-scenariobuilder.dat", tmp_path)
    scenario_builder = ScenarioBuilder.load(tmp_path / "modeler-scenariobuilder.dat")
    study, database = _build_study("with_scenarization.yml", tmp_path, scenario_builder)
    assert isinstance(database.get_data("D", "demand"), LazyTimeScenarioSeriesData)
    return study


def _sequential_config() -> OptimConfig:
    return OptimConfig.model_validate(
        {
            "time-scope": {"first-time-step": 0, "last-time-step": 1},
            "scenario-scope": {"nb-scenarios": 3},
            "resolution": {"mode": "sequential-subproblems", "block-length": 1},
        }
    )


def test_materialize_per_worker_same_result_as_lazy(tmp_path: Path) -> None:
    """MaterializationStrategy.PER_WORKER produces the same objective as NONE (lazy).

    Uses sequential-subproblems with block-length=1 over a 2-timestep horizon
    so that materialize() is called once per scenario.
    Verifies that the database entry becomes MaterializedTimeScenarioSeriesData
    and that the resulting objective is numerically identical to the lazy run.
    """
    study = _make_scenarized_study(tmp_path)
    optim_config = _sequential_config()

    lazy_table = SimulationSession(study=study, optim_config=optim_config).run()
    mat_table = SimulationSession(
        study=study, optim_config=optim_config, materialization=MaterializationStrategy.PER_WORKER
    ).run()

    mat_db = study.database.materialize([0])
    assert isinstance(mat_db.get_data("D", "demand"), MaterializedTimeScenarioSeriesData)

    obj_rows_lazy = lazy_table.data.loc[lazy_table.data["output"] == "objective-value", "value"]
    obj_rows_mat = mat_table.data.loc[mat_table.data["output"] == "objective-value", "value"]
    assert obj_rows_lazy.sum() == pytest.approx(obj_rows_mat.sum(), rel=1e-6)


def test_materialize_upfront_same_result_as_lazy(tmp_path: Path) -> None:
    """MaterializationStrategy.UPFRONT produces the same objective as NONE (lazy).

    Loads the full database into RAM before the run loop starts.
    All columns for all scenarios are materialized in one I/O pass.
    """
    study = _make_scenarized_study(tmp_path)
    optim_config = _sequential_config()

    lazy_table = SimulationSession(study=study, optim_config=optim_config).run()
    upfront_table = SimulationSession(
        study=study, optim_config=optim_config, materialization=MaterializationStrategy.UPFRONT
    ).run()

    mat_db = study.database.materialize(list(range(optim_config.scenario_scope.nb_scenarios)))
    assert isinstance(mat_db.get_data("D", "demand"), MaterializedTimeScenarioSeriesData)

    obj_rows_lazy = lazy_table.data.loc[lazy_table.data["output"] == "objective-value", "value"]
    obj_rows_up = upfront_table.data.loc[upfront_table.data["output"] == "objective-value", "value"]
    assert obj_rows_lazy.sum() == pytest.approx(obj_rows_up.sum(), rel=1e-6)
