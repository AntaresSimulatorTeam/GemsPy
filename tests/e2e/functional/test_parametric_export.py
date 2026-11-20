# Copyright (c) 2026, RTE (https://www.rte-france.com)
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
Parametric export of YAML studies: a scenarized study in parallel mode, and a
Benders study whose master, subproblem and structure.txt are exported.
"""

import json
from pathlib import Path

import highspy
import pandas as pd
import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.optim_config.parsing import (
    OptimConfig,
    ResolutionMode,
    ScenarioScopeConfig,
    load_optim_config,
)
from gems_craft.study import Study
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import build_data_base, resolve_system
from gems_craft.study.scenario_builder import ScenarioBuilder
from gems_runner.main.main import input_database, input_libs, input_system
from gems_runner.parametric import export_parametric_problems
from gems_runner.simulation import (
    TimeBlock,
    build_couplings,
    build_decomposed_problems,
    build_problem,
    dump_couplings,
)
from tests.unittests.gems_runner.parametric.reconstruct import solve_realization

STUDIES_DIR = Path(__file__).parent / "studies"


@pytest.fixture
def scenarized_study(libs_dir: Path, systems_dir: Path, series_dir: Path) -> Study:
    scenario_builder = ScenarioBuilder.load(series_dir / "modeler-scenariobuilder.dat")
    with (libs_dir / "lib_unittest.yml").open() as file:
        lib_dict = resolve_library([parse_yaml_library(file)])
    with (systems_dir / "with_scenarization.yml").open() as file:
        system_schema = parse_yaml_system(file)
    return Study(
        resolve_system(system_schema, lib_dict),
        build_data_base(system_schema, series_dir, scenario_builder),
        scenario_builder=scenario_builder,
    )


def test_scenarized_study_in_parallel_mode(
    scenarized_study: Study, tmp_path: Path
) -> None:
    """Each block and scenario rebuilt from the export has the objective of the
    block built for that scenario alone; scenarios are resolved by the
    scenario builder (MC 0 and 2 read the same data-series column)."""
    config = OptimConfig.model_validate(
        {
            "time-scope": {"first-time-step": 0, "last-time-step": 1},
            "scenario-scope": {"include": ["0-2"]},
            "resolution": {"mode": "parallel-subproblems", "block-length": 1},
        }
    )
    export_parametric_problems(scenarized_study, config, tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["blocks"] == [0, 1]
    mps = tmp_path / "problem.mps"
    parameters = pd.read_parquet(tmp_path / "problem_parameters.parquet")

    h = highspy.Highs()
    h.readModel(str(mps))
    names = set(h.getLp().row_names_) | set(h.getLp().col_names_)
    assert set(parameters["row"].dropna()) | set(parameters["col"].dropna()) <= names

    for block_id in manifest["blocks"]:
        # Blocks of one time step: block i holds time step i.
        time_block = TimeBlock(block_id, [block_id])
        for s in manifest["scenarios"]:
            reference = build_problem(scenarized_study, time_block, [s])
            reference.solve(solver_name="highs", output_flag=False)
            assert solve_realization(mps, parameters, block_id, s) == pytest.approx(
                reference.objective_value
            )


def test_benders_study_exports_master_subproblem_and_structure(
    tmp_path: Path,
) -> None:
    input_dir = STUDIES_DIR / "13_1" / "input"
    lib_dict = input_libs(sorted((input_dir / "model-libraries").glob("*.yml")))
    system = input_system(input_dir / "system.yml", lib_dict)
    study = Study(
        system, input_database(input_dir / "system.yml", timeseries_path=None)
    )
    config = load_optim_config(input_dir / "optim-config.yml")
    assert config is not None
    config.resolution.mode = ResolutionMode.BENDERS_DECOMPOSITION
    # `p_max` does not depend on the scenario: allowed, fixed by the master.
    config.scenario_scope = ScenarioScopeConfig(include=["0-1"])

    export_parametric_problems(study, config, tmp_path / "export")

    manifest = json.loads((tmp_path / "export" / "manifest.json").read_text())
    assert manifest["blocks"] == [1]
    assert manifest["problems"] == ["master", "subproblem"]
    export_dir = tmp_path / "export"
    for name in manifest["problems"]:
        assert (export_dir / f"{name}.mps").exists()
        assert (export_dir / f"{name}_parameters.parquet").exists()

    # Same structure.txt as the one written for a Benders run of one scenario.
    reference_dir = tmp_path / "reference"
    decomposed = build_decomposed_problems(study, TimeBlock(1, [0]), [0], config)
    dump_couplings(build_couplings(decomposed, config), reference_dir)
    assert (export_dir / "structure.txt").read_text() == (
        reference_dir / "structure.txt"
    ).read_text()
