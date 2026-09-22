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
Tests for per-(Monte Carlo year, week-block) Benders subproblem generation.

Study 14_1 reuses 13_1's single-candidate investment model but runs it over
2 scenarios x 2 week-blocks (``block-length: 2`` over a 4-timestep horizon),
so ``build_decomposed_problems`` must produce 2*2 = 4 independent
subproblems sharing one master, instead of 13_1/13_2's single-subproblem
degenerate case (see test_build_decomposed_problem.py).

  - input/system.yml           : network and component definitions
  - input/model-libraries/     : model library YAML files
  - input/optim-config.yml     : decomposition + resolution configuration
  - expected_outputs/master.mps                : expected MPS for the master
  - expected_outputs/subproblem_y{s}_w{b}.mps  : expected MPS per subproblem
  - expected_outputs/structure.txt              : expected Benders structure file
  - expected_outputs/options.json               : expected AntaresXpansion options file
"""

import json
from pathlib import Path

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.optim_config.validation import validate_optim_config
from gems_craft.study import Study
from gems_runner.main.main import input_database, input_libs, input_system
from gems_runner.simulation import build_decomposed_problems
from gems_runner.simulation.benders_export import export_benders_problem
from gems_runner.simulation.time_block import compute_blocks

STUDY_DIR = Path(__file__).parent / "studies" / "14_1"
INPUT_DIR = STUDY_DIR / "input"
EXPECTED_DIR = STUDY_DIR / "expected_outputs"


def _load() -> tuple:
    lib_paths = sorted((INPUT_DIR / "model-libraries").glob("*.yml"))
    lib_dict = input_libs(lib_paths)
    system_path = INPUT_DIR / "system.yml"
    system = input_system(system_path, lib_dict)
    database = input_database(system_path, timeseries_path=None)
    optim_config = load_optim_config(INPUT_DIR / "optim-config.yml")
    assert optim_config is not None
    validate_optim_config(optim_config, system)
    return system, database, optim_config


def test_build_decomposed_problems_produces_one_subproblem_per_scenario_and_block() -> (
    None
):
    system, database, optim_config = _load()
    cfg = optim_config.resolution
    blocks = compute_blocks(
        optim_config.time_scope.first_time_step,
        optim_config.time_scope.last_time_step,
        cfg.block_length,
    )
    scenario_ids = optim_config.scenario_scope.scenario_ids

    decomposed = build_decomposed_problems(
        Study(system, database), blocks, scenario_ids, optim_config
    )

    assert len(scenario_ids) == 2
    assert len(blocks) == 2
    assert len(decomposed.subproblems) == len(scenario_ids) * len(blocks)

    # Ordered scenario-major, block-minor.
    expected_names = [
        f"subproblem_y{sid}_w{block.id}" for sid in scenario_ids for block in blocks
    ]
    assert [sp.name for sp in decomposed.subproblems] == expected_names
    assert decomposed.master is not None
    assert decomposed.master.name == "master"


def test_export_matches_expected_outputs(tmp_path: Path) -> None:
    system, database, optim_config = _load()
    cfg = optim_config.resolution
    blocks = compute_blocks(
        optim_config.time_scope.first_time_step,
        optim_config.time_scope.last_time_step,
        cfg.block_length,
    )
    scenario_ids = optim_config.scenario_scope.scenario_ids

    decomposed = build_decomposed_problems(
        Study(system, database), blocks, scenario_ids, optim_config
    )
    export_benders_problem(
        decomposed, optim_config, tmp_path, n_scenarios=len(scenario_ids)
    )

    # --- Every subproblem's MPS matches byte-for-byte ---
    for subproblem in decomposed.subproblems:
        generated = (tmp_path / f"{subproblem.name}.mps").read_text()
        expected = (EXPECTED_DIR / f"{subproblem.name}.mps").read_text()
        assert generated == expected, f"{subproblem.name}.mps mismatch"

    # --- Master MPS matches ---
    generated_master = (tmp_path / f"{decomposed.master.name}.mps").read_text()
    expected_master = (EXPECTED_DIR / f"{decomposed.master.name}.mps").read_text()
    assert generated_master == expected_master

    # --- structure.txt matches, and every non-master problem_id round-trips
    # to a subproblem that was actually generated (catches ordering/naming
    # drift independent of the golden-byte comparison) ---
    generated_struct = (tmp_path / "structure.txt").read_text()
    expected_struct = (EXPECTED_DIR / "structure.txt").read_text()
    assert generated_struct == expected_struct

    subproblem_names = {sp.name for sp in decomposed.subproblems}
    for line in generated_struct.splitlines():
        problem_id = line[:24].strip()
        assert problem_id == "master" or problem_id in subproblem_names

    # --- options.json is well-formed and references files that exist ---
    options = json.loads((tmp_path / "options.json").read_text())
    assert options["SLAVE_WEIGHT_VALUE"] == len(scenario_ids)
    assert options["MASTER_NAME"] == "master"
    assert options["STRUCTURE_FILE"] == "structure.txt"
    assert (tmp_path / f"{options['MASTER_NAME']}.mps").exists()
    assert (tmp_path / options["STRUCTURE_FILE"]).exists()
    for subproblem in decomposed.subproblems:
        assert (tmp_path / f"{subproblem.name}.mps").exists()
