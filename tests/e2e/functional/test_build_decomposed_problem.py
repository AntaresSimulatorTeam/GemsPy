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
Tests for decomposed problems build.

Each study directory contains:
  - input/system.yml           : network and component definitions
  - input/model-libraries/     : model library YAML files
  - input/optim-config.yml     : decomposition configuration
  - expected_outputs/master.mps      : expected MPS for the master problem
  - expected_outputs/subproblem.mps  : expected MPS for the subproblem
  - expected_outputs/structure.txt   : expected Benders structure file

The test builds the decomposed problems from the input files, writes MPS
files via linopy's to_file(), generates structure.txt, and asserts that
the produced files match expected_outputs byte-for-byte.
"""

from pathlib import Path

import pytest

from gems.main.main import (
    _write_structure_txt,
    input_database,
    input_libs,
    input_system,
)
from gems.optim_config.parsing import load_optim_config, validate_optim_config
from gems.simulation import TimeBlock, build_decomposed_problems
from gems.study import Study

STUDIES_DIR = Path(__file__).parent / "studies"
STUDY_IDS = ["13_1", "13_2"]


@pytest.mark.parametrize("study_id", STUDY_IDS)
def test_study_mps_matches_expected(study_id: str, tmp_path: Path) -> None:
    study_dir = STUDIES_DIR / study_id
    input_dir = study_dir / "input"
    expected_dir = study_dir / "expected_outputs"

    # --- Load model libraries ---
    lib_paths = sorted((input_dir / "model-libraries").glob("*.yml"))
    lib_dict = input_libs(lib_paths)

    # --- Load system and database ---
    system_path = input_dir / "system.yml"
    system = input_system(system_path, lib_dict)
    database = input_database(system_path, timeseries_path=None)

    # --- Load and validate optim-config ---
    config_path = input_dir / "optim-config.yml"
    optim_config = load_optim_config(config_path)
    assert optim_config is not None, f"optim-config.yml not found in {input_dir}"
    validate_optim_config(optim_config, system)

    # --- Build decomposed problems (1 timestep, 1 scenario) ---
    time_block = TimeBlock(1, [0])
    scenarios = 1
    decomposed = build_decomposed_problems(
        Study(system, database), time_block, list(range(scenarios)), optim_config
    )

    # --- Write MPS files ---
    decomposed.subproblem.linopy_model.to_file(
        tmp_path / f"{decomposed.subproblem.name}.mps"
    )
    if decomposed.master is not None:
        decomposed.master.linopy_model.to_file(
            tmp_path / f"{decomposed.master.name}.mps"
        )

    # --- Write structure.txt ---
    _write_structure_txt(
        decomposed,
        optim_config,
        output_dir=tmp_path,
    )

    # --- Assert subproblem MPS matches expected ---
    sub_name = decomposed.subproblem.name
    generated_sub = (tmp_path / f"{sub_name}.mps").read_text()
    expected_sub = (expected_dir / f"{sub_name}.mps").read_text()
    assert generated_sub == expected_sub, (
        f"[{study_id}] {sub_name}.mps mismatch.\n"
        f"Generated:\n{generated_sub}\n"
        f"Expected:\n{expected_sub}"
    )

    # --- Assert master MPS matches expected (if present) ---
    if decomposed.master is not None:
        master_name = decomposed.master.name
        generated_master = (tmp_path / f"{master_name}.mps").read_text()
        expected_master = (expected_dir / f"{master_name}.mps").read_text()
        assert generated_master == expected_master, (
            f"[{study_id}] {master_name}.mps mismatch.\n"
            f"Generated:\n{generated_master}\n"
            f"Expected:\n{expected_master}"
        )

    # --- Assert structure.txt matches expected ---
    generated_struct = (tmp_path / "structure.txt").read_text()
    expected_struct = (expected_dir / "structure.txt").read_text()
    assert generated_struct == expected_struct, (
        f"[{study_id}] structure.txt mismatch.\n"
        f"Generated:\n{generated_struct}\n"
        f"Expected:\n{expected_struct}"
    )
