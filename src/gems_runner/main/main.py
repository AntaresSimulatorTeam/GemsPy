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

from pathlib import Path
from typing import Dict, List, Optional

from gems_craft.model.parsing import LibrarySchema, load_yaml_library
from gems_craft.optim_config.parsing import OptimConfig
from gems_craft.study.parsing import load_yaml_system, parse_cli
from gems_runner.model.library import Library
from gems_runner.model.resolve_library import resolve_library
from gems_runner.simulation import DecomposedProblems, build_couplings, dump_couplings
from gems_runner.study import Study
from gems_runner.study.data import DataBase
from gems_runner.study.resolve_components import build_data_base, resolve_system
from gems_runner.study.runner import run_study
from gems_runner.study.system import System


def input_libs(yaml_lib_paths: List[Path]) -> Dict[str, Library]:
    yaml_libraries = []
    yaml_library_ids = set()
    for path in yaml_lib_paths:
        yaml_lib: LibrarySchema = load_yaml_library(path)
        if yaml_lib.id in yaml_library_ids:
            raise ValueError(f"The identifier '{yaml_lib.id}' is defined twice")
        yaml_libraries.append(yaml_lib)
        yaml_library_ids.add(yaml_lib.id)
    return resolve_library(yaml_libraries)


def input_database(study_path: Path, timeseries_path: Optional[Path]) -> DataBase:
    return build_data_base(load_yaml_system(study_path), timeseries_path)


def input_system(study_path: Path, libraries: Dict[str, Library]) -> System:
    return resolve_system(load_yaml_system(study_path), libraries)


def _write_structure_txt(
    decomposed: DecomposedProblems,
    optim_config: OptimConfig,
    output_dir: Path,
) -> None:
    dump_couplings(build_couplings(decomposed, optim_config), output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main_cli() -> None:
    parsed_args = parse_cli()
    run_study(
        study_dir=parsed_args.study_dir,
        optim_config_path=parsed_args.optim_config_path,
    )


if __name__ == "__main__":
    main_cli()
