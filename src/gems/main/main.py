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

from gems.model.library import Library
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import TimeBlock, build_problem
from gems.study import DataBase
from gems.study.parsing import parse_cli, parse_yaml_components
from gems.study.resolve_components import (
    System,
    build_data_base,
    build_network,
    consistency_check,
    resolve_system,
)


class AntaresTimeSeriesImportError(Exception):
    pass


def input_libs(yaml_lib_paths: List[Path]) -> dict[str, Library]:
    yaml_libraries = []
    yaml_library_ids = set()

    for path in yaml_lib_paths:
        with path.open("r") as file:
            yaml_lib = parse_yaml_library(file)

            if yaml_lib.id in yaml_library_ids:
                raise ValueError(f"The identifier '{yaml_lib.id}' is defined twice")

            yaml_libraries.append(yaml_lib)
            yaml_library_ids.add(yaml_lib.id)

    return resolve_library(yaml_libraries)


def input_database(study_path: Path, timeseries_path: Optional[Path]) -> DataBase:
    with study_path.open() as comp:
        return build_data_base(parse_yaml_components(comp), timeseries_path)


def input_study(study_path: Path, librairies: dict[str, Library]) -> System:
    with study_path.open() as comp:
        return resolve_system(parse_yaml_components(comp), librairies)


def main_cli() -> None:
    parsed_args = parse_cli()

    lib_dict = input_libs(parsed_args.models_path)
    study = input_study(parsed_args.components_path, lib_dict)

    models = {}
    for lib in lib_dict.values():
        models.update(lib.models)

    consistency_check(study.components, models)

    try:
        database = input_database(
            parsed_args.components_path, parsed_args.timeseries_path
        )

    except UnboundLocalError:
        raise AntaresTimeSeriesImportError(
            f"An error occurred while importing time series."
        )

    network = build_network(study)

    timeblock = TimeBlock(1, list(range(parsed_args.duration)))
    scenario = parsed_args.nb_scenarios

    try:
        problem = build_problem(network, database, timeblock, scenario)

    except IndexError as e:
        raise IndexError(
            f"{e}. Did parameters '--duration' and '--scenario' were correctly set?"
        )

    status = problem.solver.Solve()
    print("status : ", status)

    print("final average cost : ", problem.solver.Objective().Value())


if __name__ == "__main__":
    main_cli()
