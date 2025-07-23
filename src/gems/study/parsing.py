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

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TextIO, Union

import pandas as pd
from pydantic import Field
from yaml import safe_load

from gems.utils import ModifiedBaseModel


def parse_yaml_components(input_study: TextIO) -> "InputSystem":
    tree = safe_load(input_study)
    return InputSystem.model_validate(tree["system"])


def parse_scenario_builder(file: Path) -> pd.DataFrame:
    sb = pd.read_csv(file, names=("name", "year", "scenario"))
    sb.rename(columns={0: "name", 1: "year", 2: "scenario"})
    return sb


class InputPortConnections(ModifiedBaseModel):
    component1: str
    port1: str
    component2: str
    port2: str


class InputComponentParameter(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False
    value: Union[float, str]
    scenario_group: Optional[str] = None


class InputComponent(ModifiedBaseModel):
    id: str
    model: str
    scenario_group: Optional[str] = None
    parameters: Optional[List[InputComponentParameter]] = None


class InputSystem(ModifiedBaseModel):
    model_libraries: Optional[str] = None  # Parsed but unused for now
    nodes: List[InputComponent] = Field(default_factory=list)
    components: List[InputComponent] = Field(default_factory=list)
    connections: List[InputPortConnections] = Field(default_factory=list)


@dataclass(frozen=True)
class ParsedArguments:
    models_path: List[Path]
    components_path: Path
    timeseries_path: Path
    duration: int
    nb_scenarios: int


def parse_cli() -> ParsedArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", type=Path, help="path to the root directory of the study"
    )
    parser.add_argument(
        "--models", nargs="+", type=Path, help="list of path to model file, *.yml"
    )
    parser.add_argument(
        "--component", type=Path, help="path to the component file, *.yml"
    )
    parser.add_argument(
        "--timeseries", type=Path, help="path to the timeseries directory"
    )
    parser.add_argument(
        "--duration", type=int, help="duration of the simulation", default=1
    )
    parser.add_argument(
        "--scenario", type=int, help="number of scenario of the simulation", default=1
    )

    args = parser.parse_args()

    if args.study:
        if args.models or args.component or args.timeseries:
            parser.error(
                "--study flag can't be use with --models, --component and --timeseries"
            )

        components_path = args.study / "input" / "components" / "components.yml"
        timeseries_dir = args.study / "input" / "components" / "series"
        model_paths = [
            args.study / "input" / "models" / file
            for file in os.listdir(args.study / "input" / "models")
        ]

    else:
        if not args.models or not args.component:
            parser.error("--models and --component must be entered")

        components_path = args.component
        timeseries_dir = args.timeseries
        model_paths = args.models

    return ParsedArguments(
        model_paths, components_path, timeseries_dir, args.duration, args.scenario
    )
