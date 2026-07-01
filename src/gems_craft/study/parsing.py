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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TextIO, Union

from pydantic import Field, ValidationError
from yaml import safe_dump, safe_load

from gems_craft.utils import ModifiedBaseModel


def _parse_yaml_components(input_study: TextIO) -> "SystemSchema":
    tree = safe_load(input_study)
    return SystemSchema.model_validate(tree["system"])


def load_yaml_system(path: Path) -> "SystemSchema":
    try:
        with path.open() as f:
            return _parse_yaml_components(f)
    except (ValidationError, ValueError) as e:
        raise ValueError(f"An error occurred during parsing: {e}") from e


class AreaConnectionsSchema(ModifiedBaseModel):
    component: str
    port: str
    area: str


class PortConnectionsSchema(ModifiedBaseModel):
    component1: str
    port1: str
    component2: str
    port2: str


class ComponentParameterSchema(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False
    value: Union[float, str]
    scenario_group: Optional[str] = None


class ComponentPropertySchema(ModifiedBaseModel):
    id: str
    value: str


class ComponentSchema(ModifiedBaseModel):
    id: str
    model: str
    scenario_group: Optional[str] = None
    parameters: Optional[List[ComponentParameterSchema]] = None
    properties: Optional[List[ComponentPropertySchema]] = None


class SystemSchema(ModifiedBaseModel):
    id: Optional[str] = None
    model_libraries: Optional[str] = None  # Parsed but unused for now
    components: List[ComponentSchema] = Field(default_factory=list)
    connections: Optional[List[PortConnectionsSchema]] = None
    area_connections: Optional[List[AreaConnectionsSchema]] = None


def write_yaml_system(system: "SystemSchema", path: Path) -> None:
    data = {"system": system.model_dump(by_alias=True, exclude_none=True, mode="json")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        safe_dump(data, f, allow_unicode=True, sort_keys=False)


@dataclass(frozen=True)
class ParsedArguments:
    study_dir: Path
    optim_config_path: Optional[Path] = None


def parse_cli() -> ParsedArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="path to the root directory of the study",
    )
    parser.add_argument(
        "--optim-config",
        type=Path,
        default=None,
        dest="optim_config",
        help="optional custom path to optim-config.yml (defaults to study_dir/input/optim-config.yml)",
    )

    args = parser.parse_args()
    return ParsedArguments(
        study_dir=args.study,
        optim_config_path=args.optim_config,
    )
