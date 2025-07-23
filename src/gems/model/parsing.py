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
import typing
from typing import List, Optional

from pydantic import Field, ValidationError
from yaml import safe_load

from gems.utils import ModifiedBaseModel, _to_kebab


def parse_yaml_library(input: typing.TextIO) -> "InputLibrary":
    tree = safe_load(input)
    try:
        return InputLibrary.model_validate(tree["library"])
    except ValidationError as e:
        raise ValueError(f"An error occurred during parsing: {e}")


class InputParameter(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False


class InputVariable(ModifiedBaseModel):
    id: str
    time_dependent: bool = True
    scenario_dependent: bool = True
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None
    variable_type: str = "continuous"

    class Config(ModifiedBaseModel.Config):
        coerce_numbers_to_str = True


class InputConstraint(ModifiedBaseModel):
    id: str
    expression: str
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None


class InputField(ModifiedBaseModel):
    id: str


class InputPortType(ModifiedBaseModel):
    id: str
    fields: List[InputField] = Field(default_factory=list)
    description: Optional[str] = None


class InputModelPort(ModifiedBaseModel):
    id: str
    type: str


class InputPortFieldDefinition(ModifiedBaseModel):
    port: str
    field: str
    definition: str


class InputModel(ModifiedBaseModel):
    id: str
    parameters: List[InputParameter] = Field(default_factory=list)
    variables: List[InputVariable] = Field(default_factory=list)
    ports: List[InputModelPort] = Field(default_factory=list)
    port_field_definitions: List[InputPortFieldDefinition] = Field(default_factory=list)
    binding_constraints: List[InputConstraint] = Field(default_factory=list)
    constraints: List[InputConstraint] = Field(default_factory=list)
    objective: Optional[str] = None
    description: Optional[str] = None


class InputLibrary(ModifiedBaseModel):
    id: str
    dependencies: List[str] = Field(default_factory=list)
    port_types: List[InputPortType] = Field(default_factory=list)
    models: List[InputModel] = Field(default_factory=list)
    description: Optional[str] = None
