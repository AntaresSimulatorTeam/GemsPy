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
from dataclasses import dataclass
from typing import List, Optional

from pydantic import ConfigDict, Field, ValidationError
from yaml import safe_load

from gems.utils import ModifiedBaseModel


def parse_yaml_library(input: typing.TextIO) -> "LibrarySchema":
    tree = safe_load(input)
    try:
        return LibrarySchema.model_validate(tree["library"])
    except ValidationError as e:
        raise ValueError(f"An error occurred during parsing: {e}")


class ParameterSchema(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False


class VariableSchema(ModifiedBaseModel):
    id: str
    time_dependent: bool = True
    scenario_dependent: bool = True
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None
    variable_type: str = "continuous"

    model_config = ConfigDict(
        **ModifiedBaseModel.model_config, coerce_numbers_to_str=True
    )


class ConstraintSchema(ModifiedBaseModel):
    id: str
    expression: str
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None


class FieldSchema(ModifiedBaseModel):
    id: str


class PortTypeSchema(ModifiedBaseModel):
    id: str
    fields: List[FieldSchema] = Field(default_factory=list)
    description: Optional[str] = None


class ModelPortSchema(ModifiedBaseModel):
    id: str
    type: str


class PortFieldDefinitionSchema(ModifiedBaseModel):
    port: str
    field: str
    definition: str


class ObjectiveContributionSchema(ModifiedBaseModel):
    id: str
    expression: str


@dataclass
class ExtraOutputSchema(ModifiedBaseModel):
    id: str
    expression: str


class ModelSchema(ModifiedBaseModel):
    id: str
    taxonomy_category: Optional[str] = None
    parameters: List[ParameterSchema] = Field(default_factory=list)
    variables: List[VariableSchema] = Field(default_factory=list)
    ports: List[ModelPortSchema] = Field(default_factory=list)
    port_field_definitions: List[PortFieldDefinitionSchema] = Field(
        default_factory=list
    )
    binding_constraints: List[ConstraintSchema] = Field(default_factory=list)
    constraints: List[ConstraintSchema] = Field(default_factory=list)
    objective_contributions: List[ObjectiveContributionSchema] = Field(
        default_factory=list, alias="objective-contributions"
    )
    description: Optional[str] = None
    extra_outputs: Optional[List[ExtraOutputSchema]] = None


class LibrarySchema(ModifiedBaseModel):
    id: str
    dependencies: List[str] = Field(default_factory=list)
    port_types: List[PortTypeSchema] = Field(default_factory=list)
    models: List[ModelSchema] = Field(default_factory=list)
    description: Optional[str] = None
