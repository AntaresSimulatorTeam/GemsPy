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
from pathlib import Path
from typing import List, Optional

from pydantic import ConfigDict, Field, ValidationError
from yaml import safe_dump, safe_load

from gems_craft.utils import ModifiedBaseModel


class ParameterSchema(ModifiedBaseModel):
    id: str
    time_dependent: bool = True
    scenario_dependent: bool = True


class _ExpressionSchema(ModifiedBaseModel):
    """Base of the schemas with expression fields.

    A plain YAML number (e.g. ``definition: 0``) is read as the expression of
    that literal, as if it were quoted.
    """

    model_config = ConfigDict(
        **ModifiedBaseModel.model_config, coerce_numbers_to_str=True
    )


class VariableSchema(_ExpressionSchema):
    id: str
    time_dependent: bool = True
    scenario_dependent: bool = True
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None
    variable_type: str = "continuous"


class ConstraintSchema(_ExpressionSchema):
    id: str
    expression: str
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None


class FieldSchema(ModifiedBaseModel):
    id: str


class AreaConnectionSchema(ModifiedBaseModel):
    injection_to_balance: Optional[str] = None
    spillage_bound: Optional[str] = None
    unsupplied_energy_bound: Optional[str] = None
    # Port field through which a component reads the marginal price of the legacy area
    price: Optional[str] = None


class PortThermalCapacitySchema(ModifiedBaseModel):
    capacity_field: str


class PortTypeSchema(ModifiedBaseModel):
    id: str
    fields: List[FieldSchema] = Field(default_factory=list)
    description: Optional[str] = None
    area_connection: Optional[AreaConnectionSchema] = None
    thermal_capacity_connection: Optional[PortThermalCapacitySchema] = None


class ModelPortSchema(ModifiedBaseModel):
    id: str
    type: str


class PortFieldDefinitionSchema(_ExpressionSchema):
    port: str
    field: str
    definition: str


class PropertySchema(ModifiedBaseModel):
    id: str


class ObjectiveContributionSchema(_ExpressionSchema):
    id: str
    expression: str


@dataclass
class ExtraOutputSchema(_ExpressionSchema):
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
    properties: List[PropertySchema] = Field(default_factory=list)


class LibrarySchema(ModifiedBaseModel):
    id: str
    dependencies: List[str] = Field(default_factory=list)
    port_types: List[PortTypeSchema] = Field(default_factory=list)
    models: List[ModelSchema] = Field(default_factory=list)
    description: Optional[str] = None
    taxonomy: Optional[str] = None
    version: Optional[str] = None


def parse_yaml_library(input: typing.TextIO) -> LibrarySchema:
    tree = safe_load(input)
    try:
        return LibrarySchema.model_validate(tree["library"])
    except ValidationError as e:
        raise ValueError(f"An error occurred during parsing: {e}")


def write_yaml_library(library: LibrarySchema, path: Path) -> None:
    for port_type in library.port_types:
        if port_type.area_connection is not None:
            port_type_field = "area-connection"
        elif port_type.thermal_capacity_connection is not None:
            port_type_field = "thermal-capacity-connection"
        else:
            continue
        raise ValueError(
            f"Library '{library.id}' has {port_type_field} on port type '{port_type.id}'. "
            "Write it with gems_craft_hybrid.model.parsing.write_yaml_hybrid_library."
        )
    data = {
        "library": library.model_dump(by_alias=True, exclude_none=True, mode="json")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        safe_dump(data, f, allow_unicode=True, sort_keys=False)
