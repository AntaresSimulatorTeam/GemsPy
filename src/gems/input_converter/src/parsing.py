# Copyright (c) 2025, RTE (https://www.rte-france.com)
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

from typing import Optional, TextIO, Union

from pydantic import BaseModel, Field
from yaml import safe_load


def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class ModifiedBaseModel(BaseModel):
    class Config:
        alias_generator = _to_kebab
        extra = "forbid"
        populate_by_name = True


class Operation(ModifiedBaseModel):
    type: Optional[str] = None
    multiply_by: Optional[Union[float, str]] = None
    divide_by: Optional[Union[float, str]] = None


class ObjectProperties(ModifiedBaseModel):
    type: str
    area: Optional[str] = None
    link: Optional[str] = None
    cluster: Optional[str] = None
    binding_constraint_id: Optional[str] = None
    field: Optional[str] = None


class ConversionValue(ModifiedBaseModel):
    object_properties: Optional[ObjectProperties] = None
    column: Optional[int] = None
    operation: Optional[Operation] = None
    constant: Optional[float] = None


class ParameterConversionConfig(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False
    value: ConversionValue


class ComponentConversionConfig(ModifiedBaseModel):
    id: str
    parameters: Optional[list[ParameterConversionConfig]] = None


class ReferencedLegacyObjects(ModifiedBaseModel):
    id: str
    object_properties: ObjectProperties


class AreaConnectionConversionConfig(ModifiedBaseModel):
    component: str
    port: str
    area: str


class PortConnectionConversionConfig(ModifiedBaseModel):
    component1: str
    port1: str
    component2: str
    port2: str


class TemplateParameter(ModifiedBaseModel):
    name: str
    description: Optional[str] = None
    cluster_type: Optional[str] = None
    exclude: Optional[list[ReferencedLegacyObjects]] = None


class ConversionTemplate(ModifiedBaseModel):
    name: str
    model: str
    generator_version_compatibility: Optional[str] = None
    template_parameters: list[TemplateParameter] = Field(default_factory=list)
    component: ComponentConversionConfig
    connections: list[PortConnectionConversionConfig] = Field(default_factory=list)
    area_connections: list[AreaConnectionConversionConfig] = Field(default_factory=list)
    legacy_objects_to_delete: list[ReferencedLegacyObjects] = Field(
        default_factory=list
    )


def parse_conversion_template(input_template: TextIO) -> ConversionTemplate:
    tree = safe_load(input_template)
    return ConversionTemplate.model_validate(tree["template"])
