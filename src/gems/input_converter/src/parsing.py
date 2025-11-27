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

from dataclasses import dataclass, field
from typing import Optional, TextIO, Union

import pandas as pd
from pydantic import BaseModel, Field
from yaml import safe_load


@dataclass
class VirtualObjectsRepository:
    areas: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    thermals: list[str] = field(default_factory=list)

    def resolve_template(
        self, template_pattern: str, value: str
    ) -> "VirtualObjectsRepository":
        thermals = []
        for thermal in self.thermals:
            thermals.append(thermal.replace(template_pattern, value))
        return VirtualObjectsRepository(
            areas=self.areas, links=self.links, thermals=thermals
        )

    def add(self, other: "VirtualObjectsRepository") -> None:
        self.areas += other.areas
        self.links += other.links
        self.thermals += other.thermals


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

    def execute(
        self,
        initial_value: Union[pd.DataFrame, pd.Series, float],
        preprocessed_values: Optional[Union[dict[str, float], float]] = None,
    ) -> Union[float, pd.Series, pd.DataFrame]:
        def resolve(value: Union[str, float]) -> Union[float, pd.Series]:
            if isinstance(value, str):
                if (
                    not isinstance(preprocessed_values, dict)
                    or value not in preprocessed_values
                ):
                    raise ValueError(
                        f"Missing value for key '{value}' in preprocessed_values"
                    )
                return preprocessed_values[value]
            return value

        if self.type == "max":
            return float(max(initial_value))  # type: ignore

        if self.multiply_by is not None:
            return initial_value * resolve(self.multiply_by)

        if self.divide_by is not None:
            return initial_value / resolve(self.divide_by)

        raise ValueError(
            "Operation must have at least one of 'multiply_by', 'divide_by', or 'type'"
        )


class ObjectProperties(ModifiedBaseModel):
    type: str
    area: Optional[str] = None
    link: Optional[str] = None
    cluster: Optional[str] = None
    binding_constraint_id: Optional[str] = None
    field: Optional[str] = None

    def resolve_template(self, template_pattern: str, value: str) -> "ObjectProperties":
        area = (
            self.area.replace(template_pattern, value)
            if self.area is not None
            else None
        )
        link = (
            self.link.replace(template_pattern, value)
            if self.link is not None
            else None
        )
        cluster = (
            self.cluster.replace(template_pattern, value)
            if self.cluster is not None
            else None
        )
        binding_constraint_id = (
            self.binding_constraint_id.replace(template_pattern, value)
            if self.binding_constraint_id is not None
            else None
        )
        field = (
            self.field.replace(template_pattern, value)
            if self.field is not None
            else None
        )
        return ObjectProperties(
            type=self.type,
            area=area,
            link=link,
            cluster=cluster,
            binding_constraint_id=binding_constraint_id,
            field=field,
        )


class ConversionValue(ModifiedBaseModel):
    object_properties: Optional[ObjectProperties] = None
    column: Optional[int] = None
    operation: Optional[Operation] = None
    constant: Optional[float] = None

    def resolve_template(self, template_pattern: str, value: str) -> "ConversionValue":
        object_properties = (
            self.object_properties.resolve_template(template_pattern, value)
            if self.object_properties is not None
            else None
        )
        return ConversionValue(
            object_properties=object_properties,
            column=self.column,
            operation=self.operation,
            constant=self.constant,
        )


class ParameterConversionConfig(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False
    value: ConversionValue

    def resolve_template(
        self, template_pattern: str, value: str
    ) -> "ParameterConversionConfig":
        return ParameterConversionConfig(
            id=self.id,
            time_dependent=self.time_dependent,
            scenario_dependent=self.scenario_dependent,
            value=self.value.resolve_template(template_pattern, value),
        )


class ComponentConversionConfig(ModifiedBaseModel):
    id: str
    parameters: list[ParameterConversionConfig]

    def resolve_template(
        self, template_pattern: str, value: str
    ) -> "ComponentConversionConfig":
        id = self.id.replace(template_pattern, value)
        parameters = []
        for param in self.parameters:
            parameters.append(param.resolve_template(template_pattern, value))
        return ComponentConversionConfig(id=id, parameters=parameters)


class ReferencedLegacyObjects(ModifiedBaseModel):
    id: str
    object_properties: ObjectProperties

    def resolve_template(
        self, template_pattern: str, value: str
    ) -> "ReferencedLegacyObjects":
        return ReferencedLegacyObjects(
            id=self.id,
            object_properties=self.object_properties.resolve_template(
                template_pattern, value
            ),
        )


class AreaConnectionConversionConfig(ModifiedBaseModel):
    component: str
    port: str
    area: str

    def resolve_template(
        self, template_pattern: str, value: str
    ) -> "AreaConnectionConversionConfig":
        return AreaConnectionConversionConfig(
            component=self.component.replace(template_pattern, value),
            port=self.port,
            area=self.area.replace(template_pattern, value),
        )


class PortConnectionConversionConfig(ModifiedBaseModel):
    component1: str
    port1: str
    component2: str
    port2: str

    def resolve_template(
        self, template_pattern: str, value: str
    ) -> "PortConnectionConversionConfig":
        return PortConnectionConversionConfig(
            component1=self.component1.replace(template_pattern, value),
            port1=self.port1,
            component2=self.component2.replace(template_pattern, value),
            port2=self.port2,
        )


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
    scenario_group: Optional[str] = None

    def resolve_template(
        self, template_pattern: str, value: str
    ) -> "ConversionTemplate":
        component = self.component.resolve_template(template_pattern, value)
        connections = []
        area_connections = []
        legacy_objects_to_delete = []
        for connection in self.connections:
            connections.append(connection.resolve_template(template_pattern, value))
        for area_connection in self.area_connections:
            area_connections.append(
                area_connection.resolve_template(template_pattern, value)
            )
        for object_to_delete in self.legacy_objects_to_delete:
            legacy_objects_to_delete.append(
                object_to_delete.resolve_template(template_pattern, value)
            )
        return ConversionTemplate(
            name=self.name,
            model=self.model,
            generator_version_compatibility=self.generator_version_compatibility,
            template_parameters=self.template_parameters,
            component=component,
            connections=connections,
            area_connections=area_connections,
            legacy_objects_to_delete=legacy_objects_to_delete,
            scenario_group=self.scenario_group,
        )

    def get_excluded_objects_ids(self) -> VirtualObjectsRepository:
        virtual_areas = []
        virtual_links = []
        virtual_thermals = []
        for param in self.template_parameters:
            if param.exclude is not None:
                for excluded_object in param.exclude:
                    exclude_prop = excluded_object.object_properties
                    if exclude_prop.type == "area" and exclude_prop.area is not None:
                        virtual_areas.append(exclude_prop.area)
                    elif exclude_prop.type == "link" and exclude_prop.link is not None:
                        virtual_links.append(exclude_prop.link)
                    elif exclude_prop.type == "thermal":
                        if (
                            exclude_prop.area is not None
                            and exclude_prop.cluster is not None
                        ):
                            virtual_thermals.append(exclude_prop.cluster)
        return VirtualObjectsRepository(virtual_areas, virtual_links, virtual_thermals)


def parse_conversion_template(input_template: TextIO) -> ConversionTemplate:
    tree = safe_load(input_template)
    return ConversionTemplate.model_validate(tree["template"])
