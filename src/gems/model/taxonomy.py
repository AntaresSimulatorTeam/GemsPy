# Copyright (c) 2026, RTE (https://www.rte-france.com)
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
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml
from pydantic import Field

from gems.model.parsing import (
    ConstraintSchema,
    LibrarySchema,
    ModelSchema,
    PortFieldDefinitionSchema,
)
from gems.utils import ModifiedBaseModel


class TaxonomyItem(ModifiedBaseModel):
    id: str


class TaxonomyCategory(ModifiedBaseModel):
    id: str
    parent_category: Optional[str] = None
    variables: List[TaxonomyItem] = Field(default_factory=list)
    parameters: List[TaxonomyItem] = Field(default_factory=list)
    ports: List[TaxonomyItem] = Field(default_factory=list)
    port_field_definitions: List[PortFieldDefinitionSchema] = Field(
        default_factory=list
    )
    constraints: List[TaxonomyItem] = Field(default_factory=list)
    binding_constraints: List[ConstraintSchema] = Field(default_factory=list)
    extra_outputs: List[TaxonomyItem] = Field(default_factory=list)
    properties: List[TaxonomyItem] = Field(default_factory=list)


class TaxonomyData(ModifiedBaseModel):
    id: str
    description: str = ""
    categories: List[TaxonomyCategory] = Field(default_factory=list)


@dataclass
class Taxonomy:
    id: str
    description: str = ""
    categories: List[TaxonomyCategory] = field(default_factory=list)


def load_taxonomy(taxonomy_file: Path) -> Taxonomy:
    with open(taxonomy_file, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "taxonomy" not in raw:
        raise ValueError(f"Missing 'taxonomy' key at root of {taxonomy_file}")
    data = TaxonomyData.model_validate(raw["taxonomy"])
    return Taxonomy(
        id=data.id, description=data.description, categories=data.categories
    )


def _missing(required: List, exposed: List, key: Callable = lambda x: x.id) -> List:
    """Return the sorted identifiers required by the taxonomy but not exposed by the model."""
    return sorted(set(map(key, required)) - set(map(key, exposed)))


def check_library_against_taxonomy(library: LibrarySchema, taxonomy: Taxonomy) -> None:
    """
    Validates that every model declaring a taxonomy_category:
      1. References a category that exists in the taxonomy.
      2. Exposes all variables, parameters, ports, port-field-definitions,
         constraints, binding-constraints, extra-outputs and properties listed
         in that taxonomy category.

    Raises ValueError describing the first violation found.
    """
    categories: Dict[str, TaxonomyCategory] = {c.id: c for c in taxonomy.categories}

    # Each entry maps a human-readable field-group name to the required items
    # (from the taxonomy category) and the items exposed by the model, plus the
    # function used to identify an item within that group.
    def field_groups(
        category: TaxonomyCategory, model_schema: "ModelSchema"
    ) -> List[tuple]:
        port_field_key: Callable = lambda d: (d.port, d.field)
        return [
            ("variable", category.variables, model_schema.variables, lambda x: x.id),
            ("parameter", category.parameters, model_schema.parameters, lambda x: x.id),
            ("port", category.ports, model_schema.ports, lambda x: x.id),
            (
                "port-field-definition",
                category.port_field_definitions,
                model_schema.port_field_definitions,
                port_field_key,
            ),
            (
                "constraint",
                category.constraints,
                model_schema.constraints,
                lambda x: x.id,
            ),
            (
                "binding-constraint",
                category.binding_constraints,
                model_schema.binding_constraints,
                lambda x: x.id,
            ),
            (
                "extra-output",
                category.extra_outputs,
                model_schema.extra_outputs or [],
                lambda x: x.id,
            ),
            ("property", category.properties, model_schema.properties, lambda x: x.id),
        ]

    for model_schema in library.models:
        cat_id = model_schema.taxonomy_category
        if cat_id is None:
            continue

        if cat_id not in categories:
            raise ValueError(
                f"Model '{model_schema.id}' references taxonomy category '{cat_id}' "
                f"which does not exist in taxonomy '{taxonomy.id}'."
            )

        category = categories[cat_id]
        for group_name, required, exposed, key in field_groups(category, model_schema):
            missing = _missing(required, exposed, key)
            if missing:
                raise ValueError(
                    f"Model '{model_schema.id}' (taxonomy-category: '{cat_id}') is "
                    f"missing {group_name}(s) required by the taxonomy: {missing}."
                )
