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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import yaml
from pydantic import Field

from gems_craft.utils import ModifiedBaseModel

if TYPE_CHECKING:
    # Annotations only — parsing.py imports this module, so this would be circular.
    from gems_craft.model.parsing import LibrarySchema, ModelSchema


class TaxonomyItem(ModifiedBaseModel):
    id: str


class TaxonomyCategory(ModifiedBaseModel):
    id: str
    parent_category: Optional[str] = None
    variables: List[TaxonomyItem] = Field(default_factory=list)
    parameters: List[TaxonomyItem] = Field(default_factory=list)
    ports: List[TaxonomyItem] = Field(default_factory=list)
    port_field_definitions: List[TaxonomyItem] = Field(default_factory=list)
    constraints: List[TaxonomyItem] = Field(default_factory=list)
    binding_constraints: List[TaxonomyItem] = Field(default_factory=list)
    extra_outputs: List[TaxonomyItem] = Field(default_factory=list)
    properties: List[TaxonomyItem] = Field(default_factory=list)


class TaxonomySchema(ModifiedBaseModel):
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
    data = TaxonomySchema.model_validate(raw["taxonomy"])
    return Taxonomy(
        id=data.id, description=data.description, categories=data.categories
    )


def _missing(
    required: List[TaxonomyItem], exposed: List, exposed_key: Callable
) -> List[str]:
    """Return the sorted taxonomy item ids not exposed by the model.

    Taxonomy items are always identified by their ``id``; ``exposed_key`` maps each
    model-side item to the identifier to compare against (e.g. the ``port.field``
    string for port-field-definitions).
    """
    return sorted(
        {item.id for item in required} - {exposed_key(item) for item in exposed}
    )


def check_library_against_taxonomy(
    library: "LibrarySchema", taxonomy: Taxonomy
) -> None:
    """
    Validates that every model declaring a taxonomy_category:
      1. References a category that exists in the taxonomy.
      2. Exposes all variables, parameters, ports, port-field-definitions,
         constraints, binding-constraints, extra-outputs and properties listed
         in that taxonomy category.

    Raises ValueError describing the first violation found.
    """
    categories: Dict[str, TaxonomyCategory] = {c.id: c for c in taxonomy.categories}

    by_id: Callable = lambda x: x.id

    # Each entry maps a human-readable field-group name to the required items
    # (from the taxonomy category) and the items exposed by the model, plus the
    # function identifying a model-side item within that group. Taxonomy items are
    # homogeneous (``TaxonomyItem``) and always identified by their ``id``.
    def field_groups(
        category: TaxonomyCategory, model_schema: "ModelSchema"
    ) -> List[tuple]:
        port_field_key: Callable = lambda d: f"{d.port}.{d.field}"
        return [
            ("variable", category.variables, model_schema.variables, by_id),
            ("parameter", category.parameters, model_schema.parameters, by_id),
            ("port", category.ports, model_schema.ports, by_id),
            (
                "port-field-definition",
                category.port_field_definitions,
                model_schema.port_field_definitions,
                port_field_key,
            ),
            ("constraint", category.constraints, model_schema.constraints, by_id),
            (
                "binding-constraint",
                category.binding_constraints,
                model_schema.binding_constraints,
                by_id,
            ),
            (
                "extra-output",
                category.extra_outputs,
                model_schema.extra_outputs or [],
                by_id,
            ),
            ("property", category.properties, model_schema.properties, by_id),
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
