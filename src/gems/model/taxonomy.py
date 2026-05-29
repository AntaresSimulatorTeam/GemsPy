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
from typing import Dict, List, Optional

import yaml
from pydantic import Field

from gems.model.parsing import LibrarySchema
from gems.utils import ModifiedBaseModel


class TaxonomyItem(ModifiedBaseModel):
    id: str


class TaxonomyCategory(ModifiedBaseModel):
    id: str
    parent_category: Optional[str] = None
    variables: List[TaxonomyItem] = Field(default_factory=list)
    parameters: List[TaxonomyItem] = Field(default_factory=list)
    ports: List[TaxonomyItem] = Field(default_factory=list)
    constraints: List[TaxonomyItem] = Field(default_factory=list)
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


def check_library_against_taxonomy(library: LibrarySchema, taxonomy: Taxonomy) -> None:
    """
    Validates that every model declaring a taxonomy_category:
      1. References a category that exists in the taxonomy.
      2. Exposes all port IDs listed in that taxonomy category.

    Raises ValueError describing the first violation found.
    """
    categories: Dict[str, TaxonomyCategory] = {c.id: c for c in taxonomy.categories}

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
        model_port_ids = {p.id for p in model_schema.ports}
        missing = sorted({item.id for item in category.ports} - model_port_ids)
        if missing:
            raise ValueError(
                f"Model '{model_schema.id}' (taxonomy-category: '{cat_id}') is missing "
                f"port(s) required by the taxonomy: {missing}."
            )
