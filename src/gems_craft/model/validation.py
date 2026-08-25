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

"""Cross-validation of parsed model libraries.

Kept apart from `parsing.py` (which only reads YAML into schemas) and from
`taxonomy.py` (which only holds the taxonomy data and reads it from disk), so
that reading and validating stay separate concerns — mirroring
`optim_config/parsing.py` and `optim_config/validation.py`.
"""

from typing import Callable, Dict, List, Optional

from gems_craft.model.parsing import LibrarySchema, ModelSchema
from gems_craft.model.taxonomy import Taxonomy, TaxonomyCategory, TaxonomyItem


def validate_libraries_against_taxonomy(
    libraries: List[LibrarySchema], taxonomy: Optional[Taxonomy]
) -> None:
    """Check every library declaring a ``taxonomy`` field against ``taxonomy``.

    Libraries that do not declare a taxonomy are left alone, even when some of
    their models carry a ``taxonomy-category``.

    Raises ValueError if a declaring library has no taxonomy to check against,
    if it declares a different taxonomy id, or if any of its models violates it.
    """
    for library in libraries:
        if library.taxonomy is None:
            continue
        declared = f"Library '{library.id}' declares taxonomy '{library.taxonomy}'"
        if taxonomy is None:
            raise ValueError(
                f"{declared} but no taxonomy was provided to check it against."
            )
        if library.taxonomy != taxonomy.id:
            raise ValueError(f"{declared} but was checked against '{taxonomy.id}'.")
        check_library_against_taxonomy(library, taxonomy)


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

    by_id: Callable = lambda x: x.id

    # Each entry maps a human-readable field-group name to the required items
    # (from the taxonomy category) and the items exposed by the model, plus the
    # function identifying a model-side item within that group. Taxonomy items are
    # homogeneous (``TaxonomyItem``) and always identified by their ``id``.
    def field_groups(
        category: TaxonomyCategory, model_schema: ModelSchema
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
