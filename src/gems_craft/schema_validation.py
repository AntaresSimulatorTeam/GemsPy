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

from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from yaml import safe_load

_SchemaName = Literal["model-library", "system", "optim-config"]
_SCHEMA_FILES = (
    "common.schema.yml",
    "model-library.schema.yml",
    "system.schema.yml",
    "optim-config.schema.yml",
)


@lru_cache(maxsize=1)
def _schema_registry() -> tuple[dict[str, dict[str, Any]], Registry]:
    schema_dir = files("gems_craft").joinpath("schemas")
    schemas = {
        filename.removesuffix(".schema.yml"): safe_load(
            schema_dir.joinpath(filename).read_text(encoding="utf-8")
        )
        for filename in _SCHEMA_FILES
    }

    registry = Registry().with_resources(
        [
            (
                schema["$id"],
                Resource.from_contents(schema, default_specification=DRAFT202012),
            )
            for schema in schemas.values()
        ]
    )
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    return schemas, registry


@lru_cache(maxsize=3)
def _validator(schema_name: _SchemaName) -> Draft202012Validator:
    schemas, registry = _schema_registry()
    return Draft202012Validator(schemas[schema_name], registry=registry)


def validate_input_schema(
    schema_name: _SchemaName,
    instance: Any,
) -> None:
    """Validate an input against its bundled GEMS schema."""
    try:
        _validator(schema_name).validate(instance)
    except JsonSchemaValidationError as error:
        raise ValueError(f"{error.json_path}: {error.message}") from error
