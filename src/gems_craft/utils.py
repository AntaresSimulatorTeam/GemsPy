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

"""
Module for technical utilities.
"""

import json
import pathlib
from typing import Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
Supplier = Callable[[], T]


def require_not_none(obj: Any, msg: Optional[str] = None) -> None:
    """
    Raises a ValueError if obj is None.
    """
    if obj is None:
        error_msg = msg if msg else "Object must not be None"
        raise ValueError(error_msg)


def get_or_add(dictionary: Dict[K, V], key: K, default_factory: Supplier[V]) -> V:
    """
    Gets value from dictionary, or inserts it if it does not exist.

    Factory is only called if value is absent.
    """
    value = dictionary.get(key)
    if not value:
        value = default_factory()
        dictionary[key] = value
    return value


def get_root_section(tree: Any, root_key: str, source: Optional[str] = None) -> Any:
    """
    Returns tree[root_key], the content of a YAML document with a single root key.

    Raises a ValueError if root_key is missing, if the root holds other keys,
    which would otherwise be silently ignored, or if the content of root_key
    is empty or not a mapping. source, the file name, is added to the message.
    """
    where = f"{root_key} file '{source}'" if source else f"{root_key} file"
    if not isinstance(tree, dict) or root_key not in tree:
        raise ValueError(f"Missing root key '{root_key}' in {where}.")
    unknown_keys = [key for key in tree if key != root_key]
    if unknown_keys:
        plural = "s" if len(unknown_keys) > 1 else ""
        keys = ", ".join(f"'{key}'" for key in unknown_keys)
        raise ValueError(
            f"Unknown root key{plural} {keys} in {where};"
            f" did you mean to put {'them' if plural else 'it'} under '{root_key}:'?"
        )
    section = tree[root_key]
    if not isinstance(section, dict):
        raise ValueError(f"'{root_key}:' in {where} is empty or not a mapping.")
    return section


def serialize(filename: str, message: str, path: pathlib.Path) -> None:
    """
    Write message to path/filename

    Will throw an exception if it fails to create dir or ro open file
    """
    path.mkdir(parents=True, exist_ok=True)
    file = (path / filename).open(mode="w")

    with file:
        file.write(message)


def serialize_json(
    filename: str, message: Dict[str, Any], path: pathlib.Path, indentation: int = 4
) -> None:
    serialize(filename, json.dumps(message, indent=indentation), path)


def read_json(filename: str, path: pathlib.Path) -> Dict[str, Any]:
    with (path / filename).open() as file:
        data = json.load(file)
    return data


"""
Only _to_kebab adn ModifiedBaseModel are used in the project.
"""


# Design note: actual parsing and validation is delegated to pydantic models
def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class ModifiedBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_kebab, extra="forbid", populate_by_name=True
    )
