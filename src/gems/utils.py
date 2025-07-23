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

from pydantic import BaseModel

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


# Design note: actual parsing and validation is delegated to pydantic models
def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class ModifiedBaseModel(BaseModel):
    class Config:
        alias_generator = _to_kebab
        extra = "forbid"
        populate_by_name = True
