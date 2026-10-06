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

import io
from typing import Optional

import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.validation import check_custom_sets

_LIB = """
library:
  id: setlib
  sets: [{id: fuel}]
  models:
    - id: m
      sets: [{id: segment}]
      parameters: [{id: p, indexed-by: [fuel, segment]}]
"""

_GLOBAL = "[{id: fuel, elements: [gas, coal]}]"
_LOCAL = "[{id: segment, elements: 0..2}]"


def _system(global_sets: Optional[str] = _GLOBAL, local_sets: Optional[str] = _LOCAL):
    lines = ["system:"]
    if global_sets:
        lines.append(f"  sets: {global_sets}")
    lines += ["  components:", "    - id: A", "      model: setlib.m"]
    if local_sets:
        lines.append(f"      sets: {local_sets}")
    lines.append("      parameters: [{id: p, value: 1.0}]")
    return parse_yaml_system(io.StringIO("\n".join(lines)))


def _check(system, lib_yaml: str = _LIB) -> None:
    lib_dict = resolve_library([parse_yaml_library(io.StringIO(lib_yaml))])
    model_dict = {}
    for library in lib_dict.values():
        model_dict |= library.models
    check_custom_sets(system, model_dict, lib_dict)


def test_valid_global_and_local_set_instantiation_ok() -> None:
    _check(_system())


@pytest.mark.parametrize(
    "system_args, error",
    [
        ({"global_sets": None}, "not instantiated"),
        ({"local_sets": None}, "missing instantiation"),
        (
            {"global_sets": "[{id: fuel, elements: [gas, gas]}]"},
            "duplicate elements",
        ),
        (
            {"global_sets": "[{id: fuel, elements: [gas, 'co|al']}]"},
            r"containing '\|'",
        ),
        (
            {
                "global_sets": "[{id: fuel, elements: [gas]}, {id: ghost, elements: [a]}]"
            },
            "not declared by any library",
        ),
        (
            {
                "local_sets": "[{id: segment, elements: 0..2}, {id: ghost, elements: [a]}]"
            },
            "not local sets of its model",
        ),
    ],
    ids=[
        "missing global set",
        "missing local set",
        "duplicate elements",
        "pipe in element",
        "undeclared global set",
        "non-local set on component",
    ],
)
def test_invalid_set_instantiation_raises(system_args: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _check(_system(**system_args))


def test_unused_global_set_not_required_to_be_instantiated() -> None:
    """A declared global set that no model references via indexed_by needn't be
    instantiated (like a time-dependent parameter needn't actually vary)."""
    lib = """
library:
  id: setlib
  sets: [{id: fuel}]
  models: [{id: m, parameters: [{id: p}]}]
"""
    _check(_system(global_sets=None, local_sets=None), lib)
