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


def _fuel_lib(model: str) -> str:
    """A library declaring the global set 'fuel' and a single model `m`."""
    return (
        f"library: {{id: setlib, sets: [{{id: fuel}}], models: [{{id: m, {model}}}]}}"
    )


_NO_SETS = {"global_sets": None, "local_sets": None}

# (system_args, library yaml, expected error regex or None when the check passes)
_CASES = {
    "valid instantiation": ({}, _LIB, None),
    "missing global set": ({"global_sets": None}, _LIB, "not instantiated"),
    "missing local set": ({"local_sets": None}, _LIB, "missing instantiation"),
    "duplicate elements": (
        {"global_sets": "[{id: fuel, elements: [gas, gas]}]"},
        _LIB,
        "duplicate elements",
    ),
    "duplicate elements as text": (
        {"global_sets": "[{id: fuel, elements: [1, '1']}]"},
        _LIB,
        "duplicate elements",
    ),
    "pipe in element": (
        {"global_sets": "[{id: fuel, elements: [gas, 'co|al']}]"},
        _LIB,
        r"containing '\|'",
    ),
    "undeclared global set": (
        {"global_sets": "[{id: fuel, elements: [gas]}, {id: ghost, elements: [a]}]"},
        _LIB,
        "not declared by any library",
    ),
    "non-local set on component": (
        {"local_sets": "[{id: segment, elements: 0..2}, {id: ghost, elements: [a]}]"},
        _LIB,
        "not local sets of its model",
    ),
    # like a time-dependent parameter needn't actually vary
    "unused global set not instantiated": (
        _NO_SETS,
        _fuel_lib("parameters: [{id: p}]"),
        None,
    ),
    "set used only by sum_over not instantiated": (
        _NO_SETS,
        _fuel_lib(
            "parameters: [{id: p}], "
            "extra-outputs: [{id: o, expression: 'sum_over(fuel, p)'}]"
        ),
        "not instantiated",
    ),
}


@pytest.mark.filterwarnings("ignore:sum_over")
@pytest.mark.parametrize("system_args, lib, error", _CASES.values(), ids=_CASES.keys())
def test_check_custom_sets(system_args: dict, lib: str, error: Optional[str]) -> None:
    system = _system(**system_args)
    if error is None:
        _check(system, lib)
    else:
        with pytest.raises(ValueError, match=error):
            _check(system, lib)
