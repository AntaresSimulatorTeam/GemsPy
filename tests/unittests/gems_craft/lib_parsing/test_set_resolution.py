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

import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library

_FLOW_PORT_TYPE = "port-types: [{id: flow, fields: [{id: flow}]}]"


def _lib(*lines: str) -> str:
    """A library `l` with the given (already indented by 2) top-level lines."""
    return "library:\n  id: l\n" + "".join(f"  {line}\n" for line in lines)


def _model_lib(model: str, *top: str) -> str:
    return _lib(*top, "models:", f"  - {{id: m, {model}}}")


def _port_lib(definition: str, *top: str) -> str:
    return _lib(
        _FLOW_PORT_TYPE,
        *top,
        "models:",
        "  - id: m",
        "    sets: [{id: segment}]",
        "    parameters: [{id: gen, indexed-by: [segment]}]",
        "    ports: [{id: injection_port, type: flow}]",
        "    port-field-definitions:",
        f"      - {{port: injection_port, field: flow, definition: '{definition}'}}",
    )


_FUEL = "sets: [{id: fuel}]"
_P_FUEL = "parameters: [{id: cap, indexed-by: [fuel]}]"

# (library yaml, expected error regex or None when resolution must succeed)
_CASES = {
    "local set collides with global set": (
        _model_lib("sets: [{id: fuel}]", _FUEL),
        "collide with",
    ),
    "local set unrelated to global set": (
        _model_lib("sets: [{id: segment}]", _FUEL),
        None,
    ),
    "indexed-by undeclared set": (
        _model_lib("parameters: [{id: p, indexed-by: [fuel]}]"),
        "undeclared set",
    ),
    "set index on variable not indexed by it": (
        _model_lib(
            "parameters: [{id: p}], variables: [{id: x}], "
            "constraints: [{id: c, expression: 'x[fuel] <= p'}]",
            _FUEL,
        ),
        "not indexed by 'fuel'",
    ),
    "set index on variable indexed by it": (
        _model_lib(
            "parameters: [{id: p}], variables: [{id: x, indexed-by: [fuel]}], "
            "constraints: [{id: c, expression: 'x[fuel] <= p'}]",
            _FUEL,
        ),
        None,
    ),
    "time index on non time-dependent parameter": (
        _model_lib(
            "parameters: [{id: p}], variables: [{id: x}], "
            "constraints: [{id: c, expression: 'p[5] <= x'}]"
        ),
        "not time-dependent",
    ),
    "time index on time-dependent parameter": (
        _model_lib(
            "parameters: [{id: p, time-dependent: true}], variables: [{id: x}], "
            "constraints: [{id: c, expression: 'p[5] <= x'}]"
        ),
        None,
    ),
    "relative shift on non time-dependent parameter": (
        _model_lib(
            "parameters: [{id: p}], variables: [{id: x}], "
            "constraints: [{id: c, expression: 'p[t+1] <= x'}]"
        ),
        "not time-dependent",
    ),
    "relative shift on time-dependent parameter": (
        _model_lib(
            "parameters: [{id: p, time-dependent: true}], variables: [{id: x}], "
            "constraints: [{id: c, expression: 'p[t+1] <= x'}]"
        ),
        None,
    ),
    "bound indexed by a set the variable lacks": (
        _model_lib(
            f"{_P_FUEL}, variables: [{{id: x, upper-bound: 'cap[fuel]'}}]", _FUEL
        ),
        "not declared in 'x'",
    ),
    "bound summed over a set the variable lacks": (
        _model_lib(
            f"{_P_FUEL}, variables: [{{id: x, upper-bound: 'sum_over(fuel, cap[fuel])'}}]",
            _FUEL,
        ),
        None,
    ),
    "bound indexed by a set the variable has": (
        _model_lib(
            f"{_P_FUEL}, variables: "
            "[{id: x, indexed-by: [fuel], upper-bound: 'cap[fuel]'}]",
            _FUEL,
        ),
        None,
    ),
    "port field indexed by local set": (
        _port_lib("gen[segment]"),
        "wrap it in sum_over",
    ),
    "port field summed over local set": (
        _port_lib("sum_over(segment, gen[segment])"),
        None,
    ),
}


@pytest.mark.parametrize("yaml, error", _CASES.values(), ids=_CASES.keys())
def test_set_resolution(yaml: str, error: "str | None") -> None:
    lib = parse_yaml_library(io.StringIO(yaml))
    if error is None:
        resolve_library([lib])
    else:
        with pytest.raises(ValueError, match=error):
            resolve_library([lib])


def test_port_field_indexed_by_global_set_is_exempt() -> None:
    """Global sets, unlike local ones, may cross a port."""
    lib = _lib(
        _FLOW_PORT_TYPE,
        _FUEL,
        "models:",
        "  - id: m",
        f"    {_P_FUEL}",
        "    ports: [{id: injection_port, type: flow}]",
        "    port-field-definitions:",
        "      - {port: injection_port, field: flow, definition: 'cap[fuel]'}",
    )
    resolve_library([parse_yaml_library(io.StringIO(lib))])


def test_duplicate_global_set_across_dependency_raises() -> None:
    lib_a = parse_yaml_library(io.StringIO(_lib(_FUEL, "models: [{id: m}]")))
    lib_b = parse_yaml_library(
        io.StringIO(
            "library:\n  id: libB\n  dependencies: [l]\n"
            "  sets: [{id: fuel}]\n  models: [{id: n}]\n"
        )
    )
    with pytest.raises(Exception, match="defined twice"):
        resolve_library([lib_a, lib_b])
