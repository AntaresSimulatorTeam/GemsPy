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


def _port_lib(definition: str) -> str:
    return _lib(
        _FLOW_PORT_TYPE,
        "sets: [{id: fuel}]",
        "models:",
        "  - id: m",
        "    sets: [{id: segment}]",
        "    parameters: [{id: gen, indexed-by: [segment]}, {id: cap, indexed-by: [fuel]}]",
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
    "constraint bound with an invalid set index": (
        _model_lib(
            "parameters: [{id: p}], variables: [{id: x}], "
            "constraints: [{id: c, expression: x, lower-bound: 'p[fuel]'}]",
            _FUEL,
        ),
        "not indexed by 'fuel'",
    ),
    "constraint bound indexed by a set the expression lacks": (
        _model_lib(
            f"{_P_FUEL}, variables: [{{id: x}}], "
            "constraints: [{id: c, expression: x, upper-bound: 'cap[fuel]'}]",
            _FUEL,
        ),
        "that constraint 'c' is not indexed by",
    ),
    "constraint bound indexed by a set the expression has": (
        _model_lib(
            f"{_P_FUEL}, variables: [{{id: x, indexed-by: [fuel]}}], "
            "constraints: [{id: c, expression: 'x[fuel]', upper-bound: 'cap[fuel]'}]",
            _FUEL,
        ),
        None,
    ),
    "bound indexed by a set the variable lacks": (
        _model_lib(
            f"{_P_FUEL}, variables: [{{id: x, upper-bound: 'cap[fuel]'}}]", _FUEL
        ),
        "that variable 'x' is not indexed by",
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
    # global sets, unlike local ones, may cross a port
    "port field indexed by global set": (_port_lib("cap[fuel]"), None),
    "set index on dual of constraint with sum_connections": (
        _lib(
            _FLOW_PORT_TYPE,
            "models:",
            "  - id: m",
            "    sets: [{id: segment}]",
            "    parameters: [{id: p, indexed-by: [segment]}]",
            "    variables: [{id: x}]",
            "    ports: [{id: in_port, type: flow}]",
            "    constraints:",
            "      - {id: c, expression: 'x + sum_connections(in_port.flow) <= p'}",
            "    extra-outputs: [{id: o, expression: '(dual(c))[segment=0]'}]",
        ),
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
