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
from typing import Set

import pytest

from gems.expression import ExpressionNode, literal, param, print_expr, var
from gems.expression.equality import expressions_equal
from gems.expression.expression import port_field
from gems.expression.parsing.parse_expression import (
    AntaresParseException,
    ModelIdentifiers,
    parse_expression,
)


@pytest.mark.parametrize(
    "variables, parameters, expression_str, expected",
    [
        ({}, {}, "1 + 2", literal(1) + 2),
        ({}, {}, "1 - 2", literal(1) - 2),
        ({}, {}, "1 - 3 + 4 - 2", literal(1) - 3 + 4 - 2),
        (
            {"x"},
            {"p"},
            "1 + 2 * x = p",
            literal(1) + 2 * var("x") == param("p"),
        ),
        (
            {},
            {},
            "port.f <= 0",
            port_field("port", "f") <= 0,
        ),
        ({"x"}, {}, "sum(x)", var("x").time_sum()),
        ({"x"}, {}, "x[-1]", var("x").eval(-literal(1))),
        ({"x"}, {}, "x[1]", var("x").eval(1)),
        ({"x"}, {}, "x[t-1]", var("x").shift(-literal(1))),
        (
            {"x"},
            {},
            "x[t-1+1]",
            var("x").shift(-literal(1) + literal(1)),
        ),
        (
            {"x"},
            {"d"},
            "x[t-d+1]",
            var("x").shift(-param("d") + literal(1)),
        ),
        (
            {"x"},
            {"d"},
            "x[t-2*d+1]",
            var("x").shift(-literal(2) * param("d") + literal(1)),
        ),
        (
            {"x"},
            {"d"},
            "x[t-1+d*2]",
            var("x").shift(-literal(1) + param("d") * literal(2)),
        ),
        (
            {"x"},
            {"d"},
            "x[t-2-d+1]",
            var("x").shift(-literal(2) - param("d") + literal(1)),
        ),
        (
            {"x"},
            {},
            "sum(t-1..t+5, x)",
            var("x").time_sum(-literal(1), literal(5)),
        ),
        (
            {"x"},
            {},
            "sum(t-1..t, x)",
            var("x").time_sum(-literal(1), literal(0)),
        ),
        (
            {"x"},
            {},
            "sum(t..t+5, x)",
            var("x").time_sum(literal(0), literal(5)),
        ),
        ({"x"}, {}, "x[t]", var("x")),
        ({"x"}, {"p"}, "x[t+p]", var("x").shift(param("p"))),
        ({}, {}, "sum_connections(port.f)", port_field("port", "f").sum_connections()),
        (
            {"level", "injection", "withdrawal"},
            {"inflows", "efficiency"},
            "level - level[-1] - efficiency * injection + withdrawal = inflows",
            var("level")
            - var("level").eval(-literal(1))
            - param("efficiency") * var("injection")
            + var("withdrawal")
            == param("inflows"),
        ),
        (
            {"nb_start", "nb_on"},
            {"d_min_up"},
            "sum(t - d_min_up + 1 .. t, nb_start) <= nb_on",
            var("nb_start").time_sum(-param("d_min_up") + 1, literal(0))
            <= var("nb_on"),
        ),
        (
            {"generation"},
            {"cost"},
            "expec(sum(cost * generation))",
            (param("cost") * var("generation")).time_sum().expec(),
        ),
    ],
)
def test_parsing_visitor(
    variables: Set[str],
    parameters: Set[str],
    expression_str: str,
    expected: ExpressionNode,
) -> None:
    identifiers = ModelIdentifiers(variables, parameters)
    expr = parse_expression(expression_str, identifiers)
    print()
    print(print_expr(expr))
    assert expressions_equal(expr, expected)


@pytest.mark.parametrize(
    "expression_str",
    [
        "1**3",
        "1 6",
        "x[t+1-t]",
        "x[2*t]",
        "x[t 4]",
    ],
)
def test_parse_cancellation_should_throw(expression_str: str) -> None:
    # Console log error is displayed !
    identifiers = ModelIdentifiers(
        variables={"x"},
        parameters=set(),
    )

    with pytest.raises(
        AntaresParseException,
        match=r"An error occurred during parsing: ParseCancellationException",
    ):
        parse_expression(expression_str, identifiers)
