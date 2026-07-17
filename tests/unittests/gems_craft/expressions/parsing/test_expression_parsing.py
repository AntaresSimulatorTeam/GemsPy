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

from gems_craft.expression import ExpressionNode, literal, param, print_expr, var
from gems_craft.expression.equality import expressions_equal
from gems_craft.expression.expression import (
    DualNode,
    ReducedCostNode,
    maximum,
    minimum,
    port_field,
)
from gems_craft.expression.parsing.parse_expression import (
    ModelIdentifiers,
    ParsingException,
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
        (
            {},
            {"p"},
            "floor(p)",
            param("p").floor(),
        ),
        (
            {},
            {"p"},
            "ceil(p)",
            param("p").ceil(),
        ),
        (
            {},
            {"p"},
            "abs(p)",
            param("p").abs(),
        ),
        (
            {},
            {"p"},
            "round(p)",
            param("p").round(),
        ),
        (
            {},
            {"p", "q"},
            "abs(p - q)",
            (param("p") - param("q")).abs(),
        ),
        (
            {},
            {"p", "q"},
            "round(p / q)",
            (param("p") / param("q")).round(),
        ),
        (
            {},
            {"p", "q"},
            "max(0, abs(p - q))",
            maximum(literal(0), (param("p") - param("q")).abs()),
        ),
        (
            {},
            {"a", "b"},
            "max(a, b)",
            maximum(param("a"), param("b")),
        ),
        (
            {},
            {"a", "b"},
            "min(a, b)",
            minimum(param("a"), param("b")),
        ),
        (
            {},
            {"p", "q"},
            "ceil(p/q)",
            (param("p") / param("q")).ceil(),
        ),
        (
            {},
            {"p", "q"},
            "max(0, ceil(p/q))",
            maximum(literal(0), (param("p") / param("q")).ceil()),
        ),
        (
            {},
            {"a", "b", "c"},
            "max(a, b, c)",
            maximum(param("a"), param("b"), param("c")),
        ),
        (
            {},
            {"a", "b", "c"},
            "min(a, b, c)",
            minimum(param("a"), param("b"), param("c")),
        ),
        (
            {"x", "y"},
            {},
            "(x + y)[t-1]",
            (var("x") + var("y")).shift(-literal(1)),
        ),
        (
            {},
            {"p", "q"},
            "(ceil(p/q))[t]",
            (param("p") / param("q")).ceil(),
        ),
        (
            {},
            {"p_max_cluster", "p_max_unit"},
            "max(0, (ceil(p_max_cluster/p_max_unit))[t-1] - (ceil(p_max_cluster/p_max_unit)))",
            maximum(
                literal(0),
                (param("p_max_cluster") / param("p_max_unit")).ceil().shift(-literal(1))
                - (param("p_max_cluster") / param("p_max_unit")).ceil(),
            ),
        ),
        (
            {"x", "y"},
            {},
            "(x + y)[1]",
            (var("x") + var("y")).eval(literal(1)),
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
    "variables, parameters, constraints, expression_str, expected",
    [
        (set(), set(), {"balance"}, "dual(balance)", DualNode("balance")),
        ({"p"}, set(), set(), "reduced_cost(p)", ReducedCostNode("p")),
    ],
)
def test_parsing_dual_and_reduced_cost(
    variables: set,
    parameters: set,
    constraints: set,
    expression_str: str,
    expected: ExpressionNode,
) -> None:
    identifiers = ModelIdentifiers(variables, parameters, constraints)
    expr = parse_expression(expression_str, identifiers)
    assert expressions_equal(expr, expected)


def test_parse_dual_unknown_constraint_raises() -> None:
    identifiers = ModelIdentifiers(set(), set(), {"other"})
    with pytest.raises(ParsingException, match="not a constraint"):
        parse_expression("dual(balance)", identifiers)


def test_parse_reduced_cost_unknown_variable_raises() -> None:
    identifiers = ModelIdentifiers({"x"}, set(), set())
    with pytest.raises(ParsingException, match="not a variable"):
        parse_expression("reduced_cost(p)", identifiers)


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
        ParsingException,
        match=r"An error occurred during parsing: ParseCancellationException",
    ):
        parse_expression(expression_str, identifiers)
