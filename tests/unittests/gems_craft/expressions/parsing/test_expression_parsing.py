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
    LowerBoundNode,
    ReducedCostNode,
    SetIndexNode,
    UpperBoundNode,
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
        # legacy absolute-time indexing by a parameter
        ({"x"}, {"p"}, "x[p]", var("x").eval(param("p"))),
        ({"x"}, {"p"}, "x[p+1]", var("x").eval(param("p") + 1)),
        ({"x"}, {"p"}, "x[p-1]", var("x").eval(param("p") - 1)),
        (
            {"x", "y"},
            {"p", "q"},
            "(x+y)[p+q]",
            (var("x") + var("y")).eval(param("p") + param("q")),
        ),
        (
            {"x"},
            {"p", "q"},
            "x[p - q*2]",
            # leading signs bind first in a shift expression: p + (-q)*2
            var("x").eval(param("p") + (-param("q")) * 2),
        ),
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
        ({"p"}, set(), set(), "lower_bound(p)", LowerBoundNode("p")),
        ({"p"}, set(), set(), "upper_bound(p)", UpperBoundNode("p")),
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


def test_parse_lower_bound_unknown_variable_raises() -> None:
    identifiers = ModelIdentifiers({"x"}, set(), set())
    with pytest.raises(ParsingException, match="not a variable"):
        parse_expression("lower_bound(p)", identifiers)


def test_parse_upper_bound_unknown_variable_raises() -> None:
    identifiers = ModelIdentifiers({"x"}, set(), set())
    with pytest.raises(ParsingException, match="not a variable"):
        parse_expression("upper_bound(p)", identifiers)


@pytest.mark.parametrize(
    "sets, expression_str, expected",
    [
        ({"fuel"}, "x[fuel]", var("x").set_index("fuel")),
        ({"fuel"}, "x[fuel=2]", var("x").set_index("fuel", position=literal(2))),
        ({"fuel"}, "x[t=2]", var("x").eval(literal(2))),
        (
            {"fuel"},
            "x[t=2, fuel=1]",
            var("x").eval(literal(2)).set_index("fuel", position=literal(1)),
        ),
        (
            {"fuel"},
            "x[fuel+1]",
            var("x").set_index("fuel", relative_shift=literal(1)),
        ),
        (
            {"fuel"},
            "x[fuel-1]",
            var("x").set_index("fuel", relative_shift=-literal(1)),
        ),
        ({"fuel"}, "x[fuel+0]", var("x").set_index("fuel")),
        pytest.param(
            {"fuel"},
            "x[t+1, fuel]",
            var("x").shift(literal(1)).set_index("fuel"),
            id="time-shift-and-set",
        ),
        pytest.param(
            {"fuel"},
            "x[fuel+1, t-1]",
            var("x").shift(-literal(1)).set_index("fuel", relative_shift=literal(1)),
            id="set-shift-and-time-shift-any-order",
        ),
        pytest.param(
            {"fuel", "segment"},
            "x[fuel=p, segment+p]",
            var("x")
            .set_index("fuel", position=param("p"))
            .set_index("segment", relative_shift=param("p")),
            id="parameter-in-set-terms",
        ),
        pytest.param(
            {"fuel", "segment"},
            "sum_over(fuel, sum_over(segment, x))",
            var("x").sum_over("segment").sum_over("fuel"),
            id="nested-sum-over",
        ),
        pytest.param(
            {"fuel"},
            "sum_over(fuel, x[fuel])",
            var("x").set_index("fuel").sum_over("fuel"),
            id="sum-over-set-index",
        ),
        pytest.param(
            {"fuel"},
            "sum(t-1..t+1, sum_over(fuel, x[fuel]))",
            var("x")
            .set_index("fuel")
            .sum_over("fuel")
            .time_sum(-literal(1), literal(1)),
            id="sum-over-in-time-window",
        ),
        (
            {"segment", "fuel"},
            "x[segment=2, fuel=1]",
            var("x")
            .set_index("fuel", position=literal(1))
            .set_index("segment", position=literal(2)),
        ),
        (
            {"fuel"},
            "x[fuel=3, 2]",
            var("x").eval(literal(2)).set_index("fuel", position=literal(3)),
        ),
        (
            {"fuel"},
            "sum_over(fuel, x)",
            var("x").sum_over("fuel"),
        ),
        (
            {"fuel"},
            "(x + p)[fuel]",
            (var("x") + param("p")).set_index("fuel"),
        ),
    ],
)
def test_parsing_visitor_with_sets(
    sets: Set[str], expression_str: str, expected: ExpressionNode
) -> None:
    identifiers = ModelIdentifiers(
        variables={"x"}, parameters={"p"}, constraints=set(), sets=sets
    )
    expr = parse_expression(expression_str, identifiers)
    assert expressions_equal(expr, expected)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ("x[t=2]", "x[2]"),  # keyword time form == legacy bare form
        ("x[segment=2, fuel=1]", "x[fuel=1, segment=2]"),  # term order is irrelevant
        ("x[fuel=3, 2]", "x[2, fuel=3]"),
    ],
)
def test_parsing_equivalent_index_forms(lhs: str, rhs: str) -> None:
    identifiers = ModelIdentifiers(
        variables={"x"}, parameters=set(), constraints=set(), sets={"segment", "fuel"}
    )
    assert expressions_equal(
        parse_expression(lhs, identifiers), parse_expression(rhs, identifiers)
    )


@pytest.mark.parametrize(
    "expression_str, match",
    [
        ("x[2, 3]", "more than one term.*got '2' and '3' in '\\[2,3\\]'"),
        ("x[t, p]", "more than one term.*got 't' and 'p' in '\\[t,p\\]'"),
        ("x[fuel, fuel=2]", "indexed more than once"),
        ("x[notaset]", "not a valid variable or parameter.*declared sets"),
        ("x[fule]", "fule is not a valid.*declared sets: \\['fuel', 'segment'\\]"),
        ("x[fuel*2]", "'fuel' is a set.*X\\[fuel=k\\].*sum_over\\(fuel"),
        ("x[2*fuel+1]", "'fuel' is a set"),
        ("x[fuel=2, fuel+1]", "indexed more than once"),
        ("x[nonexistent=2]", "neither 't' nor a declared set"),
        ("sum(fuel-p..fuel, x)", "must start with 't'"),
        ("sum(t-p..fuel, x)", "must start with 't'"),
        ("sum(foo..t, x)", "must start with 't'"),
        ("sum_over(other, x)", "'other' is not a declared set"),
        ("sum_over(p, x)", "'p' is not a declared set"),
        ("x[p=2]", "neither 't' nor a declared set"),
        ("x[fuel>=2]", "'>=' is not valid in the keyword index form"),
        ("x[t<=2]", "'<=' is not valid in the keyword index form"),
    ],
)
def test_parsing_custom_set_indexing_raises(expression_str: str, match: str) -> None:
    identifiers = ModelIdentifiers(
        variables={"x"}, parameters={"p"}, constraints=set(), sets={"fuel", "segment"}
    )
    with pytest.raises(ParsingException, match=match):
        parse_expression(expression_str, identifiers)


def test_set_index_node_rejects_position_and_relative_shift() -> None:
    with pytest.raises(ValueError, match="both an explicit position"):
        SetIndexNode(var("x"), "fuel", position=literal(1), relative_shift=literal(1))


@pytest.mark.parametrize(
    "variables, parameters, sets, match",
    [
        ({"v"}, set(), {"v"}, "clash"),
        (set(), {"fuel"}, {"fuel"}, "clash"),
        (set(), set(), {"t"}, "reserved"),
    ],
)
def test_model_identifiers_reject_ambiguous_sets(
    variables: Set[str], parameters: Set[str], sets: Set[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ModelIdentifiers(variables, parameters, sets=sets)


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
