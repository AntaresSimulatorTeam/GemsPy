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

import pytest

from gems.expression import (
    AdditionNode,
    DivisionNode,
    EvaluationContext,
    EvaluationVisitor,
    ExpressionNode,
    LiteralNode,
    ParameterNode,
    PrinterVisitor,
    ValueProvider,
    VariableNode,
    literal,
    param,
    sum_expressions,
    var,
    visit,
)
from gems.expression.equality import expressions_equal


def test_ast() -> None:
    add_node = AdditionNode([LiteralNode(1), VariableNode("x")])
    expr = DivisionNode(add_node, ParameterNode("p"))

    assert visit(expr, PrinterVisitor()) == "((1 + x) / p)"

    context = EvaluationContext(variables={"x": 3}, parameters={"p": 4})
    assert visit(expr, EvaluationVisitor(context)) == 1


def test_operators() -> None:
    x = var("x")
    p = param("p")
    expr: ExpressionNode = (5 * x + 3) / p - 2

    assert visit(expr, PrinterVisitor()) == "((((5.0 * x) + 3.0) / p) - 2.0)"

    context = EvaluationContext(variables={"x": 3}, parameters={"p": 4})
    assert visit(expr, EvaluationVisitor(context)) == pytest.approx(2.5, 1e-16)

    assert visit(-expr, EvaluationVisitor(context)) == pytest.approx(-2.5, 1e-16)


def test_sum_expressions() -> None:
    assert expressions_equal(sum_expressions([]), literal(0))
    assert expressions_equal(sum_expressions([literal(1)]), literal(1))
    assert expressions_equal(sum_expressions([literal(1), var("x")]), 1 + var("x"))
    assert expressions_equal(
        sum_expressions([literal(1), var("x"), param("p")]), 1 + (var("x") + param("p"))
    )


def test_floor_ceil_max_min() -> None:
    from gems.expression.expression import maximum, minimum

    context = EvaluationContext(parameters={"p": 2.7, "q": 1.3})

    assert visit(param("p").floor(), EvaluationVisitor(context)) == 2.0
    assert visit(param("p").ceil(), EvaluationVisitor(context)) == 3.0
    assert visit(
        maximum(param("p"), param("q")), EvaluationVisitor(context)
    ) == pytest.approx(2.7)
    assert visit(
        minimum(param("p"), param("q")), EvaluationVisitor(context)
    ) == pytest.approx(1.3)
    assert visit(
        maximum(literal(0), param("q")), EvaluationVisitor(context)
    ) == pytest.approx(1.3)
    assert visit(maximum(literal(0), -param("p")), EvaluationVisitor(context)) == 0.0
    # variadic (3+ operands)
    assert visit(
        maximum(param("p"), param("q"), literal(5.0)), EvaluationVisitor(context)
    ) == pytest.approx(5.0)
    assert visit(
        minimum(param("p"), param("q"), literal(5.0)), EvaluationVisitor(context)
    ) == pytest.approx(1.3)
